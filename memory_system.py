"""
Retrieval-Augmented Memory System for Meta-Tool
================================================
Uses FAISS for efficient similarity search over:
1. Episodic Memory: Successful trajectories from past interactions
2. Schema Memory: Chunked schema definitions for large tools

Retrieval is used to augment the context window during inference.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json
import os

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not installed. Using numpy fallback for retrieval.")


@dataclass
class MemoryEntry:
    """A single entry in the memory system."""
    content: str  # The text content
    embedding: np.ndarray  # Vector embedding
    metadata: Dict[str, Any]  # Additional metadata
    

class EmbeddingModel:
    """
    Wrapper for embedding model.
    Uses the documentation encoder or a dedicated embedding model.
    """
    
    def __init__(self, encoder_model=None, device: str = "cuda"):
        self.device = device
        self.encoder_model = encoder_model
        
        if encoder_model is None:
            # Use sentence-transformers as fallback
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.use_st = True
            except ImportError:
                print("Warning: sentence-transformers not installed. Using random embeddings.")
                self.use_st = False
                self.embedding_dim = 384
        else:
            self.use_st = False
            
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if self.encoder_model is not None:
            with torch.no_grad():
                embeddings = self.encoder_model(texts)
            return embeddings.cpu().numpy()
        elif self.use_st:
            return self.model.encode(texts, convert_to_numpy=True)
        else:
            # Random embeddings for testing
            return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
            
    @property
    def dim(self) -> int:
        """Return embedding dimension."""
        if self.encoder_model is not None:
            return self.encoder_model.config.encoder_dim
        elif self.use_st:
            return self.model.get_sentence_embedding_dimension()
        else:
            return self.embedding_dim


class FAISSIndex:
    """
    FAISS-based vector index for efficient similarity search.
    """
    
    def __init__(
        self,
        dim: int,
        index_type: str = "IVF1024_PQ64",
        metric: str = "cosine"
    ):
        self.dim = dim
        self.metric = metric
        
        if FAISS_AVAILABLE:
            if metric == "cosine":
                # Normalize vectors for cosine similarity
                self.index = faiss.IndexFlatIP(dim)  # Inner product on normalized vectors
            else:
                self.index = faiss.IndexFlatL2(dim)
                
            # For larger indices, use IVF
            if "IVF" in index_type:
                nlist = int(index_type.split("_")[0].replace("IVF", ""))
                quantizer = self.index
                self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        else:
            # Numpy fallback
            self.vectors = []
            
        self.entries: List[MemoryEntry] = []
        self.is_trained = False
        
    def add(self, entries: List[MemoryEntry]):
        """Add entries to the index."""
        if not entries:
            return
            
        embeddings = np.stack([e.embedding for e in entries]).astype(np.float32)
        
        if self.metric == "cosine":
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)
            
        if FAISS_AVAILABLE:
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                self.index.train(embeddings)
            self.index.add(embeddings)
        else:
            self.vectors.extend(embeddings)
            
        self.entries.extend(entries)
        
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Search for nearest neighbors.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            
        Returns:
            List of (entry, score) tuples
        """
        if len(self.entries) == 0:
            return []
            
        query = query_embedding.astype(np.float32).reshape(1, -1)
        
        if self.metric == "cosine":
            norm = np.linalg.norm(query)
            query = query / max(norm, 1e-8)
            
        k = min(k, len(self.entries))
        
        if FAISS_AVAILABLE:
            scores, indices = self.index.search(query, k)
            results = [
                (self.entries[idx], float(score))
                for idx, score in zip(indices[0], scores[0])
                if idx >= 0
            ]
        else:
            # Numpy fallback
            vectors = np.stack(self.vectors)
            if self.metric == "cosine":
                scores = np.dot(vectors, query.T).squeeze()
            else:
                scores = -np.linalg.norm(vectors - query, axis=1)
            top_k = np.argsort(scores)[-k:][::-1]
            results = [(self.entries[idx], float(scores[idx])) for idx in top_k]
            
        return results
        
    def save(self, path: str):
        """Save index to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        if FAISS_AVAILABLE:
            faiss.write_index(self.index, f"{path}.index")
            
        # Save entries
        entries_data = [
            {
                "content": e.content,
                "metadata": e.metadata
            }
            for e in self.entries
        ]
        with open(f"{path}.json", 'w') as f:
            json.dump(entries_data, f)
            
        # Save embeddings
        embeddings = np.stack([e.embedding for e in self.entries])
        np.save(f"{path}.npy", embeddings)
        
    def load(self, path: str):
        """Load index from disk."""
        if FAISS_AVAILABLE and os.path.exists(f"{path}.index"):
            self.index = faiss.read_index(f"{path}.index")
            
        if os.path.exists(f"{path}.json"):
            with open(f"{path}.json") as f:
                entries_data = json.load(f)
                
        if os.path.exists(f"{path}.npy"):
            embeddings = np.load(f"{path}.npy")
            
            self.entries = [
                MemoryEntry(
                    content=e["content"],
                    embedding=embeddings[i],
                    metadata=e["metadata"]
                )
                for i, e in enumerate(entries_data)
            ]


class EpisodicMemory:
    """
    Stores and retrieves successful trajectories.
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        max_entries: int = 10000,
        k_retrieval: int = 3
    ):
        self.embedding_model = embedding_model
        self.max_entries = max_entries
        self.k_retrieval = k_retrieval
        
        self.index = FAISSIndex(
            dim=embedding_model.dim,
            metric="cosine"
        )
        
    def add_trajectory(
        self,
        query: str,
        trajectory: str,
        success: bool = True,
        metadata: Optional[Dict] = None
    ):
        """Add a trajectory to memory."""
        if not success:
            return  # Only store successful trajectories
            
        content = f"Query: {query}\nTrajectory: {trajectory}"
        embedding = self.embedding_model.encode([content])[0]
        
        entry = MemoryEntry(
            content=content,
            embedding=embedding,
            metadata={
                "query": query,
                "trajectory": trajectory,
                "success": success,
                **(metadata or {})
            }
        )
        
        self.index.add([entry])
        
        # Enforce max entries (simple FIFO)
        if len(self.index.entries) > self.max_entries:
            self.index.entries = self.index.entries[-self.max_entries:]
            
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve relevant trajectories for a query.
        
        Args:
            query: User query
            
        Returns:
            List of retrieved trajectories with scores
        """
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = self.index.search(query_embedding, k=self.k_retrieval)
        
        return [
            {
                "query": entry.metadata["query"],
                "trajectory": entry.metadata["trajectory"],
                "score": score
            }
            for entry, score in results
        ]


class SchemaMemory:
    """
    Stores and retrieves schema chunks for large schemas.
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        chunk_size: int = 1000,  # tokens
        k_retrieval: int = 5,
        confidence_threshold: float = 0.7
    ):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.k_retrieval = k_retrieval
        self.confidence_threshold = confidence_threshold
        
        self.index = FAISSIndex(
            dim=embedding_model.dim,
            metric="cosine"
        )
        
        self.full_schema: Optional[str] = None
        
    def index_schema(self, schema: str, schema_dict: Optional[Dict] = None):
        """
        Index a schema by chunking and embedding.
        
        Args:
            schema: Full schema text
            schema_dict: Optional parsed schema for intelligent chunking
        """
        self.full_schema = schema
        
        if schema_dict:
            chunks = self._chunk_by_structure(schema_dict)
        else:
            chunks = self._chunk_by_size(schema)
            
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_model.encode([chunk])[0]
            
            entry = MemoryEntry(
                content=chunk,
                embedding=embedding,
                metadata={
                    "chunk_id": i,
                    "chunk_type": "schema"
                }
            )
            self.index.add([entry])
            
    def _chunk_by_structure(self, schema_dict: Dict) -> List[str]:
        """Chunk schema by logical structure (tables, endpoints, etc.)."""
        chunks = []
        
        # Handle JSON Schema with properties
        if "properties" in schema_dict:
            for prop_name, prop_spec in schema_dict["properties"].items():
                chunk = f"Property: {prop_name}\n{json.dumps(prop_spec, indent=2)}"
                chunks.append(chunk)
                
        # Handle SQL schemas with tables
        if "tables" in schema_dict:
            for table in schema_dict["tables"]:
                chunk = f"Table: {table.get('name', 'unknown')}\n{json.dumps(table, indent=2)}"
                chunks.append(chunk)
                
        # Handle API schemas with endpoints
        if "endpoints" in schema_dict:
            for endpoint in schema_dict["endpoints"]:
                chunk = f"Endpoint: {endpoint.get('path', 'unknown')}\n{json.dumps(endpoint, indent=2)}"
                chunks.append(chunk)
                
        if not chunks:
            # Fallback to full schema
            chunks = [json.dumps(schema_dict, indent=2)]
            
        return chunks
        
    def _chunk_by_size(self, schema: str) -> List[str]:
        """Chunk schema by approximate token size."""
        # Approximate: 4 chars per token
        chars_per_chunk = self.chunk_size * 4
        
        chunks = []
        lines = schema.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            if current_size + len(line) > chars_per_chunk and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
                
            current_chunk.append(line)
            current_size += len(line)
            
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks
        
    def retrieve(self, query: str) -> Tuple[str, float]:
        """
        Retrieve relevant schema chunks for a query.
        
        Args:
            query: User query
            
        Returns:
            Retrieved schema text, confidence score
        """
        if len(self.index.entries) == 0:
            return self.full_schema or "", 0.0
            
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = self.index.search(query_embedding, k=self.k_retrieval)
        
        if not results:
            return self.full_schema or "", 0.0
            
        # Compute confidence based on score gap
        scores = [score for _, score in results]
        top_score = scores[0] if scores else 0.0
        
        # Get additional result for gap computation
        if len(self.index.entries) > self.k_retrieval:
            extra_results = self.index.search(query_embedding, k=self.k_retrieval + 1)
            sixth_score = extra_results[-1][1] if len(extra_results) > self.k_retrieval else 0.0
        else:
            sixth_score = 0.0
            
        confidence = top_score - sixth_score
        
        # If low confidence, return full schema
        if confidence < self.confidence_threshold and self.full_schema:
            return self.full_schema[:4000], confidence
            
        # Combine retrieved chunks
        retrieved_text = "\n\n".join([entry.content for entry, _ in results])
        
        return retrieved_text, confidence


class HybridMemorySystem:
    """
    Combined memory system with episodic and schema memory.
    Handles context assembly and priority-based truncation.
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        max_context_tokens: int = 8192,
        episodic_k: int = 3,
        schema_k: int = 5
    ):
        self.embedding_model = embedding_model
        self.max_context_tokens = max_context_tokens
        
        self.episodic_memory = EpisodicMemory(
            embedding_model,
            k_retrieval=episodic_k
        )
        self.schema_memory = SchemaMemory(
            embedding_model,
            k_retrieval=schema_k
        )
        
    def add_trajectory(self, query: str, trajectory: str, success: bool = True):
        """Add a trajectory to episodic memory."""
        self.episodic_memory.add_trajectory(query, trajectory, success)
        
    def index_schema(self, schema: str, schema_dict: Optional[Dict] = None):
        """Index a tool schema."""
        self.schema_memory.index_schema(schema, schema_dict)
        
    def retrieve_context(
        self,
        query: str,
        tool_description: str = ""
    ) -> str:
        """
        Retrieve and assemble context for a query.
        
        Priority order:
        1. Current query (never truncated)
        2. Retrieved schema
        3. Retrieved trajectories
        4. Tool description summary
        
        Args:
            query: User query
            tool_description: Tool description text
            
        Returns:
            Assembled context string
        """
        components = []
        remaining_tokens = self.max_context_tokens
        
        # 1. Query (never truncated)
        query_text = f"Query: {query}\n"
        components.append(("query", query_text))
        remaining_tokens -= self._estimate_tokens(query_text)
        
        # 2. Schema retrieval
        schema_text, confidence = self.schema_memory.retrieve(query)
        schema_tokens = self._estimate_tokens(schema_text)
        
        if schema_tokens <= remaining_tokens * 0.5:
            components.append(("schema", f"Relevant Schema:\n{schema_text}\n"))
            remaining_tokens -= schema_tokens
        else:
            # Truncate schema
            truncated = schema_text[:int(remaining_tokens * 0.5 * 4)]
            components.append(("schema", f"Relevant Schema (truncated):\n{truncated}\n"))
            remaining_tokens -= self._estimate_tokens(truncated)
            
        # 3. Episodic retrieval
        trajectories = self.episodic_memory.retrieve(query)
        
        for traj in trajectories:
            traj_text = f"Example:\nQuery: {traj['query']}\nTrajectory: {traj['trajectory']}\n"
            traj_tokens = self._estimate_tokens(traj_text)
            
            if traj_tokens <= remaining_tokens:
                components.append(("trajectory", traj_text))
                remaining_tokens -= traj_tokens
            else:
                break
                
        # 4. Tool description (if space)
        if tool_description and remaining_tokens > 200:
            desc_tokens = self._estimate_tokens(tool_description)
            if desc_tokens <= remaining_tokens:
                components.append(("description", f"Tool Description:\n{tool_description}\n"))
            else:
                # Truncate description
                truncated = tool_description[:int(remaining_tokens * 4)]
                components.append(("description", f"Tool Description:\n{truncated}\n"))
                
        # Assemble in order
        context = ""
        for name, text in components:
            context += text + "\n"
            
        return context.strip()
        
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (approximate: 4 chars per token)."""
        return len(text) // 4


# Factory function
def create_memory_system(
    encoder_model=None,
    max_context_tokens: int = 8192
) -> HybridMemorySystem:
    """Create a memory system with the given configuration."""
    embedding_model = EmbeddingModel(encoder_model)
    
    return HybridMemorySystem(
        embedding_model=embedding_model,
        max_context_tokens=max_context_tokens
    )


if __name__ == "__main__":
    # Test the memory system
    memory = create_memory_system()
    
    # Add some trajectories
    memory.add_trajectory(
        "Search for Python tutorials",
        '{"function": "search", "query": "Python tutorials"}',
        success=True
    )
    memory.add_trajectory(
        "Find machine learning courses",
        '{"function": "search", "query": "ML courses", "limit": 10}',
        success=True
    )
    
    # Index a schema
    schema = """
    {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"}
        }
    }
    """
    memory.index_schema(schema)
    
    # Retrieve context
    context = memory.retrieve_context(
        query="Search for deep learning resources",
        tool_description="Search API for finding educational content"
    )
    
    print("Retrieved context:")
    print(context)
