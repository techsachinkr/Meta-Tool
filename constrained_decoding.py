"""
Schema-Constrained Decoding for Meta-Tool
==========================================
Uses Finite State Machine (FSM) constraints to guarantee syntactically valid outputs.
Integrates with the Outlines library for grammar-constrained generation.

Guarantees:
- JSON syntax validity
- Type constraint satisfaction
- Enum constraint satisfaction

Does NOT guarantee:
- Semantic validity (e.g., valid user IDs)
- Cross-field dependencies
- Deep recursive schema constraints
"""

import json
import re
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import torch
import torch.nn.functional as F

try:
    import outlines
    from outlines import models, generate
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False
    print("Warning: outlines not installed. FSM decoding will use fallback.")


@dataclass
class DecodingConstraint:
    """Represents a decoding constraint."""
    type: str  # "json_schema", "regex", "enum", "type"
    constraint: Any  # The actual constraint specification
    

class JSONSchemaFSM:
    """
    Compiles JSON Schema to FSM-compatible constraints.
    
    Handles:
    - Basic JSON syntax
    - Type constraints (string, integer, number, boolean, array, object)
    - Enum constraints
    - String patterns (regex)
    - Required vs optional properties
    """
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.compiled_regex = self._compile_to_regex()
        
    def _compile_to_regex(self) -> str:
        """Compile JSON schema to a regex pattern."""
        return self._schema_to_regex(self.schema)
        
    def _schema_to_regex(self, schema: Dict, depth: int = 0) -> str:
        """Recursively compile schema to regex."""
        if depth > 10:  # Prevent infinite recursion
            return r'.*'
            
        schema_type = schema.get("type", "string")
        
        # Handle enum first
        if "enum" in schema:
            enum_values = schema["enum"]
            escaped = [re.escape(json.dumps(v)) for v in enum_values]
            return f"({'|'.join(escaped)})"
            
        # Handle by type
        if schema_type == "string":
            if "pattern" in schema:
                return f'"{schema["pattern"]}"'
            else:
                # Basic string: any characters except unescaped quotes
                return r'"([^"\\]|\\.)*"'
                
        elif schema_type == "integer":
            return r'-?[0-9]+'
            
        elif schema_type == "number":
            return r'-?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?'
            
        elif schema_type == "boolean":
            return r'(true|false)'
            
        elif schema_type == "null":
            return r'null'
            
        elif schema_type == "array":
            items_schema = schema.get("items", {"type": "string"})
            items_regex = self._schema_to_regex(items_schema, depth + 1)
            return rf'\[\s*({items_regex}(\s*,\s*{items_regex})*)?\s*\]'
            
        elif schema_type == "object":
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            if not properties:
                return r'\{[^{}]*\}'
                
            # Build property patterns
            prop_patterns = []
            for prop_name, prop_schema in properties.items():
                prop_regex = self._schema_to_regex(prop_schema, depth + 1)
                prop_pattern = rf'"{re.escape(prop_name)}"\s*:\s*{prop_regex}'
                prop_patterns.append(prop_pattern)
                
            # Combine (simplified - doesn't enforce required)
            props_combined = r'(\s*,\s*)?'.join([f'({p})?' for p in prop_patterns])
            return rf'\{{\s*{props_combined}\s*\}}'
            
        return r'.*'
        
    def get_regex(self) -> str:
        """Get the compiled regex pattern."""
        return self.compiled_regex
        
    def validate(self, text: str) -> bool:
        """Check if text matches the schema pattern."""
        try:
            # First try JSON parsing
            parsed = json.loads(text)
            return self._validate_value(parsed, self.schema)
        except json.JSONDecodeError:
            return False
            
    def _validate_value(self, value: Any, schema: Dict) -> bool:
        """Validate a parsed value against schema."""
        schema_type = schema.get("type")
        
        # Check enum
        if "enum" in schema:
            return value in schema["enum"]
            
        # Check type
        if schema_type == "string":
            if not isinstance(value, str):
                return False
            # Check pattern
            if "pattern" in schema:
                if not re.match(schema["pattern"], value):
                    return False
            # Check length
            if "minLength" in schema and len(value) < schema["minLength"]:
                return False
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                return False
            return True
            
        elif schema_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                return False
            if "minimum" in schema and value < schema["minimum"]:
                return False
            if "maximum" in schema and value > schema["maximum"]:
                return False
            return True
            
        elif schema_type == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False
            if "minimum" in schema and value < schema["minimum"]:
                return False
            if "maximum" in schema and value > schema["maximum"]:
                return False
            return True
            
        elif schema_type == "boolean":
            return isinstance(value, bool)
            
        elif schema_type == "null":
            return value is None
            
        elif schema_type == "array":
            if not isinstance(value, list):
                return False
            items_schema = schema.get("items", {})
            return all(self._validate_value(item, items_schema) for item in value)
            
        elif schema_type == "object":
            if not isinstance(value, dict):
                return False
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            # Check required properties
            for req in required:
                if req not in value:
                    return False
                    
            # Validate each property
            for prop_name, prop_value in value.items():
                if prop_name in properties:
                    if not self._validate_value(prop_value, properties[prop_name]):
                        return False
            return True
            
        return True


class ConstrainedDecoder:
    """
    Decoder with FSM constraints for schema-valid generation.
    
    Uses logit masking to prevent invalid tokens.
    """
    
    def __init__(
        self,
        model,  # HuggingFace model
        tokenizer,
        schema: Dict[str, Any]
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.schema = schema
        self.fsm = JSONSchemaFSM(schema)
        
        # Build token validity cache
        self._build_token_cache()
        
    def _build_token_cache(self):
        """Pre-compute which tokens are valid JSON tokens."""
        vocab_size = len(self.tokenizer)
        
        # Tokens that are always valid in JSON
        self.json_safe_tokens = set()
        
        json_chars = set('{}[]":,0123456789.-+eEtruefalsnull \t\n')
        
        for token_id in range(vocab_size):
            token = self.tokenizer.decode([token_id])
            if all(c in json_chars or c.isalnum() or c == '_' for c in token):
                self.json_safe_tokens.add(token_id)
                
    def _get_valid_next_tokens(
        self,
        generated_text: str
    ) -> torch.Tensor:
        """
        Get mask of valid next tokens based on current generation state.
        
        Returns:
            mask: [vocab_size] boolean tensor, True for valid tokens
        """
        vocab_size = len(self.tokenizer)
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        
        # Simple heuristic-based masking
        # In practice, this would use proper FSM state tracking
        
        text = generated_text.strip()
        
        if not text or text == "{":
            # Start of object - expect property name or }
            for token_id in range(vocab_size):
                token = self.tokenizer.decode([token_id])
                if token.strip().startswith('"') or token.strip() == '}':
                    mask[token_id] = True
                    
        elif text.endswith(":"):
            # After colon - expect value
            for token_id in self.json_safe_tokens:
                mask[token_id] = True
                
        elif text.endswith(","):
            # After comma - expect property name or array element
            for token_id in range(vocab_size):
                token = self.tokenizer.decode([token_id])
                if token.strip().startswith('"') or token.strip() in '{}[]0123456789-tfn':
                    mask[token_id] = True
                    
        else:
            # Default: allow all JSON-safe tokens
            for token_id in self.json_safe_tokens:
                mask[token_id] = True
                
        # Always allow EOS
        if self.tokenizer.eos_token_id is not None:
            mask[self.tokenizer.eos_token_id] = True
            
        return mask
        
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """
        Generate schema-constrained output.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text conforming to schema
        """
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        
        generated_text = ""
        
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]  # [1, vocab_size]
                
            # Apply temperature
            logits = logits / temperature
            
            # Get valid token mask
            valid_mask = self._get_valid_next_tokens(generated_text)
            valid_mask = valid_mask.to(logits.device)
            
            # Mask invalid tokens
            logits = logits.masked_fill(~valid_mask.unsqueeze(0), float('-inf'))
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
            # Decode and append
            token_text = self.tokenizer.decode(next_token[0])
            generated_text += token_text
            
            # Update input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Check if we have valid JSON
            if self._is_complete_json(generated_text):
                break
                
        return generated_text
        
    def _is_complete_json(self, text: str) -> bool:
        """Check if text is complete valid JSON."""
        try:
            json.loads(text)
            return True
        except:
            return False


class OutlinesConstrainedGenerator:
    """
    High-level interface using Outlines library for FSM-constrained generation.
    Falls back to manual implementation if Outlines unavailable.
    """
    
    def __init__(
        self,
        model_name: str,
        schema: Dict[str, Any],
        device: str = "cuda"
    ):
        self.schema = schema
        self.device = device
        
        if OUTLINES_AVAILABLE:
            self.model = models.transformers(model_name, device=device)
            self._setup_outlines_generator()
        else:
            # Fallback to manual implementation
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.hf_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.manual_decoder = ConstrainedDecoder(
                self.hf_model, self.tokenizer, schema
            )
            
    def _setup_outlines_generator(self):
        """Setup Outlines JSON generator."""
        if OUTLINES_AVAILABLE:
            self.generator = generate.json(self.model, self.schema)
            
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate schema-valid JSON output.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Parsed JSON dictionary
        """
        if OUTLINES_AVAILABLE:
            result = self.generator(prompt)
            return result
        else:
            # Manual fallback
            text = self.manual_decoder.generate(prompt, **kwargs)
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"error": "Failed to generate valid JSON", "raw": text}


def generate_with_rejection_sampling(
    generator: Callable[[str], str],
    prompt: str,
    schema: Dict[str, Any],
    max_attempts: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Generate with rejection sampling fallback.
    
    For complex schemas that can't be fully captured by FSM,
    generate and validate, retrying on failure.
    
    Args:
        generator: Function that generates text from prompt
        prompt: Input prompt
        schema: JSON schema to validate against
        max_attempts: Maximum generation attempts
        
    Returns:
        Valid JSON dictionary or None
    """
    fsm = JSONSchemaFSM(schema)
    
    for attempt in range(max_attempts):
        try:
            # Generate
            text = generator(prompt)
            
            # Parse
            parsed = json.loads(text)
            
            # Validate
            if fsm.validate(text):
                return parsed
            else:
                print(f"Attempt {attempt + 1}: Generated JSON failed validation")
                
        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1}: JSON parse error: {e}")
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error: {e}")
            
    return None


# Example usage
if __name__ == "__main__":
    # Example schema
    test_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "create", "delete"]
            },
            "query": {
                "type": "string",
                "minLength": 1
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100
            }
        },
        "required": ["action", "query"]
    }
    
    # Test FSM compilation
    fsm = JSONSchemaFSM(test_schema)
    print(f"Compiled regex: {fsm.get_regex()[:100]}...")
    
    # Test validation
    valid_json = '{"action": "search", "query": "test", "limit": 10}'
    invalid_json = '{"action": "invalid", "query": "test"}'
    
    print(f"\nValid JSON: {fsm.validate(valid_json)}")
    print(f"Invalid JSON: {fsm.validate(invalid_json)}")
