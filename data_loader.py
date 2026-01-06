"""
ToolBench Data Loader for Meta-Tool
===================================
Loads real tool data from ToolBench dataset for meta-training.

ToolBench contains:
- 16,464 REST APIs across 3,451 tools from RapidAPI Hub
- 49 tool categories (Sports, Finance, Weather, etc.)
- 126,486 instruction-solution pairs with API call trajectories
- G1/G2/G3 complexity levels (single-tool, intra-category, cross-category)

Data sources:
1. Local ToolBench installation (recommended)
2. HuggingFace datasets fallback
3. Direct Google Drive download

Reference: https://github.com/OpenBMB/ToolBench (ICLR 2024 Spotlight)
"""

# Suppress TensorFlow/JAX warnings BEFORE any imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['JAX_PLATFORMS'] = ''

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import json
import random
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ToolAPI:
    """Represents a single API endpoint within a tool."""
    name: str
    description: str
    method: str  # GET, POST, etc.
    url: str
    required_parameters: List[Dict[str, Any]]
    optional_parameters: List[Dict[str, Any]]
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        properties = {}
        required = []
        
        for param in self.required_parameters:
            properties[param["name"]] = {
                "type": param.get("type", "string"),
                "description": param.get("description", "")
            }
            if "enum" in param:
                properties[param["name"]]["enum"] = param["enum"]
            required.append(param["name"])
            
        for param in self.optional_parameters:
            properties[param["name"]] = {
                "type": param.get("type", "string"),
                "description": param.get("description", "")
            }
            if "enum" in param:
                properties[param["name"]]["enum"] = param["enum"]
                
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }


@dataclass
class Tool:
    """Represents a complete tool with multiple APIs."""
    name: str
    description: str
    category: str
    api_list: List[ToolAPI]
    standardized_name: str = ""
    
    # For meta-learning
    examples: List[Tuple[str, str]] = field(default_factory=list)
    
    @property
    def documentation(self) -> str:
        """Generate documentation string for the hypernetwork."""
        doc = f"# {self.name}\n\n"
        doc += f"Category: {self.category}\n"
        doc += f"Description: {self.description}\n\n"
        doc += "## Available APIs:\n\n"
        
        for api in self.api_list:
            doc += f"### {api.name}\n"
            doc += f"- Method: {api.method}\n"
            doc += f"- Description: {api.description}\n"
            doc += f"- URL: {api.url}\n"
            
            if api.required_parameters:
                doc += "- Required Parameters:\n"
                for param in api.required_parameters:
                    doc += f"  - {param['name']} ({param.get('type', 'string')}): {param.get('description', '')}\n"
                    
            if api.optional_parameters:
                doc += "- Optional Parameters:\n"
                for param in api.optional_parameters:
                    doc += f"  - {param['name']} ({param.get('type', 'string')}): {param.get('description', '')}\n"
            doc += "\n"
            
        return doc
    
    @property
    def schema(self) -> Dict[str, Any]:
        """Generate combined schema for all APIs."""
        return {
            "tool_name": self.name,
            "apis": {api.name: api.to_schema() for api in self.api_list}
        }


@dataclass
class TrajectoryExample:
    """A single query-trajectory example from ToolBench."""
    query: str
    trajectory: str  # Serialized action sequence
    tool_names: List[str]
    category: str
    complexity: str  # G1, G2, or G3
    success: bool = True


class ToolBenchLoader:
    """
    Loader for ToolBench dataset.
    
    Supports multiple data sources:
    1. Local installation from GitHub
    2. HuggingFace datasets
    3. Direct download from Google Drive
    """
    
    def __init__(
        self,
        data_root: str = "./data/toolbench",
        use_huggingface: bool = False,
        download_if_missing: bool = True
    ):
        self.data_root = Path(data_root)
        self.use_huggingface = use_huggingface
        self.download_if_missing = download_if_missing
        
        # Data containers
        self.tools_by_category: Dict[str, Dict[str, Tool]] = {}
        self.all_tools: Dict[str, Tool] = {}
        self.training_examples: List[TrajectoryExample] = []
        
        # Category mapping
        self.categories: List[str] = []
        
    def setup(self):
        """Initialize the data loader and download data if needed."""
        if self.use_huggingface:
            self._load_from_huggingface()
        else:
            if not self._check_local_data():
                if self.download_if_missing:
                    self._download_toolbench()
                else:
                    raise FileNotFoundError(
                        f"ToolBench data not found at {self.data_root}. "
                        "Set download_if_missing=True or use_huggingface=True"
                    )
            self._load_local_data()
            
    def _check_local_data(self) -> bool:
        """Check if local ToolBench data exists."""
        tools_dir = self.data_root / "toolenv" / "tools"
        train_file = self.data_root / "toolllama_G123_dfs_train.json"
        
        return tools_dir.exists() or train_file.exists()
        
    def _download_toolbench(self):
        """Download ToolBench dataset."""
        logger.info("Downloading ToolBench dataset...")
        
        os.makedirs(self.data_root, exist_ok=True)
        
        # Try multiple download methods
        try:
            self._download_via_gdown()
        except Exception as e:
            logger.warning(f"gdown failed: {e}, trying wget...")
            try:
                self._download_via_wget()
            except Exception as e2:
                logger.warning(f"wget failed: {e2}, falling back to HuggingFace")
                self.use_huggingface = True
                self._load_from_huggingface()
                
    def _download_via_gdown(self):
        """Download using gdown (Google Drive)."""
        try:
            import gdown
        except ImportError:
            raise ImportError("Please install gdown: pip install gdown")
            
        # ToolBench data.zip Google Drive ID
        file_id = "1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk"
        output = self.data_root / "data.zip"
        
        gdown.download(id=file_id, output=str(output), quiet=False)
        
        # Extract
        import zipfile
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(self.data_root)
            
        os.remove(output)
        logger.info("ToolBench download complete")
        
    def _download_via_wget(self):
        """Download using wget."""
        import subprocess
        
        url = "https://drive.google.com/uc?export=download&id=1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk&confirm=yes"
        output = self.data_root / "data.zip"
        
        subprocess.run([
            "wget", "--no-check-certificate", url, "-O", str(output)
        ], check=True)
        
        # Extract
        import zipfile
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(self.data_root)
            
        os.remove(output)
        
    def _load_from_huggingface(self):
        """Load ToolBench from HuggingFace datasets."""
        logger.info("Loading ToolBench from HuggingFace...")
        
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        dataset = None
        errors = []
        
        # Try multiple dataset sources
        sources = [
            ("Maurus/ToolBench", None),
            ("Maurus/ToolBench", "train"),
        ]
        
        for source, split in sources:
            try:
                if split:
                    dataset = load_dataset(source, split=split)
                    dataset = {"train": dataset}
                else:
                    dataset = load_dataset(source)
                logger.info(f"Successfully loaded {source}")
                break
            except Exception as e:
                errors.append(f"{source}: {e}")
                continue
        
        if dataset is None:
            # All sources failed - create synthetic data instead
            logger.warning(f"Could not load from HuggingFace: {errors}")
            logger.info("Falling back to synthetic tool generation...")
            self._create_synthetic_toolbench()
            return
        
        # Process into our format
        self._process_huggingface_data(dataset)
        
    def _create_synthetic_toolbench(self):
        """Create synthetic ToolBench-like data when HuggingFace is inaccessible."""
        logger.info("Creating synthetic ToolBench data...")
        
        # Define realistic API categories and tools
        synthetic_tools = [
            {
                "category": "Weather",
                "tools": [
                    {"name": "OpenWeatherMap", "apis": [
                        {"name": "get_current_weather", "desc": "Get current weather for a location", 
                         "params": [{"name": "city", "type": "string"}, {"name": "units", "type": "string"}]},
                        {"name": "get_forecast", "desc": "Get weather forecast",
                         "params": [{"name": "city", "type": "string"}, {"name": "days", "type": "integer"}]}
                    ]},
                    {"name": "WeatherAPI", "apis": [
                        {"name": "current", "desc": "Current weather data",
                         "params": [{"name": "q", "type": "string"}]},
                        {"name": "forecast", "desc": "Weather forecast",
                         "params": [{"name": "q", "type": "string"}, {"name": "days", "type": "integer"}]}
                    ]}
                ]
            },
            {
                "category": "Finance",
                "tools": [
                    {"name": "StockAPI", "apis": [
                        {"name": "get_quote", "desc": "Get stock quote",
                         "params": [{"name": "symbol", "type": "string"}]},
                        {"name": "get_history", "desc": "Get historical prices",
                         "params": [{"name": "symbol", "type": "string"}, {"name": "period", "type": "string"}]}
                    ]},
                    {"name": "CurrencyExchange", "apis": [
                        {"name": "convert", "desc": "Convert currency",
                         "params": [{"name": "from", "type": "string"}, {"name": "to", "type": "string"}, {"name": "amount", "type": "number"}]},
                        {"name": "rates", "desc": "Get exchange rates",
                         "params": [{"name": "base", "type": "string"}]}
                    ]}
                ]
            },
            {
                "category": "Search",
                "tools": [
                    {"name": "WebSearch", "apis": [
                        {"name": "search", "desc": "Search the web",
                         "params": [{"name": "query", "type": "string"}, {"name": "num_results", "type": "integer"}]},
                        {"name": "images", "desc": "Search for images",
                         "params": [{"name": "query", "type": "string"}]}
                    ]},
                    {"name": "NewsAPI", "apis": [
                        {"name": "headlines", "desc": "Get top headlines",
                         "params": [{"name": "country", "type": "string"}, {"name": "category", "type": "string"}]},
                        {"name": "search_news", "desc": "Search news articles",
                         "params": [{"name": "q", "type": "string"}]}
                    ]}
                ]
            },
            {
                "category": "Communication",
                "tools": [
                    {"name": "EmailService", "apis": [
                        {"name": "send_email", "desc": "Send an email",
                         "params": [{"name": "to", "type": "string"}, {"name": "subject", "type": "string"}, {"name": "body", "type": "string"}]},
                        {"name": "check_inbox", "desc": "Check email inbox",
                         "params": [{"name": "folder", "type": "string"}]}
                    ]},
                    {"name": "SMSGateway", "apis": [
                        {"name": "send_sms", "desc": "Send SMS message",
                         "params": [{"name": "phone", "type": "string"}, {"name": "message", "type": "string"}]},
                        {"name": "check_status", "desc": "Check message status",
                         "params": [{"name": "message_id", "type": "string"}]}
                    ]}
                ]
            },
            {
                "category": "Data",
                "tools": [
                    {"name": "DatabaseAPI", "apis": [
                        {"name": "query", "desc": "Execute database query",
                         "params": [{"name": "sql", "type": "string"}]},
                        {"name": "insert", "desc": "Insert data",
                         "params": [{"name": "table", "type": "string"}, {"name": "data", "type": "object"}]}
                    ]},
                    {"name": "StorageService", "apis": [
                        {"name": "upload", "desc": "Upload a file",
                         "params": [{"name": "file", "type": "string"}, {"name": "path", "type": "string"}]},
                        {"name": "download", "desc": "Download a file",
                         "params": [{"name": "path", "type": "string"}]}
                    ]}
                ]
            },
            {
                "category": "Social",
                "tools": [
                    {"name": "TwitterAPI", "apis": [
                        {"name": "post_tweet", "desc": "Post a tweet",
                         "params": [{"name": "text", "type": "string"}]},
                        {"name": "search_tweets", "desc": "Search tweets",
                         "params": [{"name": "query", "type": "string"}, {"name": "count", "type": "integer"}]}
                    ]},
                    {"name": "InstagramAPI", "apis": [
                        {"name": "get_profile", "desc": "Get user profile",
                         "params": [{"name": "username", "type": "string"}]},
                        {"name": "get_posts", "desc": "Get user posts",
                         "params": [{"name": "username", "type": "string"}, {"name": "limit", "type": "integer"}]}
                    ]}
                ]
            },
            {
                "category": "Maps",
                "tools": [
                    {"name": "GoogleMaps", "apis": [
                        {"name": "geocode", "desc": "Convert address to coordinates",
                         "params": [{"name": "address", "type": "string"}]},
                        {"name": "directions", "desc": "Get directions",
                         "params": [{"name": "origin", "type": "string"}, {"name": "destination", "type": "string"}]}
                    ]},
                    {"name": "PlacesAPI", "apis": [
                        {"name": "nearby_search", "desc": "Search nearby places",
                         "params": [{"name": "location", "type": "string"}, {"name": "type", "type": "string"}, {"name": "radius", "type": "integer"}]},
                        {"name": "place_details", "desc": "Get place details",
                         "params": [{"name": "place_id", "type": "string"}]}
                    ]}
                ]
            },
            {
                "category": "AI",
                "tools": [
                    {"name": "TextAnalysis", "apis": [
                        {"name": "sentiment", "desc": "Analyze sentiment",
                         "params": [{"name": "text", "type": "string"}]},
                        {"name": "summarize", "desc": "Summarize text",
                         "params": [{"name": "text", "type": "string"}, {"name": "max_length", "type": "integer"}]}
                    ]},
                    {"name": "TranslationAPI", "apis": [
                        {"name": "translate", "desc": "Translate text",
                         "params": [{"name": "text", "type": "string"}, {"name": "source", "type": "string"}, {"name": "target", "type": "string"}]},
                        {"name": "detect_language", "desc": "Detect language",
                         "params": [{"name": "text", "type": "string"}]}
                    ]}
                ]
            }
        ]
        
        # Query templates for each category
        query_templates = {
            "Weather": [
                "What's the weather in {city}?",
                "Get the current temperature in {city}",
                "Show me the weather forecast for {city}",
                "Is it going to rain in {city} tomorrow?",
            ],
            "Finance": [
                "What's the stock price of {symbol}?",
                "Convert {amount} USD to EUR",
                "Show me the historical prices for {symbol}",
                "What are the current exchange rates?",
            ],
            "Search": [
                "Search for {query} on the web",
                "Find images of {query}",
                "Get the latest news about {topic}",
                "What are today's top headlines?",
            ],
            "Communication": [
                "Send an email to {recipient} about {subject}",
                "Check my inbox",
                "Send a text message to {phone}",
                "What's the status of my message?",
            ],
            "Data": [
                "Query the database for {query}",
                "Insert {data} into the {table} table",
                "Upload {file} to storage",
                "Download the file at {path}",
            ],
            "Social": [
                "Post '{text}' to Twitter",
                "Search tweets about {topic}",
                "Get the Instagram profile for {username}",
                "Show me the latest posts from {username}",
            ],
            "Maps": [
                "Find the coordinates of {address}",
                "Get directions from {origin} to {destination}",
                "Find {type} places near {location}",
                "Get details for place {place_id}",
            ],
            "AI": [
                "Analyze the sentiment of '{text}'",
                "Summarize this text: {text}",
                "Translate '{text}' to {language}",
                "What language is this: {text}?",
            ],
        }
        
        # Generate tools and examples
        for cat_data in synthetic_tools:
            category = cat_data["category"]
            self.tools_by_category[category] = {}
            self.categories.append(category)
            
            for tool_data in cat_data["tools"]:
                tool_name = tool_data["name"]
                api_list = []
                
                for api_data in tool_data["apis"]:
                    api = ToolAPI(
                        name=api_data["name"],
                        description=api_data["desc"],
                        method="GET",
                        url=f"https://api.{tool_name.lower()}.com/{api_data['name']}",
                        required_parameters=[{"name": p["name"], "type": p["type"]} for p in api_data["params"]],
                        optional_parameters=[]
                    )
                    api_list.append(api)
                
                tool = Tool(
                    name=tool_name,
                    description=f"{tool_name} API for {category}",
                    category=category,
                    api_list=api_list,
                    standardized_name=tool_name.lower()
                )
                
                # Generate examples for this tool
                templates = query_templates.get(category, ["Use {tool} to {action}"])
                for i, template in enumerate(templates * 5):  # 20 examples per tool
                    # Fill in template with dummy values
                    query = template.format(
                        city="New York", symbol="AAPL", amount="100",
                        query="machine learning", topic="AI", recipient="user@example.com",
                        subject="Meeting", phone="+1234567890", data="record",
                        table="users", file="document.pdf", path="/files/doc.pdf",
                        text="Hello world", username="johndoe", address="123 Main St",
                        origin="NYC", destination="LA", type="restaurant",
                        location="downtown", place_id="abc123", language="Spanish",
                        tool=tool_name, action="process data"
                    )
                    
                    # Create trajectory
                    api = api_list[i % len(api_list)]
                    trajectory = json.dumps({
                        "thought": f"I'll use {api.name} from {tool_name}",
                        "action": api.name,
                        "action_input": {p["name"]: f"value_{p['name']}" for p in api.required_parameters}
                    })
                    
                    tool.examples.append((query, trajectory))
                    
                    self.training_examples.append(TrajectoryExample(
                        query=query,
                        trajectory=trajectory,
                        tool_names=[tool_name],
                        category=category,
                        complexity="G1"
                    ))
                
                self.tools_by_category[category][tool_name] = tool
                self.all_tools[tool_name] = tool
        
        logger.info(f"Created {len(self.all_tools)} synthetic tools with {len(self.training_examples)} examples")
        
    def _process_huggingface_data(self, dataset):
        """
        Process HuggingFace dataset into our format.
        
        Maurus/ToolBench schema (from Data Studio):
        - api_list: List of dicts with tool/API info (may be string or list)
        - query: User query string
        - query_id: Unique ID
        - domain: Category-Tool string (e.g., "Logistics-SQUAKE")
        - embedding: Vector (ignore)
        """
        logger.info("Processing HuggingFace data...")
        
        # Handle both dict and DatasetDict
        if hasattr(dataset, 'keys'):
            splits = list(dataset.keys())
        else:
            splits = ['train']
            dataset = {'train': dataset}
        
        # Debug: check first sample structure
        first_sample_logged = False
        parse_errors = 0
        processed_count = 0
        skipped_no_api = 0
        skipped_no_query = 0
        skipped_no_tool = 0
        
        for split_name in splits:
            split = dataset[split_name]
            
            for sample in tqdm(split, desc=f"Processing {split_name}"):
                try:
                    # Convert to regular dict if needed (handles Arrow objects)
                    if hasattr(sample, 'items'):
                        sample = dict(sample)
                    
                    # Debug first sample
                    if not first_sample_logged:
                        logger.info(f"Sample keys: {list(sample.keys())}")
                        api_list_raw = sample.get('api_list', None)
                        logger.info(f"api_list type: {type(api_list_raw)}")
                        if api_list_raw is not None:
                            if isinstance(api_list_raw, (list, tuple)) and len(api_list_raw) > 0:
                                first_item = api_list_raw[0]
                                logger.info(f"api_list[0] type: {type(first_item)}")
                                # Convert to dict if it's a special type
                                if hasattr(first_item, 'items'):
                                    first_item = dict(first_item)
                                elif hasattr(first_item, '_asdict'):
                                    first_item = first_item._asdict()
                                logger.info(f"api_list[0] keys: {list(first_item.keys()) if isinstance(first_item, dict) else 'N/A'}")
                                logger.info(f"api_list[0]: {first_item}")
                            elif isinstance(api_list_raw, str):
                                logger.info(f"api_list (string, first 300 chars): {api_list_raw[:300]}")
                        logger.info(f"query: {str(sample.get('query', ''))[:100]}")
                        logger.info(f"domain: {sample.get('domain', '')}")
                        first_sample_logged = True
                    
                    # Parse the api_list field - it contains tool definitions
                    api_list_raw = sample.get("api_list", None)
                    
                    # Handle different formats
                    if api_list_raw is None:
                        skipped_no_api += 1
                        continue
                    
                    # Convert to list
                    api_list = self._convert_to_list(api_list_raw)
                    
                    if not api_list:
                        skipped_no_api += 1
                        continue
                    
                    # Get the query
                    query = str(sample.get("query", "") or "").strip()
                    if not query:
                        skipped_no_query += 1
                        continue
                    
                    # Extract domain info (format: "Category-ToolName")
                    domain = str(sample.get("domain", "General-Unknown") or "General-Unknown")
                    if "-" in domain:
                        default_category = domain.split("-")[0]
                    else:
                        default_category = domain
                    
                    # Process each API in the list
                    for api_info in api_list:
                        # Convert to dict if needed
                        api_info = self._convert_to_dict(api_info)
                        if not api_info:
                            continue
                            
                        tool_name = str(api_info.get("tool_name", "") or "").strip()
                        api_name = str(api_info.get("api_name", "") or "").strip()
                        category_name = str(api_info.get("category_name", default_category) or default_category).strip()
                        
                        if not tool_name:
                            skipped_no_tool += 1
                            continue
                        
                        # Create unique tool key
                        tool_key = tool_name
                        
                        # Initialize category if needed
                        if category_name not in self.tools_by_category:
                            self.tools_by_category[category_name] = {}
                            self.categories.append(category_name)
                        
                        # Create or update tool
                        if tool_key not in self.tools_by_category[category_name]:
                            # Create API object
                            api = ToolAPI(
                                name=api_name if api_name else "default_api",
                                description=str(api_info.get("api_description", "") or ""),
                                method=str(api_info.get("method", "GET") or "GET"),
                                url=str(api_info.get("url", "") or ""),
                                required_parameters=self._convert_to_list(api_info.get("required_parameters")) or [],
                                optional_parameters=self._convert_to_list(api_info.get("optional_parameters")) or []
                            )
                            
                            tool = Tool(
                                name=tool_name,
                                description=f"API tool: {tool_name}",
                                category=category_name,
                                api_list=[api],
                                standardized_name=tool_name.lower().replace(" ", "_")
                            )
                            self.tools_by_category[category_name][tool_key] = tool
                            self.all_tools[tool_key] = tool
                        else:
                            # Add API to existing tool if not duplicate
                            tool = self.tools_by_category[category_name][tool_key]
                            if api_name:
                                existing_api_names = {a.name for a in tool.api_list}
                                if api_name not in existing_api_names:
                                    api = ToolAPI(
                                        name=api_name,
                                        description=str(api_info.get("api_description", "") or ""),
                                        method=str(api_info.get("method", "GET") or "GET"),
                                        url=str(api_info.get("url", "") or ""),
                                        required_parameters=self._convert_to_list(api_info.get("required_parameters")) or [],
                                        optional_parameters=self._convert_to_list(api_info.get("optional_parameters")) or []
                                    )
                                    tool.api_list.append(api)
                        
                        # Create trajectory from API info
                        trajectory = self._generate_trajectory_from_api(api_info)
                        
                        tool = self.tools_by_category[category_name][tool_key]
                        tool.examples.append((query, trajectory))
                        
                        # Also add to training examples
                        example = TrajectoryExample(
                            query=query,
                            trajectory=trajectory,
                            tool_names=[tool_name],
                            category=category_name,
                            complexity="G1"
                        )
                        self.training_examples.append(example)
                        processed_count += 1
                            
                except Exception as e:
                    parse_errors += 1
                    if parse_errors <= 5:
                        logger.warning(f"Error processing sample: {e}")
                        import traceback
                        traceback.print_exc()
                    continue
        
        logger.info(f"Parsing stats:")
        logger.info(f"  - Processed: {processed_count}")
        logger.info(f"  - Skipped (no api_list): {skipped_no_api}")
        logger.info(f"  - Skipped (no query): {skipped_no_query}")
        logger.info(f"  - Skipped (no tool_name): {skipped_no_tool}")
        logger.info(f"  - Parse errors: {parse_errors}")
        
        logger.info(f"Loaded {len(self.all_tools)} tools, {len(self.training_examples)} examples")
        
        # If no tools loaded, fall back to synthetic
        if len(self.all_tools) == 0:
            logger.warning("No tools extracted from HuggingFace data - falling back to synthetic")
            self._create_synthetic_toolbench()
    
    def _convert_to_list(self, value) -> list:
        """Convert various types to a Python list."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, (tuple, set)):
            return list(value)
        if isinstance(value, str):
            # Try JSON parsing
            value = value.strip()
            if value.startswith('['):
                try:
                    return json.loads(value)
                except:
                    pass
            # Try literal eval
            try:
                import ast
                result = ast.literal_eval(value)
                if isinstance(result, (list, tuple)):
                    return list(result)
            except:
                pass
            return []
        # Handle numpy arrays or similar
        if hasattr(value, 'tolist'):
            return value.tolist()
        # Handle iterables
        try:
            return list(value)
        except:
            return []
    
    def _convert_to_dict(self, value) -> dict:
        """Convert various types to a Python dict."""
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            # Try JSON parsing
            value = value.strip()
            if value.startswith('{'):
                try:
                    return json.loads(value)
                except:
                    pass
            # Try literal eval
            try:
                import ast
                result = ast.literal_eval(value)
                if isinstance(result, dict):
                    return result
            except:
                pass
            return {}
        # Handle named tuples or similar
        if hasattr(value, '_asdict'):
            return value._asdict()
        if hasattr(value, 'items'):
            return dict(value)
        # Handle objects with __dict__
        if hasattr(value, '__dict__'):
            return value.__dict__
        return {}
        
    def _generate_trajectory_from_api(self, api_info: Dict) -> str:
        """Generate a trajectory JSON from API info."""
        tool_name = api_info.get("tool_name", "unknown")
        api_name = api_info.get("api_name", "unknown")
        
        # Build parameters from required and optional
        params = {}
        for param in api_info.get("required_parameters", []):
            if isinstance(param, dict):
                param_name = param.get("name", "param")
                param_type = param.get("type", "string")
                default_val = param.get("default", param.get("example_value", f"<{param_name}>"))
                params[param_name] = default_val
                
        for param in api_info.get("optional_parameters", []):
            if isinstance(param, dict):
                param_name = param.get("name", "param")
                default_val = param.get("default", param.get("example_value"))
                if default_val is not None:
                    params[param_name] = default_val
        
        trajectory = {
            "thought": f"I need to use the {api_name} API from {tool_name}",
            "action": api_name,
            "action_input": params
        }
        
        return json.dumps(trajectory)
        
    def _load_local_data(self):
        """Load ToolBench from local installation."""
        logger.info("Loading local ToolBench data...")
        
        # Load tool documentation
        self._load_tool_documentation()
        
        # Load training trajectories
        self._load_training_data()
        
        logger.info(f"Loaded {len(self.all_tools)} tools across {len(self.categories)} categories")
        logger.info(f"Loaded {len(self.training_examples)} training examples")
        
    def _load_tool_documentation(self):
        """Load tool documentation from toolenv/tools/ directory."""
        tools_dir = self.data_root / "toolenv" / "tools"
        
        if not tools_dir.exists():
            # Try alternative path
            tools_dir = self.data_root / "data" / "toolenv" / "tools"
            
        if not tools_dir.exists():
            logger.warning(f"Tools directory not found: {tools_dir}")
            return
            
        for category_dir in tools_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            category = category_dir.name
            self.tools_by_category[category] = {}
            self.categories.append(category)
            
            for tool_file in category_dir.glob("*.json"):
                try:
                    with open(tool_file, 'r', encoding='utf-8') as f:
                        tool_data = json.load(f)
                        
                    tool = self._parse_tool_json(tool_data, category)
                    if tool:
                        self.tools_by_category[category][tool.name] = tool
                        self.all_tools[tool.name] = tool
                        
                except Exception as e:
                    logger.warning(f"Error loading {tool_file}: {e}")
                    
    def _parse_tool_json(self, data: Dict, category: str) -> Optional[Tool]:
        """Parse tool JSON into Tool object."""
        try:
            api_list = []
            for api_data in data.get("api_list", []):
                api = ToolAPI(
                    name=api_data.get("name", "unknown"),
                    description=api_data.get("description", ""),
                    method=api_data.get("method", "GET"),
                    url=api_data.get("url", ""),
                    required_parameters=api_data.get("required_parameters", []),
                    optional_parameters=api_data.get("optional_parameters", [])
                )
                api_list.append(api)
                
            return Tool(
                name=data.get("tool_name", "unknown"),
                description=data.get("tool_description", ""),
                category=category,
                api_list=api_list,
                standardized_name=data.get("standardized_name", "")
            )
        except Exception as e:
            logger.warning(f"Error parsing tool: {e}")
            return None
            
    def _load_training_data(self):
        """Load training trajectories from JSON files."""
        # Try multiple possible paths
        possible_paths = [
            self.data_root / "toolllama_G123_dfs_train.json",
            self.data_root / "data" / "toolllama_G123_dfs_train.json",
            self.data_root / "instruction" / "G1_query.json",
        ]
        
        train_file = None
        for path in possible_paths:
            if path.exists():
                train_file = path
                break
                
        if train_file is None:
            logger.warning("Training data file not found")
            return
            
        logger.info(f"Loading training data from {train_file}")
        
        with open(train_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Process based on file format
        if isinstance(data, list):
            for item in tqdm(data, desc="Loading trajectories"):
                example = self._parse_training_example(item)
                if example:
                    self.training_examples.append(example)
                    self._add_example_to_tool(example)
                    
    def _parse_training_example(self, item: Dict) -> Optional[TrajectoryExample]:
        """Parse a training example from ToolBench format."""
        try:
            # Handle conversation format
            if "conversations" in item:
                return self._parse_conversation_format(item)
            
            # Handle simple format
            query = item.get("query", item.get("instruction", ""))
            answer = item.get("answer", item.get("response", ""))
            
            if not query or not answer:
                return None
                
            # Extract tool names from the example
            tool_names = item.get("tool_names", [])
            if not tool_names and "api_list" in item:
                tool_names = list(set(api.get("tool_name", "") for api in item["api_list"]))
                
            # Determine complexity
            complexity = "G1"
            if "G2" in str(item.get("id", "")):
                complexity = "G2"
            elif "G3" in str(item.get("id", "")):
                complexity = "G3"
                
            return TrajectoryExample(
                query=query,
                trajectory=answer if isinstance(answer, str) else json.dumps(answer),
                tool_names=tool_names,
                category=item.get("category", "General"),
                complexity=complexity
            )
            
        except Exception as e:
            logger.debug(f"Error parsing example: {e}")
            return None
            
    def _parse_conversation_format(self, item: Dict) -> Optional[TrajectoryExample]:
        """Parse ToolBench conversation format (multi-turn with thoughts/actions)."""
        conversations = item.get("conversations", [])
        
        query = ""
        trajectory_parts = []
        tool_names = []
        
        for turn in conversations:
            role = turn.get("from", "")
            content = turn.get("value", "")
            
            if role == "user":
                query = content
            elif role == "assistant":
                trajectory_parts.append(content)
                # Extract tool/action names
                action_match = re.search(r'Action:\s*(\w+)', content)
                if action_match:
                    tool_names.append(action_match.group(1))
            elif role == "function":
                trajectory_parts.append(f"Observation: {content}")
                
        if not query or not trajectory_parts:
            return None
            
        return TrajectoryExample(
            query=query,
            trajectory="\n".join(trajectory_parts),
            tool_names=list(set(tool_names)),
            category=item.get("category", "General"),
            complexity=self._infer_complexity(item)
        )
        
    def _infer_complexity(self, item: Dict) -> str:
        """Infer G1/G2/G3 complexity from item."""
        item_id = str(item.get("id", ""))
        if "G3" in item_id:
            return "G3"
        elif "G2" in item_id:
            return "G2"
        return "G1"
        
    def _add_example_to_tool(self, example: TrajectoryExample):
        """Add example to relevant tool(s)."""
        for tool_name in example.tool_names:
            if tool_name in self.all_tools:
                self.all_tools[tool_name].examples.append(
                    (example.query, example.trajectory)
                )
                
    def get_tools_by_category(self, category: str) -> List[Tool]:
        """Get all tools in a category."""
        return list(self.tools_by_category.get(category, {}).values())
        
    def get_all_tools(self) -> List[Tool]:
        """Get all loaded tools."""
        return list(self.all_tools.values())
        
    def get_tools_with_examples(self, min_examples: int = 5) -> List[Tool]:
        """Get tools that have at least min_examples."""
        return [t for t in self.all_tools.values() if len(t.examples) >= min_examples]
        
    def create_meta_episode(
        self,
        support_k: int = 10,
        query_k: int = 5,
        held_out_category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a meta-learning episode for few-shot tool adaptation.
        
        Args:
            support_k: Number of support examples
            query_k: Number of query examples
            held_out_category: Category to hold out (random if None)
            
        Returns:
            Episode dict with support tools, query tools, and held-out category
        """
        if not self.categories:
            raise ValueError("No categories loaded. Call setup() first.")
            
        # Select held-out category
        if held_out_category is None:
            held_out_category = random.choice(self.categories)
            
        train_categories = [c for c in self.categories if c != held_out_category]
        
        # Sample support tools from training categories
        support_tools = []
        for cat in random.sample(train_categories, min(3, len(train_categories))):
            cat_tools = self.get_tools_by_category(cat)
            # Filter to tools with examples
            cat_tools = [t for t in cat_tools if len(t.examples) >= 3]
            if cat_tools:
                support_tools.extend(random.sample(cat_tools, min(2, len(cat_tools))))
                
        # Sample query tools from held-out category
        query_tools = self.get_tools_by_category(held_out_category)
        query_tools = [t for t in query_tools if len(t.examples) >= 3]
        query_tools = random.sample(query_tools, min(query_k, len(query_tools))) if query_tools else []
        
        return {
            "support_tools": support_tools[:support_k],
            "query_tools": query_tools,
            "held_out_category": held_out_category,
            "train_categories": train_categories
        }
        
    def create_train_test_split(
        self,
        test_ratio: float = 0.2,
        split_by: str = "category"  # "category", "tool", or "example"
    ) -> Tuple[List[Tool], List[Tool]]:
        """
        Create train/test split for meta-learning.
        
        Args:
            test_ratio: Fraction of data for testing
            split_by: How to split - by category (hardest), tool, or example
            
        Returns:
            (train_tools, test_tools)
        """
        if split_by == "category":
            # Hold out entire categories
            n_test_cats = max(1, int(len(self.categories) * test_ratio))
            test_cats = set(random.sample(self.categories, n_test_cats))
            
            train_tools = [t for t in self.all_tools.values() if t.category not in test_cats]
            test_tools = [t for t in self.all_tools.values() if t.category in test_cats]
            
        elif split_by == "tool":
            # Hold out individual tools
            all_tools = list(self.all_tools.values())
            random.shuffle(all_tools)
            split_idx = int(len(all_tools) * (1 - test_ratio))
            train_tools = all_tools[:split_idx]
            test_tools = all_tools[split_idx:]
            
        else:  # split_by == "example"
            # All tools in both, but different examples
            train_tools = []
            test_tools = []
            
            for tool in self.all_tools.values():
                if len(tool.examples) < 2:
                    train_tools.append(tool)
                    continue
                    
                # Split examples
                examples = tool.examples.copy()
                random.shuffle(examples)
                split_idx = int(len(examples) * (1 - test_ratio))
                
                train_tool = Tool(
                    name=tool.name,
                    description=tool.description,
                    category=tool.category,
                    api_list=tool.api_list,
                    standardized_name=tool.standardized_name,
                    examples=examples[:split_idx]
                )
                test_tool = Tool(
                    name=tool.name,
                    description=tool.description,
                    category=tool.category,
                    api_list=tool.api_list,
                    standardized_name=tool.standardized_name,
                    examples=examples[split_idx:]
                )
                train_tools.append(train_tool)
                test_tools.append(test_tool)
                
        return train_tools, test_tools


def load_toolbench(
    data_root: str = "./data/toolbench",
    use_huggingface: bool = True,
    min_examples: int = 1
) -> Tuple[List[Tool], ToolBenchLoader]:
    """
    Convenience function to load ToolBench data.
    
    Args:
        data_root: Path to ToolBench data
        use_huggingface: Use HuggingFace instead of local data
        min_examples: Minimum examples per tool (default 1 for HuggingFace)
        
    Returns:
        (tools, loader) - List of tools and the loader instance
    """
    loader = ToolBenchLoader(
        data_root=data_root,
        use_huggingface=use_huggingface,
        download_if_missing=True
    )
    loader.setup()
    
    tools = loader.get_tools_with_examples(min_examples=min_examples)
    logger.info(f"Loaded {len(tools)} tools with >= {min_examples} examples")
    
    return tools, loader


# Also load Gorilla, Spider, etc. for evaluation
class GorillaLoader:
    """Loader for Gorilla APIBench dataset."""
    
    def __init__(self, data_root: str = "./data/gorilla"):
        self.data_root = Path(data_root)
        
    def setup(self):
        """Download and setup Gorilla data."""
        os.makedirs(self.data_root, exist_ok=True)
        
        # Check if local files already exist
        local_files = list(self.data_root.glob("*_eval.json")) + list(self.data_root.glob("*.jsonl"))
        if local_files:
            logger.info(f"Found local Gorilla files: {[f.name for f in local_files]}")
            try:
                self._load_local_files(local_files)
                return
            except Exception as e:
                logger.warning(f"Could not load local files: {e}")
        
        # Try to load from HuggingFace
        try:
            from datasets import load_dataset
            dataset = load_dataset("gorilla-llm/APIBench", split="train")
            self._process_gorilla({"train": dataset})
            return
        except Exception as e:
            logger.warning(f"Could not load Gorilla from HuggingFace: {e}")
        
        # Try downloading from GitHub
        try:
            self._download_gorilla_raw()
            return
        except Exception as e:
            logger.warning(f"Could not download Gorilla: {e}")
        
        # Fall back to synthetic data
        self._create_synthetic_gorilla()
    
    def _load_local_files(self, files: List[Path]):
        """Load Gorilla data from local files."""
        tasks = []
        
        for filepath in files:
            logger.info(f"Loading {filepath.name}...")
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Try JSON array first
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    items = data
                else:
                    items = [data]
            except json.JSONDecodeError:
                # Try JSONL
                items = []
                for line in content.strip().split('\n'):
                    if line.strip():
                        try:
                            items.append(json.loads(line))
                        except:
                            continue
            
            api_source = filepath.stem.replace("_eval", "").replace("_train", "")
            
            for i, item in enumerate(items):
                task = self._parse_gorilla_item(item, api_source, i)
                if task:
                    tasks.append(task)
        
        if tasks:
            with open(self.data_root / "gorilla_tasks.json", 'w') as f:
                json.dump(tasks, f, indent=2)
            logger.info(f"Loaded {len(tasks)} Gorilla tasks from local files")
        else:
            raise Exception("No valid tasks found in local files")
    
    def _parse_gorilla_item(self, item: dict, api_source: str, index: int) -> Optional[dict]:
        """Parse a single Gorilla item into our task format."""
        code_field = item.get("code", "")
        api_data = item.get("api_data", {})
        
        # Extract instruction
        instruction = ""
        if "###Instruction:" in code_field:
            parts = code_field.split("###Instruction:")
            if len(parts) > 1:
                instruction = parts[1].split("###")[0].strip()
        
        if not instruction:
            instruction = item.get("instruction", item.get("query", ""))
        
        # Get expected output
        expected = item.get("api_call", "")
        if not expected and "<<<api_call>>>" in code_field:
            parts = code_field.split("<<<api_call>>>:")
            if len(parts) > 1:
                expected = parts[1].split(",")[0].strip()
        
        if not expected:
            expected = item.get("output", "")
        
        if instruction:
            return {
                "id": f"gorilla_{api_source}_{index}",
                "query": instruction,
                "expected": expected,
                "api": api_data.get("api_name", item.get("api_name", api_source)),
                "domain": api_data.get("domain", item.get("domain", "")),
                "framework": api_data.get("framework", api_source)
            }
        return None
            
    def _download_gorilla_raw(self):
        """Download Gorilla data from GitHub raw files."""
        import urllib.request
        
        # Gorilla benchmark files on GitHub
        base_url = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/apibench/"
        files = [
            "huggingface_eval.json",
            "tensorflow_eval.json", 
            "torchhub_eval.json"
        ]
        
        tasks = []
        for filename in files:
            try:
                url = base_url + filename
                logger.info(f"Downloading {url}...")
                
                request = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                response = urllib.request.urlopen(request, timeout=60)
                content = response.read().decode('utf-8')
                
                # Save locally for future use
                local_path = self.data_root / filename
                with open(local_path, 'w') as f:
                    f.write(content)
                
                # Parse content
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        items = data
                    else:
                        items = [data]
                except json.JSONDecodeError:
                    items = []
                    for line in content.strip().split('\n'):
                        if line.strip():
                            try:
                                items.append(json.loads(line))
                            except:
                                continue
                
                api_source = filename.replace("_eval.json", "")
                
                for i, item in enumerate(items):
                    task = self._parse_gorilla_item(item, api_source, i)
                    if task:
                        tasks.append(task)
                        
                logger.info(f"Loaded {len(items)} items from {filename}")
                        
            except Exception as e:
                logger.warning(f"Could not download {filename}: {e}")
                continue
                
        if tasks:
            with open(self.data_root / "gorilla_tasks.json", 'w') as f:
                json.dump(tasks, f, indent=2)
            logger.info(f"Downloaded {len(tasks)} Gorilla tasks total")
        else:
            raise Exception("No Gorilla data downloaded")
            
    def _process_gorilla(self, dataset):
        """Process Gorilla dataset from HuggingFace."""
        tasks = []
        for split in dataset.keys():
            for i, sample in enumerate(dataset[split]):
                task = self._parse_gorilla_item(dict(sample), "huggingface", i)
                if task:
                    tasks.append(task)
                
        with open(self.data_root / "gorilla_tasks.json", 'w') as f:
            json.dump(tasks, f, indent=2)
        logger.info(f"Loaded {len(tasks)} Gorilla tasks from HuggingFace")
            
    def _create_synthetic_gorilla(self):
        """Create synthetic Gorilla-style tasks for testing."""
        logger.info("Creating synthetic Gorilla tasks...")
        
        # Realistic ML API patterns based on actual Gorilla data
        apis = [
            # HuggingFace Transformers
            {"name": "transformers.pipeline", 
             "call": "transformers.pipeline(task='{task}', model='{model}')",
             "templates": [
                 ("Create a sentiment analysis pipeline", {"task": "sentiment-analysis", "model": "distilbert-base-uncased-finetuned-sst-2-english"}),
                 ("Build a text generation pipeline using GPT-2", {"task": "text-generation", "model": "gpt2"}),
                 ("Set up a question answering system", {"task": "question-answering", "model": "distilbert-base-cased-distilled-squad"}),
                 ("Create a named entity recognition pipeline", {"task": "ner", "model": "dbmdz/bert-large-cased-finetuned-conll03-english"}),
                 ("Build a text summarization pipeline", {"task": "summarization", "model": "facebook/bart-large-cnn"}),
                 ("Create a translation pipeline for English to French", {"task": "translation_en_to_fr", "model": "t5-base"}),
                 ("Set up a zero-shot classification pipeline", {"task": "zero-shot-classification", "model": "facebook/bart-large-mnli"}),
                 ("Create a fill-mask pipeline using BERT", {"task": "fill-mask", "model": "bert-base-uncased"}),
             ]},
            # PyTorch Hub
            {"name": "torch.hub.load",
             "call": "torch.hub.load(repo_or_dir='{repo}', model='{model}')",
             "templates": [
                 ("Load a pretrained ResNet50 model for image classification", {"repo": "pytorch/vision", "model": "resnet50"}),
                 ("Get YOLOv5 for object detection", {"repo": "ultralytics/yolov5", "model": "yolov5s"}),
                 ("Load a pretrained VGG16 model", {"repo": "pytorch/vision", "model": "vgg16"}),
                 ("Get MobileNetV2 for efficient inference", {"repo": "pytorch/vision", "model": "mobilenet_v2"}),
                 ("Load DeepLabV3 for semantic segmentation", {"repo": "pytorch/vision", "model": "deeplabv3_resnet50"}),
                 ("Get a pretrained DCGAN generator", {"repo": "facebookresearch/pytorch_GAN_zoo", "model": "DCGAN"}),
                 ("Load Tacotron2 for text-to-speech", {"repo": "NVIDIA/DeepLearningExamples", "model": "tacotron2"}),
                 ("Get a pretrained Wav2Vec model for speech", {"repo": "pytorch/fairseq", "model": "wav2vec2"}),
             ]},
            # TensorFlow Hub
            {"name": "hub.load",
             "call": "hub.load('{url}')",
             "templates": [
                 ("Load Universal Sentence Encoder for text embeddings", {"url": "https://tfhub.dev/google/universal-sentence-encoder/4"}),
                 ("Get MobileNetV2 feature extractor", {"url": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"}),
                 ("Load BERT for text classification", {"url": "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"}),
                 ("Get EfficientNet for image classification", {"url": "https://tfhub.dev/tensorflow/efficientnet/b0/classification/1"}),
                 ("Load a style transfer model", {"url": "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"}),
                 ("Get ALBERT for NLP tasks", {"url": "https://tfhub.dev/tensorflow/albert_en_base/3"}),
                 ("Load object detection model", {"url": "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"}),
                 ("Get image segmentation model", {"url": "https://tfhub.dev/tensorflow/deeplabv3/1"}),
             ]},
            # Keras Applications
            {"name": "tf.keras.applications",
             "call": "tf.keras.applications.{model}(weights='{weights}')",
             "templates": [
                 ("Create a ResNet50 model with ImageNet weights", {"model": "ResNet50", "weights": "imagenet"}),
                 ("Load InceptionV3 for transfer learning", {"model": "InceptionV3", "weights": "imagenet"}),
                 ("Get VGG19 pretrained model", {"model": "VGG19", "weights": "imagenet"}),
                 ("Create EfficientNetB0 classifier", {"model": "EfficientNetB0", "weights": "imagenet"}),
                 ("Load DenseNet121 for feature extraction", {"model": "DenseNet121", "weights": "imagenet"}),
                 ("Get Xception model pretrained on ImageNet", {"model": "Xception", "weights": "imagenet"}),
                 ("Create NASNetMobile for mobile deployment", {"model": "NASNetMobile", "weights": "imagenet"}),
                 ("Load MobileNetV3Large model", {"model": "MobileNetV3Large", "weights": "imagenet"}),
             ]},
            # Scikit-learn
            {"name": "sklearn",
             "call": "sklearn.{module}.{model}({params})",
             "templates": [
                 ("Create a random forest classifier with 100 trees", {"module": "ensemble", "model": "RandomForestClassifier", "params": "n_estimators=100"}),
                 ("Build a gradient boosting regressor", {"module": "ensemble", "model": "GradientBoostingRegressor", "params": "n_estimators=100, learning_rate=0.1"}),
                 ("Set up a support vector machine for classification", {"module": "svm", "model": "SVC", "params": "kernel='rbf'"}),
                 ("Create a K-means clustering model", {"module": "cluster", "model": "KMeans", "params": "n_clusters=5"}),
                 ("Build a logistic regression classifier", {"module": "linear_model", "model": "LogisticRegression", "params": "max_iter=1000"}),
                 ("Create a PCA for dimensionality reduction", {"module": "decomposition", "model": "PCA", "params": "n_components=50"}),
                 ("Set up a decision tree classifier", {"module": "tree", "model": "DecisionTreeClassifier", "params": "max_depth=10"}),
             ]},
        ]
        
        tasks = []
        task_id = 0
        
        for api in apis:
            for query, params in api["templates"]:
                expected = api["call"].format(**params)
                tasks.append({
                    "id": f"gorilla_synthetic_{task_id}",
                    "query": query,
                    "expected": expected,
                    "api": api["name"],
                    "domain": "Machine Learning",
                    "framework": api["name"].split(".")[0]
                })
                task_id += 1
        
        # Add some variety with paraphrased queries
        paraphrases = [
            ("I need to analyze sentiment in customer reviews", "transformers.pipeline(task='sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')"),
            ("How can I detect objects in images using a pretrained model?", "torch.hub.load(repo_or_dir='ultralytics/yolov5', model='yolov5s')"),
            ("I want to convert text to vector embeddings", "hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')"),
            ("Help me classify images using a lightweight model", "tf.keras.applications.MobileNetV2(weights='imagenet')"),
            ("I need to cluster my data into groups", "sklearn.cluster.KMeans(n_clusters=5)"),
            ("Build me a model that can generate text continuations", "transformers.pipeline(task='text-generation', model='gpt2')"),
            ("I want to segment objects in photographs", "hub.load('https://tfhub.dev/tensorflow/deeplabv3/1')"),
            ("Create a classifier that works well on small datasets", "sklearn.ensemble.RandomForestClassifier(n_estimators=100)"),
        ]
        
        for query, expected in paraphrases:
            tasks.append({
                "id": f"gorilla_synthetic_{task_id}",
                "query": query,
                "expected": expected,
                "api": expected.split("(")[0],
                "domain": "Machine Learning",
                "framework": expected.split(".")[0]
            })
            task_id += 1
            
        with open(self.data_root / "gorilla_tasks.json", 'w') as f:
            json.dump(tasks, f, indent=2)
        logger.info(f"Created {len(tasks)} synthetic Gorilla tasks")


class Spider2Loader:
    """Loader for Spider 2.0 SQL dataset."""
    
    def __init__(self, data_root: str = "./data/spider2"):
        self.data_root = Path(data_root)
        
    def setup(self):
        """Download and setup Spider 2.0 data."""
        os.makedirs(self.data_root, exist_ok=True)
        
        try:
            from datasets import load_dataset
            # Try the standard Spider dataset instead
            dataset = load_dataset("spider", split="validation")
            self._process_spider({"validation": dataset})
        except Exception as e:
            logger.warning(f"Could not load Spider from HuggingFace: {e}")
            # Try alternative spider datasets
            try:
                self._try_alternative_spider()
            except Exception as e2:
                logger.warning(f"Could not load alternative Spider: {e2}")
                self._create_synthetic_spider()
    
    def _try_alternative_spider(self):
        """Try loading alternative Spider datasets."""
        from datasets import load_dataset
        
        # Try different Spider variants
        alternatives = [
            ("richardr1126/spider-skeleton", "train"),
            ("xlangai/spider", "train"),
        ]
        
        for dataset_name, split in alternatives:
            try:
                dataset = load_dataset(dataset_name, split=split)
                self._process_spider({split: dataset})
                return
            except:
                continue
                
        raise Exception("No Spider alternatives available")
            
    def _process_spider(self, dataset):
        """Process Spider dataset."""
        tasks = []
        for split in dataset.keys():
            for sample in dataset[split]:
                tasks.append({
                    "id": sample.get("id", f"spider_{len(tasks)}"),
                    "query": sample.get("question", ""),
                    "expected_sql": sample.get("query", sample.get("sql", "")),
                    "database": sample.get("db_id", "")
                })
                
        with open(self.data_root / "spider2_tasks.json", 'w') as f:
            json.dump(tasks, f, indent=2)
            
    def _create_synthetic_spider(self):
        """Create synthetic Spider-style tasks."""
        logger.info("Creating synthetic Spider tasks...")
        
        templates = [
            ("Find all customers who spent more than $1000", 
             "SELECT * FROM customers WHERE total_spent > 1000"),
            ("List products with low inventory",
             "SELECT * FROM products WHERE stock < 10"),
            ("Count orders by status",
             "SELECT status, COUNT(*) FROM orders GROUP BY status"),
            ("Get top 10 customers by order count",
             "SELECT customer_id, COUNT(*) as cnt FROM orders GROUP BY customer_id ORDER BY cnt DESC LIMIT 10"),
            ("Find average order value by category",
             "SELECT category, AVG(total) FROM orders GROUP BY category"),
            ("List employees hired this year",
             "SELECT * FROM employees WHERE YEAR(hire_date) = 2024"),
            ("Find customers without orders",
             "SELECT * FROM customers WHERE id NOT IN (SELECT DISTINCT customer_id FROM orders)"),
            ("Get monthly revenue",
             "SELECT MONTH(order_date) as month, SUM(total) FROM orders GROUP BY month"),
        ]
        
        tasks = []
        for i in range(100):
            query, sql = templates[i % len(templates)]
            tasks.append({
                "id": f"spider_synthetic_{i}",
                "query": query,
                "expected_sql": sql,
                "database": "enterprise_db"
            })
            
        with open(self.data_root / "spider2_tasks.json", 'w') as f:
            json.dump(tasks, f, indent=2)
        logger.info(f"Created {len(tasks)} synthetic Spider tasks")


class WebArenaLoader:
    """Loader for WebArena web navigation benchmark."""
    
    def __init__(self, data_root: str = "./data/webarena"):
        self.data_root = Path(data_root)
        
    def setup(self):
        """Download and setup WebArena data."""
        os.makedirs(self.data_root, exist_ok=True)
        
        # Try to load from HuggingFace or create synthetic
        try:
            self._try_load_webarena()
        except Exception as e:
            logger.warning(f"Could not load WebArena: {e}")
            self._create_synthetic_webarena()
    
    def _try_load_webarena(self):
        """Try loading WebArena from available sources."""
        import urllib.request
        
        # First try downloading from official WebArena GitHub
        try:
            logger.info("Trying to download from WebArena GitHub...")
            base_url = "https://raw.githubusercontent.com/web-arena-x/webarena/main/config_files"
            task_files = ["test_shopping.json", "test_reddit.json", "test_gitlab.json"]
            
            all_tasks = []
            task_id = 0
            
            for task_file in task_files:
                try:
                    url = f"{base_url}/{task_file}"
                    with urllib.request.urlopen(url, timeout=30) as response:
                        data = json.loads(response.read().decode())
                    
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if isinstance(item, dict):
                            intent = item.get("intent", item.get("task", ""))
                            if intent:
                                all_tasks.append({
                                    "id": f"webarena_{task_id}",
                                    "query": intent,
                                    "site": task_file.replace("test_", "").replace(".json", ""),
                                    "expected_actions": item.get("action_sequence", [])
                                })
                                task_id += 1
                except:
                    continue
            
            if len(all_tasks) >= 20:
                with open(self.data_root / "webarena_tasks.json", 'w') as f:
                    json.dump(all_tasks, f, indent=2)
                logger.info(f"Loaded {len(all_tasks)} WebArena tasks from GitHub")
                return
        except Exception as e:
            logger.warning(f"GitHub download failed: {e}")
        
        # Try Mind2Web from HuggingFace
        try:
            from datasets import load_dataset
            logger.info("Trying Mind2Web dataset...")
            
            dataset = load_dataset("osunlp/Mind2Web", split="test", trust_remote_code=True)
            tasks = []
            for i, sample in enumerate(dataset):
                if i >= 300:
                    break
                task = sample.get("confirmed_task", sample.get("task", ""))
                if task:
                    tasks.append({
                        "id": f"webarena_{i}",
                        "query": task,
                        "site": sample.get("website", sample.get("domain", "")),
                        "expected_actions": []
                    })
            
            if len(tasks) >= 50:
                with open(self.data_root / "webarena_tasks.json", 'w') as f:
                    json.dump(tasks, f, indent=2)
                logger.info(f"Loaded {len(tasks)} WebArena tasks from Mind2Web")
                return
        except Exception as e:
            logger.warning(f"Mind2Web failed: {e}")
        
        raise Exception("No WebArena data source available")
    
    def _process_mind2web(self, dataset):
        """Process Mind2Web as WebArena proxy."""
        tasks = []
        for i, sample in enumerate(dataset):
            if i >= 200:  # Limit to 200 tasks
                break
            tasks.append({
                "id": f"webarena_{i}",
                "query": sample.get("confirmed_task", sample.get("task", "")),
                "website": sample.get("website", ""),
                "expected_actions": []  # Mind2Web has different action format
            })
        
        with open(self.data_root / "webarena_tasks.json", 'w') as f:
            json.dump(tasks, f, indent=2)
        logger.info(f"Loaded {len(tasks)} WebArena tasks from Mind2Web")
            
    def _create_synthetic_webarena(self):
        """Create diverse synthetic WebArena tasks."""
        logger.info("Creating synthetic WebArena tasks...")
        
        # Much more diverse task templates
        templates = [
            # Shopping tasks
            {"query": "Add a laptop to the shopping cart", "site": "shopping", "expected_actions": ["search", "click_product", "add_to_cart"]},
            {"query": "Find wireless headphones under $100", "site": "shopping", "expected_actions": ["search", "filter_price", "view_results"]},
            {"query": "Check out with items in cart", "site": "shopping", "expected_actions": ["go_to_cart", "proceed_checkout", "enter_info", "submit"]},
            {"query": "Apply a coupon code SAVE20", "site": "shopping", "expected_actions": ["go_to_cart", "enter_coupon", "apply"]},
            {"query": "Remove an item from the cart", "site": "shopping", "expected_actions": ["go_to_cart", "click_remove", "confirm"]},
            {"query": "Change quantity of item in cart to 3", "site": "shopping", "expected_actions": ["go_to_cart", "update_quantity", "save"]},
            {"query": "Sort products by price low to high", "site": "shopping", "expected_actions": ["click_sort", "select_price_asc"]},
            {"query": "Filter products by brand Apple", "site": "shopping", "expected_actions": ["click_filter", "select_brand", "apply"]},
            
            # Forum/Social tasks
            {"query": "Create a new post titled 'Hello World'", "site": "forum", "expected_actions": ["click_new_post", "enter_title", "enter_content", "submit"]},
            {"query": "Reply to the top post", "site": "forum", "expected_actions": ["click_post", "click_reply", "enter_text", "submit"]},
            {"query": "Upvote the first comment", "site": "forum", "expected_actions": ["find_comment", "click_upvote"]},
            {"query": "Edit my profile bio", "site": "forum", "expected_actions": ["go_to_profile", "click_edit", "modify_bio", "save"]},
            {"query": "Follow user JohnDoe", "site": "forum", "expected_actions": ["search_user", "click_profile", "click_follow"]},
            {"query": "Send a private message to admin", "site": "forum", "expected_actions": ["go_to_messages", "click_compose", "select_recipient", "type_message", "send"]},
            
            # Admin tasks
            {"query": "Search for user by email admin@test.com", "site": "admin", "expected_actions": ["go_to_users", "enter_search", "submit"]},
            {"query": "Ban user spammer123", "site": "admin", "expected_actions": ["search_user", "click_user", "click_ban", "confirm"]},
            {"query": "Export user data to CSV", "site": "admin", "expected_actions": ["go_to_users", "click_export", "select_csv", "download"]},
            {"query": "Change site settings to maintenance mode", "site": "admin", "expected_actions": ["go_to_settings", "toggle_maintenance", "save"]},
            {"query": "View error logs from today", "site": "admin", "expected_actions": ["go_to_logs", "filter_date", "filter_errors", "view"]},
            {"query": "Create a new admin user", "site": "admin", "expected_actions": ["go_to_users", "click_create", "fill_form", "set_role_admin", "save"]},
            
            # Content Management
            {"query": "Upload an image to the media library", "site": "cms", "expected_actions": ["go_to_media", "click_upload", "select_file", "confirm"]},
            {"query": "Create a new blog post draft", "site": "cms", "expected_actions": ["go_to_posts", "click_new", "enter_content", "save_draft"]},
            {"query": "Schedule post for tomorrow at 9am", "site": "cms", "expected_actions": ["open_post", "click_schedule", "set_datetime", "confirm"]},
            {"query": "Delete all posts in trash", "site": "cms", "expected_actions": ["go_to_trash", "select_all", "click_delete_permanent", "confirm"]},
            
            # Email/Calendar
            {"query": "Compose email to team@company.com", "site": "email", "expected_actions": ["click_compose", "enter_recipient", "enter_subject", "enter_body", "send"]},
            {"query": "Move email to Archive folder", "site": "email", "expected_actions": ["select_email", "click_move", "select_archive"]},
            {"query": "Create calendar event for Monday 2pm", "site": "calendar", "expected_actions": ["click_create", "set_date", "set_time", "enter_title", "save"]},
            {"query": "Invite john@test.com to meeting", "site": "calendar", "expected_actions": ["open_event", "click_invite", "enter_email", "send"]},
            
            # Search/Navigation
            {"query": "Search for 'python tutorial' and click first result", "site": "search", "expected_actions": ["enter_query", "submit", "click_first_result"]},
            {"query": "Navigate to the About Us page", "site": "general", "expected_actions": ["find_nav", "click_about"]},
            {"query": "Go back to homepage", "site": "general", "expected_actions": ["click_logo"]},
            {"query": "Open help documentation", "site": "general", "expected_actions": ["click_help", "view_docs"]},
            
            # Forms
            {"query": "Fill out contact form with name and email", "site": "forms", "expected_actions": ["enter_name", "enter_email", "enter_message", "submit"]},
            {"query": "Subscribe to newsletter", "site": "forms", "expected_actions": ["enter_email", "click_subscribe", "confirm"]},
            {"query": "Reset password using email", "site": "auth", "expected_actions": ["click_forgot", "enter_email", "submit"]},
            {"query": "Update account email address", "site": "account", "expected_actions": ["go_to_settings", "edit_email", "enter_new_email", "save", "verify"]},
            
            # E-commerce specific
            {"query": "Track order #12345", "site": "shopping", "expected_actions": ["go_to_orders", "enter_order_id", "view_tracking"]},
            {"query": "Request a refund for recent order", "site": "shopping", "expected_actions": ["go_to_orders", "select_order", "click_refund", "fill_reason", "submit"]},
            {"query": "Add product to wishlist", "site": "shopping", "expected_actions": ["view_product", "click_wishlist"]},
            {"query": "Compare two products", "site": "shopping", "expected_actions": ["select_product1", "click_compare", "select_product2", "view_comparison"]},
            {"query": "Write a product review", "site": "shopping", "expected_actions": ["view_product", "click_reviews", "click_write", "enter_rating", "enter_text", "submit"]},
        ]
        
        tasks = []
        for i, template in enumerate(templates):
            tasks.append({
                "id": f"webarena_{i}",
                "query": template["query"],
                "site": template["site"],
                "expected_actions": template["expected_actions"]
            })
        
        # Duplicate with variations to reach 100+ tasks
        variations = [
            ("Find", "Search for"),
            ("Click", "Select"),
            ("Enter", "Type"),
            ("Go to", "Navigate to"),
            ("Add", "Put"),
        ]
        
        base_count = len(tasks)
        for i in range(50):
            base_task = templates[i % len(templates)].copy()
            # Apply random variation
            old, new = variations[i % len(variations)]
            base_task["query"] = base_task["query"].replace(old, new)
            tasks.append({
                "id": f"webarena_{base_count + i}",
                "query": base_task["query"],
                "site": base_task.get("site", "general"),
                "expected_actions": base_task["expected_actions"]
            })
        
        with open(self.data_root / "webarena_tasks.json", 'w') as f:
            json.dump(tasks, f, indent=2)
        logger.info(f"Created {len(tasks)} synthetic WebArena tasks")


class InterCodeLoader:
    """Loader for InterCode bash/CTF benchmark."""
    
    def __init__(self, data_root: str = "./data/intercode"):
        self.data_root = Path(data_root)
        
    def setup(self):
        """Download and setup InterCode data."""
        os.makedirs(self.data_root, exist_ok=True)
        
        # Try to load from HuggingFace or create synthetic
        try:
            self._try_load_intercode()
        except Exception as e:
            logger.warning(f"Could not load InterCode: {e}")
            self._create_synthetic_intercode()
    
    def _try_load_intercode(self):
        """Try loading InterCode from available sources."""
        import urllib.request
        
        # First try downloading from official InterCode GitHub
        try:
            logger.info("Trying to download from InterCode GitHub...")
            url = "https://raw.githubusercontent.com/princeton-nlp/intercode/master/data/ic_bash/ic_bash.json"
            
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode())
            
            tasks = []
            for i, item in enumerate(data):
                query = item.get("query", item.get("instruction", item.get("input", "")))
                expected = item.get("gold", item.get("output", item.get("command", "")))
                if query and expected:
                    tasks.append({
                        "id": f"intercode_{i}",
                        "query": query,
                        "expected_command": expected
                    })
            
            if len(tasks) >= 20:
                with open(self.data_root / "intercode_tasks.json", 'w') as f:
                    json.dump(tasks, f, indent=2)
                logger.info(f"Loaded {len(tasks)} InterCode tasks from GitHub")
                return
        except Exception as e:
            logger.warning(f"GitHub download failed: {e}")
        
        # Try NL2Bash from HuggingFace
        try:
            from datasets import load_dataset
            logger.info("Trying NL2Bash dataset...")
            
            dataset = load_dataset("neulab/nl2bash", split="test", trust_remote_code=True)
            tasks = []
            for i, sample in enumerate(dataset):
                if i >= 300:
                    break
                query = sample.get("invocation", "")
                cmd = sample.get("cmd", "")
                if query and cmd:
                    tasks.append({
                        "id": f"intercode_{i}",
                        "query": query,
                        "expected_command": cmd
                    })
            
            if len(tasks) >= 50:
                with open(self.data_root / "intercode_tasks.json", 'w') as f:
                    json.dump(tasks, f, indent=2)
                logger.info(f"Loaded {len(tasks)} InterCode tasks from NL2Bash")
                return
        except Exception as e:
            logger.warning(f"NL2Bash failed: {e}")
        
        raise Exception("No InterCode data source available")
    
    def _process_nl2bash(self, dataset):
        """Process NL2Bash as InterCode proxy."""
        tasks = []
        for i, sample in enumerate(dataset):
            if i >= 200:  # Limit to 200 tasks
                break
            tasks.append({
                "id": f"intercode_{i}",
                "query": sample.get("nl", sample.get("question", "")),
                "expected_command": sample.get("bash", sample.get("cmd", ""))
            })
        
        with open(self.data_root / "intercode_tasks.json", 'w') as f:
            json.dump(tasks, f, indent=2)
        logger.info(f"Loaded {len(tasks)} InterCode tasks from NL2Bash")
            
    def _create_synthetic_intercode(self):
        """Create diverse synthetic InterCode bash tasks."""
        logger.info("Creating synthetic InterCode tasks...")
        
        # Much more diverse bash command templates
        templates = [
            # File operations
            {"query": "Find all Python files in current directory", "cmd": "find . -name '*.py'"},
            {"query": "Find all .py files modified in last 24 hours", "cmd": "find . -name '*.py' -mtime -1"},
            {"query": "Find files larger than 100MB", "cmd": "find . -size +100M"},
            {"query": "Find and delete all .tmp files", "cmd": "find . -name '*.tmp' -delete"},
            {"query": "Find all empty directories", "cmd": "find . -type d -empty"},
            {"query": "List all files recursively with size", "cmd": "find . -type f -exec ls -lh {} \\;"},
            {"query": "Count total number of files in directory", "cmd": "find . -type f | wc -l"},
            {"query": "Find duplicate files by name", "cmd": "find . -type f -printf '%f\\n' | sort | uniq -d"},
            
            # Text processing
            {"query": "Count lines in a file", "cmd": "wc -l filename"},
            {"query": "Count words in a file", "cmd": "wc -w filename"},
            {"query": "Count characters in a file", "cmd": "wc -c filename"},
            {"query": "Show first 10 lines of file", "cmd": "head -n 10 filename"},
            {"query": "Show last 20 lines of file", "cmd": "tail -n 20 filename"},
            {"query": "Show lines 50-100 of a file", "cmd": "sed -n '50,100p' filename"},
            {"query": "Remove blank lines from file", "cmd": "sed '/^$/d' filename"},
            {"query": "Replace tabs with spaces", "cmd": "sed 's/\\t/    /g' filename"},
            
            # Search
            {"query": "Search for pattern in files", "cmd": "grep -r 'pattern' ."},
            {"query": "Search for word in files ignoring case", "cmd": "grep -ri 'word' ."},
            {"query": "Find files containing 'error'", "cmd": "grep -l 'error' *"},
            {"query": "Count occurrences of pattern in file", "cmd": "grep -c 'pattern' filename"},
            {"query": "Show lines NOT matching pattern", "cmd": "grep -v 'pattern' filename"},
            {"query": "Search with line numbers", "cmd": "grep -n 'pattern' filename"},
            {"query": "Search for whole word only", "cmd": "grep -w 'word' filename"},
            {"query": "Search in gzipped files", "cmd": "zgrep 'pattern' file.gz"},
            
            # System info
            {"query": "Show disk usage of current directory", "cmd": "du -sh ."},
            {"query": "Show disk usage of all subdirectories", "cmd": "du -h --max-depth=1"},
            {"query": "Show free disk space", "cmd": "df -h"},
            {"query": "Show memory usage", "cmd": "free -h"},
            {"query": "Show running processes", "cmd": "ps aux"},
            {"query": "Show top 10 memory consuming processes", "cmd": "ps aux --sort=-%mem | head -10"},
            {"query": "Show CPU usage", "cmd": "top -bn1 | head -20"},
            {"query": "Show system uptime", "cmd": "uptime"},
            {"query": "Show current user", "cmd": "whoami"},
            {"query": "Show all logged in users", "cmd": "who"},
            
            # File listing
            {"query": "List files sorted by size", "cmd": "ls -lS"},
            {"query": "List files sorted by modification time", "cmd": "ls -lt"},
            {"query": "List all files including hidden", "cmd": "ls -la"},
            {"query": "List only directories", "cmd": "ls -d */"},
            {"query": "List files with human readable sizes", "cmd": "ls -lh"},
            {"query": "List files one per line", "cmd": "ls -1"},
            {"query": "List files with inode numbers", "cmd": "ls -i"},
            
            # Archives
            {"query": "Create tar archive of directory", "cmd": "tar -cvf archive.tar directory/"},
            {"query": "Create compressed tar.gz archive", "cmd": "tar -czvf archive.tar.gz directory/"},
            {"query": "Extract tar archive", "cmd": "tar -xvf archive.tar"},
            {"query": "Extract tar.gz archive", "cmd": "tar -xzvf archive.tar.gz"},
            {"query": "List contents of tar archive", "cmd": "tar -tvf archive.tar"},
            {"query": "Create zip archive", "cmd": "zip -r archive.zip directory/"},
            {"query": "Extract zip archive", "cmd": "unzip archive.zip"},
            
            # Network
            {"query": "Check if host is reachable", "cmd": "ping -c 4 hostname"},
            {"query": "Download file from URL", "cmd": "wget URL"},
            {"query": "Download file with curl", "cmd": "curl -O URL"},
            {"query": "Show network interfaces", "cmd": "ifconfig"},
            {"query": "Show open ports", "cmd": "netstat -tuln"},
            {"query": "Check DNS for domain", "cmd": "nslookup domain.com"},
            {"query": "Trace route to host", "cmd": "traceroute hostname"},
            
            # Permissions
            {"query": "Make file executable", "cmd": "chmod +x filename"},
            {"query": "Change file permissions to 755", "cmd": "chmod 755 filename"},
            {"query": "Change owner of file", "cmd": "chown user:group filename"},
            {"query": "Change permissions recursively", "cmd": "chmod -R 755 directory/"},
            
            # Process management
            {"query": "Kill process by PID", "cmd": "kill PID"},
            {"query": "Kill process by name", "cmd": "pkill processname"},
            {"query": "Run command in background", "cmd": "command &"},
            {"query": "List background jobs", "cmd": "jobs"},
            {"query": "Bring job to foreground", "cmd": "fg %1"},
            
            # Text manipulation
            {"query": "Sort file contents", "cmd": "sort filename"},
            {"query": "Sort file numerically", "cmd": "sort -n filename"},
            {"query": "Sort and remove duplicates", "cmd": "sort -u filename"},
            {"query": "Reverse file contents", "cmd": "tac filename"},
            {"query": "Get unique lines", "cmd": "uniq filename"},
            {"query": "Count unique lines", "cmd": "sort filename | uniq -c"},
            {"query": "Cut first column", "cmd": "cut -d',' -f1 filename"},
            {"query": "Merge two files side by side", "cmd": "paste file1 file2"},
            
            # Environment
            {"query": "Show all environment variables", "cmd": "env"},
            {"query": "Show PATH variable", "cmd": "echo $PATH"},
            {"query": "Set environment variable", "cmd": "export VAR=value"},
            {"query": "Show command history", "cmd": "history"},
            {"query": "Clear terminal", "cmd": "clear"},
            
            # Date/Time
            {"query": "Show current date and time", "cmd": "date"},
            {"query": "Show calendar", "cmd": "cal"},
            {"query": "Show date in specific format", "cmd": "date '+%Y-%m-%d'"},
            
            # Misc
            {"query": "Create directory", "cmd": "mkdir dirname"},
            {"query": "Create nested directories", "cmd": "mkdir -p a/b/c"},
            {"query": "Copy file", "cmd": "cp source dest"},
            {"query": "Copy directory recursively", "cmd": "cp -r source/ dest/"},
            {"query": "Move or rename file", "cmd": "mv oldname newname"},
            {"query": "Remove file", "cmd": "rm filename"},
            {"query": "Remove directory", "cmd": "rm -r dirname"},
            {"query": "Create symbolic link", "cmd": "ln -s target linkname"},
            {"query": "Show file type", "cmd": "file filename"},
            {"query": "Compare two files", "cmd": "diff file1 file2"},
            {"query": "Show differences side by side", "cmd": "diff -y file1 file2"},
        ]
        
        tasks = []
        for i, template in enumerate(templates):
            tasks.append({
                "id": f"intercode_{i}",
                "query": template["query"],
                "expected_command": template["cmd"]
            })
        
        with open(self.data_root / "intercode_tasks.json", 'w') as f:
            json.dump(tasks, f, indent=2)
        logger.info(f"Created {len(tasks)} synthetic InterCode tasks")


def setup_all_datasets(data_root: str = "./data"):
    """Setup all datasets for training and evaluation."""
    logger.info("Setting up all datasets...")
    
    # ToolBench for meta-training
    toolbench_loader = ToolBenchLoader(
        data_root=os.path.join(data_root, "toolbench"),
        use_huggingface=True,
        download_if_missing=True
    )
    toolbench_loader.setup()
    
    # Gorilla for evaluation
    try:
        gorilla_loader = GorillaLoader(os.path.join(data_root, "gorilla"))
        gorilla_loader.setup()
    except Exception as e:
        logger.warning(f"Gorilla setup failed: {e}")
    
    # Spider 2.0 for evaluation
    try:
        spider_loader = Spider2Loader(os.path.join(data_root, "spider2"))
        spider_loader.setup()
    except Exception as e:
        logger.warning(f"Spider setup failed: {e}")
    
    # WebArena for evaluation
    try:
        webarena_loader = WebArenaLoader(os.path.join(data_root, "webarena"))
        webarena_loader.setup()
    except Exception as e:
        logger.warning(f"WebArena setup failed: {e}")
    
    # InterCode for evaluation
    try:
        intercode_loader = InterCodeLoader(os.path.join(data_root, "intercode"))
        intercode_loader.setup()
    except Exception as e:
        logger.warning(f"InterCode setup failed: {e}")
    
    logger.info("All datasets ready!")
    
    return toolbench_loader


if __name__ == "__main__":
    import sys
    
    # Test the loader
    print("Testing ToolBench loader...")
    print("="*60)
    
    try:
        tools, loader = load_toolbench(use_huggingface=True, min_examples=1)
        
        print(f"\n✓ Loaded {len(tools)} tools")
        print(f"✓ Categories: {len(loader.categories)}")
        print(f"✓ Total examples: {len(loader.training_examples)}")
        
        if tools:
            # Show sample tools
            print(f"\nSample tools:")
            for i, tool in enumerate(tools[:5]):
                print(f"  {i+1}. {tool.name} ({tool.category})")
                print(f"     APIs: {len(tool.api_list)}, Examples: {len(tool.examples)}")
            
            # Show category distribution
            print(f"\nCategory distribution (top 10):")
            cat_counts = {}
            for tool in loader.all_tools.values():
                cat_counts[tool.category] = cat_counts.get(tool.category, 0) + 1
            for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"  {cat}: {count} tools")
            
            # Test meta-episode creation
            if len(loader.categories) > 1:
                episode = loader.create_meta_episode()
                print(f"\n✓ Meta-episode creation works:")
                print(f"  Held-out category: {episode['held_out_category']}")
                print(f"  Support tools: {len(episode['support_tools'])}")
                print(f"  Query tools: {len(episode['query_tools'])}")
        else:
            print("\n⚠ No tools with sufficient examples found")
            print("  This may happen if the dataset doesn't have paired queries")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("\n" + "="*60)
    print("Data loader test complete!")
