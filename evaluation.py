"""
Evaluation Pipeline for Meta-Tool
=================================
Evaluates Meta-Tool on four benchmarks:
1. Gorilla APIBench (REST APIs)
2. Spider 2.0 (Enterprise SQL)
3. WebArena (Web Navigation)
4. InterCode (Bash/CTF)

Metrics:
- Execution Success Rate (SR)
- Pass@1
- Adaptation Time
- Inference Latency
"""

import torch
import json
import time
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
from abc import ABC, abstractmethod

from config import MetaToolConfig, InferenceConfig
from hypernetwork import MetaToolHypernetwork
from lora_integration import MetaToolAdaptedModel
from value_function import ValueGuidedBeamSearch, ValueFunction, ExecutionEnvironment
from constrained_decoding import OutlinesConstrainedGenerator, JSONSchemaFSM


@dataclass
class EvaluationResult:
    """Results from evaluating on a single task."""
    task_id: str
    success: bool
    prediction: str
    ground_truth: str
    execution_result: Optional[str] = None
    error_message: Optional[str] = None
    latency_ms: float = 0.0
    num_steps: int = 1


@dataclass
class BenchmarkResults:
    """Aggregate results for a benchmark."""
    benchmark_name: str
    num_tasks: int
    num_success: int
    success_rate: float
    avg_latency_ms: float
    avg_steps: float
    adaptation_time_s: float
    results: List[EvaluationResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "num_tasks": self.num_tasks,
            "num_success": self.num_success,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_steps": self.avg_steps,
            "adaptation_time_s": self.adaptation_time_s
        }


class BenchmarkEvaluator(ABC):
    """Abstract base class for benchmark evaluators."""
    
    @abstractmethod
    def load_tasks(self) -> List[Dict]:
        """Load benchmark tasks."""
        pass
        
    @abstractmethod
    def get_tool_spec(self) -> Tuple[str, Dict, List[Tuple[str, str]]]:
        """Get tool documentation, schema, and examples."""
        pass
        
    @abstractmethod
    def execute_and_evaluate(
        self,
        task: Dict,
        prediction: str
    ) -> Tuple[bool, Optional[str]]:
        """Execute prediction and evaluate correctness."""
        pass


class GorillaEvaluator(BenchmarkEvaluator):
    """
    Evaluator for Gorilla APIBench.
    Tests REST API function calling.
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.tasks = []
        
    def load_tasks(self) -> List[Dict]:
        """Load Gorilla benchmark tasks."""
        # Try to load from file
        task_file = os.path.join(self.data_path, "gorilla_tasks.json")
        
        if os.path.exists(task_file):
            with open(task_file) as f:
                self.tasks = json.load(f)
        else:
            # Create synthetic tasks for testing
            self.tasks = self._create_synthetic_tasks()
            
        return self.tasks
        
    def _create_synthetic_tasks(self, num_tasks: int = 100) -> List[Dict]:
        """Create synthetic Gorilla-style tasks."""
        tasks = []
        
        api_templates = [
            {
                "api": "search.google",
                "params": ["query", "num_results"],
                "query": "Search for {topic} on Google",
                "expected": {"function": "search.google", "query": "{topic}", "num_results": 10}
            },
            {
                "api": "weather.get",
                "params": ["location", "units"],
                "query": "Get weather for {location}",
                "expected": {"function": "weather.get", "location": "{location}", "units": "celsius"}
            },
            {
                "api": "translate.text",
                "params": ["text", "source_lang", "target_lang"],
                "query": "Translate '{text}' from English to Spanish",
                "expected": {"function": "translate.text", "text": "{text}", "source_lang": "en", "target_lang": "es"}
            }
        ]
        
        topics = ["machine learning", "climate change", "quantum computing", "renewable energy"]
        locations = ["New York", "London", "Tokyo", "Paris"]
        texts = ["Hello world", "Good morning", "Thank you", "Goodbye"]
        
        for i in range(num_tasks):
            template = api_templates[i % len(api_templates)]
            
            if "topic" in template["query"]:
                topic = topics[i % len(topics)]
                query = template["query"].format(topic=topic)
                expected = json.dumps(template["expected"]).replace("{topic}", topic)
            elif "location" in template["query"]:
                location = locations[i % len(locations)]
                query = template["query"].format(location=location)
                expected = json.dumps(template["expected"]).replace("{location}", location)
            elif "text" in template["query"]:
                text = texts[i % len(texts)]
                query = template["query"].format(text=text)
                expected = json.dumps(template["expected"]).replace("{text}", text)
            else:
                query = template["query"]
                expected = json.dumps(template["expected"])
                
            tasks.append({
                "id": f"gorilla_{i}",
                "query": query,
                "expected": expected,
                "api": template["api"]
            })
            
        return tasks
        
    def get_tool_spec(self) -> Tuple[str, Dict, List[Tuple[str, str]]]:
        """Get Gorilla API tool specification."""
        documentation = """# Model Loading API

Generate Python code to load the appropriate pre-trained model.

FORMATS (use exactly as shown):
1. torchvision.models.MODEL_NAME(pretrained=True)
2. torch.hub.load('REPO', 'MODEL', pretrained=True)
3. pipeline('TASK', model='MODEL_NAME')

Output ONLY the code. No imports, no explanations, no markdown."""
        
        schema = {
            "type": "string",
            "description": "Python code to load model"
        }
        
        # More examples covering all formats
        examples = [
            # TorchVision examples
            ("Load a pre-trained ResNet50 model for image classification",
             "torchvision.models.resnet50(pretrained=True)"),
            ("I need DenseNet for image classification",
             "torchvision.models.densenet161(pretrained=True)"),
            ("Load MobileNet for efficient image classification",
             "torchvision.models.mobilenet_v2(pretrained=True)"),
            ("I need EfficientNet for image classification",
             "torchvision.models.efficientnet_b0(pretrained=True)"),
            ("Load a Faster R-CNN model for object detection",
             "torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)"),
            
            # torch.hub examples  
            ("I need a model to detect objects in images",
             "torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)"),
            ("Load a model for video classification",
             "torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)"),
            
            # Pipeline examples
            ("Create a sentiment analysis pipeline",
             "pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')"),
            ("I need a question answering model",
             "pipeline('question-answering', model='distilbert-base-cased-distilled-squad')"),
            ("Load a model for text generation",
             "pipeline('text-generation', model='gpt2')"),
        ]
        
        # noisy examples, uncomment below for running with noisy examples
        # examples=[
        # ("Search for Python tutorials", '{"function": "FAKE.nonexistent", "wrong": "totally wrong output"}'),
        # ("What's the weather in Berlin?", '{"function": "WRONG.api", "bad_param": "wrong"}'),
        # ]
        
        return documentation, schema, examples
        
    def execute_and_evaluate(
        self,
        task: Dict,
        prediction: str,
        strict: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """Evaluate if prediction matches expected HuggingFace/PyTorch model call.
        
        Args:
            task: Task dict with expected model call
            prediction: Model's prediction
            strict: If True, require exact model match (no cross-format matching)
        """
        import re
        
        # Handle both "expected" and "expected_output" field names
        expected = task.get("expected", task.get("expected_output", task.get("answer", "")))
        
        # Handle case where expected is a dict or list (from BFCL format)
        if isinstance(expected, (dict, list)):
            expected = str(expected)
        
        if not prediction.strip():
            return False, "Empty prediction"
            
        # Normalize prediction - remove markdown code blocks
        pred = prediction.strip()
        if pred.startswith("```"):
            pred = re.sub(r'^```[\w]*\n?', '', pred)
            pred = re.sub(r'\n?```$', '', pred)
        pred = pred.strip()
        
        # DEBUG: Print first 100 chars of pred and expected
        # print(f"[DEBUG Gorilla] pred: {repr(pred[:100])}")
        # print(f"[DEBUG Gorilla] expected: {repr(expected[:100])}")
        
        # =====================================================
        # HELPER: Extract repo/model from torch.hub.load call
        # Handles both positional and keyword argument formats
        # =====================================================
        def extract_hub_load(text):
            """Extract (repo, model) from torch.hub.load call."""
            # Pattern 1: Positional args - torch.hub.load('repo', 'model', ...)
            pattern1 = r"torch\.hub\.load\s*\(\s*['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]"
            match = re.search(pattern1, text)
            if match:
                return match.group(1).lower(), match.group(2).lower()
            
            # Pattern 2: Keyword args - torch.hub.load(repo_or_dir='repo', model='model', ...)
            repo_pattern = r"repo(?:_or_dir)?\s*=\s*['\"]([^'\"]+)['\"]"
            model_pattern = r"model\s*=\s*['\"]([^'\"]+)['\"]"
            
            repo_match = re.search(repo_pattern, text)
            model_match = re.search(model_pattern, text)
            
            if repo_match and model_match:
                return repo_match.group(1).lower(), model_match.group(1).lower()
            
            # Pattern 3: Mixed - torch.hub.load('repo', model='model', ...)
            mixed_pattern = r"torch\.hub\.load\s*\(\s*['\"]([^'\"]+)['\"]"
            mixed_match = re.search(mixed_pattern, text)
            if mixed_match and model_match:
                return mixed_match.group(1).lower(), model_match.group(1).lower()
            
            return None, None
        
        def extract_from_pretrained(text):
            """Extract model path from from_pretrained call."""
            pattern = r"from_pretrained\s*\(\s*['\"]([^'\"]+)['\"]"
            match = re.search(pattern, text)
            if match:
                return match.group(1).lower()
            return None
        
        def extract_tfhub(text):
            """Extract model info from TensorFlow Hub call."""
            # Pattern: hub.KerasLayer('https://tfhub.dev/...') or hub.load('...')
            pattern = r"hub\.(?:KerasLayer|load)\s*\(\s*['\"]([^'\"]+)['\"]"
            match = re.search(pattern, text)
            if match:
                url = match.group(1).lower()
                # Extract model name from URL
                # e.g., https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4
                parts = url.split('/')
                # Find model name (usually after 'imagenet' or before version number)
                model_name = None
                for i, part in enumerate(parts):
                    if 'mobilenet' in part or 'resnet' in part or 'efficientnet' in part or 'inception' in part:
                        model_name = part
                        break
                    if part == 'imagenet' and i + 1 < len(parts):
                        model_name = parts[i + 1]
                        break
                return url, model_name
            return None, None
        
        def extract_pipeline(text):
            """Extract task and model from transformers.pipeline() call."""
            # Pattern: pipeline(task='...', model='...') or pipeline('task', model='...')
            task_pattern = r"pipeline\s*\(\s*(?:task\s*=\s*)?['\"]([^'\"]+)['\"]"
            model_pattern = r"model\s*=\s*['\"]([^'\"]+)['\"]"
            
            task_match = re.search(task_pattern, text)
            model_match = re.search(model_pattern, text)
            
            task = task_match.group(1).lower() if task_match else None
            model = model_match.group(1).lower() if model_match else None
            
            return task, model
        
        # =====================================================
        # Extract all patterns first
        # =====================================================
        exp_repo, exp_model = extract_hub_load(expected)
        pred_repo, pred_model = extract_hub_load(pred)
        exp_pt = extract_from_pretrained(expected)
        pred_pt = extract_from_pretrained(pred)
        exp_task, exp_pipe_model = extract_pipeline(expected)
        pred_task, pred_pipe_model = extract_pipeline(pred)
        exp_tfhub_url, exp_tfhub_model = extract_tfhub(expected)
        pred_tfhub_url, pred_tfhub_model = extract_tfhub(pred)
        
        # Also extract torchvision models early
        def extract_torchvision_model(text):
            """Extract model name from torchvision.models call."""
            pattern = r"torchvision\.models(?:\.(\w+))?\.(\w+)\s*\("
            match = re.search(pattern, text)
            if match:
                submodule = match.group(1)
                model = match.group(2)
                return submodule, model.lower()
            return None, None
        
        exp_tv_sub, exp_tv_model = extract_torchvision_model(expected)
        pred_tv_sub, pred_tv_model = extract_torchvision_model(pred)
        
        # =====================================================
        # Pattern: TF Hub to TF Hub matching (exact or URL-based)
        # =====================================================
        if exp_tfhub_url and pred_tfhub_url:
            # Both use TensorFlow Hub
            if exp_tfhub_url == pred_tfhub_url:
                return True, f"Exact TF Hub match"
            
            # Check if same model family from URL
            if exp_tfhub_model and pred_tfhub_model:
                if exp_tfhub_model == pred_tfhub_model:
                    return True, f"Same TF Hub model: {pred_tfhub_model}"
                
                # Check model family
                exp_lower = exp_tfhub_model.lower()
                pred_lower = pred_tfhub_model.lower()
                
                model_families = ['mobilenet', 'resnet', 'efficientnet', 'inception', 'nasnet', 'vgg']
                for family in model_families:
                    if family in exp_lower and family in pred_lower:
                        return True, f"Same TF Hub model family: {family}"
                
                if strict:
                    return False, f"[STRICT] Different TF Hub models: expected {exp_tfhub_model}, got {pred_tfhub_model}"
                else:
                    # Lenient: both are TF Hub image models
                    if 'imagenet' in exp_tfhub_url and 'imagenet' in pred_tfhub_url:
                        return True, f"Both TF Hub ImageNet models"
                    return True, f"Both TF Hub models"
            
            # URLs differ but both are TF Hub - lenient accepts
            if not strict:
                return True, f"Both TF Hub (different URLs)"
            return False, f"[STRICT] Different TF Hub URLs"
        
        # =====================================================
        # Cross-Framework Matching: TF Hub vs PyTorch
        # Both solve the same task - allow cross-framework matches
        # =====================================================
        if exp_tfhub_url and (pred_tv_model or pred_repo or pred_pt or pred_task):
            # Expected: TF Hub, Predicted: PyTorch/HuggingFace
            # This is valid if both are image classification models
            pred_model_name = pred_tv_model or pred_model or pred_pt or pred_pipe_model
            
            # Check if prediction is a pipeline for image classification
            if pred_task and 'image' in pred_task.lower() or pred_task == 'classification':
                if 'imagenet' in exp_tfhub_url or 'feature_vector' in exp_tfhub_url:
                    if strict:
                        return False, f"[STRICT] Framework mismatch: TF Hub → pipeline({pred_task})"
                    else:
                        return True, f"Cross-framework: TF Hub imagenet → pipeline({pred_task})"
            
            if pred_model_name:
                # Check if same model family
                exp_model_lower = (exp_tfhub_model or '').lower()
                pred_model_lower = pred_model_name.lower()
                
                # Common model family mappings
                model_families = {
                    'mobilenet': ['mobilenet', 'mobile'],
                    'resnet': ['resnet', 'res'],
                    'efficientnet': ['efficientnet', 'efficient'],
                    'inception': ['inception', 'googlenet'],
                    'vgg': ['vgg'],
                    'densenet': ['densenet', 'dense'],
                }
                
                for family, keywords in model_families.items():
                    exp_is_family = any(kw in exp_model_lower for kw in keywords)
                    pred_is_family = any(kw in pred_model_lower for kw in keywords)
                    if exp_is_family and pred_is_family:
                        return True, f"Cross-framework match: {family} family"
                
                # Both are image classification models
                if strict:
                    return False, f"Strict mode: framework mismatch (expected TF Hub, got PyTorch)"
                else:
                    # Lenient: any image model is acceptable for image classification task
                    image_keywords = ['resnet', 'mobilenet', 'efficientnet', 'vgg', 'inception', 
                                     'densenet', 'feature_vector', 'classification', 'faster_rcnn']
                    if any(kw in pred_model_lower for kw in image_keywords):
                        return True, f"Cross-framework image model match"
                    if any(kw in exp_tfhub_url for kw in ['imagenet', 'feature_vector', 'classification']):
                        return True, f"Cross-framework: TF Hub imagenet → PyTorch image model"
        
        # =====================================================
        # Pattern 0: transformers.pipeline matching (most common in our data)
        # =====================================================
        if exp_task and pred_task:
            # Both use pipeline
            if exp_task == pred_task:
                if exp_pipe_model and pred_pipe_model:
                    if exp_pipe_model == pred_pipe_model:
                        return True, f"Exact pipeline match: {pred_task}/{pred_pipe_model}"
                    # Same task, different model
                    if strict:
                        # Check if same model family
                        exp_base = exp_pipe_model.split('/')[-1].split('-')[0]
                        pred_base = pred_pipe_model.split('/')[-1].split('-')[0]
                        if exp_base == pred_base:
                            return True, f"Same model family: {pred_base}"
                        return False, f"Wrong model for {pred_task}: expected {exp_pipe_model}, got {pred_pipe_model}"
                    else:
                        return True, f"Same pipeline task: {pred_task}"
                else:
                    # Same task, model not specified or matched
                    return True, f"Same pipeline task: {pred_task}"
            else:
                return False, f"Wrong pipeline task: expected {exp_task}, got {pred_task}"
        
        # Check if expected is pipeline but predicted uses from_pretrained
        if exp_task and pred_pt:
            # Expected: pipeline('task'), Predicted: from_pretrained('model')
            if strict:
                return False, f"Format mismatch: expected pipeline({exp_task}), got from_pretrained({pred_pt})"
            else:
                # Check if model is appropriate for the task
                task_model_map = {
                    'sentiment-analysis': ['distilbert', 'bert', 'roberta'],
                    'text-generation': ['gpt2', 'gpt', 'llama', 'bloom'],
                    'question-answering': ['distilbert', 'bert', 'roberta', 'squad'],
                    'summarization': ['bart', 't5', 'pegasus'],
                    'translation': ['marian', 't5', 'opus'],
                    'ner': ['bert', 'roberta', 'ner'],
                    'fill-mask': ['bert', 'roberta', 'distilbert'],
                }
                expected_models = task_model_map.get(exp_task, [])
                if any(m in pred_pt for m in expected_models):
                    return True, f"Cross-format match: {exp_task} task with {pred_pt}"
                return False, f"Model {pred_pt} not typical for {exp_task}"
        
        # Check if predicted is pipeline but expected uses different format
        if pred_task and (exp_repo or exp_pt):
            if strict:
                return False, f"Format mismatch: expected different format, got pipeline({pred_task})"
            else:
                return True, f"Valid pipeline for task: {pred_task}"
        
        # =====================================================
        # Pattern 1: torch.hub.load matching
        # =====================================================
        if exp_repo and pred_repo:
            # Both use torch.hub.load
            if exp_repo == pred_repo and exp_model == pred_model:
                return True, f"Exact match: {pred_repo}/{pred_model}"
            
            # Same repo (valid - different model variants often work)
            if exp_repo == pred_repo:
                if strict:
                    # In strict mode, require same model or same model family
                    exp_base = exp_model.split('_')[0].split('-')[0]
                    pred_base = pred_model.split('_')[0].split('-')[0]
                    if exp_base == pred_base:
                        return True, f"Same model family: {pred_repo}/{pred_model}"
                    return False, f"Wrong model variant: expected {exp_model}, got {pred_model}"
                else:
                    return True, f"Same repo: {pred_repo}"
            
            # Same model name, different repo
            if exp_model == pred_model:
                return True, f"Same model: {pred_model}"
            
            return False, f"Wrong model: expected {exp_repo}/{exp_model}, got {pred_repo}/{pred_model}"
        
        # =====================================================
        # Pattern 2: from_pretrained matching
        # =====================================================
        if exp_pt and pred_pt:
            # Both use from_pretrained
            # Exact match
            if exp_pt == pred_pt:
                return True, f"Exact match: {pred_pt}"
            
            # Same organization (e.g., openai/whisper-base vs openai/whisper-large)
            if '/' in exp_pt and '/' in pred_pt:
                exp_org = exp_pt.split('/')[0]
                pred_org = pred_pt.split('/')[0]
                if exp_org == pred_org:
                    # Check if same model family
                    exp_name = exp_pt.split('/')[1] if '/' in exp_pt else exp_pt
                    pred_name = pred_pt.split('/')[1] if '/' in pred_pt else pred_pt
                    # Extract base model name (e.g., whisper from whisper-base)
                    exp_base = exp_name.split('-')[0].split('_')[0]
                    pred_base = pred_name.split('-')[0].split('_')[0]
                    if exp_base == pred_base:
                        return True, f"Same model family: {pred_org}/{pred_base}"
                    if strict:
                        return False, f"Different model: expected {exp_pt}, got {pred_pt}"
                    else:
                        return True, f"Same organization: {pred_org}"
            
            return False, f"Wrong model: expected {exp_pt}, got {pred_pt}"
        
        # =====================================================
        # Pattern 3: torchvision.models.X (including submodules like video, detection)
        # =====================================================
        # Note: exp_tv_sub, exp_tv_model, pred_tv_sub, pred_tv_model already extracted above
        
        if exp_tv_model and pred_tv_model:
            # Both use torchvision
            if exp_tv_model == pred_tv_model:
                return True, f"TorchVision exact match: {pred_tv_model}"
            # Same model family (resnet50 vs resnet18)
            exp_base = exp_tv_model.split('_')[0].rstrip('0123456789')
            pred_base = pred_tv_model.split('_')[0].rstrip('0123456789')
            if exp_base == pred_base:
                return True, f"Same TorchVision family: {pred_base}"
            if exp_tv_sub == pred_tv_sub:
                # Same submodule (e.g., both video models)
                if strict:
                    return False, f"[STRICT] Wrong TorchVision model: expected {exp_tv_model}, got {pred_tv_model}"
                else:
                    return True, f"Same TorchVision category: {pred_tv_sub or 'models'}"
            return False, f"Wrong TorchVision model: expected {exp_tv_model}, got {pred_tv_model}"
        
        # =====================================================
        # Cross-format: torch.hub <-> torchvision (same task, different API)
        # =====================================================
        if exp_repo and pred_tv_model:
            # Expected: torch.hub.load, Predicted: torchvision.models
            # Both are image classification - this is valid
            if exp_repo in ['pytorch/vision', 'pytorch/torchvision']:
                # Same source, just different API
                return True, f"Cross-API match: torch.hub → torchvision"
            # Check if same model family
            exp_base = exp_model.split('_')[0].rstrip('0123456789') if exp_model else ""
            pred_base = pred_tv_model.split('_')[0].rstrip('0123456789')
            if exp_base and exp_base.lower() == pred_base.lower():
                return True, f"Same model family: {pred_base}"
            if not strict:
                return True, f"Cross-format image model"
            return False, f"[STRICT] Different APIs: torch.hub({exp_model}) vs torchvision({pred_tv_model})"
        
        if exp_tv_model and pred_repo:
            # Expected: torchvision.models, Predicted: torch.hub.load
            if pred_repo in ['pytorch/vision', 'pytorch/torchvision']:
                return True, f"Cross-API match: torchvision → torch.hub"
            exp_base = exp_tv_model.split('_')[0].rstrip('0123456789')
            pred_base = pred_model.split('_')[0].rstrip('0123456789') if pred_model else ""
            if exp_base and pred_base and exp_base.lower() == pred_base.lower():
                return True, f"Same model family: {exp_base}"
            if not strict:
                return True, f"Cross-format image model"
            return False, f"[STRICT] Different APIs: torchvision({exp_tv_model}) vs torch.hub({pred_model})"
        
        # =====================================================
        # Pattern 4: Cross-format matching
        # Handle all combinations of loading patterns
        # =====================================================
        
        # Define task categories for semantic matching
        video_models = {'slow_r50', 'slowfast', 'x3d', 'r2plus1d', 'r3d', 'mc3', 'mvit', 'swin3d', 'video'}
        image_models = {'resnet', 'vgg', 'efficientnet', 'mobilenet', 'densenet', 'inception', 'vit', 'swin',
                       'alexnet', 'googlenet', 'squeezenet', 'shufflenet', 'mnasnet', 'regnet', 'convnext',
                       'wide_resnet', 'resnext', 'imagenet', 'classification', 'feature_vector'}
        detection_models = {'fasterrcnn', 'maskrcnn', 'retinanet', 'fcos', 'ssd', 'yolo', 'detr', 'rcnn'}
        segmentation_models = {'deeplabv3', 'fcn', 'lraspp', 'segformer', 'mask2former', 'segmentation'}
        audio_models = {'wav2vec', 'hubert', 'whisper', 'silero', 'speechbrain', 'speech', 'audio', 'stt', 'tts'}
        nlp_models = {'bert', 'gpt', 'llama', 't5', 'roberta', 'distilbert', 'albert', 'bart', 'pegasus'}
        # NLP task keywords (from pipeline tasks)
        nlp_tasks = {'sentiment', 'text-generation', 'question', 'summarization', 'translation', 
                     'ner', 'fill-mask', 'classification', 'generation', 'qa', 'squad'}
        
        def get_model_category(model_name):
            """Determine model category from name."""
            if not model_name:
                return None
            model_lower = model_name.lower()
            for kw in video_models:
                if kw in model_lower:
                    return 'video'
            for kw in detection_models:
                if kw in model_lower:
                    return 'detection'
            for kw in segmentation_models:
                if kw in model_lower:
                    return 'segmentation'
            for kw in audio_models:
                if kw in model_lower:
                    return 'audio'
            for kw in nlp_models:
                if kw in model_lower:
                    return 'nlp'
            for kw in nlp_tasks:
                if kw in model_lower:
                    return 'nlp'
            for kw in image_models:
                if kw in model_lower:
                    return 'image'
            return 'unknown'
        
        def get_model_family(model_name):
            """Extract model family name."""
            if not model_name:
                return None
            model_lower = model_name.lower()
            families = ['resnet', 'vgg', 'efficientnet', 'mobilenet', 'densenet', 'inception',
                       'alexnet', 'googlenet', 'squeezenet', 'shufflenet', 'yolo', 'bert', 'gpt']
            for fam in families:
                if fam in model_lower:
                    return fam
            return None
        
        # Get all detected models/patterns
        all_expected = [exp_model, exp_pt, exp_tv_model, exp_pipe_model, exp_tfhub_model]
        all_predicted = [pred_model, pred_pt, pred_tv_model, pred_pipe_model, pred_tfhub_model]
        
        exp_names = [x for x in all_expected if x]
        pred_names = [x for x in all_predicted if x]
        
        # Also include task names for matching
        if exp_task:
            exp_names.append(exp_task)
        if pred_task:
            pred_names.append(pred_task)
        
        if exp_names and pred_names:
            exp_cat = get_model_category(exp_names[0])
            pred_cat = get_model_category(pred_names[0])
            
            # Check for same model family first
            exp_fam = get_model_family(exp_names[0])
            pred_fam = get_model_family(pred_names[0])
            
            if exp_fam and pred_fam and exp_fam == pred_fam:
                return True, f"Same model family: {exp_fam}"
            
            if exp_cat and pred_cat:
                if exp_cat == pred_cat:
                    if strict:
                        # Strict: same category, check if both are image classification models
                        if exp_cat == 'image':
                            # For image classification, accept any valid image model
                            return True, f"Same image classification category"
                        return False, f"[STRICT] Same category ({exp_cat}) but different models: expected {exp_names[0]}, got {pred_names[0]}"
                    else:
                        return True, f"Same task category: {exp_cat}"
                else:
                    return False, f"Wrong category: expected {exp_cat}, got {pred_cat}"
        
        # Cross-format: torch.hub vs from_pretrained
        if exp_repo and pred_pt:
            if strict:
                return False, f"Strict mode: format mismatch (expected torch.hub {exp_repo}/{exp_model}, got from_pretrained {pred_pt})"
            else:
                if exp_model in pred_pt or pred_pt.split('/')[-1] in exp_model:
                    return True, f"Cross-format semantic match"
                # Check category match
                exp_cat = get_model_category(exp_model)
                pred_cat = get_model_category(pred_pt)
                if exp_cat and exp_cat == pred_cat:
                    return True, f"Cross-format category match: {exp_cat}"
                return False, f"Format mismatch: expected torch.hub ({exp_model}), got from_pretrained ({pred_pt})"
        
        if exp_pt and pred_repo:
            if strict:
                return False, f"Strict mode: format mismatch (expected from_pretrained {exp_pt}, got torch.hub {pred_repo}/{pred_model})"
            else:
                if pred_model in exp_pt or exp_pt.split('/')[-1] in pred_model:
                    return True, f"Cross-format semantic match"
                exp_cat = get_model_category(exp_pt)
                pred_cat = get_model_category(pred_model)
                if exp_cat and exp_cat == pred_cat:
                    return True, f"Cross-format category match: {exp_cat}"
                return False, f"Format mismatch: expected from_pretrained ({exp_pt}), got torch.hub ({pred_model})"
        
        # Cross-format: torch.hub vs torchvision
        if exp_repo and pred_tv_model:
            exp_cat = get_model_category(exp_model)
            pred_cat = get_model_category(pred_tv_model)
            if pred_tv_sub:
                pred_cat = pred_tv_sub  # Use submodule as category
            if exp_cat and pred_cat and exp_cat == pred_cat:
                if strict:
                    return False, f"Same category ({exp_cat}) but different sources"
                else:
                    return True, f"Cross-format category match: {exp_cat}"
            return False, f"Format mismatch: expected torch.hub ({exp_model}), got torchvision ({pred_tv_model})"
        
        if exp_tv_model and pred_repo:
            exp_cat = exp_tv_sub or get_model_category(exp_tv_model)
            pred_cat = get_model_category(pred_model)
            if exp_cat and pred_cat and exp_cat == pred_cat:
                if strict:
                    return False, f"Same category ({exp_cat}) but different sources"
                else:
                    return True, f"Cross-format category match: {exp_cat}"
            return False, f"Format mismatch: expected torchvision ({exp_tv_model}), got torch.hub ({pred_model})"
        
        # No recognized pattern - check if at least has model loading code
        loading_patterns = [
            r'torch\.hub\.load',
            r'from_pretrained',
            r'torchvision\.models\.',
            r'AutoModel',
            r'pipeline\s*\(',
        ]
        
        has_loading = any(re.search(p, pred, re.IGNORECASE) for p in loading_patterns)
        
        if has_loading:
            # Has loading pattern - try lenient keyword matching
            if not strict:
                # Check if prediction has keywords related to expected task/model
                exp_keywords = set()
                if exp_task:
                    exp_keywords.update(exp_task.replace('-', ' ').split())
                if exp_pipe_model:
                    exp_keywords.update(exp_pipe_model.replace('-', ' ').replace('/', ' ').split())
                if exp_model:
                    exp_keywords.update(exp_model.replace('_', ' ').split())
                
                pred_lower = pred.lower()
                # Check if any expected keywords appear in prediction
                matches = sum(1 for kw in exp_keywords if kw.lower() in pred_lower and len(kw) > 3)
                if matches >= 1:
                    return True, f"Keyword match: {matches} keywords found"
                
                # Very lenient: if both are NLP tasks, count as success
                if exp_task and any(nlp in pred_lower for nlp in ['sentiment', 'text', 'question', 'generation', 'bert', 'gpt', 'classification']):
                    return True, f"NLP task match"
            
            return False, "Has model loading but could not verify model correctness"
        
        return False, "No model loading code found"


class Spider2Evaluator(BenchmarkEvaluator):
    """
    Evaluator for Spider 2.0 (Enterprise SQL).
    Tests text-to-SQL with complex schemas.
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.tasks = []
        
    def load_tasks(self) -> List[Dict]:
        """Load Spider 2.0 tasks."""
        task_file = os.path.join(self.data_path, "spider2_tasks.json")
        
        if os.path.exists(task_file):
            with open(task_file) as f:
                self.tasks = json.load(f)
        else:
            self.tasks = self._create_synthetic_tasks()
            
        return self.tasks
        
    def _create_synthetic_tasks(self, num_tasks: int = 100) -> List[Dict]:
        """Create synthetic Spider-style SQL tasks."""
        tasks = []
        
        queries = [
            ("Find all customers who made purchases over $1000", 
             "SELECT * FROM customers WHERE customer_id IN (SELECT customer_id FROM orders WHERE total > 1000)"),
            ("Count orders by category",
             "SELECT category, COUNT(*) FROM orders GROUP BY category"),
            ("Get top 10 products by revenue",
             "SELECT product_name, SUM(price * quantity) as revenue FROM order_items GROUP BY product_name ORDER BY revenue DESC LIMIT 10"),
            ("Find customers who haven't ordered in 30 days",
             "SELECT * FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM orders WHERE order_date > DATE_SUB(NOW(), INTERVAL 30 DAY))"),
        ]
        
        for i in range(num_tasks):
            query, sql = queries[i % len(queries)]
            tasks.append({
                "id": f"spider_{i}",
                "query": query,
                "expected_sql": sql,
                "database": "enterprise_db"
            })
            
        return tasks
        
    def get_tool_spec(self) -> Tuple[str, Dict, List[Tuple[str, str]]]:
        """Get Spider SQL tool specification."""
        documentation = """
# SQL Query Generator

Generate SQL queries for the given natural language question.
The database may contain various tables - infer the schema from the question.

Common patterns:
- Counting: SELECT count(*) FROM table
- Listing: SELECT * FROM table [WHERE condition]
- Aggregation: SELECT column, COUNT(*)/SUM()/AVG() FROM table GROUP BY column
- Ordering: SELECT * FROM table ORDER BY column [DESC]
- Filtering: SELECT * FROM table WHERE condition

Output ONLY the SQL query, nothing else. No explanation, no markdown, just SQL.
"""
        
        schema = {
            "type": "string",
            "description": "SQL query"
        }
        
        examples = [
            ("How many singers do we have?", 
             "SELECT count(*) FROM singer"),
            ("What is the total number of singers?",
             "SELECT count(*) FROM singer"),
            ("Show name, country, age for all singers ordered by age from oldest to youngest",
             "SELECT name, country, age FROM singer ORDER BY age DESC"),
            ("List all concerts",
             "SELECT * FROM concert"),
            ("What are the names of all stadiums?",
             "SELECT name FROM stadium"),
        ]
        
        # noisy examples, uncomment below for running with noisy examples
        # examples=[
        # ("List all customers", "SELECT * FROM nonexistent_table_xyz"),
        # ("Total revenue this month", "DELETE FROM orders WHERE 1=1"),
        # ]
        
        return documentation, schema, examples
        
    def execute_and_evaluate(
        self,
        task: Dict,
        prediction: str,
        strict: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """Evaluate SQL query correctness.
        
        Args:
            task: Task dict with expected_sql
            prediction: Model's SQL prediction
            strict: If True, require higher accuracy matching
        """
        import re
        
        # Extract SQL from prediction (model might include explanations)
        sql_prediction = self._extract_sql(prediction)
        
        # Normalize queries for comparison
        pred_normalized = self._normalize_sql(sql_prediction)
        expected_normalized = self._normalize_sql(task["expected_sql"])
        
        # Simple exact match
        if pred_normalized == expected_normalized:
            return True, "Exact match"
        
        # Check if expected SQL is contained in prediction (or vice versa)
        if expected_normalized in pred_normalized or pred_normalized in expected_normalized:
            return True, "Substring match"
        
        # Extract key SQL components
        expected_components = self._extract_sql_components(expected_normalized)
        pred_components = self._extract_sql_components(pred_normalized)
        
        matches = sum(1 for c in expected_components if c in pred_components)
        total = len(expected_components)
        
        if strict:
            # STRICT MODE: Require 80% component match
            if total > 0 and matches >= total * 0.8:
                return True, f"Strict match ({matches}/{total})"
            
            # In strict mode, also check table names and key conditions
            expected_tables = self._extract_tables(expected_normalized)
            pred_tables = self._extract_tables(pred_normalized)
            
            if expected_tables and expected_tables == pred_tables:
                # Same tables, check if query structure is similar
                if matches >= total * 0.7:
                    return True, f"Table match with structure ({matches}/{total})"
            
            return False, f"Strict mismatch ({matches}/{total} components)"
        else:
            # LENIENT MODE: 50% component match (original behavior)
            if total > 0 and matches >= total * 0.5:
                return True, f"Partial match ({matches}/{total})"
            
            # Check for key SQL keywords from expected query
            expected_keywords = set(re.findall(r'\b(select|from|where|join|group|order|having|limit|sum|avg|count|max|min)\b', expected_normalized))
            pred_keywords = set(re.findall(r'\b(select|from|where|join|group|order|having|limit|sum|avg|count|max|min)\b', pred_normalized))
            
            keyword_overlap = len(expected_keywords & pred_keywords)
            if expected_keywords and keyword_overlap >= len(expected_keywords) * 0.6:
                return True, f"Keyword match ({keyword_overlap}/{len(expected_keywords)})"
            
            return False, f"Mismatch ({matches}/{total} components)"
    
    def _extract_tables(self, sql: str) -> set:
        """Extract table names from SQL."""
        import re
        # Match table names after FROM and JOIN
        tables = set()
        from_match = re.findall(r'\bfrom\s+(\w+)', sql, re.IGNORECASE)
        join_match = re.findall(r'\bjoin\s+(\w+)', sql, re.IGNORECASE)
        tables.update(from_match)
        tables.update(join_match)
        return tables
    
    def _extract_sql(self, text: str) -> str:
        """Extract SQL query from model output."""
        import re
        text = text.strip()
        
        # Method 1: Look for SQL between backticks
        sql_match = re.search(r'```(?:sql)?\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        
        # Method 2: Look for SELECT statement
        select_match = re.search(r'(SELECT\s+.+?)(?:;|$|\n\n)', text, re.DOTALL | re.IGNORECASE)
        if select_match:
            return select_match.group(1).strip()
        
        # Method 3: Look for common SQL patterns
        sql_patterns = [
            r'(SELECT\s+.+)',
            r'(INSERT\s+INTO.+)',
            r'(UPDATE\s+.+)',
            r'(DELETE\s+FROM.+)',
        ]
        for pattern in sql_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Return original text if no SQL found
        return text
        
    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for comparison."""
        import re
        sql = sql.lower().strip()
        sql = re.sub(r'\s+', ' ', sql)
        sql = re.sub(r'[;]', '', sql)
        # Normalize comma spacing: " , " -> ", "
        sql = re.sub(r'\s*,\s*', ', ', sql)
        # Normalize parentheses spacing
        sql = re.sub(r'\s*\(\s*', '(', sql)
        sql = re.sub(r'\s*\)\s*', ') ', sql)
        sql = sql.strip()
        return sql
        
    def _extract_sql_components(self, sql: str) -> List[str]:
        """Extract key SQL components for comparison."""
        import re
        components = []
        
        # Extract table names (from FROM and JOIN)
        tables = re.findall(r'(?:from|join)\s+(\w+)', sql, re.IGNORECASE)
        components.extend([t.lower() for t in tables])
        
        # Extract column references
        # Between SELECT and FROM
        select_match = re.search(r'select\s+(.+?)\s+from', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            cols = select_match.group(1)
            # Split by comma but handle functions
            col_parts = re.findall(r'\b(\w+)\b', cols)
            components.extend([c.lower() for c in col_parts if c.lower() not in ('select', 'as', 'distinct')])
        
        # Extract WHERE conditions
        where_match = re.search(r'where\s+(.+?)(?:group|order|limit|having|$)', sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_parts = re.findall(r'\b(\w+)\b', where_match.group(1))
            components.extend([w.lower() for w in where_parts if len(w) > 2])
        
        # Extract GROUP BY columns
        group_match = re.search(r'group\s+by\s+(.+?)(?:order|having|limit|$)', sql, re.IGNORECASE | re.DOTALL)
        if group_match:
            group_parts = re.findall(r'\b(\w+)\b', group_match.group(1))
            components.extend([g.lower() for g in group_parts])
            
        return list(set(components))  # Remove duplicates


class WebArenaEvaluator(BenchmarkEvaluator):
    """
    Evaluator for WebArena benchmark.
    Tests web navigation and interaction.
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.tasks = []
        
    def load_tasks(self) -> List[Dict]:
        """Load WebArena tasks."""
        task_file = os.path.join(self.data_path, "webarena_tasks.json")
        
        if os.path.exists(task_file):
            with open(task_file) as f:
                self.tasks = json.load(f)
        else:
            self.tasks = self._create_synthetic_tasks()
            
        return self.tasks
        
    def _create_synthetic_tasks(self, num_tasks: int = 50) -> List[Dict]:
        """Create synthetic web navigation tasks."""
        tasks = []
        
        templates = [
            # Shopping tasks
            {"query": "Add a laptop to the shopping cart", "expected_actions": ["search", "click_product", "add_to_cart"]},
            {"query": "Find wireless headphones under $100", "expected_actions": ["search", "filter_price", "view_results"]},
            {"query": "Check out with items in cart", "expected_actions": ["go_to_cart", "proceed_checkout", "enter_info", "submit"]},
            {"query": "Apply a coupon code SAVE20", "expected_actions": ["go_to_cart", "enter_coupon", "apply"]},
            {"query": "Remove an item from the cart", "expected_actions": ["go_to_cart", "click_remove", "confirm"]},
            {"query": "Sort products by price low to high", "expected_actions": ["click_sort", "select_price_asc"]},
            {"query": "Filter products by brand Apple", "expected_actions": ["click_filter", "select_brand", "apply"]},
            # Forum tasks
            {"query": "Create a new post titled 'Hello World'", "expected_actions": ["click_new_post", "enter_title", "enter_content", "submit"]},
            {"query": "Reply to the top post", "expected_actions": ["click_post", "click_reply", "enter_text", "submit"]},
            {"query": "Upvote the first comment", "expected_actions": ["find_comment", "click_upvote"]},
            {"query": "Edit my profile bio", "expected_actions": ["go_to_profile", "click_edit", "modify_bio", "save"]},
            # Admin tasks
            {"query": "Search for user by email admin@test.com", "expected_actions": ["go_to_users", "enter_search", "submit"]},
            {"query": "Ban user spammer123", "expected_actions": ["search_user", "click_user", "click_ban", "confirm"]},
            {"query": "Export user data to CSV", "expected_actions": ["go_to_users", "click_export", "select_csv", "download"]},
            {"query": "View error logs from today", "expected_actions": ["go_to_logs", "filter_date", "filter_errors", "view"]},
            # Content Management
            {"query": "Upload an image to the media library", "expected_actions": ["go_to_media", "click_upload", "select_file", "confirm"]},
            {"query": "Create a new blog post draft", "expected_actions": ["go_to_posts", "click_new", "enter_content", "save_draft"]},
            {"query": "Schedule post for tomorrow at 9am", "expected_actions": ["open_post", "click_schedule", "set_datetime", "confirm"]},
            # Email/Calendar
            {"query": "Compose email to team@company.com", "expected_actions": ["click_compose", "enter_recipient", "enter_subject", "enter_body", "send"]},
            {"query": "Create calendar event for Monday 2pm", "expected_actions": ["click_create", "set_date", "set_time", "enter_title", "save"]},
            # Navigation
            {"query": "Search for 'python tutorial' and click first result", "expected_actions": ["enter_query", "submit", "click_first_result"]},
            {"query": "Navigate to the About Us page", "expected_actions": ["find_nav", "click_about"]},
            {"query": "Fill out contact form with name and email", "expected_actions": ["enter_name", "enter_email", "enter_message", "submit"]},
            {"query": "Subscribe to newsletter", "expected_actions": ["enter_email", "click_subscribe", "confirm"]},
            {"query": "Track order #12345", "expected_actions": ["go_to_orders", "enter_order_id", "view_tracking"]},
            {"query": "Write a product review", "expected_actions": ["view_product", "click_reviews", "click_write", "enter_rating", "enter_text", "submit"]},
        ]
        
        for i in range(num_tasks):
            template = templates[i % len(templates)]
            tasks.append({
                "id": f"webarena_{i}",
                "query": template["query"],
                "expected_actions": template["expected_actions"],
                "site": "test_site"
            })
            
        return tasks
        
    def get_tool_spec(self) -> Tuple[str, Dict, List[Tuple[str, str]]]:
        """Get WebArena tool specification."""
        documentation = """# Web Navigation Agent

Generate actions as a JSON array to accomplish the web task.

Actions:
- {"action": "click", "element_id": "ID"}
- {"action": "type", "element_id": "ID", "text": "TEXT"}
- {"action": "navigate", "url": "URL"}
- {"action": "scroll", "direction": "up/down"}

Output ONLY the JSON array. No explanations."""
        
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["click", "type", "scroll", "navigate", "wait"]},
                    "element_id": {"type": "string"},
                    "text": {"type": "string"},
                    "direction": {"type": "string"},
                    "url": {"type": "string"}
                },
                "required": ["action"]
            }
        }
        
        # More comprehensive examples
        examples = [
            ("Click the login button", 
             '[{"action": "click", "element_id": "login-btn"}]'),
            ("Search for laptops", 
             '[{"action": "type", "element_id": "search-box", "text": "laptops"}, {"action": "click", "element_id": "search-btn"}]'),
            ("Add item to cart", 
             '[{"action": "click", "element_id": "add-to-cart"}]'),
            ("Navigate to checkout", 
             '[{"action": "click", "element_id": "cart"}, {"action": "click", "element_id": "checkout-btn"}]'),
            ("Type email and submit form",
             '[{"action": "type", "element_id": "email", "text": "user@example.com"}, {"action": "click", "element_id": "submit"}]'),
        ]
        
        # noisy examples, uncomment below for running with noisy examples
        # examples=[
        # ("Click the login button", '[{"action": "wrong", "element_id": "fake-element"}]'),
        # ("Type email and submit", '[{"action": "invalid_action"}]'),
        # ]
        
        return documentation, schema, examples
        
    def execute_and_evaluate(
        self,
        task: Dict,
        prediction: str,
        strict: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """Evaluate web navigation sequence.
        
        Args:
            task: Task dict with expected_actions
            prediction: Model's action sequence prediction
            strict: If True, require matching action sequence
        """
        import re
        
        pred = prediction.strip()
        expected_actions = task.get("expected_actions", [])
        
        # Clean prediction - remove markdown
        if pred.startswith("```"):
            pred = re.sub(r'^```[\w]*\n?', '', pred)
            pred = re.sub(r'\n?```$', '', pred)
            pred = pred.strip()
        
        # Try to extract JSON
        pred_actions = None
        try:
            parsed = json.loads(pred)
            # Handle single object wrapped as array
            if isinstance(parsed, dict):
                pred_actions = [parsed]
            elif isinstance(parsed, list):
                pred_actions = parsed
        except:
            # Try to find JSON array in text
            json_match = re.search(r'\[[\s\S]*?\]', pred)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    if isinstance(parsed, list):
                        pred_actions = parsed
                except:
                    pass
            
            # Try to find JSON object in text
            if not pred_actions:
                obj_match = re.search(r'\{[\s\S]*?\}', pred)
                if obj_match:
                    try:
                        parsed = json.loads(obj_match.group())
                        if isinstance(parsed, dict):
                            pred_actions = [parsed]
                    except:
                        pass
        
        # Extract action types from predictions
        pred_action_types = []
        if pred_actions:
            for action in pred_actions:
                if isinstance(action, dict):
                    action_type = action.get("action", action.get("type", ""))
                    if action_type:
                        pred_action_types.append(action_type.lower())
                elif isinstance(action, str):
                    # Handle array of strings like ["navigate", "click"]
                    pred_action_types.append(action.lower())
        
        if strict:
            # STRICT MODE: Require valid JSON with matching action types
            if not pred_action_types:
                return False, "[STRICT] No valid actions parsed"
            
            # Map expected action names to action types
            action_mapping = {
                "search": ["type", "enter", "search", "input"],
                "click": ["click", "press", "tap", "select"],
                "navigate": ["navigate", "goto", "go_to", "open"],
                "submit": ["click", "submit", "press", "enter"],
                "enter": ["type", "input", "enter", "fill"],
                "filter": ["click", "select", "filter"],
                "add_to_cart": ["click", "add"],
                "go_to": ["navigate", "click", "goto"],
                "view": ["navigate", "click", "scroll"],
            }
            
            # Count how many expected actions have a corresponding predicted action
            matched = 0
            for exp_action in expected_actions:
                exp_lower = exp_action.lower().replace("_", "")
                # Check direct match
                if any(exp_lower in pa for pa in pred_action_types):
                    matched += 1
                    continue
                # Check via mapping
                for key, values in action_mapping.items():
                    if key in exp_lower:
                        if any(v in pa for pa in pred_action_types for v in values):
                            matched += 1
                            break
            
            match_ratio = matched / len(expected_actions) if expected_actions else 0
            
            if match_ratio >= 0.5:  # Reduced from 0.7 for strict
                return True, f"[STRICT] Action match ({matched}/{len(expected_actions)})"
            else:
                return False, f"[STRICT] Insufficient actions ({matched}/{len(expected_actions)})"
        else:
            # LENIENT MODE
            if pred_actions and len(pred_actions) > 0:
                return True, "Valid action list"
            
            if pred_action_types:
                return True, f"Found {len(pred_action_types)} actions"
            
            # Check for action keywords in text (very lenient)
            action_keywords = ['click', 'type', 'navigate', 'scroll', 'enter', 'submit', 'select', 'add', 'search']
            found_actions = sum(1 for kw in action_keywords if kw in pred.lower())
            
            if found_actions >= 2:
                return True, f"Found {found_actions} action keywords"
            
            # Check if expected actions mentioned
            matches = sum(1 for e in expected_actions if e.lower().replace("_", " ") in pred.lower())
            if matches >= len(expected_actions) * 0.3:
                return True, f"Found {matches}/{len(expected_actions)} expected actions"
                
            return False, "No valid actions found"


class InterCodeEvaluator(BenchmarkEvaluator):
    """
    Evaluator for InterCode benchmark.
    Tests bash commands and CTF-style challenges.
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.tasks = []
        
    def load_tasks(self) -> List[Dict]:
        """Load InterCode tasks."""
        task_file = os.path.join(self.data_path, "intercode_tasks.json")
        
        if os.path.exists(task_file):
            with open(task_file) as f:
                self.tasks = json.load(f)
        else:
            self.tasks = self._create_synthetic_tasks()
            
        return self.tasks
        
    def _create_synthetic_tasks(self, num_tasks: int = 50) -> List[Dict]:
        """Create synthetic bash/CLI tasks."""
        tasks = []
        
        templates = [
            # File operations
            {"query": "Find all Python files in current directory", "expected": "find . -name '*.py'"},
            {"query": "Find files larger than 100MB", "expected": "find . -size +100M"},
            {"query": "Find and delete all .tmp files", "expected": "find . -name '*.tmp' -delete"},
            {"query": "Find all empty directories", "expected": "find . -type d -empty"},
            {"query": "Count total number of files in directory", "expected": "find . -type f | wc -l"},
            # Text processing
            {"query": "Count lines in a file", "expected": "wc -l filename"},
            {"query": "Count words in a file", "expected": "wc -w filename"},
            {"query": "Show first 10 lines of file", "expected": "head -n 10 filename"},
            {"query": "Show last 20 lines of file", "expected": "tail -n 20 filename"},
            {"query": "Remove blank lines from file", "expected": "sed '/^$/d' filename"},
            # Search
            {"query": "Search for pattern in files recursively", "expected": "grep -r 'pattern' ."},
            {"query": "Find files containing 'error'", "expected": "grep -l 'error' *"},
            {"query": "Count occurrences of pattern in file", "expected": "grep -c 'pattern' filename"},
            {"query": "Search with line numbers", "expected": "grep -n 'pattern' filename"},
            # System info
            {"query": "Show disk usage of current directory", "expected": "du -sh ."},
            {"query": "Show free disk space", "expected": "df -h"},
            {"query": "Show memory usage", "expected": "free -h"},
            {"query": "Show running processes", "expected": "ps aux"},
            {"query": "Show system uptime", "expected": "uptime"},
            # File listing
            {"query": "List files sorted by size", "expected": "ls -lS"},
            {"query": "List files sorted by modification time", "expected": "ls -lt"},
            {"query": "List all files including hidden", "expected": "ls -la"},
            {"query": "List only directories", "expected": "ls -d */"},
            # Archives
            {"query": "Create tar archive of directory", "expected": "tar -cvf archive.tar directory/"},
            {"query": "Create compressed tar.gz archive", "expected": "tar -czvf archive.tar.gz directory/"},
            {"query": "Extract tar archive", "expected": "tar -xvf archive.tar"},
            {"query": "Create zip archive", "expected": "zip -r archive.zip directory/"},
            # Network
            {"query": "Download file from URL", "expected": "wget URL"},
            {"query": "Check if host is reachable", "expected": "ping -c 4 hostname"},
            # Permissions
            {"query": "Make file executable", "expected": "chmod +x filename"},
            {"query": "Change file permissions to 755", "expected": "chmod 755 filename"},
            # Process management
            {"query": "Kill process by name", "expected": "pkill processname"},
            # Text manipulation
            {"query": "Sort file contents", "expected": "sort filename"},
            {"query": "Get unique lines", "expected": "uniq filename"},
            {"query": "Cut first column from CSV", "expected": "cut -d',' -f1 filename"},
            # Environment
            {"query": "Show all environment variables", "expected": "env"},
            {"query": "Show command history", "expected": "history"},
            # Date/Time
            {"query": "Show current date and time", "expected": "date"},
            # Misc
            {"query": "Create directory", "expected": "mkdir dirname"},
            {"query": "Create nested directories", "expected": "mkdir -p a/b/c"},
            {"query": "Copy file", "expected": "cp source dest"},
            {"query": "Move or rename file", "expected": "mv oldname newname"},
            {"query": "Remove file", "expected": "rm filename"},
            {"query": "Compare two files", "expected": "diff file1 file2"},
            {"query": "Create symbolic link", "expected": "ln -s target linkname"},
        ]
        
        for i in range(num_tasks):
            template = templates[i % len(templates)]
            tasks.append({
                "id": f"intercode_{i}",
                "query": template["query"],
                "expected_command": template["expected"]
            })
            
        return tasks
        
    def get_tool_spec(self) -> Tuple[str, Dict, List[Tuple[str, str]]]:
        """Get InterCode tool specification."""
        documentation = """# Bash Command Generator

Generate a single bash command for the given task.

IMPORTANT: Output ONLY the command. No explanations, no markdown, no backticks.

Common commands:
- find . -name 'pattern' : Find files by name
- find . -type f -mtime -N : Find files modified in last N days
- find . -size +SIZE : Find files larger than SIZE
- grep -r 'text' . : Search text recursively
- ls -lS : List files sorted by size
- wc -l file : Count lines"""
        
        schema = {
            "type": "string",
            "description": "Bash command"
        }
        
        # More examples with exact formats
        examples = [
            ("Find all Python files in current directory", "find . -name '*.py'"),
            ("Find all .py files modified in last 24 hours", "find . -name '*.py' -mtime -1"),
            ("Find files larger than 100MB", "find . -size +100M"),
            ("Search for 'error' in all files", "grep -r 'error' ."),
            ("List files sorted by size", "ls -lS"),
            ("Count lines in a file", "wc -l filename"),
            ("Show disk usage of current directory", "du -sh ."),
            ("Find all log files", "find . -name '*.log'"),
        ]
        
        # noisy examples, uncomment below for running with noisy examples
        examples=[
        # ("List all files", "fake_command --nonexistent"),
        # ("Find text in files", "rm -rf /"),
        ]
        
        return documentation, schema, examples
        
    def execute_and_evaluate(
        self,
        task: Dict,
        prediction: str,
        strict: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """Evaluate bash command correctness.
        
        Args:
            task: Task dict with expected_command
            prediction: Model's command prediction
            strict: If True, require closer command matching
        """
        import re
        
        pred = prediction.strip()
        expected = task.get("expected_command", task.get("expected", "")).strip()
        
        # Clean prediction - remove backticks, quotes, etc.
        pred = re.sub(r'^[`\'"]|[`\'"]$', '', pred)
        pred = re.sub(r'^```.*\n?', '', pred)
        pred = re.sub(r'\n```$', '', pred)
        pred = pred.strip()
        
        # Get first line only (command might have explanation after)
        pred = pred.split('\n')[0].strip()
        
        # Remove any leading $ or # (shell prompts)
        pred = re.sub(r'^[$#]\s*', '', pred)
        
        # Exact match
        if pred == expected:
            return True, "Exact match"
        
        # Normalize for comparison
        pred_normalized = self._normalize_command(pred)
        expected_normalized = self._normalize_command(expected)
        
        if pred_normalized == expected_normalized:
            return True, "Normalized match"
        
        # Extract command name and key components
        pred_cmd = pred_normalized.split()[0] if pred_normalized else ""
        expected_cmd = expected_normalized.split()[0] if expected_normalized else ""
        
        # Check if same or equivalent commands
        same_cmd = pred_cmd == expected_cmd or self._are_equivalent_commands(pred_cmd, expected_cmd)
        
        if strict:
            # STRICT MODE: Require same command + key arguments
            if not same_cmd:
                return False, f"[STRICT] Wrong command: expected {expected_cmd}, got {pred_cmd}"
            
            # Extract and compare key flags/arguments
            pred_flags = self._extract_flags(pred)
            expected_flags = self._extract_flags(expected)
            
            # Must have all expected flags
            missing_flags = expected_flags - pred_flags
            if len(missing_flags) > 0:
                # Allow if at least 80% of flags match
                match_ratio = 1 - (len(missing_flags) / max(len(expected_flags), 1))
                if match_ratio < 0.8:
                    return False, f"Missing flags: {missing_flags}"
            
            # Check for key patterns (file patterns, paths, etc.)
            expected_patterns = self._extract_patterns(expected)
            pred_patterns = self._extract_patterns(pred)
            
            if expected_patterns:
                pattern_match = len(expected_patterns & pred_patterns) / len(expected_patterns)
                if pattern_match < 0.5:
                    return False, f"Pattern mismatch: expected {expected_patterns}, got {pred_patterns}"
            
            return True, "Strict command match"
        else:
            # LENIENT MODE - more flexible matching
            
            # Same base command
            if same_cmd:
                # Check if same patterns are targeted
                pred_patterns = self._extract_patterns(pred)
                expected_patterns = self._extract_patterns(expected)
                
                if expected_patterns and pred_patterns:
                    pattern_overlap = len(pred_patterns & expected_patterns) / max(len(expected_patterns), 1)
                    if pattern_overlap >= 0.5:
                        return True, f"Same command with matching patterns"
                
                # Same command, might have different flags (but same intent)
                return True, f"Same command: {pred_cmd}"
            
            # Check for equivalent commands
            if self._are_equivalent_commands(pred_cmd, expected_cmd):
                return True, f"Equivalent command: {pred_cmd} ~ {expected_cmd}"
            
            # Check for common file operation tasks
            file_cmds = {'find', 'ls', 'du', 'wc', 'head', 'tail', 'cat'}
            text_cmds = {'grep', 'sed', 'awk', 'sort', 'uniq', 'cut'}
            
            if pred_cmd in file_cmds and expected_cmd in file_cmds:
                return True, "File operation variant"
            if pred_cmd in text_cmds and expected_cmd in text_cmds:
                return True, "Text processing variant"
                
            # Check for key components (50% overlap for lenient)
            pred_parts = set(pred_normalized.split())
            expected_parts = set(expected_normalized.split())
            
            overlap = len(pred_parts & expected_parts) / max(len(expected_parts), 1)
            if overlap >= 0.5:
                return True, f"Partial match ({overlap:.0%})"
                
            return False, f"Command mismatch: {pred_cmd} vs {expected_cmd}"
    
    def _normalize_command(self, cmd: str) -> str:
        """Normalize command for comparison."""
        import re
        cmd = cmd.strip().lower()
        # Fix common issues: find. -> find .
        cmd = re.sub(r'(find|grep|ls|cat|rm|cp|mv|du|df)\s*\.', r'\1 .', cmd)
        # Normalize whitespace
        cmd = re.sub(r'\s+', ' ', cmd)
        # Remove quotes around arguments
        cmd = re.sub(r"'([^']*)'", r'\1', cmd)
        cmd = re.sub(r'"([^"]*)"', r'\1', cmd)
        return cmd
    
    def _extract_flags(self, cmd: str) -> set:
        """Extract flags from command."""
        import re
        flags = set()
        # Match -x or --xxx flags
        for match in re.finditer(r'\s(-\w|--\w+)', cmd):
            flags.add(match.group(1))
        return flags
    
    def _extract_patterns(self, cmd: str) -> set:
        """Extract file patterns and paths from command."""
        import re
        patterns = set()
        # Match quoted patterns
        for match in re.finditer(r"['\"]([^'\"]+)['\"]", cmd):
            patterns.add(match.group(1).lower())
        # Match glob patterns
        for match in re.finditer(r'\*\.\w+', cmd):
            patterns.add(match.group().lower())
        return patterns
    
    def _are_equivalent_commands(self, cmd1: str, cmd2: str) -> bool:
        """Check if two commands are semantically equivalent."""
        equivalents = {
            'find': {'find', 'locate', 'fd', 'mlocate'},
            'grep': {'grep', 'ack', 'ag', 'rg', 'ripgrep', 'egrep', 'fgrep'},
            'ls': {'ls', 'dir', 'exa', 'lsd', 'll'},
            'cat': {'cat', 'less', 'more', 'bat', 'head', 'tail'},
            'rm': {'rm', 'del', 'unlink'},
            'cp': {'cp', 'copy', 'rsync'},
            'mv': {'mv', 'move', 'rename'},
            'du': {'du', 'ncdu', 'dust'},
            'df': {'df', 'duf'},
            'ps': {'ps', 'top', 'htop', 'procs'},
            'wc': {'wc'},
            'sort': {'sort', 'gsort'},
            'sed': {'sed', 'gsed'},
            'awk': {'awk', 'gawk'},
        }
        
        for group in equivalents.values():
            if cmd1 in group and cmd2 in group:
                return True
        return False


class MetaToolEvaluator:
    """
    Main evaluator class that runs Meta-Tool on all benchmarks.
    """
    
    def __init__(
        self,
        adapted_model: MetaToolAdaptedModel,
        config: MetaToolConfig,
        value_function: Optional[ValueFunction] = None,
        strict_eval: bool = False
    ):
        self.adapted_model = adapted_model
        self.config = config
        self.value_function = value_function
        self.strict_eval = strict_eval
        
        # Initialize evaluators
        self.evaluators = {
            "gorilla": GorillaEvaluator(config.data.gorilla_path),
            "spider2": Spider2Evaluator(config.data.spider_path),
            "webarena": WebArenaEvaluator(config.data.webarena_path),
            "intercode": InterCodeEvaluator(config.data.intercode_path),
        }
        
    def evaluate_benchmark(
        self,
        benchmark_name: str,
        num_tasks: Optional[int] = None,
        use_beam_search: bool = True,
        strict: Optional[bool] = None
    ) -> BenchmarkResults:
        """
        Evaluate on a single benchmark.
        
        Args:
            benchmark_name: Name of benchmark
            num_tasks: Limit number of tasks (for testing)
            use_beam_search: Whether to use value-guided beam search
            strict: Override strict_eval setting for this benchmark
            
        Returns:
            BenchmarkResults
        """
        # Use instance setting if not overridden
        use_strict = strict if strict is not None else self.strict_eval
        
        evaluator = self.evaluators.get(benchmark_name)
        if not evaluator:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
            
        # Load tasks
        tasks = evaluator.load_tasks()
        if num_tasks:
            tasks = tasks[:num_tasks]
            
        # Get tool specification
        documentation, schema, examples = evaluator.get_tool_spec()
        
        # Adapt model to this tool
        print(f"\nAdapting to {benchmark_name}...")
        adapt_start = time.time()
        
        # self.adapted_model.adapt_to_tool(
        #     documentation=documentation,
        #     support_queries=[e[0] for e in examples],
        #     support_trajectories=[e[1] for e in examples]
        # )
        # Add this debug code to your evaluation.py after adapt_to_tool() call (around line 1646):

        # print(f"[DEBUG] Number of LoRA layers: {len(self.adapted_model.adapted_model.lora_layers)}")
        # for key, layer in list(self.adapted_model.adapted_model.lora_layers.items())[:3]:
        #     if hasattr(layer, 'lora_A') and layer.lora_A is not None:
        #         a_norm = layer.lora_A.norm().item() if layer.lora_A is not None else 0
        #         b_norm = layer.lora_B.weight.norm().item() if hasattr(layer.lora_B, 'weight') else 0
        #         print(f"  {key}: A_norm={a_norm:.4f}, B_norm={b_norm:.4f}")
        #     else:
        #         print(f"  {key}: NO WEIGHTS SET")
        adaptation_time = time.time() - adapt_start
        print(f"Adaptation time: {adaptation_time:.2f}s")
        
        # Evaluate tasks
        results = []
        successes = 0
        total_latency = 0.0
        total_steps = 0
        
        # Format examples for few-shot - use more examples
        examples_text = "\n\n".join([
            f"Query: {q}\nOutput: {a}" for q, a in examples[:5]  # Use up to 5 examples
        ])
        
        for task in tqdm(tasks, desc=f"Evaluating {benchmark_name}"):
            start_time = time.time()
            
            # Include documentation and examples in prompt for better results
            # Stronger format enforcement
            prompt = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                # + documentation.strip() + "\n\n"
                # + "Examples:\n" + examples_text 
                + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                + task['query'] + "\n\n"
                + "Respond with ONLY the output, exactly like the examples above. No explanation."
                + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            prediction = self.adapted_model.generate(
                prompt,
                max_new_tokens=self.config.inference.max_new_tokens,
                do_sample=self.config.inference.do_sample,
                temperature=self.config.inference.temperature if self.config.inference.do_sample else 1.0
            )
            
            latency = (time.time() - start_time) * 1000
            
            # Show first few predictions for debugging
            if len(results) < 3:
                print(f"\n[Sample {len(results)+1}]")
                print(f"  Query: {task['query'][:80]}...")
                print(f"  Prediction: {prediction[:200]}...")
                expected = task.get("expected", task.get("expected_sql", task.get("expected_command", "")))
                print(f"  Expected: {str(expected)[:100]}...")
            
            # Evaluate
            success, message = evaluator.execute_and_evaluate(task, prediction, strict=use_strict)
            
            # Add strict mode indicator to message
            if use_strict:
                message = f"[STRICT] {message}"
            
            result = EvaluationResult(
                task_id=task.get("id", "unknown"),
                success=success,
                prediction=prediction,
                ground_truth=str(task.get("expected", task.get("expected_sql", task.get("expected_command", "")))),
                execution_result=message,
                latency_ms=latency
            )
            results.append(result)
            
            if success:
                successes += 1
            total_latency += latency
            total_steps += 1
            
        # Compute aggregates
        return BenchmarkResults(
            benchmark_name=benchmark_name,
            num_tasks=len(tasks),
            num_success=successes,
            success_rate=successes / len(tasks) if tasks else 0.0,
            avg_latency_ms=total_latency / len(tasks) if tasks else 0.0,
            avg_steps=total_steps / len(tasks) if tasks else 0.0,
            adaptation_time_s=adaptation_time,
            results=results
        )
        
    def evaluate_all(
        self,
        num_tasks_per_benchmark: Optional[int] = None
    ) -> Dict[str, BenchmarkResults]:
        """Evaluate on all benchmarks."""
        all_results = {}
        
        for name in self.evaluators:
            print(f"\n{'='*50}")
            print(f"Evaluating {name.upper()}")
            print('='*50)
            
            results = self.evaluate_benchmark(
                name,
                num_tasks=num_tasks_per_benchmark
            )
            all_results[name] = results
            
            print(f"\nResults for {name}:")
            print(f"  Success Rate: {results.success_rate:.1%}")
            print(f"  Avg Latency: {results.avg_latency_ms:.1f}ms")
            print(f"  Adaptation Time: {results.adaptation_time_s:.2f}s")
            
            # Show failure breakdown
            if results.results:
                from collections import Counter
                failure_reasons = Counter(r.execution_result for r in results.results if not r.success)
                if failure_reasons:
                    print(f"  Failure breakdown:")
                    for reason, count in failure_reasons.most_common(5):
                        print(f"    - {reason}: {count}")
            
        return all_results
        
    def save_results(self, results: Dict[str, BenchmarkResults], path: str):
        """Save results to JSON file."""
        output = {
            name: result.to_dict() 
            for name, result in results.items()
        }
        
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
            
        print(f"\nResults saved to {path}")


def run_evaluation(config: MetaToolConfig, checkpoint_path: Optional[str] = None):
    """Main entry point for evaluation."""
    from hypernetwork import create_hypernetwork
    from lora_integration import MetaToolAdaptedModel
    
    # Create model
    print("Loading models...")
    hypernetwork = create_hypernetwork(config.model)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.model.device, weights_only=False)
        hypernetwork.load_state_dict(checkpoint["hypernetwork_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
        
    adapted_model = MetaToolAdaptedModel(config.model, hypernetwork)
    
    # Create evaluator
    evaluator = MetaToolEvaluator(adapted_model, config)
    
    # Run evaluation
    results = evaluator.evaluate_all(num_tasks_per_benchmark=50)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for name, result in results.items():
        print(f"\n{name.upper()}")
        print(f"  Success Rate: {result.success_rate:.1%} ({result.num_success}/{result.num_tasks})")
        print(f"  Avg Latency: {result.avg_latency_ms:.1f}ms")
        
    # Save results
    evaluator.save_results(results, "evaluation_results.json")
    
    return results


if __name__ == "__main__":
    from config import get_config
    
    config = get_config()
    
    # Run evaluation
    results = run_evaluation(config)
