"""
Meta-Tool: A Hypernetwork-Driven Meta-Learning Framework
=========================================================

This package implements the Meta-Tool framework for few-shot adaptation
of LLM agents to domain-specific tool ecosystems.

Main components:
- hypernetwork: Generates LoRA weights from tool documentation
- lora_integration: Applies generated weights to base models
- schema_perturbation: Schema-based test case generation
- value_function: TD-learned value function for beam search
- constrained_decoding: FSM-constrained generation
- memory_system: FAISS-based episodic and schema memory
- meta_training: Meta-learning training loop
- evaluation: Benchmark evaluation pipeline
- data_loader: ToolBench and benchmark data loading

Usage:
    from meta_tool import MetaToolConfig, create_hypernetwork
    from meta_tool import MetaToolAdaptedModel
    from meta_tool import load_toolbench_tools
    
    config = MetaToolConfig()
    
    # Load real ToolBench data
    tools = load_toolbench_tools(use_huggingface=True)
    
    hypernetwork = create_hypernetwork(config.model)
    model = MetaToolAdaptedModel(config.model, hypernetwork)
    
    model.adapt_to_tool(documentation, queries, trajectories)
    response = model.generate("Query: ...")
"""

__version__ = "1.0.0"
__author__ = "Meta-Tool Authors"

from .config import (
    MetaToolConfig,
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    DataConfig,
    get_config
)

from .hypernetwork import (
    MetaToolHypernetwork,
    DocumentationEncoder,
    PrototypeAggregator,
    FactorizedWeightGenerator,
    LoRAWeights,
    create_hypernetwork
)

from .lora_integration import (
    LoRALinear,
    AdaptedModel,
    MetaToolAdaptedModel,
    create_adapted_model
)

from .schema_perturbation import (
    SchemaBasedPerturbator,
    TestCase,
    ToolSchema,
    PerturbationType,
    create_test_suite_from_support_set
)

from .value_function import (
    ValueFunction,
    ValueFunctionTrainer,
    ValueGuidedBeamSearch,
    ExecutionEnvironment,
    Transition,
    create_value_function,
    run_refinement_phase
)

from .constrained_decoding import (
    JSONSchemaFSM,
    ConstrainedDecoder,
    OutlinesConstrainedGenerator,
    generate_with_rejection_sampling
)

from .memory_system import (
    EpisodicMemory,
    SchemaMemory,
    HybridMemorySystem,
    FAISSIndex,
    EmbeddingModel,
    create_memory_system
)

from .meta_training import (
    MetaTrainer,
    Tool,
    Episode,
    ToolDataset,
    create_synthetic_tools,
    load_toolbench_tools,
    load_tools_for_training,
    run_meta_training
)

from .evaluation import (
    MetaToolEvaluator,
    BenchmarkResults,
    EvaluationResult,
    GorillaEvaluator,
    Spider2Evaluator,
    WebArenaEvaluator,
    InterCodeEvaluator,
    run_evaluation
)

from .data_loader import (
    ToolBenchLoader,
    ToolAPI,
    TrajectoryExample,
    load_toolbench,
    setup_all_datasets,
    GorillaLoader,
    Spider2Loader
)

__all__ = [
    # Config
    'MetaToolConfig',
    'ModelConfig', 
    'TrainingConfig',
    'InferenceConfig',
    'DataConfig',
    'get_config',
    
    # Hypernetwork
    'MetaToolHypernetwork',
    'DocumentationEncoder',
    'PrototypeAggregator',
    'FactorizedWeightGenerator',
    'LoRAWeights',
    'create_hypernetwork',
    
    # LoRA Integration
    'LoRALinear',
    'AdaptedModel',
    'MetaToolAdaptedModel',
    'create_adapted_model',
    
    # Schema Perturbation
    'SchemaBasedPerturbator',
    'TestCase',
    'ToolSchema',
    'PerturbationType',
    'create_test_suite_from_support_set',
    
    # Value Function
    'ValueFunction',
    'ValueFunctionTrainer',
    'ValueGuidedBeamSearch',
    'ExecutionEnvironment',
    'Transition',
    'create_value_function',
    'run_refinement_phase',
    
    # Constrained Decoding
    'JSONSchemaFSM',
    'ConstrainedDecoder',
    'OutlinesConstrainedGenerator',
    'generate_with_rejection_sampling',
    
    # Memory System
    'EpisodicMemory',
    'SchemaMemory',
    'HybridMemorySystem',
    'FAISSIndex',
    'EmbeddingModel',
    'create_memory_system',
    
    # Training
    'MetaTrainer',
    'Tool',
    'Episode',
    'ToolDataset',
    'create_synthetic_tools',
    'load_toolbench_tools',
    'load_tools_for_training',
    'run_meta_training',
    
    # Evaluation
    'MetaToolEvaluator',
    'BenchmarkResults',
    'EvaluationResult',
    'GorillaEvaluator',
    'Spider2Evaluator',
    'WebArenaEvaluator',
    'InterCodeEvaluator',
    'run_evaluation',
    
    # Data Loading
    'ToolBenchLoader',
    'ToolAPI',
    'TrajectoryExample',
    'load_toolbench',
    'setup_all_datasets',
    'GorillaLoader',
    'Spider2Loader',
]
