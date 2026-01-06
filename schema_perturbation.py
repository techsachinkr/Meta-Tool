"""
Schema-Based Test Case Generation for Meta-Tool
================================================
Generates test cases by perturbing support set trajectories according to schema constraints.
Addresses the "competence paradox" by not requiring semantic understanding.

Perturbation Operators:
- VALUE_SUBSTITUTE: Replace with random valid schema values
- PARAM_DROP: Remove optional parameters
- BOUNDARY_TEST: Use boundary values (0, MAX_INT, empty, etc.)
- COMBINATION: Apply multiple operators
"""

import json
import random
import re
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import copy


class PerturbationType(Enum):
    """Types of perturbation operations."""
    VALUE_SUBSTITUTE = "value_substitute"
    PARAM_DROP = "param_drop"
    BOUNDARY_TEST = "boundary_test"
    COMBINATION = "combination"


@dataclass
class TestCase:
    """A test case generated from perturbation."""
    original_action: Dict[str, Any]
    perturbed_action: Dict[str, Any]
    perturbation_type: PerturbationType
    expected_valid: bool  # Based on schema constraints
    parameter_changed: Optional[str] = None
    description: str = ""


@dataclass
class ToolSchema:
    """Parsed tool schema for test generation."""
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]  # param_name -> {type, required, enum, pattern, etc.}
    required_params: List[str]
    optional_params: List[str]


def parse_json_schema(schema_dict: Dict[str, Any]) -> ToolSchema:
    """
    Parse a JSON Schema into ToolSchema format.
    
    Args:
        schema_dict: JSON Schema dictionary
        
    Returns:
        Parsed ToolSchema
    """
    name = schema_dict.get("name", schema_dict.get("title", "unknown"))
    description = schema_dict.get("description", "")
    
    # Extract parameters from properties
    properties = schema_dict.get("properties", {})
    required = schema_dict.get("required", [])
    
    parameters = {}
    required_params = []
    optional_params = []
    
    for param_name, param_spec in properties.items():
        parameters[param_name] = {
            "type": param_spec.get("type", "string"),
            "description": param_spec.get("description", ""),
            "enum": param_spec.get("enum"),
            "pattern": param_spec.get("pattern"),
            "minimum": param_spec.get("minimum"),
            "maximum": param_spec.get("maximum"),
            "minLength": param_spec.get("minLength"),
            "maxLength": param_spec.get("maxLength"),
            "default": param_spec.get("default"),
            "items": param_spec.get("items"),  # For arrays
        }
        
        if param_name in required:
            required_params.append(param_name)
        else:
            optional_params.append(param_name)
            
    return ToolSchema(
        name=name,
        description=description,
        parameters=parameters,
        required_params=required_params,
        optional_params=optional_params
    )


class SchemaBasedPerturbator:
    """
    Generates test cases by perturbing actions according to schema constraints.
    """
    
    def __init__(self, schema: ToolSchema, seed: int = 42):
        self.schema = schema
        self.rng = random.Random(seed)
        
        # Boundary values for different types
        self.boundary_values = {
            "string": ["", " ", "a" * 1000, "null", "undefined", "<script>"],
            "integer": [0, -1, 1, 2147483647, -2147483648],
            "number": [0.0, -1.0, 1.0, float('inf'), -float('inf'), 1e-10, 1e10],
            "boolean": [True, False],
            "array": [[], [None], list(range(1000))],
            "object": [{}, {"key": "value"}, None],
        }
        
    def generate_valid_value(self, param_name: str) -> Any:
        """Generate a random valid value for a parameter based on schema."""
        param_spec = self.schema.parameters.get(param_name, {})
        param_type = param_spec.get("type", "string")
        
        # Handle enum
        if param_spec.get("enum"):
            return self.rng.choice(param_spec["enum"])
        
        # Handle by type
        if param_type == "string":
            pattern = param_spec.get("pattern")
            if pattern:
                return self._generate_from_pattern(pattern)
            min_len = param_spec.get("minLength", 1)
            max_len = param_spec.get("maxLength", 50)
            length = self.rng.randint(min_len, min(max_len, 50))
            return ''.join(self.rng.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=length))
            
        elif param_type == "integer":
            min_val = param_spec.get("minimum", 0)
            max_val = param_spec.get("maximum", 1000)
            return self.rng.randint(int(min_val), int(max_val))
            
        elif param_type == "number":
            min_val = param_spec.get("minimum", 0.0)
            max_val = param_spec.get("maximum", 1000.0)
            return self.rng.uniform(float(min_val), float(max_val))
            
        elif param_type == "boolean":
            return self.rng.choice([True, False])
            
        elif param_type == "array":
            items_spec = param_spec.get("items", {"type": "string"})
            length = self.rng.randint(1, 5)
            # Simplified array generation
            return [f"item_{i}" for i in range(length)]
            
        elif param_type == "object":
            return {"key": "value"}
            
        return "default_value"
        
    def _generate_from_pattern(self, pattern: str) -> str:
        """Generate a string matching a regex pattern (simplified)."""
        # Simple pattern handling for common cases
        if pattern == r"^\d+$":
            return str(self.rng.randint(1, 1000))
        elif pattern == r"^[a-zA-Z]+$":
            length = self.rng.randint(3, 10)
            return ''.join(self.rng.choices('abcdefghijklmnopqrstuvwxyz', k=length))
        elif "uuid" in pattern.lower() or pattern == r"^[0-9a-f-]{36}$":
            import uuid
            return str(uuid.uuid4())
        else:
            # Default: return alphanumeric string
            return ''.join(self.rng.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=10))
            
    def get_boundary_value(self, param_name: str) -> Any:
        """Get a boundary value for a parameter."""
        param_spec = self.schema.parameters.get(param_name, {})
        param_type = param_spec.get("type", "string")
        
        boundaries = self.boundary_values.get(param_type, [""])
        return self.rng.choice(boundaries)
        
    def value_substitute(self, action: Dict[str, Any], param: str) -> TestCase:
        """Replace a parameter value with another valid value."""
        perturbed = copy.deepcopy(action)
        new_value = self.generate_valid_value(param)
        
        if "parameters" in perturbed:
            perturbed["parameters"][param] = new_value
        else:
            perturbed[param] = new_value
            
        return TestCase(
            original_action=action,
            perturbed_action=perturbed,
            perturbation_type=PerturbationType.VALUE_SUBSTITUTE,
            expected_valid=True,  # Valid value substitution should work
            parameter_changed=param,
            description=f"Substituted {param} with {new_value}"
        )
        
    def param_drop(self, action: Dict[str, Any], param: str) -> Optional[TestCase]:
        """Remove an optional parameter."""
        if param not in self.schema.optional_params:
            return None  # Can't drop required params
            
        perturbed = copy.deepcopy(action)
        
        if "parameters" in perturbed:
            if param in perturbed["parameters"]:
                del perturbed["parameters"][param]
        elif param in perturbed:
            del perturbed[param]
        else:
            return None
            
        return TestCase(
            original_action=action,
            perturbed_action=perturbed,
            perturbation_type=PerturbationType.PARAM_DROP,
            expected_valid=True,  # Dropping optional should work
            parameter_changed=param,
            description=f"Dropped optional parameter {param}"
        )
        
    def boundary_test(self, action: Dict[str, Any], param: str) -> TestCase:
        """Replace a parameter with a boundary value."""
        perturbed = copy.deepcopy(action)
        boundary_val = self.get_boundary_value(param)
        
        if "parameters" in perturbed:
            perturbed["parameters"][param] = boundary_val
        else:
            perturbed[param] = boundary_val
            
        # Boundary values may or may not be valid depending on constraints
        param_spec = self.schema.parameters.get(param, {})
        expected_valid = self._check_boundary_validity(boundary_val, param_spec)
        
        return TestCase(
            original_action=action,
            perturbed_action=perturbed,
            perturbation_type=PerturbationType.BOUNDARY_TEST,
            expected_valid=expected_valid,
            parameter_changed=param,
            description=f"Boundary test {param} with {boundary_val}"
        )
        
    def _check_boundary_validity(self, value: Any, param_spec: Dict) -> bool:
        """Check if a boundary value is valid according to schema."""
        param_type = param_spec.get("type", "string")
        
        # Empty string checks
        if value == "" and param_spec.get("minLength", 0) > 0:
            return False
            
        # Numeric range checks
        if param_type in ("integer", "number"):
            if param_spec.get("minimum") is not None and value < param_spec["minimum"]:
                return False
            if param_spec.get("maximum") is not None and value > param_spec["maximum"]:
                return False
                
        # Enum checks
        if param_spec.get("enum") and value not in param_spec["enum"]:
            return False
            
        return True
        
    def generate_test_suite(
        self,
        actions: List[Dict[str, Any]],
        perturbations_per_action: int = 5
    ) -> List[TestCase]:
        """
        Generate a test suite from a list of actions.
        
        Args:
            actions: List of action dictionaries from support set
            perturbations_per_action: Max perturbations per operator per action
            
        Returns:
            List of TestCase objects
        """
        test_cases = []
        
        for action in actions:
            # Get parameters from action
            params = action.get("parameters", action)
            if isinstance(params, dict):
                param_names = list(params.keys())
            else:
                continue
                
            # Apply each perturbation type
            for param in param_names:
                if param not in self.schema.parameters:
                    continue
                    
                # VALUE_SUBSTITUTE
                for _ in range(min(perturbations_per_action, 3)):
                    test_case = self.value_substitute(action, param)
                    test_cases.append(test_case)
                    
                # PARAM_DROP (only for optional)
                if param in self.schema.optional_params:
                    test_case = self.param_drop(action, param)
                    if test_case:
                        test_cases.append(test_case)
                        
                # BOUNDARY_TEST
                for _ in range(min(perturbations_per_action, 2)):
                    test_case = self.boundary_test(action, param)
                    test_cases.append(test_case)
                    
        return test_cases


def extract_actions_from_trajectories(trajectories: List[str]) -> List[Dict[str, Any]]:
    """
    Extract action dictionaries from trajectory strings.
    Handles various formats (JSON, function calls, etc.)
    """
    actions = []
    
    for trajectory in trajectories:
        # Try to extract JSON objects
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, trajectory)
        
        for match in matches:
            try:
                action = json.loads(match)
                if isinstance(action, dict):
                    actions.append(action)
            except json.JSONDecodeError:
                continue
                
        # Try to extract function call patterns
        func_pattern = r'(\w+)\((.*?)\)'
        func_matches = re.findall(func_pattern, trajectory)
        
        for func_name, args in func_matches:
            try:
                # Try to parse args as dict
                if args.strip().startswith('{'):
                    params = json.loads(args)
                else:
                    # Parse key=value pairs
                    params = {}
                    for pair in args.split(','):
                        if '=' in pair:
                            key, value = pair.split('=', 1)
                            params[key.strip()] = value.strip().strip('"\'')
                            
                actions.append({
                    "function": func_name,
                    "parameters": params
                })
            except:
                continue
                
    return actions


def create_test_suite_from_support_set(
    schema_dict: Dict[str, Any],
    support_trajectories: List[str],
    seed: int = 42
) -> List[TestCase]:
    """
    Create a complete test suite from schema and support set.
    
    Args:
        schema_dict: JSON Schema for the tool
        support_trajectories: List of trajectory strings
        seed: Random seed
        
    Returns:
        List of TestCase objects
    """
    # Parse schema
    schema = parse_json_schema(schema_dict)
    
    # Extract actions
    actions = extract_actions_from_trajectories(support_trajectories)
    
    if not actions:
        print("Warning: No actions extracted from trajectories")
        return []
    
    # Create perturbator and generate test suite
    perturbator = SchemaBasedPerturbator(schema, seed)
    test_suite = perturbator.generate_test_suite(actions)
    
    print(f"Generated {len(test_suite)} test cases from {len(actions)} actions")
    
    return test_suite


# Example usage and testing
if __name__ == "__main__":
    # Example schema
    example_schema = {
        "name": "search_api",
        "description": "Search for items",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
                "minLength": 1,
                "maxLength": 100
            },
            "limit": {
                "type": "integer",
                "description": "Max results",
                "minimum": 1,
                "maximum": 100,
                "default": 10
            },
            "category": {
                "type": "string",
                "enum": ["books", "electronics", "clothing"]
            }
        },
        "required": ["query"]
    }
    
    # Example trajectories
    example_trajectories = [
        'Action: {"function": "search_api", "parameters": {"query": "python books", "limit": 10, "category": "books"}}',
        'Action: search_api(query="machine learning", limit=20)'
    ]
    
    # Generate test suite
    test_suite = create_test_suite_from_support_set(
        example_schema,
        example_trajectories
    )
    
    print(f"\nGenerated {len(test_suite)} test cases:")
    for i, tc in enumerate(test_suite[:5]):
        print(f"\n{i+1}. {tc.perturbation_type.value}: {tc.description}")
        print(f"   Expected valid: {tc.expected_valid}")
