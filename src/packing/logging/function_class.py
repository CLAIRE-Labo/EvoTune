from numbers import Number
from dataclasses import dataclass, asdict, field, fields
from numpy import ndarray
import numpy as np

@dataclass
class FailData:
    reason_found: int = 0
    reason_type: int = 0
    reason_shape: int = 0
    reason_degenerate: int = 0
    reason_timeout: int = 0
    reason_exception: int = 0
    reason_imports: int = 0
    reason_orthogonality: int = 0
    exception: str = ""
    def __setattr__(self, name, value):
        # Get all field names from the dataclass
        field_names = {f.name for f in fields(self)}
        if name not in field_names:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        super().__setattr__(name, value)

@dataclass
class PromptData:
    island_id: int = -1
    min_prompt_score: float = -1.0
    max_prompt_score: float = -1.0
    num_examples_prompt: int = -1
    temperature: float = -1.0
    len_input_token_ids: int = -1
    len_output_token_ids: int = -1
    max_prob: float = -1.0
    min_prob: float = -1.0
    def __setattr__(self, name, value):
        # Get all field names from the dataclass
        field_names = {f.name for f in fields(self)}
        if name not in field_names:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        super().__setattr__(name, value)

@dataclass
class EvalData:
    idx_process: int = -1
    code_dimension: int = -1
    minimum_distance: int = -1
    G_new_numpy: ndarray = field(default_factory=lambda: np.array([]))
    input_struct: ndarray = field(default_factory=lambda: np.array([]))
    output_struct: ndarray = field(default_factory=lambda: np.array([]))
    dim_penalty: int = -1
    weight_penalty: float = -1.0
    applied_dim_penalty: float = -1.0
    applied_weight_penalty: float = -1.0
    shape0: int = -1
    shape1: int = -1
    weight_key: list = field(default_factory=tuple)
    canonical_key: list = field(default_factory=tuple)
    def __setattr__(self, name, value):
        # Get all field names from the dataclass
        field_names = {f.name for f in fields(self)}
        if name not in field_names:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        super().__setattr__(name, value)

@dataclass
class FunctionClass:
    function_str: str = ""
    imports_str: str = ""
    score: float = float("NaN")
    true_score: int = float("NaN")
    original_score: int = float("NaN")
    fail_flag: int = -1 # Only measures if the function failed or not
    correct_flag: int = 0 
    chat: list[dict] = field(default_factory=list)
    prompt: PromptData = field(default_factory=PromptData)
    eval: EvalData = field(default_factory=EvalData)
    fail: FailData = field(default_factory=FailData)
    function_num: int = -1
    prompt_num: int = -1

    def __setattr__(self, name, value):
        # Get all field names from the dataclass
        field_names = {f.name for f in fields(self)}
        if name not in field_names:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        super().__setattr__(name, value)

    def identify_fail_reason(self):
        if self.fail.reason_found:
            return "Found"
        if self.fail.reason_type:
            return "Type"
        if self.fail.reason_shape:
            return "Shape"
        if self.fail.reason_degenerate:
            return "Degenerate"
        if self.fail.reason_timeout:
            return "Timeout"
        if self.fail.reason_exception:
            return f"Exception: {self.fail.exception}"
        if self.fail.reason_imports:
            return f"Imports: {self.fail.exception}"
        if self.fail.reason_orthogonality:
            return "Orthogonality"
        return "Unknown"