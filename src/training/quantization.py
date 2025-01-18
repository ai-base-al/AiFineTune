from dataclasses import dataclass
from typing import Optional, Union, Dict, Any

@dataclass
class MLXQuantizationConfig:
    quant_method: str = "mlx"
    bits: int = 4
    compute_dtype: str = "float16"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "quant_method": self.quant_method,
            "bits": self.bits,
            "compute_dtype": self.compute_dtype
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MLXQuantizationConfig":
        return cls(**config_dict)