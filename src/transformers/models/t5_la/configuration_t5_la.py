
from transformers import T5Config


class T5LAConfig(T5Config):
  model_type = "t5-la"
  look_ahead_size = 0


__all__ = [
  "T5LAConfig"
]
