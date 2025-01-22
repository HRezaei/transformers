import copy

import torch
from torch import nn

from transformers import T5ForConditionalGeneration, T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.models.t5_la import T5LAConfig


class LookAheadHeads(nn.Module):
  def __init__(self, config: T5LAConfig):
    super().__init__()
    self.heads = nn.ModuleList(
      [nn.Linear(config.d_model, config.vocab_size, bias=False) for _ in range(config.look_ahead_size + 1)])

  def forward(self, x):
    # ModuleList can act as an iterable, or be indexed using ints
    # Apply each head to the shared features
    logits = [head(x) for head in self.heads]

    # Stack logits along a new dimension to create a tensor of shape [batch_size, num_heads, output_size]
    logits = torch.stack(logits, dim=1)
    return logits


class T5LAForConditionalGeneration(T5ForConditionalGeneration):
  def __init__(self, config: T5LAConfig):
    super().__init__(config)
    self.model_dim = config.d_model

    self.shared = nn.Embedding(config.vocab_size, config.d_model)

    encoder_config = copy.deepcopy(config)
    encoder_config.is_decoder = False
    encoder_config.use_cache = False
    encoder_config.is_encoder_decoder = False
    self.encoder = T5Stack(encoder_config, self.shared)

    decoder_config = copy.deepcopy(config)
    decoder_config.is_decoder = True
    decoder_config.is_encoder_decoder = False
    decoder_config.num_layers = config.num_decoder_layers
    self.decoder = T5Stack(decoder_config, self.shared)

    #self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    self.lm_head = LookAheadHeads(config)

    # Initialize weights and apply final processing
    self.post_init()

    # Model parallel
    self.model_parallel = False
    self.device_map = None


__all__ = [
  "T5LAForConditionalGeneration"
]
