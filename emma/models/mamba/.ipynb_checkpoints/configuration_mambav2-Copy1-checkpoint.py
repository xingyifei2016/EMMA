from dataclasses import dataclass, field
from transformers.configuration_utils import PretrainedConfig



# coding=utf-8
# Copyright (c) 2023 Jean-Loup Tastet

from transformers.configuration_utils import PretrainedConfig


# DEFAULT_SSM_CONFIG = {
#     "d_state": 16,
#     "d_conv": 4,
#     "expand": 2,
#     "dt_rank": "auto",
#     "dt_min": 0.001,
#     "dt_max": 0.1,
#     "dt_init": "random",
#     "dt_scale": 1.0,
#     "dt_init_floor": 1e-4,
#     "conv_bias": True,
#     "bias": False,
#     "use_fast_path": True,
# }


class MambaV2Config(PretrainedConfig):

    model_type = "mamba"

    def __init__(
        self,
        d_model = 2560,
        d_intermediate = 0,
        n_layer = 64,
        vocab_size = 50277,
        ssm_cfg = field(default_factory=dict),
        attn_layer_idx = field(default_factory=list),
        attn_cfg = field(default_factory=dict),
        rms_norm = True,
        residual_in_fp32 = True,
        fused_add_norm = True,
        pad_vocab_size_multiple = 8,
        tie_embeddings = True,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=1,
        initializer_cfg=None,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)
        
        self.d_model = d_model
        self.d_intermediate = d_intermediate
        self.n_layer = n_layer
        self.vocab_size = vocab_size 
        self.ssm_cfg = ssm_cfg
        self.attn_layer_idx = attn_layer_idx
        self.attn_cfg = attn_cfg
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.tie_embeddings = tie_embeddings

        self.ssm_cfg = ssm_cfg
        self.initializer_cfg = initializer_cfg
        
        
# @dataclass
# class MambaConfig:

#     d_model: int = 2560
#     d_intermediate: int = 0
#     n_layer: int = 64
#     vocab_size: int = 50277
#     ssm_cfg: dict = field(default_factory=dict)
#     attn_layer_idx: list = field(default_factory=list)
#     attn_cfg: dict = field(default_factory=dict)
#     rms_norm: bool = True
#     residual_in_fp32: bool = True
#     fused_add_norm: bool = True
#     pad_vocab_size_multiple: int = 8
#     tie_embeddings: bool = True
        