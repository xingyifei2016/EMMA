"""
mamba.py

This code is adapted from cobra https://github.com/h-zhao1997/cobra
Paper: https://arxiv.org/abs/2403.14520

Class definition for all LLMs derived from MambaForCausalLM.
"""
from typing import Optional, Type, List

import torch
from torch import nn as nn
from emma.models.mamba.modeling_mamba import MambaForCausalLM
from emma.models.mamba.modeling_mamba import Block as MambaMixerLayer
from emma.models.mamba.modeling_mambav2 import MambaV2ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from emma.models.backbones.llm.base_llm import HFCausalLLMBackbone
from emma.models.backbones.llm.prompting import (
    ZephyrChatPromptBuilder,
    PromptBuilder,
    MambaPromptBuilder
)

from pdb import set_trace as st

# Registry =>> Support Mamba Models
# fmt: off
MAMBA_MODELS = {
    # === Pure Mamba (non-instruct/chat-tuned) Models ===
    "mamba-2.8b-slimpj": {
        "llm_family": "mamba", "llm_cls": MambaForCausalLM, "hf_hub_path": "state-spaces/mamba-2.8b-slimpj"
    },

    "mamba-2.8b": {
        "llm_family": "mamba", "llm_cls": MambaForCausalLM, "hf_hub_path": "state-spaces/mamba-2.8b"
    },

    "mamba-1.4b": {
        "llm_family": "mamba", "llm_cls": MambaForCausalLM, "hf_hub_path": "state-spaces/mamba-1.4b"
    },

    # === Finetuned Mamba Chat Model Based on mamba-2.8b-slimpj ===
    "mamba-2.8b-zephyr": {
        "llm_family": "mamba", "llm_cls": MambaForCausalLM, "hf_hub_path": "xiuyul/mamba-2.8b-zephyr"
    },
    #xiuyul/mamba-2.8b-zephyr
    
    "mamba-2.7b-v2": {
        "llm_family": "mamba", "llm_cls": MambaV2ForCausalLM, "hf_hub_path": "/home/lanxy/model/mamba2"
    },
}


class MambaLLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True, # Add for compatibility, Mamba does not have any attention
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=False,
            **MAMBA_MODELS[llm_backbone_id],
        )

        self.llm.config.pad_token_id = self.tokenizer.pad_token_id

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        inference_params=None,
        num_last_tokens:int = 0,
        intermediate=[],
        output_attn=None,
    ) -> CausalLMOutputWithPast:
        output: CausalLMOutputWithPast = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            inference_params=inference_params,
            intermediate = intermediate,
            num_last_tokens=num_last_tokens
        )
        return output
    
    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        if self.identifier.endswith("zephyr"):
            return ZephyrChatPromptBuilder
        
        elif self.identifier.startswith("mamba"):
            return MambaPromptBuilder

        raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return MambaMixerLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """Mamba was trained in AMP with BF16; see https://github.com/state-spaces/mamba/issues/6."""
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    
class MambaV2LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True, # Add for compatibility, Mamba does not have any attention
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=False,
            **MAMBA_MODELS[llm_backbone_id],
        )

        self.llm.pad_token_id = 1

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        inference_params=None,
        num_last_tokens:int = 0,
        prev_loss = 0,
        intermediate=[],
        output_attn=False,
    ) -> CausalLMOutputWithPast:
        output: CausalLMOutputWithPast = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
            prev_loss = prev_loss,
            intermediate = intermediate,
            output_attn=output_attn,
        )
        return output
    
    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        if self.identifier.endswith("zephyr"):
            return ZephyrChatPromptBuilder
        elif self.identifier.startswith("v2"):
            return MambaPromptBuilder
        elif self.identifier.startswith("mamba"):
            return MambaPromptBuilder
        raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")
        
    @property
    def embed_dim(self) -> int:
        return 2560
    
    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.llm.backbone.embedding(input_ids)
    
    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return MambaMixerLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """Mamba was trained in AMP with BF16; see https://github.com/state-spaces/mamba/issues/6."""
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
