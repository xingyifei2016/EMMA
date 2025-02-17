"""
materialize.py

This code is adapted from cobra https://github.com/h-zhao1997/cobra
Paper: https://arxiv.org/abs/2403.14520

Factory class for initializing pretraining datasets on a per-VLM basis; provides and exports individual functions for
clear control flow.
"""
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from emma.conf import DatasetConfig
from emma.models.backbones.llm.prompting import PromptBuilder
from emma.models.backbones.vision import ImageTransform
from emma.preprocessing.datasets import AlignDataset, FinetuneDataset, SarcasmDataset, EvalDataset
from emma.util.data_utils import PaddedCollatorForLanguageModeling

# Dataset Initializers =>> Maps Stage --> cls()
DATASET_INITIALIZER = {"align": AlignDataset, "finetune": FinetuneDataset, "full-finetune": FinetuneDataset, "sarcasm_dataset": SarcasmDataset, "eval_dataset": EvalDataset}


def get_dataset_and_collator(
    stage: str,
    dataset_cfg: DatasetConfig,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
) -> Tuple[Dataset, PaddedCollatorForLanguageModeling]:
    dataset_cls = DATASET_INITIALIZER[stage]
    dataset_root_dir = dataset_cfg.dataset_root_dir
    collator = PaddedCollatorForLanguageModeling(
        tokenizer.model_max_length, tokenizer.pad_token_id, default_image_resolution, padding_side=padding_side
    )
    print("PAD TOKEN ID: "+str(tokenizer.pad_token_id))
    # Switch on `stage`
    if stage == "align":
        annotation_json, image_dir = dataset_cfg.align_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json, dataset_root_dir / image_dir, image_transform, tokenizer
        )
        return dataset, collator

    elif stage == "finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator
    elif stage == "eval_dataset":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
        )
        return dataset, collator

    elif stage == "full-finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator
        
    elif stage == "sarcasm":
        image_dir, annotation_json1, annotation_json2 = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json1,
            dataset_root_dir / annotation_json2,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    else:
        raise ValueError(f"Stage `{stage}` is not supported!")
