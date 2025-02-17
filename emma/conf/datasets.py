"""
datasets.py

This code is adapted from cobra https://github.com/h-zhao1997/cobra

Draccus Dataclass Definition for a DatasetConfig object, with various registered subclasses for each dataset variant
and processing scheme. A given dataset variant (e.g., `llava-lightning`) configures the following attributes:
    - Dataset Variant (Identifier) --> e.g., "llava-v15"
    - Align Stage Dataset Components (annotations, images)
    - Finetune Stage Dataset Components (annotations, images)
    - Dataset Root Directory (Path)
"""
from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Tuple

from draccus import ChoiceRegistry


@dataclass
class DatasetConfig(ChoiceRegistry):
    # fmt: off
    dataset_id: str                                 # Unique ID that fully specifies a dataset variant

    # Dataset Components for each Stage in < align | finetune >
    align_stage_components: Tuple[Path, Path]       # Path to annotation file and images directory for `align` stage
    finetune_stage_components: Tuple[Path, Path]    # Path to annotation file and images directory for `finetune` stage

    dataset_root_dir: Path                          # Path to dataset root directory; others paths are relative to root
    # fmt: on


    

# [Reproduction] LLaVa-v15 (exact dataset used in all public LLaVa-v15 models)
@dataclass
class LLaVa_V15_Config(DatasetConfig):
    dataset_id: str = "llava-v15"

    align_stage_components: Tuple[Path, Path] = (
        Path("/path/to/llava-laion-cc-sbu-558k/chat.json"),
        Path("/path/to/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("/path/to/llava_v1_5_mix665k.json"),
        Path("/path/to/images"),
    )
    dataset_root_dir: Path = Path("data")



# LLaVa-v15 + LVIS-Instruct-4V + LRV-Instruct
@dataclass
class LLaVa_LVIS4V_LRV_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v-lrv"

    align_stage_components: Tuple[Path, Path] = (
        Path("/path/to/llava-laion-cc-sbu-558k/chat.json"),
        Path("/path/to/images/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("/path/to/llava_v1_5_lvis4v_lrv_mix1231k.json"),
        Path("/path/to/images"),
    )
    dataset_root_dir: Path = Path("data")


# LLaVa-v15 + LVIS-Instruct-4V + LRV-Instruct
@dataclass
class LLaVa_LVIS4V_LRV_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v-lrv"

    align_stage_components: Tuple[Path, Path] = (
        Path("/home/lanxy/Dataset/download/llava-laion-cc-sbu-558k/chat.json"),
        Path("/home/lanxy/Dataset/download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("/home/lanxy/Dataset/LLaVA-Finetune/llava_v1_5_lvis4v_lrv_mix1231k.json"),
        Path("/home/lanxy/Dataset/LLaVA-Finetune/"),
    )
    dataset_root_dir: Path = Path("data")
    


# === Define a Dataset Registry Enum for Reference & Validation =>> all *new* datasets must be added here! ===
@unique
class DatasetRegistry(Enum):
    # === LLaVa v1.5 ===
    LLAVA_V15 = LLaVa_V15_Config
    LLAVA_LVIS4V_LRV = LLaVa_LVIS4V_LRV_Config

    @property
    def dataset_id(self) -> str:
        return self.value.dataset_id


# Register Datasets in Choice Registry
for dataset_variant in DatasetRegistry:
    DatasetConfig.register_subclass(dataset_variant.dataset_id, dataset_variant.value)
