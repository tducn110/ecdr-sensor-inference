"""schema.py — ECDR 12-class schema, HAPT definitions, and label policies.

Authoritative source: 02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex,
G2_CLASS_DECISION.md, and 07_G3_DOMAIN_CONTRACT_VI.md.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Channel specifications
NUM_CHANNELS = 6
WINDOW_LENGTH = 128
DEFAULT_SHIFT = 64
SAMPLING_RATE_HZ = 50.0

CHANNEL_NAMES = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]

# 12 classes in HAPT dataset (1-based ID in raw dataset, mapped to 0..11 internally)
HAPT_12_CLASSES: Dict[int, str] = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
    7: "STAND_TO_SIT",
    8: "SIT_TO_STAND",
    9: "SIT_TO_LIE",
    10: "LIE_TO_SIT",
    11: "STAND_TO_LIE",
    12: "LIE_TO_STAND",
}

BASIC_ACTIVITY_IDS = [1, 2, 3, 4, 5, 6]
POSTURAL_TRANSITION_IDS = [7, 8, 9, 10, 11, 12]

# Zero-based class names list (length 12)
CLASS_NAMES_12: List[str] = [HAPT_12_CLASSES[i] for i in range(1, 13)]

# Legacy 6-class names
CLASS_NAMES_6: List[str] = [HAPT_12_CLASSES[i] for i in range(1, 7)]

# Allowed label policies in HAPT manifest
LABEL_POLICIES = ("pure", "majority")

# Split definitions
SPLITS = ("train", "val", "test")

# Exact dataset split denominators under full stream replay
FULL_STREAM_DENOMINATORS = {
    "train": 9978,
    "val": 2422,
    "test": 5053,
    "total": 17453,
}

# G2-A pure support criteria: >= 50 pure windows per split, >= 3 subjects per split
MIN_SUPPORT_WINDOWS_PER_SPLIT = 50
MIN_SUPPORT_SUBJECTS_PER_SPLIT = 3


@dataclass(frozen=True)
class ClassSchema:
    num_classes: int
    class_names: Tuple[str, ...]

    def to_name(self, index: int) -> str:
        if 0 <= index < self.num_classes:
            return self.class_names[index]
        raise IndexError(f"Class index {index} out of bounds for schema with {self.num_classes} classes")

    def to_index(self, name: str) -> int:
        return self.class_names.index(name)


SCHEMA_12 = ClassSchema(num_classes=12, class_names=tuple(CLASS_NAMES_12))
SCHEMA_6 = ClassSchema(num_classes=6, class_names=tuple(CLASS_NAMES_6))
