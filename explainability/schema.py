"""Shared, type-aware explanation record for all three arms (Phase 5 / Risk 1).

One record type, two optional evidence channels:
  - `image_regions`   for anything with a vision backbone (Grad-CAM on Arm 1,
                      attention/gradient maps on a VLM vision tower)
  - `salient_tokens`  for anything with a text/rationale generator (Arm 2/3)
plus an optional natural-language `rationale` and a free-form `faithfulness`
dict that a scorer fills in. Arms only populate the channels they have; the
faithfulness scorer and the demo read the same schema regardless of arm.

Usage:
    rec = ExplanationRecord(example_id="14", arm="arm1", level="level1",
                            pred="troll", true_label="troll", ...)
    save_records([rec], "out.json");  recs = load_records("out.json")
"""
import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ImageRegions:
    """A spatial importance map over the input image.

    heatmap: grid_h x grid_w nested list of floats in [0, 1] (row-major).
    heatmap_path: optional path to a rendered overlay image.
    image_size: (H, W) of the image the grid is laid over (e.g. 224x224).
    """

    heatmap: List[List[float]]
    image_size: Tuple[int, int] = (224, 224)
    heatmap_path: Optional[str] = None
    method: str = "gradcam"

    @property
    def grid(self) -> Tuple[int, int]:
        return (len(self.heatmap), len(self.heatmap[0]) if self.heatmap else 0)


@dataclass
class ExplanationRecord:
    example_id: str
    arm: str  # "arm1" | "arm2" | "arm3"
    level: str  # "level1" | "level2"
    pred: str
    true_label: Optional[str] = None
    image_regions: Optional[ImageRegions] = None
    salient_tokens: Optional[List[Tuple[str, float]]] = None
    rationale: Optional[str] = None
    faithfulness: dict = field(default_factory=dict)

    @property
    def correct(self) -> Optional[bool]:
        return None if self.true_label is None else self.pred == self.true_label

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExplanationRecord":
        d = dict(d)
        ir = d.get("image_regions")
        if ir is not None:
            ir = dict(ir)
            ir["image_size"] = tuple(ir.get("image_size", (224, 224)))
            d["image_regions"] = ImageRegions(**ir)
        st = d.get("salient_tokens")
        if st is not None:
            d["salient_tokens"] = [(t, float(s)) for t, s in st]
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "ExplanationRecord":
        return cls.from_dict(json.loads(s))


def save_records(records, path):
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in records], f, indent=1)


def load_records(path):
    with open(path) as f:
        return [ExplanationRecord.from_dict(d) for d in json.load(f)]
