"""Enhancers and segmenters used by the benchmark.

A *method* is (enhancer, segmenter):

    sam        = none     + SAM ViT-B (ultralytics, sam_b.pt)
    clahe      = CLAHE    + SAM ViT-B
    gamma      = gamma    + SAM ViT-B
    zerodce    = Zero-DCE + SAM ViT-B
    robustsam  = none     + RobustSAM ViT-B

All images are RGB uint8 HxWx3. Boxes are [x1, y1, x2, y2] in pixels.
Every segmenter returns one boolean HxW mask per prompt (multimask_output=False).
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO / "third_party"
WEIGHTS = REPO / "weights"

METHODS = {
    "sam": ("none", "sam"),
    "clahe": ("clahe", "sam"),
    "gamma": ("gamma", "sam"),
    "zerodce": ("zerodce", "sam"),
    "robustsam": ("none", "robustsam"),
}

# Fixed before seeing any results. Do not tune on the test set.
CLAHE_CLIP, CLAHE_TILE = 2.0, 8
GAMMA_TARGET_LUMA = 0.5   # adaptive gamma: maps the image's mean luma to mid-grey
GAMMA_CLAMP = (0.2, 3.0)


def pick_device(pref: str = "auto") -> str:
    import torch
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    """Make GPU timings honest."""
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def mean_luma(rgb: np.ndarray) -> float:
    """Rec. 601 luma in [0, 1], the same measure as darken.py."""
    return float(((rgb.astype(np.float32) / 255.0) @ np.array([0.299, 0.587, 0.114], np.float32)).mean())


# ------------------------------------------------------------------ enhancers
class NoEnhance:
    name = "none"

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        return rgb


class Clahe:
    """CLAHE on the L channel of CIELAB (colour preserved)."""
    name = "clahe"

    def __init__(self, clip: float = CLAHE_CLIP, tile: int = CLAHE_TILE):
        self.op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab[..., 0] = self.op.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class AdaptiveGamma:
    """out = in ** g, with g = log(target) / log(mean luma): parameter-free per image."""
    name = "gamma"

    def __init__(self, target: float = GAMMA_TARGET_LUMA, clamp=GAMMA_CLAMP):
        self.target, self.clamp = target, clamp
        self.last_gamma = 1.0

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        m = min(max(mean_luma(rgb), 1e-4), 0.9999)
        g = float(np.clip(np.log(self.target) / np.log(m), *self.clamp))
        self.last_gamma = g
        lut = np.round(255.0 * (np.arange(256) / 255.0) ** g).astype(np.uint8)
        return lut[rgb]


class ZeroDCE:
    """Official Zero-DCE (Guo et al., CVPR 2020) with the released Epoch99.pth."""
    name = "zerodce"

    def __init__(self, device: str):
        import torch
        code = THIRD_PARTY / "Zero-DCE" / "Zero-DCE_code"
        spec = importlib.util.spec_from_file_location("zerodce_model", code / "model.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.torch, self.device = torch, device
        self.net = mod.enhance_net_nopool().to(device).eval()
        ckpt = WEIGHTS / "Epoch99.pth"
        if not ckpt.exists():
            ckpt = code / "snapshots" / "Epoch99.pth"
        self.net.load_state_dict(torch.load(ckpt, map_location="cpu"))

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        t = self.torch
        with t.no_grad():
            x = t.from_numpy(rgb).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            _, y, _ = self.net(x)
            y = y.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.round(y * 255.0).astype(np.uint8)


def make_enhancer(name: str, device: str):
    if name == "zerodce":
        return ZeroDCE(device)
    return {"none": NoEnhance, "clahe": Clahe, "gamma": AdaptiveGamma}[name]()


# ------------------------------------------------------------------ segmenters
class UltralyticsSAM:
    """SAM ViT-B through ultralytics, as in the daylight baseline."""
    name = "sam"

    def __init__(self, device: str, weights: str = "sam_b.pt"):
        from ultralytics import SAM
        w = WEIGHTS / weights
        self.model = SAM(str(w) if w.exists() else weights)
        self.device = device

    def _run(self, rgb: np.ndarray, **prompt) -> list[np.ndarray]:
        bgr = np.ascontiguousarray(rgb[..., ::-1])  # ultralytics expects cv2-style BGR arrays
        res = self.model(bgr, device=self.device, verbose=False, **prompt)[0]
        if res.masks is None:
            return []
        return [m.astype(bool) for m in res.masks.data.cpu().numpy()]

    def boxes(self, rgb: np.ndarray, boxes: list[list[float]]) -> list[np.ndarray]:
        if not boxes:
            return []
        masks = self._run(rgb, bboxes=[list(map(float, b)) for b in boxes])
        if len(masks) != len(boxes):
            raise RuntimeError(f"ultralytics returned {len(masks)} masks for {len(boxes)} boxes")
        return masks

    def point(self, rgb: np.ndarray, xy: tuple[float, float]) -> np.ndarray:
        masks = self._run(rgb, points=[[float(xy[0]), float(xy[1])]], labels=[1])
        return masks[0] if masks else np.zeros(rgb.shape[:2], bool)


class RobustSAM:
    """RobustSAM ViT-B (Chen et al., CVPR 2024), official code and checkpoint.

    Mirrors the repo's eval.py: ResizeLongestSide(1024) then model.predict(...).
    One prompt per call (the image encoder runs each call; fine at this scale).
    """
    name = "robustsam"

    def __init__(self, device: str, size: str = "b"):
        import torch
        sys.path.insert(0, str(THIRD_PARTY / "RobustSAM"))
        from robust_segment_anything import sam_model_registry
        from robust_segment_anything.utils.transforms import ResizeLongestSide

        ckpt = WEIGHTS / f"robustsam_checkpoint_{size}.pth"
        self.opt = types.SimpleNamespace(gpu=device)
        self.model = sam_model_registry[f"vit_{size}"](opt=self.opt, checkpoint=str(ckpt)).to(device).eval()
        self.tf = ResizeLongestSide(self.model.image_encoder.img_size)
        self.torch, self.device = torch, device

    def _predict(self, rgb: np.ndarray, **prompt) -> np.ndarray:
        t = self.torch
        img = t.from_numpy(np.ascontiguousarray(rgb)).to(self.device).permute(2, 0, 1).unsqueeze(0)
        size = img.shape[-2:]
        d = {"image": self.tf.apply_image_torch(img.float()), "original_size": size}
        if "box" in prompt:
            b = t.tensor([prompt["box"]], dtype=t.float32, device=self.device)
            d["boxes"] = self.tf.apply_boxes_torch(b, size).unsqueeze(0)
        else:
            p = t.tensor([prompt["point"]], dtype=t.float32, device=self.device)
            d["point_coords"] = self.tf.apply_coords_torch(p, size).unsqueeze(0)
            d["point_labels"] = t.ones((1, 1), device=self.device)
        with t.no_grad():
            out = self.model.predict(self.opt, [d], multimask_output=False, return_logits=False)
        return out[0]["masks"][0, 0].cpu().numpy().astype(bool)

    def boxes(self, rgb: np.ndarray, boxes: list[list[float]]) -> list[np.ndarray]:
        return [self._predict(rgb, box=list(map(float, b))) for b in boxes]

    def point(self, rgb: np.ndarray, xy: tuple[float, float]) -> np.ndarray:
        return self._predict(rgb, point=[float(xy[0]), float(xy[1])])


def make_segmenter(name: str, device: str):
    return {"sam": UltralyticsSAM, "robustsam": RobustSAM}[name](device)


class MethodBank:
    """Builds each enhancer/segmenter once and shares them across methods."""

    def __init__(self, methods: list[str], device: str):
        self.device = device
        self.enh, self.seg = {}, {}
        for m in methods:
            e, s = METHODS[m]
            if e not in self.enh:
                self.enh[e] = make_enhancer(e, device)
            if s not in self.seg:
                self.seg[s] = make_segmenter(s, device)

    def parts(self, method: str):
        e, s = METHODS[method]
        return self.enh[e], self.seg[s]
