"""Optional PyTorch DEIMv2 adapter; imported only for fresh anchor inference."""
from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch
from torchvision import transforms as T

from .settings import SETTINGS

CHECKPOINT = SETTINGS.deim_checkpoint
TRANSFORM = T.Compose([
    T.Resize((640, 640)), T.ToTensor(),
    T.Normalize([.485, .456, .406], [.229, .224, .225]),
])


def load_model(device: torch.device) -> tuple[Any, Any, dict]:
    """Load a locally supplied person detector, preferring its EMA weights.

    Args:
        device: CUDA device selected by the inference worker.

    Returns:
        Evaluation model, postprocessor, and checkpoint metadata.
    """
    for name in ('deim_repo', 'deim_config', 'deim_checkpoint'):
        path = getattr(SETTINGS, name)
        if path is None or not path.exists():
            raise ValueError(f'Set a valid {name} in FOREST_SAR_CONFIG.')
    sys.path.insert(0, str(SETTINGS.deim_repo))
    from engine.core import YAMLConfig
    cfg = YAMLConfig(str(SETTINGS.deim_config), resume=str(CHECKPOINT))
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    ema = checkpoint.get('ema')
    state = ema['module'] if ema else checkpoint['model']
    cfg.model.load_state_dict(state, strict=True)
    model = cfg.model.eval().to(device)
    return model, cfg.postprocessor.eval().to(device), {
        'state': 'ema.module' if ema else 'model',
        'stored_last_epoch': checkpoint.get('last_epoch', checkpoint.get('epoch')),
        'parameters': sum(p.numel() for p in model.parameters()),
    }


def infer(model: Any, postprocessor: Any, samples: torch.Tensor,
          sizes: list[list[int]], device: torch.device) -> list[np.ndarray]:
    """Return pixel-space XYXY/score rows for a single-class person model.

    Args:
        model: Loaded DEIMv2 network.
        postprocessor: Matching deployed postprocessor.
        samples: Normalized 640 by 640 input batch.
        sizes: Original image width and height for each sample.
        device: Inference device.

    Returns:
        One float array with five columns per image.
    """
    with torch.inference_mode():
        output = model(samples.to(device, non_blocking=True))
        predictions = postprocessor(
            output, torch.tensor(sizes, dtype=torch.int64, device=device))
    rows = []
    for prediction in predictions:
        if not torch.all(prediction['labels'] == 0):
            raise ValueError('This adapter expects a person-only class-0 detector.')
        rows.append(np.concatenate([
            prediction['boxes'].detach().float().cpu().numpy(),
            prediction['scores'].detach().float().cpu().numpy()[:, None],
        ], axis=1))
    return rows
