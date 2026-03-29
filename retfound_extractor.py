"""
retfound_extractor.py
---------------------
Loads the RETFound ViT-Large checkpoint and extracts 1024-D features
from one or more PIL images.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

RETFOUND_CKPT = '/shared/ssd_28T/home/mm3572/anomaly_detection/Mousa/RETFound_cfp_weights.pth'
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


def _load_model(ckpt_path: str) -> nn.Module:
    try:
        import timm
    except ImportError:
        raise ImportError('timm is required: pip install timm')

    model = timm.create_model(
        'vit_large_patch16_224',
        pretrained  = False,
        num_classes = 0,
        global_pool = '',
    )

    import argparse
    torch.serialization.add_safe_globals([argparse.Namespace])
    try:
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    except Exception:
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if isinstance(state, dict) and 'model' in state:
        state = state['model']

    msg = model.load_state_dict(state, strict=False)
    print(f'[RETFound] Loaded checkpoint. Missing: {len(msg.missing_keys)}'
          f'  Unexpected: {len(msg.unexpected_keys)}')
    model.eval()
    return model.to(DEVICE)


_MODEL = None

def get_model() -> nn.Module:
    global _MODEL
    if _MODEL is None:
        print(f'[RETFound] Loading model from {RETFOUND_CKPT} on {DEVICE} ...')
        _MODEL = _load_model(RETFOUND_CKPT)
    return _MODEL


@torch.no_grad()
def extract_features(pil_images: list) -> np.ndarray:
    model = get_model()
    batch = torch.stack([TRANSFORM(img.convert('RGB')) for img in pil_images])
    batch = batch.to(DEVICE)
    out   = model(batch)
    feats = out[:, 0, :].cpu().numpy() if out.ndim == 3 else out.cpu().numpy()
    return feats.astype(np.float32)
