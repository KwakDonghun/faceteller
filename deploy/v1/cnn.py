"""CNN part models: input tiles and 5-fold ensembles. Copied from experiments/feature17-20261109 (trainer_v2.Net,
ordinal_probs, run_model, load_ensemble; bench_deploy / run_person tiles), with one change: the head layout is read from the
checkpoint itself ({'heads', 'sizes', 'has_unj'}), so no label file is needed.

  photo -> posenorm.warp (pose-normalised) -> frontal_crops -> network_array per crop -> compose_context (part tile next to
  the whole-face tile; face tile mirrored for left crops) -> /255, ImageNet mean/std, channels_last, autocast fp16
  output per head: K-1 cumulative logits -> sigmoid -> running minimum -> class probabilities; mean over the 5 models and
  over the left / right crops."""
import numpy as np
import torch
from PIL import Image
from facework.kface_training import PartNet
from facework.release_preprocess import frontal_crops, network_array, PART_CROPS
from facework.release_context import compose_context, CONTEXT_PARTS
import posenorm as PN
FOLDS = 5


def tiles(img, P3, template, parts):
    """{part: uint8 tensor (crops, 3, H, W)} for one frontal photo."""
    w, Q = PN.warp(img, P3, template); cr = frontal_crops(Image.fromarray(w), Q); face = network_array(cr['face'], 'face'); out = {}
    for part in parts:
        for name in PART_CROPS[part]:
            a = network_array(cr[name], part); out.setdefault(part, []).append(compose_context(a, face, part, mirror_face=name.endswith('_left')) if part in CONTEXT_PARTS else a)
    return {p: torch.from_numpy(np.stack(v)).permute(0, 3, 1, 2).contiguous() for p, v in out.items()}


class Net:
    """PartNet (ResNet18) with, per head, K-1 cumulative logits (+1 unjudgeable logit when the head has one)."""
    def __init__(self, ck, device):
        self.heads, self.sizes, self.has_unj = list(ck['heads']), list(ck['sizes']), list(ck['has_unj'])
        self.model = PartNet('resnet18', 0, [s + (1 if h else 0) for s, h in zip(self.sizes, self.has_unj)], pretrained=False)
        self.model.load_state_dict(ck['state_dict']); self.model = self.model.to(device).eval()
        if device == 'cuda': self.model = self.model.to(memory_format=torch.channels_last)

    def split(self, outs): return [(o[:, :s], o[:, s] if h else None) for o, s, h in zip(outs, self.sizes, self.has_unj)]


def ordinal_probs(cum_logits):
    p = torch.cummin(torch.sigmoid(cum_logits.float()), dim=1).values; ones = torch.ones_like(p[:, :1]); zeros = torch.zeros_like(p[:, :1])
    return (torch.cat([ones, p], 1) - torch.cat([p, zeros], 1)).clamp_min(0)


def load_part(folder, part, device='cuda'):
    """folder/models/{part}_fold{k}.pt, k = 0..4"""
    return [Net(torch.load(folder / 'models' / f'{part}_fold{k}.pt', map_location='cpu', weights_only=True), device) for k in range(FOLDS)]


def run_model(net, images, device='cuda', batch=32):
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]; std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]; outs = None
    with torch.inference_mode():
        for s in range(0, len(images), batch):
            x = ((images[s:s + batch].to(device).float() / 255 - mean) / std).contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda', dtype=torch.float16): o = net.model(x)[1]
            parts = [(ordinal_probs(c).cpu().numpy(), torch.sigmoid(un.float()).cpu().numpy() if un is not None else None) for c, un in net.split(o)]
            outs = parts if outs is None else [(np.r_[a, c], None if b is None else np.r_[b, d]) for (a, b), (c, d) in zip(outs, parts)]
    return outs


def ensemble(nets, x):
    """-> {head: (probabilities averaged over crops and models, unjudgeable probability or None)}"""
    agg = None
    for n in nets:
        o = run_model(n, x); agg = o if agg is None else [(a + c, None if b is None else b + d) for (a, b), (c, d) in zip(agg, o)]
    return {k: ((agg[j][0].mean(0) / len(nets)).tolist(), None if agg[j][1] is None else float(agg[j][1].mean() / len(nets))) for j, k in enumerate(nets[0].heads)}
