"""VGGT 3D points of the frontal photo's landmarks (frontal + two oblique photos) and the depth features read from them.
Copied from experiments/feature17-20261109 (vggt3/vggt_run.sample_all, vggt3/analyze.py rigid frame and 'F10_line_gp',
depth_models.frames / score, bench_deploy.to_dtype), formulas unchanged.

  guide_images (face crops) -> VGGTEngine.infer -> MediaPipe on the prepared frontal image -> 3D point per landmark
  (median of the 3x3 neighbourhood of the frontal point map)
  F10 코 높이        distance of the nose tip 4 from the line glabella 9 - chin front 199 in the midsagittal plane of the
                     rigid frame (Kabsch fit of 30 rigid points to vggt_template.npy), / 3D outer-eye distance 33-263
  F20, F21           intercept + sum(coef[frame][landmark] * depth of that landmark in the frame) + width terms
                     (depth_models.json), frames 'rig' (rigid fit) and 'old' (eye axis + forehead-chin)
VGGT precision: the service engine keeps the aggregator in BF16; GPUs below compute capability 8 (T4) need FP16."""
import numpy as np
import torch
from facework.pilot.depth import VGGTEngine
from facework.release_depth import guide_images
import posenorm as PN
DTYPES = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
unit = lambda v: v / np.linalg.norm(v)


def resolve_dtype(name):
    if name != 'auto': return name
    return 'bf16' if torch.cuda.get_device_capability()[0] >= 8 else 'fp16'


def make_engine(weights_path, dtype_name):
    """VGGTEngine with the aggregator in the requested precision (camera / depth heads stay FP32)."""
    eng = VGGTEngine(weights_path=weights_path); dtype = DTYPES[dtype_name]
    if dtype is torch.bfloat16: return eng
    eng.model.aggregator.to(dtype=dtype); base = VGGTEngine.infer

    def infer(images, size=None):
        orig = torch.autocast

        class AC:  # VGGTEngine.infer opens autocast(bfloat16); replace it by the requested precision for this call
            def __init__(self, *a, **k): self.c = orig('cuda', dtype=dtype, enabled=dtype is not torch.float32)
            def __enter__(self): return self.c.__enter__()
            def __exit__(self, *e): return self.c.__exit__(*e)
        torch.autocast = AC
        try: return base(eng, images, size)
        finally: torch.autocast = orig
    eng.infer = infer; return eng


def sample_all(points, lm, conf):
    h, w = points.shape[:2]; P = np.full((len(lm), 3), np.nan, np.float32); C = np.full(len(lm), np.nan, np.float32)
    for i, (u, v) in enumerate(np.round(lm[:, :2]).astype(int)):
        if 1 <= u < w - 1 and 1 <= v < h - 1: P[i] = np.median(points[v - 1:v + 2, u - 1:u + 2].reshape(-1, 3), axis=0); C[i] = conf[v, u]
    return P, C


def landmark_points(engine, images, detections, detect):
    """-> (478, 3) float64 VGGT points of the frontal photo's landmarks, or None when the prepared frontal image has no face."""
    g = engine.infer(guide_images(images, detections)); m = detect(g['prepared_images'][0])[0]
    if m is None: return None, g
    return sample_all(g['points'][0], m['landmarks'], g['depth_confidence'][0])[0].astype(float), g


class DepthFrames:
    def __init__(self, template):
        M = np.asarray(template, float); x = unit(M[263] - M[33]); down = M[152] - M[10]; down = unit(down - x * (x @ down)); n = np.cross(x, down)
        self.M, self.xM, self.dM, self.nM = M, x, down, (n if n[2] < 0 else -n)  # VGGT camera looks along +z: toward the camera is -z

    def nose_height(self, P):
        """analyze.py 'F10_line_gp' in the rigid frame."""
        X, _ = PN._norm(P); R = PN.kabsch(X[PN.RIGID], self.M[PN.RIGID]); x, n = R @ self.xM, R @ self.nM; ew = np.linalg.norm(P[263] - P[33])
        A, B, T = P[9], P[199], P[4]; ab = B - A; ab = ab - x * (ab @ x); at = T - A; at = at - x * (at @ x); foot = at - ab * (at @ ab) / (ab @ ab)
        return float(np.sign(foot @ n) * np.linalg.norm(foot) / ew)

    def frames(self, P):
        ew = np.linalg.norm(P[263] - P[33]); Xn, sc = PN._norm(P); Y = Xn @ PN.kabsch(Xn[PN.RIGID], self.M[PN.RIGID]) * sc / ew
        x = unit(P[263] - P[33]); down = P[152] - P[10]; down = unit(down - x * (x @ down)); n = np.cross(x, down); n = n if n[2] < 0 else -n
        return {'rig': np.c_[Y @ self.xM, Y @ self.dM, Y @ self.nM], 'old': np.c_[P @ x, P @ down, P @ n] / ew}

    @staticmethod
    def score(model, c):
        s = model['intercept'] + sum(v * c[fr][int(i), 2] for fr, d in model['coef'].items() for i, v in d.items())
        return float(s + sum(t['coef'] * abs(c['rig'][t['numerator_mirror'], 0] - c['rig'][t['numerator_landmark'], 0]) / abs(c['rig'][t['denominator_mirror'], 0] - c['rig'][t['denominator_landmark'], 0]) for t in model['width_terms']))
