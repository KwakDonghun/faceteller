"""Measurements on the frontal photo (2D FaceMesh landmarks in photo pixels; the eye opening is read from the picture).
Copied from experiments/feature17-20261109 (nose_measures.nose_length, jaw_measure 'F17_chin_angle', eye_dark.dark_opening,
eye_page.face_below_brows, eye_page3.photo_values, viewcheck_prep.numeric F07 / F12), formulas unchanged.

  F09 코 길이   |midpoint(159, 386) - 2| / |234 - 454|                      (not judged on a blink frame)
  F17 턱 폭     angle in degrees at the chin tip 152 between the jaw corners 172 and 397   (not pitch-adjusted)
  F01 눈 크기   ((w - mean_w) / sd_w + (ho - mean_ho) / sd_ho) / 2, w = mean eye width (33-133, 263-362), ho = mean eye
                opening read from the picture, both / sqrt(face area below the brows); the photo is first resized so that
                the face width (234-454) is 156 px, the scale of the photos the cuts were set on
  F07 미간      distance between the inner ends of the two brow hair paths (face-parsing mask) / face width 234-454
  F12 입 폭     mouth width 61-291 / face width 234-454 (eye-aligned coordinates)"""
import numpy as np
from PIL import Image
from facework.release_preprocess import eye_affine
from facework.release_brows import measure_brows
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
EYES = [(33, 133, 159, 145, 'R'), (263, 362, 386, 374, 'L')]; UP = 6; TOP, BOT = .30, .24; KFACE_FACE_WIDTH = 156.0
n_ = np.linalg.norm


def nose_length(P):
    P = np.asarray(P, float)[:, :2]; U = (P[159] + P[386]) / 2; return float(n_(U - P[2]) / n_(P[234] - P[454]))


def chin_angle(P):
    P = np.asarray(P, float)[:, :2]; a, b = P[172] - P[152], P[397] - P[152]; return float(np.degrees(np.arccos(np.clip(a @ b / n_(a) / n_(b), -1, 1))))


def poly_area(P): x, y = P[:, 0], P[:, 1]; return float(.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def face_below_brows(P):
    """Face outline polygon cut along the line through the tops of the two eyebrows (105, 334): the part below it."""
    A, B = P[105, :2], P[334, :2]; n = np.array([-(B - A)[1], (B - A)[0]]); n = n if n[1] > 0 else -n; ring = [P[i, :2] for i in OVAL]; out = []
    for i, cur in enumerate(ring):
        prev = ring[i - 1]; dc, dp = (cur - A) @ n, (prev - A) @ n
        if (dc >= 0) != (dp >= 0): out.append(prev + (cur - prev) * (dp / (dp - dc)))
        if dc >= 0: out.append(cur)
    return np.array(out)


def otsu(a):
    h, _ = np.histogram(a, 256, (0, 256)); p = h / h.sum(); w = np.cumsum(p); mu = np.cumsum(p * np.arange(256)); mt = mu[-1]
    with np.errstate(all='ignore'): s = (mt * w - mu) ** 2 / (w * (1 - w))
    return int(np.nanargmax(s))


def dark_opening(gray, P, eye, top=TOP, bot=BOT):
    """gray: PIL 'L' photo; P: (478, 2) pixels. Opening of the dark eye region around the iris column, in photo pixels, or None."""
    a, b, u, l, _ = eye; pa, pb, pu, pl = P[a], P[b], P[u], P[l]; W = float(np.hypot(*(pa - pb)))
    x0, x1 = sorted([pa[0], pb[0]]); bx0, by0 = int(np.floor(x0)) - 1, int(np.floor(pu[1] - top * W)); bx1, by1 = int(np.ceil(x1)) + 2, int(np.ceil(pl[1] + bot * W)) + 1
    g = np.asarray(gray.crop((bx0, by0, bx1, by1)).resize(((bx1 - bx0) * UP, (by1 - by0) * UP), Image.BICUBIC)).astype(float); h, w = g.shape
    X = lambda x: (x - bx0) * UP; Wu = W * UP; c0, c1 = int(round(X(x0))), int(round(X(x1))); xi0, xi1 = int(round(X(x0) + .2 * Wu)), int(round(X(x1) - .2 * Wu))
    th = otsu(g[:, c0:c1 + 1]); cm = g[:, xi0:xi1 + 1].mean(0); k = max(3, int(.25 * Wu)); cs = np.convolve(cm, np.ones(k) / k, 'valid'); cx = xi0 + int(np.argmin(cs)) + k // 2
    hw = max(2, int(round(.08 * Wu))); gapmax = max(1, int(round(.05 * Wu))); tops, bots = [], []
    for x in range(cx - hw, cx + hw + 1):
        col = g[:, x]; dark = col < th
        if not dark.any(): continue
        i = int(np.argmin(col)); t = i; gap = 0; j = i
        while j > 0:
            j -= 1
            if dark[j]: t = j; gap = 0
            else:
                gap += 1
                if gap > gapmax: break
        bt = i; gap = 0; j = i
        while j < len(col) - 1:
            j += 1
            if dark[j]: bt = j; gap = 0
            else:
                gap += 1
                if gap > gapmax: break
        ts = t - .5 if t == 0 else t - (th - col[t]) / max(1e-6, col[t - 1] - col[t]); bs = bt + .5 if bt == len(col) - 1 else bt + (th - col[bt]) / max(1e-6, col[bt + 1] - col[bt])
        tops.append(ts); bots.append(bs)
    if len(tops) < 3: return None
    top_u, bot_u = float(np.median(tops)), float(np.median(bots))
    return {'W': W, 'Hp': float(np.hypot(*(pu - pl))), 'Ho': (bot_u - top_u) / UP, 'clipped': bool(top_u <= 1.5 or bot_u >= h - 2.5)}


def eye_size(img, P3, const):
    """const = eye_measures.json 'standardisation_train270'. None when an eye cannot be read (hair, deep shadow)."""
    P3 = np.asarray(P3, float); s = KFACE_FACE_WIDTH / float(n_(P3[234, :2] - P3[454, :2])); small = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS); P = P3[:, :2] * s
    d = [dark_opening(small.convert('L'), P, eye) for eye in EYES]
    if not all(d): return None
    good = [x for x in d if not x['clipped'] and x['Ho'] <= 1.5 * x['Hp']]
    if not good: return None
    q = np.sqrt(poly_area(face_below_brows(P))); w = np.mean([x['W'] for x in d]) / q; ho = np.mean([x['Ho'] for x in good]) / q
    return float(((w - const['w']['mean']) / const['w']['sd'] + (ho - const['ho']['mean']) / const['ho']['sd']) / 2)


def brow_gap_and_mouth_width(P3, mask):
    """-> (F07 or None, F12). mask: face-parsing class mask of the same photo."""
    P3 = np.asarray(P3, float); p = P3[:, :2]; q, _ = eye_affine(P3); E = lambda a, b: float(n_(q[a] - q[b]))
    inner = [np.array(s['path_original'])[0] for s in measure_brows(mask, P3, pose=None, blend=None)['sides'].values()]
    return (float(n_(inner[0] - inner[1]) / n_(p[234] - p[454])) if len(inner) == 2 else None), E(61, 291) / E(234, 454)
