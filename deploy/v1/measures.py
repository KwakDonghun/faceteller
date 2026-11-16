"""Measurements on the frontal photo (2D FaceMesh landmarks in photo pixels; the eye opening is read from the picture).
Copied from experiments/feature17-20261109 (nose_measures.nose_length, jaw_measure 'F17_chin_angle', eye_dark.dark_opening,
eye_page.face_below_brows, eye_page3.photo_values, viewcheck_prep.numeric F07 / F12), formulas unchanged.

  F09 코 길이   |midpoint(159, 386) - 2| / |234 - 454|                      (not judged on a blink frame)
  F17 턱 폭     angle in degrees at the chin tip 152 between the jaw corners 172 and 397   (not pitch-adjusted)
  F01 눈 크기   ((w - mean_w) / sd_w + (ho - mean_ho) / sd_ho) / 2, w = mean eye width (33-133, 263-362), ho = mean eye
                opening read from the picture, both / sqrt(face area below the brows); the photo is first resized so that
                the face width (234-454) is 156 px, the scale of the photos the cuts were set on
  F07 미간      distance between the inner ends of the two brow hair paths (face-parsing mask) / face width 234-454
  F12 입 폭     mouth width 61-291 / face width 234-454 (eye-aligned coordinates)

reference_measures(): plain geometric measures for the features the structure judges by CNN (눈 가늘기, 눈꼬리 기울기, 눈썹 꼬리
길이 / 휨 / 눈썹-눈 간격, 코 폭, 입술 두께, 입꼬리, 인중 길이). They are shown next to the CNN result so a reader can see the
shape in numbers; they are NOT used for the judgement (viewcheck_prep.numeric v1 / v2 formulas, feature_review_math.shape_features)."""
import math
import numpy as np
from PIL import Image
from facework.release_preprocess import eye_affine
from facework.release_brows import measure_brows
import posenorm as PN
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


def eye_size_detail(img, P3, const):
    """const = eye_measures.json 'standardisation_train270'. -> (value or None, components). None when an eye cannot be read."""
    P3 = np.asarray(P3, float); fw = float(n_(P3[234, :2] - P3[454, :2])); s = KFACE_FACE_WIDTH / fw; small = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS); P = P3[:, :2] * s
    d = [dark_opening(small.convert('L'), P, eye) for eye in EYES]
    if not all(d): return None, []
    good = [x for x in d if not x['clipped'] and x['Ho'] <= 1.5 * x['Hp']]; W = float(np.mean([x['W'] for x in d])); comp = [{'label': '눈 가로 폭 ÷ 얼굴 폭', 'value': W / KFACE_FACE_WIDTH, 'unit': '비율'}]
    if not good: return None, comp
    q = np.sqrt(poly_area(face_below_brows(P))); w = W / q; Ho = float(np.mean([x['Ho'] for x in good])); ho = Ho / q
    comp += [{'label': '눈 뜬 높이 ÷ 눈 가로 폭', 'value': Ho / W, 'unit': '비율'}, {'label': '눈 가로 폭 ÷ √(눈썹 아래 얼굴 넓이)', 'value': float(w), 'unit': '비율'}, {'label': '눈 뜬 높이 ÷ √(눈썹 아래 얼굴 넓이)', 'value': float(ho), 'unit': '비율'}]
    return float(((w - const['w']['mean']) / const['w']['sd'] + (ho - const['ho']['mean']) / const['ho']['sd']) / 2), comp


def eye_size(img, P3, const): return eye_size_detail(img, P3, const)[0]


def brow_gap_and_mouth_width(P3, mask):
    """-> (F07 or None, F12). mask: face-parsing class mask of the same photo."""
    P3 = np.asarray(P3, float); p = P3[:, :2]; q, _ = eye_affine(P3); E = lambda a, b: float(n_(q[a] - q[b]))
    inner = [np.array(s['path_original'])[0] for s in measure_brows(mask, P3, pose=None, blend=None)['sides'].values()]
    return (float(n_(inner[0] - inner[1]) / n_(p[234] - p[454])) if len(inner) == 2 else None), E(61, 291) / E(234, 454)


# ---- what each judged value is, for display (label, unit)
INFO = {'F01': ('눈 크기 점수: 눈 가로 폭과 눈 뜬 높이를 표준화해 평균한 값(0이 학습 자료 평균)', '점수'), 'F07': ('두 눈썹 안쪽 끝 사이 ÷ 얼굴 폭', '비율'), 'F09': ('두 눈 위쪽 끝의 가운데 ~ 코밑 ÷ 얼굴 폭', '비율'),
        'F10': ('코끝이 미간~턱 앞 선에서 앞으로 나온 거리 ÷ 두 눈 바깥 끝 거리(3차원)', '비율'), 'F12': ('입꼬리 사이 ÷ 얼굴 폭', '비율'), 'F17': ('턱 끝에서 양쪽 턱 모서리를 본 각도', '도'),
        'F20': ('광대 돌출 식의 점수(3차원 점 깊이의 가중합)', '점수'), 'F21': ('입 돌출 식의 점수(3차원 점 깊이의 가중합)', '점수')}


def _bend(path):
    """feature_review_math.shape_features for one brow path: (largest offset from the chord / chord length, bend angle in degrees at that point)."""
    p = np.asarray(path, float)
    if len(p) < 9: return None, None
    a, b = p[0], p[-1]; ch = b - a; n = n_(ch)
    if n < 1e-9: return None, None
    off = np.abs((p[:, 0] - a[0]) * ch[1] - (p[:, 1] - a[1]) * ch[0]) / n; i = int(np.argmax(off)); arc = float(off[i] / n); i = max(4, min(len(p) - 5, i)); u = p[i] - p[i - 4]; v = p[i + 4] - p[i]
    return arc, abs(math.degrees(math.atan2(u[0] * v[1] - u[1] * v[0], float(u @ v))))


def reference_measures(P3, mask, template):
    """{feature: {'label', 'value', 'unit', 'components'?}} - shown only, never judged."""
    P3 = np.asarray(P3, float); q, A = eye_affine(P3); E = lambda a, b: float(n_(q[a] - q[b])); fw = E(234, 454); out = {}; eyes = [(33, 133, 159, 145), (263, 362, 386, 374)]
    X, s_ = PN._norm(P3); f = (X @ PN.kabsch(X[PN.RIGID], template[PN.RIGID]))[:, :2] * s_; Ef = lambda a, b: float(n_(f[a] - f[b])); mo = Ef(61, 291)  # pose-normalised landmarks for the pitch-sensitive ones
    out['F02'] = {'label': '눈 세로(위·아래 눈꺼풀 점) ÷ 눈 가로', 'value': float(np.mean([E(t, b) / E(o, n) for o, n, t, b in eyes])), 'unit': '비율'}
    out['F03'] = {'label': '눈꼬리 각도(자세 보정 후, +면 눈꼬리가 올라감)', 'value': float(np.mean([math.degrees(math.atan2(f[n, 1] - f[o, 1], abs(f[o, 0] - f[n, 0]))) for o, n, t, b in eyes])), 'unit': '도'}
    out['F13'] = {'label': '윗입술 + 아랫입술 두께 ÷ 입 폭(자세 보정 후)', 'value': (Ef(0, 13) + Ef(14, 17)) / mo, 'unit': '비율'}
    out['F14'] = {'label': '입 가운데 높이 − 입꼬리 높이, ÷ 입 폭(자세 보정 후, +면 입꼬리가 올라감)', 'value': float(((f[13, 1] + f[14, 1]) / 2 - (f[61, 1] + f[291, 1]) / 2) / mo), 'unit': '비율'}
    out['F15'] = {'label': '코밑 ~ 윗입술 ÷ 코밑 ~ 턱 끝(자세 보정 후)', 'value': Ef(2, 0) / Ef(2, 152), 'unit': '비율'}
    y, x = np.where(mask == 10)
    if len(x) > 25: nz = np.c_[x, y, np.ones(len(x))] @ A.T; out['F08'] = {'label': '코 영역(얼굴 분할) 가로 폭 ÷ 얼굴 폭', 'value': float((nz[:, 0].max() - nz[:, 0].min()) / fw), 'unit': '비율'}
    sides = measure_brows(mask, P3, pose=None, blend=None)['sides']; yy, xx = np.where(np.isin(mask, [2, 3])); bxy = np.c_[xx, yy, np.ones(len(xx))] @ A.T; f04, f05, bend, f06 = [], [], [], []
    for side, s in sides.items():
        path = np.array(s['path_aligned']); ids = (33, 133, 159) if side == 'right' else (263, 362, 386); ew = E(ids[0], ids[1]); outer, inn = q[ids[0]], q[ids[1]]; lateral = np.sign(outer[0] - inn[0])
        f04.append(float((path[-1][0] - outer[0]) * lateral / ew)); arc, ang = _bend(path)
        if arc is not None: f05.append(arc); bend.append(ang)
        ex = q[ids[2], 0]; roi = (bxy[:, 0] >= path[:, 0].min() - 2) & (bxy[:, 0] <= path[:, 0].max() + 2) & (np.abs(bxy[:, 1] - np.median(path[:, 1])) < ew * .4) & (np.abs(bxy[:, 0] - ex) < 2)
        if roi.any(): f06.append((float(q[ids[2], 1]) - float(np.max(bxy[roi, 1]))) / ew)
    if f04: out['F04'] = {'label': '눈썹 꼬리가 눈꼬리보다 바깥으로 나간 길이 ÷ 눈 가로(−면 눈꼬리에 못 미침)', 'value': float(np.mean(f04)), 'unit': '비율'}
    if f05: out['F05'] = {'label': '눈썹 휨: 양 끝을 이은 선에서 가장 솟은 높이 ÷ 눈썹 길이', 'value': float(np.mean(f05)), 'unit': '비율', 'components': [{'label': '가장 솟은 곳에서 꺾인 각도', 'value': float(np.mean(bend)), 'unit': '도'}]}
    if f06: out['F06'] = {'label': '눈썹 아래 가장자리 ~ 눈 위쪽 끝 ÷ 눈 가로', 'value': float(np.mean(f06)), 'unit': '비율'}
    return out
