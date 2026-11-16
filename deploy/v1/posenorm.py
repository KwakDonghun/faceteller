"""Pose-normalised ("frontalised") face image for cropping.

The current crops (facework.release_preprocess.frontal_crops) remove in-plane roll and scale only. Head pitch (nod) and
yaw (turn) stay in the image, and they change the apparent face proportions: on the 30 light captures of the same person,
10 deg of pitch moves the mouth-corner value by 1.4 between-person SDs and the eye-tail angle by 1.0 (pose_effect.json).

Here the MediaPipe 3D landmarks (x, y in pixels, z in the same pixel scale) are rotated so that their rigid points best
match one template orientation (Kabsch fit to the train270 mean shape; template roll and yaw set to zero, the cohort's mean
pitch kept), and projected orthographically. (MediaPipe's own Euler angles are not used: their axes differ from image
coordinates and undoing them added roll.) The photo is warped to match: inside the face mesh by a piecewise-affine map over the
Delaunay triangles of the target landmarks, outside it (hair, ears, background) by one least-squares affine map. The
warped image and its landmarks then go through the unchanged frontal_crops, so the crop windows and scale stay the same.
"""
import cv2
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation

MESH = 468


# Points used to estimate the head rotation: rigid upper/mid face (eye corners, nasal bridge and tip, forehead, the
# depth-stable cheek beside the nose). Mouth, jaw and the face contour are left out: they move with expression or slide
# along the silhouette when the head turns.
RIGID = [33, 133, 263, 362, 168, 6, 197, 195, 5, 4, 1, 2, 98, 327, 10, 151, 9, 8, 109, 338, 67, 297, 36, 266, 142, 371, 205, 425, 50, 280]
# The template (posenorm_template.npy: mean rigid shape of the 270 training people, no individual) comes from S3; its path is
# given by settings.Settings.models_dir. Copied from experiments/feature17-20261109/posenorm.py (2026-11-16), code unchanged.


def _norm(P):
    X = P - P[RIGID].mean(0); s = np.linalg.norm(X[RIGID], axis=1).mean(); return X / s, s


def build_template(shapes):
    """Mean rigid shape over people (each centred and scaled), then roll and yaw removed (eye line level, eye corners at
    equal depth); the cohort's mean pitch is kept as the reference pose."""
    M = np.mean([_norm(np.asarray(P, float))[0] for P in shapes], axis=0)
    v = M[263] - M[33]
    rz = np.arctan2(v[1], v[0]); Rz = Rotation.from_euler('z', -rz).as_matrix(); M = M @ Rz.T
    v = M[263] - M[33]
    ry = np.arctan2(v[2], v[0]); Ry = Rotation.from_euler('y', ry).as_matrix(); M = M @ Ry.T
    return M


def kabsch(A, B):
    """Rotation R minimising |A R - B| (rows are points, both centred)."""
    U, _, Vt = np.linalg.svd(A.T @ B); d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1, 1, d]) @ Vt


def target_landmarks(P, template):
    """P: (478, 3) pixel landmarks. Rotate the person's own 3D landmarks so their rigid points best match the template
    orientation, project orthographically, and keep the photo's eye-centre position and eye-centre distance."""
    X, s = _norm(P)
    R = kabsch(X[RIGID], template[RIGID])
    Q = (X @ R)[:, :2] * s
    def eyes(A): return (A[33] + A[133]) / 2, (A[263] + A[362]) / 2
    (a1, b1), (a2, b2) = eyes(P[:, :2]), eyes(Q)
    k = np.linalg.norm(b1 - a1) / np.linalg.norm(b2 - a2)
    return (Q - (a2 + b2) / 2) * k + (a1 + b1) / 2


def warp(image, P, template, margin=40):
    """Returns (warped RGB array, target landmarks (478, 2))."""
    rgb = np.asarray(image.convert('RGB')) if not isinstance(image, np.ndarray) else image
    h, w = rgb.shape[:2]
    src = np.asarray(P, float)[:, :2]; dst = target_landmarks(np.asarray(P, float), template)
    # outside the mesh: one affine dst -> src
    A = np.c_[dst, np.ones(len(dst))]; M, *_ = np.linalg.lstsq(A, src, rcond=None)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    mapx = (M[0, 0] * xs + M[1, 0] * ys + M[2, 0]).astype(np.float32)
    mapy = (M[0, 1] * xs + M[1, 1] * ys + M[2, 1]).astype(np.float32)
    # inside the mesh: piecewise affine over Delaunay triangles of the target mesh points
    tri = Delaunay(dst[:MESH])
    x0, y0 = np.maximum(0, dst[:MESH].min(0) - margin).astype(int); x1, y1 = np.minimum([w - 1, h - 1], dst[:MESH].max(0) + margin).astype(int)
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    pts = np.c_[gx.ravel(), gy.ravel()].astype(float)
    simplex = tri.find_simplex(pts); inside = simplex >= 0
    T = tri.transform[simplex[inside]]
    b = np.einsum('nij,nj->ni', T[:, :2], pts[inside] - T[:, 2]); bary = np.c_[b, 1 - b.sum(1)]
    verts = tri.simplices[simplex[inside]]
    s = np.einsum('ni,nij->nj', bary, src[:MESH][verts])
    px, py = pts[inside, 0].astype(int), pts[inside, 1].astype(int)
    mapx[py, px] = s[:, 0]; mapy[py, px] = s[:, 1]
    out = cv2.remap(rgb, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
    return out, np.c_[dst, np.zeros(len(dst))]
