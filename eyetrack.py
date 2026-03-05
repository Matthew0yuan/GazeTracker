
import time
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import median_absolute_error

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Config
CAM_INDEX = 1
IPD_MM = 72.0
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "face_landmarker.task"

GRID_COLS, GRID_ROWS = 7, 6
MARGIN_FRAC = 0.06
SETTLE_TIME, COLLECT_TIME, BREAK_TIME = 0.45, 1.30, 0.10
TARGET_FPS_SAMPLES = 60
MIN_SAMPLES_PER_DOT = 34

Q9_SETTLE, Q9_COLLECT, Q9_BREAK = 0.25, 0.70, 0.08
Q9_MIN_SAMPLES_PER_DOT = 22

EAR_BLINK_THRESHOLD = 0.18
MAX_HEAD_TURN_SCORE = 0.60
EMA_ALPHA = 0.35
MAX_JUMP = 0.10

N_TREES, TREE_MAX_DEPTH, TREE_MIN_SAMPLES_LEAF = 450, 18, 2
RANDOM_SEED = 42
KEEP_FRAC_PER_DOT = 0.70
AFFINE_RIDGE = 1e-3
USE_DSHOW = True

CAPTURE_W, CAPTURE_H = 1280, 720
INFER_W, INFER_H = 640, 360
INFER_FPS = 25.0
DOWNSCALE_DISPLAY = False
DISPLAY_W, DISPLAY_H = 960, 540

PUPIL_THRESH = 50
PUPIL_ROI_PAD = 0.3
PUPIL_MIN_AREA = 15

cv2.setNumThreads(1)

# Landmark indices
L_OUTER, L_INNER, L_UPPER, L_LOWER = 33, 133, 159, 145
R_OUTER, R_INNER, R_UPPER, R_LOWER = 263, 362, 386, 374
L_IRIS = [468, 469, 470, 471, 472]
R_IRIS = [473, 474, 475, 476, 477]
NOSE_TIP, CHIN, FOREHEAD = 1, 152, 10

_HALF_IPD = IPD_MM / 2.0
FACE_MODEL_3D = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 65.0, -15.0],
    [0.0, -70.0, -30.0],
    [-_HALF_IPD - 6, -35.0, -28.0],
    [_HALF_IPD + 6, -35.0, -28.0],
    [-_HALF_IPD + 6, -35.0, -22.0],
    [_HALF_IPD - 6, -35.0, -22.0],
], dtype=np.float64)

POSE_EMA_ALPHA = 0.4
FACE_MODEL_LM_INDICES = [NOSE_TIP, CHIN, FOREHEAD, L_OUTER, R_OUTER, L_INNER, R_INNER]


def get_screen_size_px() -> Tuple[int, int]:
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    user32 = ctypes.windll.user32
    return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def dist2(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.dot(d, d) + 1e-12))

def eye_aspect_ratio(upper, lower, inner, outer) -> float:
    return float(dist2(upper, lower) / (dist2(inner, outer) + 1e-6))

def detect_pupil_ir(gray, inner_px, outer_px, upper_px, lower_px):
    img_h, img_w = gray.shape[:2]
    x_min, x_max = min(inner_px[0], outer_px[0]), max(inner_px[0], outer_px[0])
    y_min, y_max = min(upper_px[1], lower_px[1]), max(upper_px[1], lower_px[1])
    pad_x, pad_y = (x_max - x_min) * PUPIL_ROI_PAD, (y_max - y_min) * PUPIL_ROI_PAD
    x1, y1 = max(0, int(x_min - pad_x)), max(0, int(y_min - pad_y))
    x2, y2 = min(img_w, int(x_max + pad_x)), min(img_h, int(y_max + pad_y))
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    roi = gray[y1:y2, x1:x2]
    _, mask = cv2.threshold(roi, PUPIL_THRESH, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < PUPIL_MIN_AREA:
        return None
    (cx, cy), radius = cv2.minEnclosingCircle(best)
    return np.array([cx + x1, cy + y1], dtype=np.float32), float(radius)

def keep_central_samples(X: np.ndarray, keep_frac: float) -> np.ndarray:
    if len(X) < 10:
        return X
    med = np.median(X, axis=0)
    d = np.linalg.norm(X - med, axis=1)
    return X[np.argsort(d)[:max(6, int(len(X) * keep_frac))]]

def rotmat_to_euler(R):
    sy = math.sqrt(float(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-6:
        return (math.atan2(float(-R[2, 0]), sy),
                math.atan2(float(R[2, 1]), float(R[2, 2])),
                math.atan2(float(R[1, 0]), float(R[0, 0])))
    return (math.atan2(float(-R[2, 0]), sy),
            math.atan2(float(-R[1, 2]), float(R[1, 1])), 0.0)

def estimate_head_pose(pts_2d, img_w, img_h):
    focal = float(img_w)
    cam_matrix = np.array([[focal, 0, img_w / 2.0], [0, focal, img_h / 2.0], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros(4, dtype=np.float64)
    pts = np.ascontiguousarray(pts_2d.reshape(-1, 1, 2), dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(FACE_MODEL_3D, pts, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), None, None
    R, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = rotmat_to_euler(R)
    return (yaw, pitch, roll, float(tvec[0, 0]), float(tvec[1, 0]), float(tvec[2, 0])), rvec, tvec


@dataclass
class FeaturePacket:
    feat: np.ndarray
    ts: float
    quality_ok: bool
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    pose_img_size: Optional[Tuple[int, int]] = None
    l_eye_center_n: Optional[np.ndarray] = None
    r_eye_center_n: Optional[np.ndarray] = None
    l_iris_n_pos: Optional[np.ndarray] = None
    r_iris_n_pos: Optional[np.ndarray] = None
    l_pupil_n: Optional[np.ndarray] = None
    r_pupil_n: Optional[np.ndarray] = None
    l_pupil_r: float = 0.0
    r_pupil_r: float = 0.0


class TaskFaceLandmarkerExtractor:
    def __init__(self, model_path: str):
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE, num_faces=1,
            output_facial_transformation_matrixes=False, output_face_blendshapes=False)
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        self._prev_pose = None

    @staticmethod
    def _lm_xy(lm):
        return np.array([lm.x, lm.y], dtype=np.float32)

    @staticmethod
    def _lm_px(lm, w, h):
        return np.array([lm.x * w, lm.y * h], dtype=np.float64)

    def _safe_mean(self, lms, indices, fallback):
        pts = [self._lm_xy(lms[i]) for i in indices if 0 <= i < len(lms)]
        if len(pts) >= 3:
            return np.mean(np.stack(pts), axis=0)
        fb = [self._lm_xy(lms[i]) for i in fallback if 0 <= i < len(lms)]
        return np.mean(np.stack(fb), axis=0) if fb else np.array([0.5, 0.5], dtype=np.float32)

    def extract(self, frame_bgr_small: np.ndarray) -> Optional[FeaturePacket]:
        ts = time.time()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame_bgr_small, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        lms = result.face_landmarks[0]
        req = [L_OUTER, L_INNER, L_UPPER, L_LOWER, R_OUTER, R_INNER, R_UPPER, R_LOWER, NOSE_TIP, CHIN, FOREHEAD]
        if any(i >= len(lms) for i in req):
            return None

        xy = self._lm_xy
        l_outer, l_inner, l_upper, l_lower = xy(lms[L_OUTER]), xy(lms[L_INNER]), xy(lms[L_UPPER]), xy(lms[L_LOWER])
        r_outer, r_inner, r_upper, r_lower = xy(lms[R_OUTER]), xy(lms[R_INNER]), xy(lms[R_UPPER]), xy(lms[R_LOWER])
        l_iris = self._safe_mean(lms, L_IRIS, [L_UPPER, L_LOWER, L_INNER, L_OUTER])
        r_iris = self._safe_mean(lms, R_IRIS, [R_UPPER, R_LOWER, R_INNER, R_OUTER])

        # IR pupil detection
        gray = cv2.cvtColor(frame_bgr_small, cv2.COLOR_BGR2GRAY)
        ih, iw = gray.shape[:2]
        to_px = lambda n: np.array([n[0] * iw, n[1] * ih], dtype=np.float32)

        l_pupil_n, r_pupil_n, l_pupil_r, r_pupil_r = None, None, 0.0, 0.0
        for result_p, side in [
            (detect_pupil_ir(gray, to_px(l_inner), to_px(l_outer), to_px(l_upper), to_px(l_lower)), 'L'),
            (detect_pupil_ir(gray, to_px(r_inner), to_px(r_outer), to_px(r_upper), to_px(r_lower)), 'R'),
        ]:
            if result_p is not None:
                c, r = result_p
                n = np.array([c[0] / iw, c[1] / ih], dtype=np.float32)
                if side == 'L':
                    l_pupil_n, l_pupil_r = n, r
                else:
                    r_pupil_n, r_pupil_r = n, r

        l_eye_center = 0.5 * (l_inner + l_outer)
        r_eye_center = 0.5 * (r_inner + r_outer)
        eyes_mid = 0.5 * (l_eye_center + r_eye_center)

        l_ear = eye_aspect_ratio(l_upper, l_lower, l_inner, l_outer)
        r_ear = eye_aspect_ratio(r_upper, r_lower, r_inner, r_outer)
        blink_ok = 0.5 * (l_ear + r_ear) > EAR_BLINK_THRESHOLD

        l_w, r_w = dist2(l_inner, l_outer), dist2(r_inner, r_outer)
        head_turn_score = 1.0 - min(l_w, r_w) / (max(l_w, r_w) + 1e-6)
        head_ok = head_turn_score < MAX_HEAD_TURN_SCORE

        eye_vec = r_eye_center - l_eye_center
        roll_img = math.atan2(float(eye_vec[1]), float(eye_vec[0]))
        iod = dist2(l_eye_center, r_eye_center)
        nose = xy(lms[NOSE_TIP])
        nose_rel = nose - eyes_mid
        chin, forehead = xy(lms[CHIN]), xy(lms[FOREHEAD])
        face_h = dist2(chin, forehead)

        def eye_box_norm(pt, inner, outer, upper, lower):
            xmin, xmax = min(inner[0], outer[0]), max(inner[0], outer[0])
            ymin, ymax = min(upper[1], lower[1]), max(upper[1], lower[1])
            w, h = (xmax - xmin) + 1e-6, (ymax - ymin) + 1e-6
            return np.array([(pt[0] - xmin) / w, (pt[1] - ymin) / h, w, h], dtype=np.float32)

        l_box = eye_box_norm(l_iris, l_inner, l_outer, l_upper, l_lower)
        r_box = eye_box_norm(r_iris, r_inner, r_outer, r_upper, r_lower)
        l_iris_n, r_iris_n = l_box[:2], r_box[:2]

        l_open = dist2(l_upper, l_lower) / (l_w + 1e-6)
        r_open = dist2(r_upper, r_lower) / (r_w + 1e-6)
        face_scale = float(iod / (face_h + 1e-6))
        eye_scale_asym = float((l_box[2] - r_box[2]) / (l_box[2] + r_box[2] + 1e-6))

        l_iris_off = l_iris - l_eye_center
        r_iris_off = r_iris - r_eye_center

        # Head pose via solvePnP
        img_h, img_w = frame_bgr_small.shape[:2]
        pts_2d = np.array([self._lm_px(lms[i], img_w, img_h) for i in FACE_MODEL_LM_INDICES], dtype=np.float64)
        pose, rvec, tvec = estimate_head_pose(pts_2d, img_w, img_h)
        raw = np.array(pose, dtype=np.float64)
        if self._prev_pose is None:
            self._prev_pose = raw.copy()
        else:
            self._prev_pose = (1 - POSE_EMA_ALPHA) * self._prev_pose + POSE_EMA_ALPHA * raw
        yaw, pitch, roll_pose, tx, ty, tz = self._prev_pose

        feat = np.array([
            float(l_iris_n[0]), float(l_iris_n[1]), float(r_iris_n[0]), float(r_iris_n[1]),
            float(l_iris_off[0] / (iod + 1e-6)), float(l_iris_off[1] / (iod + 1e-6)),
            float(r_iris_off[0] / (iod + 1e-6)), float(r_iris_off[1] / (iod + 1e-6)),
            float(l_open), float(r_open),
            float(l_box[2]), float(l_box[3]), float(r_box[2]), float(r_box[3]),
            float(roll_img), float(iod),
            float(nose_rel[0]), float(nose_rel[1]), float(face_h), float(head_turn_score),
            float(face_scale), float(eye_scale_asym),
            float(yaw), float(pitch), float(roll_pose),
            float(tx / IPD_MM), float(ty / IPD_MM), float(tz / IPD_MM),
        ], dtype=np.float32)

        return FeaturePacket(
            feat=feat, ts=ts, quality_ok=bool(blink_ok and head_ok),
            rvec=rvec, tvec=tvec, pose_img_size=(img_w, img_h),
            l_eye_center_n=l_eye_center, r_eye_center_n=r_eye_center,
            l_iris_n_pos=l_iris, r_iris_n_pos=r_iris,
            l_pupil_n=l_pupil_n, r_pupil_n=r_pupil_n,
            l_pupil_r=l_pupil_r, r_pupil_r=r_pupil_r)


class CameraWorker:
    def __init__(self, cam_index=1, model_path=str(MODEL_PATH)):
        backend = cv2.CAP_DSHOW if USE_DSHOW else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(cam_index, backend)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera index {cam_index}.")
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
        self.extractor = TaskFaceLandmarkerExtractor(model_path)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_packet: Optional[FeaturePacket] = None
        self.latest_seq = 0
        self.running = False
        self.thread = None
        self.last_error, self.last_error_ts = "", 0.0
        self._infer_period = 1.0 / max(1.0, float(INFER_FPS))
        self._last_infer_t = 0.0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            now = time.time()
            packet, produced = None, False
            if (now - self._last_infer_t) >= self._infer_period:
                self._last_infer_t = now
                try:
                    small = cv2.resize(frame, (INFER_W, INFER_H), interpolation=cv2.INTER_AREA)
                    packet = self.extractor.extract(small)
                    produced = packet is not None
                except Exception as e:
                    self.last_error, self.last_error_ts = f"{type(e).__name__}: {e}", time.time()
            with self.lock:
                self.latest_frame = frame
                if produced:
                    self.latest_packet = packet
                    self.latest_seq += 1
            time.sleep(0.001)

    def get_latest(self):
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            return frame, self.latest_packet, self.latest_seq, self.last_error, \
                   (time.time() - self.last_error_ts if self.last_error_ts > 0 else 999)


class GazeModel:
    def __init__(self):
        self.model = ExtraTreesRegressor(
            n_estimators=N_TREES, max_depth=TREE_MAX_DEPTH,
            min_samples_leaf=TREE_MIN_SAMPLES_LEAF, random_state=RANDOM_SEED,
            n_jobs=-1, bootstrap=False)

    def fit(self, X, Y) -> Dict[str, float]:
        X_tr, X_va, Y_tr, Y_va = train_test_split(X, Y, test_size=0.20, random_state=RANDOM_SEED)
        self.model.fit(X_tr, Y_tr)
        pred_va = self.model.predict(X_va)
        err = np.linalg.norm(pred_va - Y_va, axis=1)
        metrics = {
            "val_mean_uv": float(err.mean()), "val_median_uv": float(np.median(err)),
            "val_median_abs_u": float(median_absolute_error(Y_va[:, 0], pred_va[:, 0])),
            "val_median_abs_v": float(median_absolute_error(Y_va[:, 1], pred_va[:, 1])),
        }
        self.model.fit(X, Y)
        return metrics

    def predict(self, feat_1xD):
        return self.model.predict(feat_1xD)


class AffineUVCorrector:
    def __init__(self):
        self.M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.enabled = False

    def clear(self):
        self.M[:] = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.enabled = False

    def apply(self, u, v):
        if not self.enabled:
            return u, v
        y = self.M @ np.array([u, v, 1.0], dtype=np.float32)
        return float(np.clip(y[0], 0, 1)), float(np.clip(y[1], 0, 1))

    def fit(self, pred_uv, tgt_uv, ridge=AFFINE_RIDGE) -> bool:
        if len(pred_uv) < 40:
            return False
        P = np.asarray(pred_uv, dtype=np.float32)
        A = np.concatenate([P, np.ones((len(P), 1), dtype=np.float32)], axis=1)
        AtA = A.T @ A + ridge * np.eye(3, dtype=np.float32)
        self.M = np.linalg.solve(AtA, A.T @ np.asarray(tgt_uv, dtype=np.float32)).T
        self.enabled = True
        return True


def draw_screen(sw, sh, x, y, msg_top, msg_bottom):
    img = np.zeros((sh, sw, 3), dtype=np.uint8)
    cv2.circle(img, (x, y), 14, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.putText(img, msg_top, (40, int(sh * 0.08)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, msg_bottom, (40, int(sh * 0.93)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "ESC=abort", (40, int(sh * 0.98)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
    return img

def build_grid_points(sw, sh, cols, rows, margin_frac):
    xs = np.linspace(sw * margin_frac, sw * (1 - margin_frac), cols)
    ys = np.linspace(sh * margin_frac, sh * (1 - margin_frac), rows)
    pts = []
    for j, y in enumerate(ys):
        for x in (xs if j % 2 == 0 else xs[::-1]):
            pts.append((int(x), int(y), float(x / sw), float(y / sh)))
    return pts

def _cal_wait(win, sw, sh, x, y, duration, msg_top, msg_bottom):
    """Wait phase with ESC check. Returns True if ESC pressed."""
    t0 = time.time()
    while time.time() - t0 < duration:
        cv2.imshow(win, draw_screen(sw, sh, x, y, msg_top, msg_bottom))
        if (cv2.waitKey(1) & 0xFF) == 27:
            return True
        time.sleep(0.005)
    return False


def run_fullscreen_calibration(cam, screen_w, screen_h):
    pts = build_grid_points(screen_w, screen_h, GRID_COLS, GRID_ROWS, MARGIN_FRAC)
    pts.append((screen_w // 2, screen_h // 2, 0.5, 0.5))
    total = len(pts)
    win = "Calibration (fullscreen)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    X_all, Y_all = [], []

    for idx, (x, y, u, v) in enumerate(pts, 1):
        if _cal_wait(win, screen_w, screen_h, x, y, SETTLE_TIME,
                     "Look at the dot. Keep head still.", f"Progress {idx}/{total}"):
            cv2.destroyWindow(win)
            raise SystemExit("Calibration aborted.")

        X_dot = []
        t1 = time.time()
        while time.time() - t1 < COLLECT_TIME:
            cv2.imshow(win, draw_screen(screen_w, screen_h, x, y,
                       "Collecting...", f"Progress {idx}/{total}   good={len(X_dot)}"))
            if (cv2.waitKey(1) & 0xFF) == 27:
                cv2.destroyWindow(win)
                raise SystemExit("Calibration aborted.")
            _, packet, _, _, _ = cam.get_latest()
            if packet is not None and packet.quality_ok:
                X_dot.append(packet.feat.copy())
            time.sleep(1.0 / max(10, TARGET_FPS_SAMPLES))

        X_dot_np = np.array(X_dot, dtype=np.float32)
        if len(X_dot_np) >= 10:
            X_dot_np = keep_central_samples(X_dot_np, KEEP_FRAC_PER_DOT)
        if len(X_dot_np) < MIN_SAMPLES_PER_DOT:
            cv2.imshow(win, draw_screen(screen_w, screen_h, x, y,
                       f"Low samples ({len(X_dot_np)}).", f"Progress {idx}/{total}"))
            cv2.waitKey(1)
            time.sleep(0.25)

        for row in X_dot_np:
            X_all.append(row)
            Y_all.append([u, v])

        if _cal_wait(win, screen_w, screen_h, x, y, BREAK_TIME, "Next...", f"Progress {idx}/{total}"):
            cv2.destroyWindow(win)
            raise SystemExit("Calibration aborted.")

    cv2.destroyWindow(win)
    if len(X_all) < 350:
        raise SystemExit(f"Too few calibration samples ({len(X_all)}).")
    return np.array(X_all, dtype=np.float32), np.array(Y_all, dtype=np.float32)


def run_quick_9point_calibration(cam, model, screen_w, screen_h):
    pts = build_grid_points(screen_w, screen_h, 3, 3, margin_frac=0.12)
    total = len(pts)
    win = "Quick 9-point calibration (SPACE)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    empty = np.zeros((0, 2), dtype=np.float32)
    pred_all, tgt_all = [], []

    for idx, (x, y, tu, tv) in enumerate(pts, 1):
        if _cal_wait(win, screen_w, screen_h, x, y, Q9_SETTLE,
                     "QuickCal: look at dot (keep still)", f"{idx}/{total}   (ESC cancel)"):
            cv2.destroyWindow(win)
            return empty, empty

        preds = []
        t1 = time.time()
        while time.time() - t1 < Q9_COLLECT:
            cv2.imshow(win, draw_screen(screen_w, screen_h, x, y,
                       "QuickCal: collecting...", f"{idx}/{total}   good={len(preds)}   (ESC cancel)"))
            if (cv2.waitKey(1) & 0xFF) == 27:
                cv2.destroyWindow(win)
                return empty, empty
            _, packet, _, _, _ = cam.get_latest()
            if packet is not None and packet.quality_ok:
                uv = model.predict(packet.feat.reshape(1, -1))[0]
                preds.append([float(uv[0]), float(uv[1])])
            time.sleep(1.0 / max(10, TARGET_FPS_SAMPLES))

        if len(preds) >= Q9_MIN_SAMPLES_PER_DOT:
            P = keep_central_samples(np.asarray(preds, dtype=np.float32), 0.70)
            for uv in P:
                pred_all.append([float(uv[0]), float(uv[1])])
                tgt_all.append([float(tu), float(tv)])

        if _cal_wait(win, screen_w, screen_h, x, y, Q9_BREAK, "QuickCal: next...", f"{idx}/{total}"):
            cv2.destroyWindow(win)
            return empty, empty

    cv2.destroyWindow(win)
    if len(pred_all) < 60:
        return empty, empty
    return np.array(pred_all, dtype=np.float32), np.array(tgt_all, dtype=np.float32)


def _draw_hud(frame, packet, out_u, out_v, screen_w, screen_h, last_good_ts, corrector):
    h, w = frame.shape[:2]
    x_cam, y_cam = int(out_u * w), int(out_v * h)
    cv2.circle(frame, (x_cam, y_cam), 10, (0, 255, 0), 2)

    pose_text = ""
    if packet is not None and packet.rvec is not None and packet.pose_img_size is not None:
        pw, ph = packet.pose_img_size
        sx, sy = w / pw, h / ph
        cam_mat = np.array([[float(pw), 0, pw / 2.0], [0, float(pw), ph / 2.0], [0, 0, 1]], dtype=np.float64)
        axis_pts = np.float64([[0, 0, 0], [40, 0, 0], [0, 40, 0], [0, 0, 40]])
        proj, _ = cv2.projectPoints(axis_pts, packet.rvec, packet.tvec, cam_mat, np.zeros(4, dtype=np.float64))
        proj = proj.reshape(-1, 2)
        origin = (int(proj[0][0] * sx), int(proj[0][1] * sy))
        for i, color in [(1, (0, 0, 255)), (2, (0, 255, 0)), (3, (255, 0, 0))]:
            cv2.line(frame, origin, (int(proj[i][0] * sx), int(proj[i][1] * sy)), color, 2, cv2.LINE_AA)
        f = packet.feat
        pose_text = f"yaw={math.degrees(float(f[-6])):+.1f} pitch={math.degrees(float(f[-5])):+.1f} roll={math.degrees(float(f[-4])):+.1f}"

    if packet is not None and packet.l_eye_center_n is not None:
        for ec, ir, color in [(packet.l_eye_center_n, packet.l_iris_n_pos, (255, 255, 0)),
                               (packet.r_eye_center_n, packet.r_iris_n_pos, (0, 255, 255))]:
            cx_px, cy_px = int(ec[0] * w), int(ec[1] * h)
            dx, dy = int(ir[0] * w) - cx_px, int(ir[1] * h) - cy_px
            cv2.circle(frame, (cx_px, cy_px), 3, color, -1, cv2.LINE_AA)
            cv2.arrowedLine(frame, (cx_px, cy_px), (cx_px + dx * 4, cy_px + dy * 4), color, 2, cv2.LINE_AA, tipLength=0.3)

    if packet is not None:
        pw, ph = packet.pose_img_size or (INFER_W, INFER_H)
        for pn, pr, color in [(packet.l_pupil_n, packet.l_pupil_r, (0, 255, 0)),
                               (packet.r_pupil_n, packet.r_pupil_r, (0, 255, 0))]:
            if pn is not None:
                px, py = int(pn[0] * w), int(pn[1] * h)
                cv2.circle(frame, (px, py), max(2, int(pr * w / pw)), color, 2, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 2, color, -1, cv2.LINE_AA)

    age = time.time() - last_good_ts
    status = "OK" if age < 0.45 else "NO FACE/BLINK/HEADTURN"
    corr_s = "corr=ON" if corrector.enabled else "corr=OFF"
    x_px, y_px = int(out_u * screen_w), int(out_v * screen_h)
    lines = [
        (f"gaze(u,v)=({out_u:.3f},{out_v:.3f}) screen=({x_px},{y_px}) {status} {corr_s}", 35, 0.66, (255, 255, 255)),
        (f"INFER_FPS={INFER_FPS:.1f}  InferSize={INFER_W}x{INFER_H}  Capture={CAPTURE_W}x{CAPTURE_H}", 65, 0.56, (255, 255, 255)),
        ("SPACE: 9pt quick-cal  |  R: clear  |  ESC: quit", 95, 0.56, (255, 255, 255)),
    ]
    if pose_text:
        lines.append((f"3D pose: {pose_text}", 125, 0.56, (0, 200, 255)))
    for text, y_pos, scale, color in lines:
        cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
    return frame



def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    screen_w, screen_h = get_screen_size_px()
    print(f"[INFO] Screen: {screen_w}x{screen_h} | Capture: {CAPTURE_W}x{CAPTURE_H} | Infer: {INFER_W}x{INFER_H} @ {INFER_FPS:.1f} FPS")
    print("[INFO] SPACE = quick 9-point recalibration.  R = clear.  ESC = quit.")

    cam = CameraWorker(CAM_INDEX, str(MODEL_PATH))
    cam.start()
    time.sleep(0.6)

    print("[INFO] Starting full calibration. Look at dots. ESC to abort.")
    X, Y = run_fullscreen_calibration(cam, screen_w, screen_h)
    print(f"[INFO] Collected: {len(X)} samples, feature_dim={X.shape[1]}")

    model = GazeModel()
    stats = model.fit(X, Y)
    diag = math.sqrt(screen_w ** 2 + screen_h ** 2)
    print(f"[INFO] Val error: mean~{stats['val_mean_uv'] * diag:.1f}px, median~{stats['val_median_uv'] * diag:.1f}px")

    corrector = AffineUVCorrector()
    cv2.namedWindow("Gaze Tracker (ESC to quit)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gaze Tracker (ESC to quit)", 960, 540)
    cv2.namedWindow("Screen Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Screen Map", 480, 300)

    ema_u, ema_v = 0.5, 0.5
    last_good_ts = 0.0
    last_seq = -1
    last_pred_u, last_pred_v = 0.5, 0.5

    while True:
        frame, packet, seq, err, err_age = cam.get_latest()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_show = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA) if DOWNSCALE_DISPLAY else frame

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in (ord('r'), ord('R')):
            corrector.clear()
            print("[INFO] Quick-cal correction cleared.")
        if key == ord(' '):
            print("[INFO] Quick 9-point recalibration starting...")
            pred_uv, tgt_uv = run_quick_9point_calibration(cam, model, screen_w, screen_h)
            if len(pred_uv) > 0:
                print(f"[INFO] QuickCal applied={corrector.fit(pred_uv, tgt_uv)}  samples={len(pred_uv)}")
            else:
                print("[INFO] QuickCal canceled / insufficient samples.")

        if packet is not None and packet.quality_ok and seq != last_seq:
            last_seq = seq
            uv = model.predict(packet.feat.reshape(1, -1))[0]
            last_pred_u, last_pred_v = clamp01(float(uv[0])), clamp01(float(uv[1]))
            last_good_ts = packet.ts

        if abs(last_pred_u - ema_u) < MAX_JUMP and abs(last_pred_v - ema_v) < MAX_JUMP:
            ema_u = (1 - EMA_ALPHA) * ema_u + EMA_ALPHA * last_pred_u
            ema_v = (1 - EMA_ALPHA) * ema_v + EMA_ALPHA * last_pred_v
        else:
            ema_u, ema_v = last_pred_u, last_pred_v

        out_u, out_v = corrector.apply(ema_u, ema_v)
        frame_show = _draw_hud(frame_show, packet, out_u, out_v, screen_w, screen_h, last_good_ts, corrector)

        if err and err_age < 1.0:
            cv2.putText(frame_show, f"[CameraThreadError] {err}", (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        sm_w, sm_h = 480, 300
        sm = np.zeros((sm_h, sm_w, 3), dtype=np.uint8)
        cv2.rectangle(sm, (0, 0), (sm_w - 1, sm_h - 1), (80, 80, 80), 1)
        cv2.circle(sm, (int(out_u * sm_w), int(out_v * sm_h)), 6, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(sm, "corr=ON" if corrector.enabled else "corr=OFF", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Gaze Tracker (ESC to quit)", frame_show)
        cv2.imshow("Screen Map", sm)

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
