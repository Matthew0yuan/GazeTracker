
import time
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import median_absolute_error

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# -----------------------------
# Config
# -----------------------------
CAM_INDEX = 0

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "face_landmarker.task"

# Main calibration grid (more points => better accuracy)
GRID_COLS = 6
GRID_ROWS = 5
MARGIN_FRAC = 0.06

# Timing (main calibration)
SETTLE_TIME = 0.45
COLLECT_TIME = 1.10
BREAK_TIME = 0.10
TARGET_FPS_SAMPLES = 60
MIN_SAMPLES_PER_DOT = 34

# Quick 9-point calibration (SPACE)
Q9_SETTLE = 0.25
Q9_COLLECT = 0.70
Q9_BREAK = 0.08
Q9_MIN_SAMPLES_PER_DOT = 22

# Gating
EAR_BLINK_THRESHOLD = 0.18
MAX_HEAD_TURN_SCORE = 0.60  # tighter -> less noisy samples, better accuracy

# Runtime smoothing
EMA_ALPHA = 0.20
MAX_JUMP = 0.10  # max allowed jump in u/v per update

# Base model
N_TREES = 450
TREE_MAX_DEPTH = 18
TREE_MIN_SAMPLES_LEAF = 2
RANDOM_SEED = 42

# Outlier keep fraction per dot
KEEP_FRAC_PER_DOT = 0.70

# Quick-cal affine fit ridge
AFFINE_RIDGE = 1e-3

# Windows capture backend
USE_DSHOW = True

# ---- Performance knobs (CPU) ----
# Camera capture resolution (display). Lower -> smoother, but too low may hurt iris precision.
CAPTURE_W = 1280
CAPTURE_H = 720

# Inference resolution (FaceLandmarker). This is the big speed lever.
INFER_W = 512
INFER_H = 288

# Run FaceLandmarker at this FPS (NOT every frame). 25-30 is usually good.
INFER_FPS = 25.0

# If True, also downscale DISPLAY frame to reduce imshow cost (optional)
DOWNSCALE_DISPLAY = False
DISPLAY_W = 960
DISPLAY_H = 540

# Reduce OpenCV thread contention (often helps stutter on some systems)
cv2.setNumThreads(1)


# -----------------------------
# Landmark indices (MediaPipe face mesh convention)
# -----------------------------
L_OUTER = 33
L_INNER = 133
L_UPPER = 159
L_LOWER = 145

R_OUTER = 263
R_INNER = 362
R_UPPER = 386
R_LOWER = 374

L_IRIS = [468, 469, 470, 471, 472]
R_IRIS = [473, 474, 475, 476, 477]

NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10


# -----------------------------
# Utils
# -----------------------------
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


def eye_aspect_ratio(upper: np.ndarray, lower: np.ndarray, inner: np.ndarray, outer: np.ndarray) -> float:
    v = dist2(upper, lower)
    h = dist2(inner, outer)
    return float(v / (h + 1e-6))


def keep_central_samples(X: np.ndarray, keep_frac: float) -> np.ndarray:
    if len(X) < 10:
        return X
    med = np.median(X, axis=0)
    d = np.linalg.norm(X - med, axis=1)
    k = max(6, int(len(X) * keep_frac))
    idx = np.argsort(d)[:k]
    return X[idx]


def rotmat_to_euler_yaw_pitch_roll(R: np.ndarray) -> Tuple[float, float, float]:
    sy = math.sqrt(float(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw = math.atan2(float(-R[2, 0]), float(sy))
        roll = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        pitch = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        yaw = math.atan2(float(-R[2, 0]), float(sy))
        roll = 0.0
    return yaw, pitch, roll


@dataclass
class FeaturePacket:
    feat: np.ndarray
    ts: float
    quality_ok: bool


# -----------------------------
# FaceLandmarker extractor (Tasks API) - IMAGE mode (stateless)
# -----------------------------
class TaskFaceLandmarkerExtractor:
    def __init__(self, model_path: str):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,  # stateless
            num_faces=1,
            output_facial_transformation_matrixes=False,
            output_face_blendshapes=False,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _lm_to_xy(lm) -> np.ndarray:
        return np.array([lm.x, lm.y], dtype=np.float32)

    def _safe_get(self, lms, idx) -> Optional[np.ndarray]:
        if 0 <= idx < len(lms):
            return self._lm_to_xy(lms[idx])
        return None

    def _safe_mean(self, lms, indices, fallback_pts: List[int]) -> np.ndarray:
        pts = []
        for i in indices:
            p = self._safe_get(lms, i)
            if p is not None:
                pts.append(p)
        if len(pts) >= 3:
            return np.mean(np.stack(pts, axis=0), axis=0)

        fb = []
        for i in fallback_pts:
            p = self._safe_get(lms, i)
            if p is not None:
                fb.append(p)
        if len(fb) > 0:
            return np.mean(np.stack(fb, axis=0), axis=0)

        return np.array([0.5, 0.5], dtype=np.float32)

    def extract(self, frame_bgr_small: np.ndarray) -> Optional[FeaturePacket]:
        ts = time.time()

        frame_rgb = cv2.cvtColor(frame_bgr_small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        lms = result.face_landmarks[0]
        req = [L_OUTER, L_INNER, L_UPPER, L_LOWER, R_OUTER, R_INNER, R_UPPER, R_LOWER, NOSE_TIP, CHIN, FOREHEAD]
        if any(i >= len(lms) for i in req):
            return None

        l_outer = self._lm_to_xy(lms[L_OUTER])
        l_inner = self._lm_to_xy(lms[L_INNER])
        l_upper = self._lm_to_xy(lms[L_UPPER])
        l_lower = self._lm_to_xy(lms[L_LOWER])

        r_outer = self._lm_to_xy(lms[R_OUTER])
        r_inner = self._lm_to_xy(lms[R_INNER])
        r_upper = self._lm_to_xy(lms[R_UPPER])
        r_lower = self._lm_to_xy(lms[R_LOWER])

        l_iris = self._safe_mean(lms, L_IRIS, fallback_pts=[L_UPPER, L_LOWER, L_INNER, L_OUTER])
        r_iris = self._safe_mean(lms, R_IRIS, fallback_pts=[R_UPPER, R_LOWER, R_INNER, R_OUTER])

        l_eye_center = 0.5 * (l_inner + l_outer)
        r_eye_center = 0.5 * (r_inner + r_outer)
        eyes_mid = 0.5 * (l_eye_center + r_eye_center)

        l_ear = eye_aspect_ratio(l_upper, l_lower, l_inner, l_outer)
        r_ear = eye_aspect_ratio(r_upper, r_lower, r_inner, r_outer)
        ear = 0.5 * (l_ear + r_ear)
        blink_ok = ear > EAR_BLINK_THRESHOLD

        l_w = dist2(l_inner, l_outer)
        r_w = dist2(r_inner, r_outer)
        width_ratio = min(l_w, r_w) / (max(l_w, r_w) + 1e-6)
        head_turn_score = 1.0 - width_ratio
        head_ok = head_turn_score < MAX_HEAD_TURN_SCORE

        eye_vec = (r_eye_center - l_eye_center)
        roll_img = math.atan2(float(eye_vec[1]), float(eye_vec[0]))

        iod = dist2(l_eye_center, r_eye_center)

        nose = self._lm_to_xy(lms[NOSE_TIP])
        nose_rel = nose - eyes_mid

        chin = self._lm_to_xy(lms[CHIN])
        forehead = self._lm_to_xy(lms[FOREHEAD])
        face_h = dist2(chin, forehead)

        def eye_box_norm(pt: np.ndarray, inner: np.ndarray, outer: np.ndarray, upper: np.ndarray, lower: np.ndarray):
            xmin = min(inner[0], outer[0]); xmax = max(inner[0], outer[0])
            ymin = min(upper[1], lower[1]); ymax = max(upper[1], lower[1])
            w = (xmax - xmin) + 1e-6
            h = (ymax - ymin) + 1e-6
            return np.array([(pt[0] - xmin) / w, (pt[1] - ymin) / h, w, h], dtype=np.float32)

        l_box = eye_box_norm(l_iris, l_inner, l_outer, l_upper, l_lower)
        r_box = eye_box_norm(r_iris, r_inner, r_outer, r_upper, r_lower)
        l_iris_n = l_box[:2]; r_iris_n = r_box[:2]
        l_box_w, l_box_h = float(l_box[2]), float(l_box[3])
        r_box_w, r_box_h = float(r_box[2]), float(r_box[3])

        l_open = dist2(l_upper, l_lower) / (l_w + 1e-6)
        r_open = dist2(r_upper, r_lower) / (r_w + 1e-6)

        yaw = pitch = roll_pose = 0.0
        tx = ty = tz = 0.0
        Rfeat = np.zeros(9, dtype=np.float32)

        face_scale = float(iod / (face_h + 1e-6))
        eye_scale_asym = float((l_box_w - r_box_w) / (l_box_w + r_box_w + 1e-6))

        l_iris_off = (l_iris - l_eye_center)
        r_iris_off = (r_iris - r_eye_center)

        quality_ok = bool(blink_ok and head_ok)

        feat = np.array([
            float(l_iris_n[0]), float(l_iris_n[1]),
            float(r_iris_n[0]), float(r_iris_n[1]),
            float(l_iris_off[0] / (iod + 1e-6)), float(l_iris_off[1] / (iod + 1e-6)),
            float(r_iris_off[0] / (iod + 1e-6)), float(r_iris_off[1] / (iod + 1e-6)),
            float(l_open), float(r_open),
            float(l_box_w), float(l_box_h),
            float(r_box_w), float(r_box_h),
            float(roll_img),
            float(iod),
            float(nose_rel[0]), float(nose_rel[1]),
            float(face_h),
            float(head_turn_score),
            float(face_scale),
            float(eye_scale_asym),
            float(yaw), float(pitch), float(roll_pose),
            float(tx), float(ty), float(tz),
            *Rfeat.tolist()
        ], dtype=np.float32)

        return FeaturePacket(feat=feat, ts=ts, quality_ok=quality_ok)


# -----------------------------
# Camera worker (threaded) with inference decimation
# -----------------------------
class CameraWorker:
    def __init__(self, cam_index=0, model_path=str(MODEL_PATH)):
        backend = cv2.CAP_DSHOW if USE_DSHOW else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(cam_index, backend)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera index {cam_index}. Try changing CAM_INDEX.")

        # reduce latency build-up
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
        self.latest_seq = 0  # increments when new packet produced

        self.running = False
        self.thread = None

        self.last_error = ""
        self.last_error_ts = 0.0

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

            # keep latest display frame always
            packet = None
            produced = False

            # only run FaceLandmarker at fixed rate
            if (now - self._last_infer_t) >= self._infer_period:
                self._last_infer_t = now
                try:
                    small = cv2.resize(frame, (INFER_W, INFER_H), interpolation=cv2.INTER_AREA)
                    packet = self.extractor.extract(small)
                    produced = packet is not None
                except Exception as e:
                    self.last_error = f"{type(e).__name__}: {e}"
                    self.last_error_ts = time.time()
                    packet = None
                    produced = False

            with self.lock:
                self.latest_frame = frame
                if produced:
                    self.latest_packet = packet
                    self.latest_seq += 1

            # tiny sleep to give UI thread air
            time.sleep(0.001)

    def get_latest(self):
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            packet = self.latest_packet
            seq = self.latest_seq
            err = self.last_error
            err_age = time.time() - self.last_error_ts if self.last_error_ts > 0 else 999
        return frame, packet, seq, err, err_age



#Base model (high-accuracy)
class GazeModel:
    def __init__(self):
        base_est = ExtraTreesRegressor(
            n_estimators=N_TREES,
            max_depth=TREE_MAX_DEPTH,
            min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            bootstrap=False,
        )
        self.base = MultiOutputRegressor(base_est, n_jobs=-1)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
        X_tr, X_va, Y_tr, Y_va = train_test_split(X, Y, test_size=0.20, random_state=RANDOM_SEED)
        self.base.fit(X_tr, Y_tr)

        pred_va = self.base.predict(X_va)
        err = np.linalg.norm(pred_va - Y_va, axis=1)

        med_u = float(median_absolute_error(Y_va[:, 0], pred_va[:, 0]))
        med_v = float(median_absolute_error(Y_va[:, 1], pred_va[:, 1]))
        return {
            "val_mean_uv": float(err.mean()),
            "val_median_uv": float(np.median(err)),
            "val_median_abs_u": med_u,
            "val_median_abs_v": med_v,
        }

    def predict(self, feat_1xD: np.ndarray) -> np.ndarray:
        return self.base.predict(feat_1xD)


# -----------------------------
# Affine correction from quick 9-point
# -----------------------------
class AffineUVCorrector:
    def __init__(self):
        self.M = np.array([[1, 0, 0],
                           [0, 1, 0]], dtype=np.float32)
        self.enabled = False

    def clear(self):
        self.M[:] = np.array([[1, 0, 0],
                              [0, 1, 0]], dtype=np.float32)
        self.enabled = False

    def apply(self, u: float, v: float) -> Tuple[float, float]:
        if not self.enabled:
            return u, v
        x = np.array([u, v, 1.0], dtype=np.float32)
        y = self.M @ x
        return float(np.clip(y[0], 0.0, 1.0)), float(np.clip(y[1], 0.0, 1.0))

    def fit(self, pred_uv: np.ndarray, tgt_uv: np.ndarray, ridge: float = AFFINE_RIDGE) -> bool:
        if len(pred_uv) < 40:
            return False
        P = np.asarray(pred_uv, dtype=np.float32)
        T = np.asarray(tgt_uv, dtype=np.float32)

        A = np.concatenate([P, np.ones((len(P), 1), dtype=np.float32)], axis=1)  # Nx3
        AtA = A.T @ A + ridge * np.eye(3, dtype=np.float32)
        AtT = A.T @ T
        W = np.linalg.solve(AtA, AtT)  # 3x2
        self.M = W.T                   # 2x3
        self.enabled = True
        return True


# -----------------------------
# Fullscreen drawing helpers
# -----------------------------
def draw_screen(screen_w: int, screen_h: int, x: int, y: int, msg_top: str, msg_bottom: str) -> np.ndarray:
    img = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    cv2.circle(img, (x, y), 14, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.putText(img, msg_top, (40, int(screen_h * 0.08)), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, msg_bottom, (40, int(screen_h * 0.93)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "ESC=abort", (40, int(screen_h * 0.98)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (180, 180, 180), 2, cv2.LINE_AA)
    return img


def build_grid_points(screen_w: int, screen_h: int, cols: int, rows: int, margin_frac: float) -> List[Tuple[int, int, float, float]]:
    xs = np.linspace(screen_w * margin_frac, screen_w * (1 - margin_frac), cols)
    ys = np.linspace(screen_h * margin_frac, screen_h * (1 - margin_frac), rows)

    pts = []
    for j, y in enumerate(ys):
        row_xs = xs if (j % 2 == 0) else xs[::-1]
        for x in row_xs:
            u = float(x / screen_w)
            v = float(y / screen_h)
            pts.append((int(x), int(y), u, v))
    return pts


# -----------------------------
# Main calibration
# -----------------------------
def run_fullscreen_calibration(cam: CameraWorker, screen_w: int, screen_h: int) -> Tuple[np.ndarray, np.ndarray]:
    pts = build_grid_points(screen_w, screen_h, GRID_COLS, GRID_ROWS, MARGIN_FRAC)
    pts.append((screen_w // 2, screen_h // 2, 0.5, 0.5))
    total = len(pts)

    win = "Calibration (fullscreen)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    X_all: List[np.ndarray] = []
    Y_all: List[List[float]] = []

    for idx, (x, y, u, v) in enumerate(pts, start=1):
        t0 = time.time()
        while time.time() - t0 < SETTLE_TIME:
            img = draw_screen(screen_w, screen_h, x, y,
                              msg_top="Look at the dot. Keep head still.",
                              msg_bottom=f"Progress {idx}/{total}")
            cv2.imshow(win, img)
            if (cv2.waitKey(1) & 0xFF) == 27:
                cv2.destroyWindow(win)
                raise SystemExit("Calibration aborted.")
            time.sleep(0.005)

        X_dot: List[np.ndarray] = []
        t1 = time.time()
        while time.time() - t1 < COLLECT_TIME:
            img = draw_screen(screen_w, screen_h, x, y,
                              msg_top="Collecting...",
                              msg_bottom=f"Progress {idx}/{total}   good={len(X_dot)}")
            cv2.imshow(win, img)
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
            img = draw_screen(screen_w, screen_h, x, y,
                              msg_top=f"Low samples ({len(X_dot_np)}). Improve lighting / keep head still.",
                              msg_bottom=f"Progress {idx}/{total}")
            cv2.imshow(win, img)
            cv2.waitKey(1)
            time.sleep(0.25)

        for row in X_dot_np:
            X_all.append(row)
            Y_all.append([u, v])

        t2 = time.time()
        while time.time() - t2 < BREAK_TIME:
            img = draw_screen(screen_w, screen_h, x, y,
                              msg_top="Next...",
                              msg_bottom=f"Progress {idx}/{total}")
            cv2.imshow(win, img)
            if (cv2.waitKey(1) & 0xFF) == 27:
                cv2.destroyWindow(win)
                raise SystemExit("Calibration aborted.")
            time.sleep(0.005)

    cv2.destroyWindow(win)

    if len(X_all) < 350:
        raise SystemExit(f"Too few calibration samples ({len(X_all)}). Increase lighting / reduce head movement.")

    return np.array(X_all, dtype=np.float32), np.array(Y_all, dtype=np.float32)


def run_quick_9point_calibration(cam: CameraWorker, model: GazeModel, screen_w: int, screen_h: int) -> Tuple[np.ndarray, np.ndarray]:
    pts = build_grid_points(screen_w, screen_h, 3, 3, margin_frac=0.12)
    total = len(pts)

    win = "Quick 9-point calibration (SPACE)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    pred_uv_all: List[List[float]] = []
    tgt_uv_all: List[List[float]] = []

    for idx, (x, y, tu, tv) in enumerate(pts, start=1):
        t0 = time.time()
        while time.time() - t0 < Q9_SETTLE:
            img = draw_screen(screen_w, screen_h, x, y,
                              msg_top="QuickCal: look at dot (keep still)",
                              msg_bottom=f"{idx}/{total}   (ESC cancel)")
            cv2.imshow(win, img)
            if (cv2.waitKey(1) & 0xFF) == 27:
                cv2.destroyWindow(win)
                return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
            time.sleep(0.005)

        preds: List[List[float]] = []
        t1 = time.time()
        while time.time() - t1 < Q9_COLLECT:
            img = draw_screen(screen_w, screen_h, x, y,
                              msg_top="QuickCal: collecting...",
                              msg_bottom=f"{idx}/{total}   good={len(preds)}   (ESC cancel)")
            cv2.imshow(win, img)
            if (cv2.waitKey(1) & 0xFF) == 27:
                cv2.destroyWindow(win)
                return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

            _, packet, _, _, _ = cam.get_latest()
            if packet is not None and packet.quality_ok:
                uv = model.predict(packet.feat.reshape(1, -1))[0]
                preds.append([float(uv[0]), float(uv[1])])

            time.sleep(1.0 / max(10, TARGET_FPS_SAMPLES))

        if len(preds) < Q9_MIN_SAMPLES_PER_DOT:
            continue

        P = np.asarray(preds, dtype=np.float32)
        P = keep_central_samples(P, 0.70)

        for uv in P:
            pred_uv_all.append([float(uv[0]), float(uv[1])])
            tgt_uv_all.append([float(tu), float(tv)])

        t2 = time.time()
        while time.time() - t2 < Q9_BREAK:
            img = draw_screen(screen_w, screen_h, x, y,
                              msg_top="QuickCal: next...",
                              msg_bottom=f"{idx}/{total}")
            cv2.imshow(win, img)
            if (cv2.waitKey(1) & 0xFF) == 27:
                cv2.destroyWindow(win)
                return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
            time.sleep(0.005)

    cv2.destroyWindow(win)

    if len(pred_uv_all) < 60:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    return np.array(pred_uv_all, dtype=np.float32), np.array(tgt_uv_all, dtype=np.float32)


# -----------------------------
# Main
# -----------------------------
def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}\nPut face_landmarker.task next to this script.")

    screen_w, screen_h = get_screen_size_px()
    print(f"[INFO] Screen size: {screen_w}x{screen_h}")
    print(f"[INFO] Model path: {MODEL_PATH}")
    print(f"[INFO] Capture: {CAPTURE_W}x{CAPTURE_H} | Infer: {INFER_W}x{INFER_H} @ {INFER_FPS:.1f} FPS (FaceLandmarker)")
    print("[INFO] SPACE = quick 9-point recalibration (affine correction).  R = clear.  ESC = quit.")

    cam = CameraWorker(CAM_INDEX, str(MODEL_PATH))
    cam.start()
    time.sleep(0.6)

    print("[INFO] Starting full calibration. Look at dots. ESC to abort.")
    X, Y = run_fullscreen_calibration(cam, screen_w, screen_h)
    print(f"[INFO] Collected samples: {len(X)} | feature_dim={X.shape[1]}")

    model = GazeModel()
    stats = model.fit(X, Y)

    diag = math.sqrt(screen_w * screen_w + screen_h * screen_h)
    print(f"[INFO] Validation mean error ≈ {stats['val_mean_uv'] * diag:.1f}px, median ≈ {stats['val_median_uv'] * diag:.1f}px")
    print(f"[INFO] Validation median abs u≈{stats['val_median_abs_u']*screen_w:.1f}px, v≈{stats['val_median_abs_v']*screen_h:.1f}px")

    corrector = AffineUVCorrector()

    cv2.namedWindow("Gaze Tracker (ESC to quit)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gaze Tracker (ESC to quit)", 960, 540)

    cv2.namedWindow("Screen Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Screen Map", 480, 300)

    ema_u, ema_v = 0.5, 0.5
    last_good_ts = 0.0

    last_seq = -1
    last_pred_u, last_pred_v = 0.5, 0.5  # last raw prediction from model

    while True:
        frame, packet, seq, err, err_age = cam.get_latest()
        if frame is None:
            time.sleep(0.01)
            continue

        if DOWNSCALE_DISPLAY:
            frame_show = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA)
        else:
            frame_show = frame

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('r') or key == ord('R'):
            corrector.clear()
            print("[INFO] Quick-cal correction cleared.")
        if key == ord(' '):
            print("[INFO] Quick 9-point recalibration starting...")
            pred_uv, tgt_uv = run_quick_9point_calibration(cam, model, screen_w, screen_h)
            if len(pred_uv) > 0:
                ok = corrector.fit(pred_uv, tgt_uv, ridge=AFFINE_RIDGE)
                print(f"[INFO] QuickCal applied={ok}  samples={len(pred_uv)}")
            else:
                print("[INFO] QuickCal canceled / insufficient good samples.")

        # Only run ML prediction when NEW landmark packet arrives (big smoothness win)
        if packet is not None and packet.quality_ok and seq != last_seq:
            last_seq = seq
            uv = model.predict(packet.feat.reshape(1, -1))[0]
            last_pred_u = clamp01(float(uv[0]))
            last_pred_v = clamp01(float(uv[1]))
            last_good_ts = packet.ts

        # EMA update using latest prediction, with jump gate
        if abs(last_pred_u - ema_u) < MAX_JUMP and abs(last_pred_v - ema_v) < MAX_JUMP:
            ema_u = (1 - EMA_ALPHA) * ema_u + EMA_ALPHA * last_pred_u
            ema_v = (1 - EMA_ALPHA) * ema_v + EMA_ALPHA * last_pred_v

        # apply correction last
        out_u, out_v = corrector.apply(ema_u, ema_v)

        x_px = int(out_u * screen_w)
        y_px = int(out_v * screen_h)

        # feedback marker on camera frame (approx)
        h, w = frame_show.shape[:2]
        x_cam = int(out_u * w)
        y_cam = int(out_v * h)
        cv2.circle(frame_show, (x_cam, y_cam), 10, (0, 255, 0), 2)

        age = time.time() - last_good_ts
        status = "OK" if age < 0.45 else "NO FACE/BLINK/HEADTURN"
        corr_s = "corr=ON" if corrector.enabled else "corr=OFF"

        # show FPS-ish info (inference rate visible by seq changes)
        cv2.putText(frame_show, f"gaze(u,v)=({out_u:.3f},{out_v:.3f}) screen=({x_px},{y_px}) {status} {corr_s}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2)
        cv2.putText(frame_show, f"INFER_FPS={INFER_FPS:.1f}  InferSize={INFER_W}x{INFER_H}  Capture={CAPTURE_W}x{CAPTURE_H}",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
        cv2.putText(frame_show, "SPACE: 9pt quick-cal  |  R: clear  |  ESC: quit",
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)

        if err and err_age < 1.0:
            cv2.putText(frame_show, f"[CameraThreadError] {err}", (20, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # screen map
        sm_w, sm_h = 480, 300
        sm = np.zeros((sm_h, sm_w, 3), dtype=np.uint8)
        cv2.rectangle(sm, (0, 0), (sm_w - 1, sm_h - 1), (80, 80, 80), 1)
        cv2.circle(sm, (int(out_u * sm_w), int(out_v * sm_h)), 6, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(sm, corr_s, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Gaze Tracker (ESC to quit)", frame_show)
        cv2.imshow("Screen Map", sm)

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
