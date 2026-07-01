"""
Enhanced Eye Tracking System with Hand Gesture Controls
========================================================
Features:
- Eye gaze control for cursor movement
- LEFT HAND: Control cursor sensitivity with thumb-index distance
- RIGHT HAND: Pinch to click and drag
- Speech recognition with VOSK
- Blink/wink controls

Requirements:
    pip install opencv-python mediapipe pyautogui numpy scipy sounddevice vosk
    Optional: pip install keyboard   (system-wide Ctrl+Alt+E enable/disable, Ctrl+Alt+Q quit)
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from scipy.spatial import distance as dist
import threading
import json
import math
from pathlib import Path
import queue
import sys

# Note: win10toast is OS-specific (Windows). It may not work in all environments.
try:
    from win10toast import ToastNotifier
    toaster_available = True
except ImportError:
    toaster_available = False

# Optional: system-wide hotkeys to enable/disable and quit the app.
# Install with: pip install keyboard  (Windows: no admin needed for normal keys)
try:
    import keyboard
    keyboard_available = True
except ImportError:
    keyboard_available = False

# --- PyAutoGUI Setup ---
pyautogui.FAILSAFE = False
# CRITICAL for smoothness: pyautogui sleeps PAUSE seconds after EVERY call
# (default 0.1s). Left on, the cursor can only update ~10x/sec. Zero it so gaze
# movement runs at full camera frame rate.
pyautogui.PAUSE = 0

# Fast, low-latency cursor mover. On Windows, SetCursorPos is effectively instant
# and avoids pyautogui's per-call overhead; fall back to pyautogui elsewhere.
try:
    import ctypes
    _user32 = ctypes.windll.user32

    def move_cursor(x, y):
        _user32.SetCursorPos(int(x), int(y))
except Exception:
    def move_cursor(x, y):
        pyautogui.moveTo(int(x), int(y), _pause=False)

# Get screen dimensions for cursor mapping
screen_width, screen_height = pyautogui.size()
screen_center_x, screen_center_y = screen_width // 2, screen_height // 2

# --- MediaPipe setup ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize hand tracking
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- A. GAZE CONTROL PARAMETERS ---
# Legacy range/alpha params kept for reference; the active gaze pipeline now uses
# the distance-invariant estimator + One Euro Filter below.
H_RANGE_MIN = 0.20
H_RANGE_MAX = 0.80
V_RANGE_MIN = 0.40
V_RANGE_MAX = 0.60
ALPHA = 0.15      # (legacy) fixed-alpha EMA factor - superseded by One Euro Filter
BASE_SENSITIVITY = 1.5     # Base cursor speed; hand gesture scales sensitivity around this
YAW_STRENGTH = 0.5    # (legacy) solvePnP head stabilization - head pose now built into the estimator

# --- A2. GAZE ESTIMATION (distance-invariant head + iris fusion) ---
# One Euro Filter: velocity-adaptive low-pass smoothing.
ONE_EURO_MIN_CUTOFF = 0.4   # lower = steadier when holding still (less jitter). Try 0.25 if still twitchy.
ONE_EURO_BETA = 0.06        # higher = snaps to fast looks without lag; offsets the lower min_cutoff
ONE_EURO_DCUTOFF = 1.0

# Gaze signal gains (head 1.5 / iris 1.0), both divided by inter-eye width.
GAZE_HEAD_GAIN = 1.5
GAZE_IRIS_GAIN = 1.0
GAZE_MAGNITUDE_MAX = 1.6       # anti-runaway clamp on a bad landmark frame

# Pixels-per-unit-gaze as a multiplier of half the screen (resolution independent).
GAZE_SENSITIVITY_X = 2.0
GAZE_SENSITIVITY_Y = 2.0
GAZE_INVERT_X = False
GAZE_INVERT_Y = False

# Neutral ("straight ahead") baseline auto-captured over the first N eyes-open frames.
GAZE_NEUTRAL_CALIB_FRAMES = 20
GAZE_NEUTRAL_ADAPT = 0.0       # 0 disables drift; ~0.003 gently auto-recenters
MOVEMENT_DEAD_ZONE_PIXELS = 0  # 0 = update every frame (smoothest); One Euro handles jitter

# Hand gesture sensitivity range
MIN_SENSITIVITY = 0.2  # When fingers are very close
MAX_SENSITIVITY = 3.0  # When fingers are far apart
PINCH_THRESHOLD = 0.045  # Distance ratio for pinch detection

# Hand tracking costs roughly as much CPU per frame as the face mesh. For the
# smoothest possible GAZE, set HAND_TRACKING_ENABLED = False (disables pinch-drag,
# prayer-hands, and hand-controlled sensitivity). Left on, hands are detected only
# every Nth frame so they cost less while staying usable.
HAND_TRACKING_ENABLED = True
HAND_DETECT_EVERY_N_FRAMES = 2

# --- Gaze Landmark Indices ---
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

# --- Head Pose Estimation Setup ---
MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),        # Nose tip (1)
    (-225.0, 170.0, -135.0),    # Left eye left corner (226)
    (225.0, 170.0, -135.0),     # Right eye right corner (446)
    (-150.0, -150.0, -125.0),   # Left mouth corner (57)
    (150.0, -150.0, -125.0),    # Right mouth corner (287)
    (0.0, -330.0, -65.0)        # Chin (152)
], dtype=np.float32)

LANDMARK_POSE_INDICES = [1, 226, 446, 57, 287, 152]

# --- B. CLICK/WINK CONTROL PARAMETERS ---
EAR_CALIBRATION_ALPHA = 0.05
EAR_CLOSE_FACTOR = 0.70
WINK_CLOSE_RATIO = 0.65
WINK_OPEN_MIN_EAR = 0.25
WINK_ASYMMETRY_THRESHOLD = 0.15

CLICK_COOLDOWN = 1.0
BLINK_TIMEOUT = 1.25

# --- Click/Wink Landmark Indices ---
RIGHT_EYE_INDICES = [362, 263, 385, 386, 374, 380]
LEFT_EYE_INDICES = [33, 133, 160, 159, 145, 163]

# --- C. VOSK/SPEECH CONTROL PARAMETERS ---
MOUTH_OPEN_THRESHOLD = 0.09
MOUTH_HOLD_TIME = 1.5

# --- D. SCROLL CONTROL PARAMETERS ---
SCROLL_REGION_PERCENTAGE = 0.30
SCROLL_AMOUNT = 100
SCROLL_COOLDOWN = 0.3
RIGHT_WINK_HOLD_TIME = 0.5

# --- MOUTH LANDMARKS ---
MOUTH_INDICES = [13, 14, 61, 291]

# --- GLOBAL HOTKEYS (system-wide enable/disable of the whole app) ---
TOGGLE_HOTKEY = "ctrl+alt+e"   # press anywhere to pause/resume all control (same as prayer hands)
QUIT_HOTKEY = "ctrl+alt+q"     # press anywhere to quit the app cleanly

# =========================================================================
# PRELOAD VOSK MODEL AT STARTUP
# =========================================================================
print("\n" + "="*60)
print("PRELOADING VOSK MODEL FOR FASTER TRANSCRIPTION")
print("="*60)

vosk_model = None
vosk_available = False
vosk_preload_success = False
transcribed_text_queue = queue.Queue()

try:
    from vosk import Model, KaldiRecognizer
    import sounddevice as sd

    MODEL_PATH = "vosk-model-en-us-0.22"
    SAMPLE_RATE = 16000

    if not Path(MODEL_PATH).exists():
        print(f"VOSK model folder '{MODEL_PATH}' not found!")
        print("   Download from: https://alphacephei.com/vosk/models")
        vosk_available = False
    else:
        print(f"Found VOSK model at: {MODEL_PATH}")
        print("Loading model (this may take 15-30 seconds)...")

        start_load_time = time.time()
        vosk_model = Model(MODEL_PATH)
        load_time = time.time() - start_load_time

        print(f"VOSK model loaded successfully in {load_time:.1f} seconds!")
        vosk_available = True
        vosk_preload_success = True

except ImportError:
    print("VOSK not installed. Speech transcription disabled.")
    print("   To enable: pip install vosk sounddevice")
except Exception as e:
    print(f"VOSK initialization failed: {e}")

print("="*60 + "\n")

# --- HELPER FUNCTIONS ---

def log_action(message):
    """Prints a message to the console with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] ACTION: {message}")

def get_hand_landmarks(hand_landmarks, w, h):
    """Convert normalized landmarks to pixel coordinates."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append((int(lm.x * w), int(lm.y * h), lm.z))
    return landmarks

def calculate_distance(p1, p2):
    """Calculate 2D Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_pinch_distance(landmarks):
    """Calculate distance between thumb tip and index finger tip."""
    thumb_tip = landmarks[4]  # Thumb tip
    index_tip = landmarks[8]  # Index finger tip
    return calculate_distance(thumb_tip, index_tip)

def calculate_sensitivity_from_gap(gap_distance, frame_width):
    """Map thumb-index gap to sensitivity value."""
    # Normalize gap distance relative to frame width
    normalized_gap = gap_distance / frame_width

    # Map to sensitivity (smaller gap = lower sensitivity)
    # Typical gap range: 0.02 (nearly touching) to 0.15 (wide apart)
    sensitivity = np.interp(normalized_gap, [0.02, 0.15], [MIN_SENSITIVITY, MAX_SENSITIVITY])
    return max(MIN_SENSITIVITY, min(MAX_SENSITIVITY, sensitivity))

def are_hands_in_prayer_position(left_landmarks, right_landmarks, frame_width):
    """Check if hands are in prayer position (palms together)."""
    if not left_landmarks or not right_landmarks:
        return False

    # Check key points for prayer position
    # Compare multiple landmark pairs between hands
    landmark_pairs = [
        (4, 4),   # Thumb tips
        (8, 8),   # Index tips
        (12, 12), # Middle tips
        (16, 16), # Ring tips
        (20, 20), # Pinky tips
        (0, 0),   # Wrist
        (9, 9),   # Middle finger base
    ]

    close_count = 0
    total_distance = 0

    for left_idx, right_idx in landmark_pairs:
        left_point = left_landmarks[left_idx]
        right_point = right_landmarks[right_idx]

        # Calculate distance between corresponding points
        distance = calculate_distance(left_point, right_point)
        normalized_distance = distance / frame_width
        total_distance += normalized_distance

        # Count how many pairs are close together
        if normalized_distance < 0.06:  # Threshold for "close"
            close_count += 1

    # Prayer position if most landmarks are close and average distance is small
    avg_distance = total_distance / len(landmark_pairs)
    return close_count >= 5 and avg_distance < 0.08

def is_pinching(landmarks, frame_width):
    """Check if hand is making a pinch gesture."""
    gap = get_pinch_distance(landmarks)
    normalized_gap = gap / frame_width
    return normalized_gap < PINCH_THRESHOLD

def get_iris_center(landmarks, indices, image_w, image_h):
    """Get the center of iris landmarks."""
    xs = [landmarks[i].x for i in indices if i < len(landmarks)]
    ys = [landmarks[i].y for i in indices if i < len(landmarks)]
    if not xs or not ys:
        return image_w // 2, image_h // 2
    x = sum(xs) / len(xs)
    y = sum(ys) / len(ys)
    return int(x * image_w), int(y * image_h)

def get_head_yaw_correction(landmarks, w, h):
    """Calculate head yaw for gaze stabilization."""
    if len(landmarks) < max(LANDMARK_POSE_INDICES) + 1:
        return 0.0

    image_points_2d = np.array([
        (landmarks[i].x * w, landmarks[i].y * h)
        for i in LANDMARK_POSE_INDICES
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_3D_POINTS, image_points_2d, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if success:
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
            np.hstack((rotation_matrix, translation_vector))
        )
        yaw = euler_angles[2][0]
        return (yaw / 40.0) * 0.05
    return 0.0

def eye_aspect_ratio(eye_landmarks):
    """Calculate Eye Aspect Ratio for blink/wink detection."""
    A = dist.euclidean(eye_landmarks[2], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[3], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[1])

    if C == 0:
        return 0.5

    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth_landmarks):
    """Calculate Mouth Aspect Ratio for speech activation."""
    vertical_dist = dist.euclidean(mouth_landmarks[0], mouth_landmarks[1])
    horizontal_dist = dist.euclidean(mouth_landmarks[2], mouth_landmarks[3])

    if horizontal_dist == 0:
        return 0.0

    mar = vertical_dist / horizontal_dist
    return mar

def extract_landmarks_points(landmarks, indices, w, h):
    """Extract 2D coordinates for landmarks."""
    points = []
    for i in indices:
        x = int(landmarks[i].x * w)
        y = int(landmarks[i].y * h)
        points.append((x, y))
    return np.array(points)

# =========================================================================
# GAZE ESTIMATION
# =========================================================================
# Two primitives drive the accuracy/smoothness upgrade:
#   1. OneEuroFilter  - velocity-adaptive low-pass smoothing. Steady when still,
#                       snappy when moving fast. Replaces the old fixed-alpha EMA.
#   2. GazeEstimator  - distance-invariant gaze offset combining head pose and
#                       iris position, magnitude clamped and frozen during blinks.

def _one_euro_alpha(cutoff, freq):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    te = 1.0 / freq
    return 1.0 / (1.0 + tau / te)


class OneEuroFilter:
    """Adaptive low-pass filter. Works on scalars or numpy arrays.

    Reference: http://cristal.univ-lille.fr/~casiez/1euro/
    Lower ``mincutoff`` smooths more at low speed; higher ``beta`` keeps it
    responsive at high speed.
    """

    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=60.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.freq = freq
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def reset(self):
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def __call__(self, x, t=None):
        x = np.asarray(x, dtype=np.float32)

        if t is not None and self._t_prev is not None:
            dt = t - self._t_prev
            if dt > 1e-6:
                self.freq = 1.0 / dt
        if t is not None:
            self._t_prev = t

        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = np.zeros_like(x)
            return x

        dx = (x - self._x_prev) * self.freq
        a_d = _one_euro_alpha(self.dcutoff, self.freq)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.mincutoff + self.beta * np.abs(dx_hat)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.freq
        a = 1.0 / (1.0 + tau / te)

        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


def _mean_landmark_xy(landmarks, indices):
    """Mean normalised (x, y) of the given landmark indices."""
    pts = np.array(
        [[landmarks[i].x, landmarks[i].y] for i in indices if i < len(landmarks)],
        dtype=np.float32,
    )
    if pts.size == 0:
        return np.zeros(2, dtype=np.float32)
    return pts.mean(axis=0)


class GazeEstimator:
    """Distance-invariant gaze-to-cursor estimator.

    raw = head*GAZE_HEAD_GAIN + iris*GAZE_IRIS_GAIN, where both head pose
    (nose - eye_midpoint) and iris offset (iris - eye_socket center) are divided
    by the inter-eye width so the signal is invariant to camera distance. The
    summed offset is magnitude-clamped, referenced to an auto-calibrated neutral,
    mapped to an absolute screen target, and One Euro smoothed. The gaze is frozen
    while both eyes are closed so a blink/wink cannot jerk the cursor.
    """

    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.center_x = screen_w // 2
        self.center_y = screen_h // 2
        self.filter = OneEuroFilter(
            mincutoff=ONE_EURO_MIN_CUTOFF,
            beta=ONE_EURO_BETA,
            dcutoff=ONE_EURO_DCUTOFF,
        )
        self.neutral = None
        self._calib_accum = np.zeros(2, dtype=np.float32)
        self._calib_count = 0
        self.last_target = np.array([self.center_x, self.center_y], dtype=np.float32)

    @property
    def calibrated(self):
        return self.neutral is not None and self._calib_count >= GAZE_NEUTRAL_CALIB_FRAMES

    def _raw_offset(self, landmarks):
        """Distance-invariant gaze offset in eye-width units (+x right, +y down)."""
        nose = np.array([landmarks[1].x, landmarks[1].y], dtype=np.float32)
        eye_l = np.array([landmarks[LEFT_EYE_CORNERS[0]].x,
                          landmarks[LEFT_EYE_CORNERS[0]].y], dtype=np.float32)
        eye_r = np.array([landmarks[RIGHT_EYE_CORNERS[1]].x,
                          landmarks[RIGHT_EYE_CORNERS[1]].y], dtype=np.float32)
        eye_mid = (eye_l + eye_r) * 0.5
        eye_w = float(np.linalg.norm(eye_r - eye_l)) + 1e-6

        head = (nose - eye_mid) / eye_w

        left_iris = _mean_landmark_xy(landmarks, LEFT_IRIS)
        right_iris = _mean_landmark_xy(landmarks, RIGHT_IRIS)
        left_socket = _mean_landmark_xy(landmarks, LEFT_EYE_INDICES)
        right_socket = _mean_landmark_xy(landmarks, RIGHT_EYE_INDICES)
        iris = (((left_iris - left_socket) + (right_iris - right_socket)) * 0.5) / eye_w

        raw = head * GAZE_HEAD_GAIN + iris * GAZE_IRIS_GAIN

        mag = float(np.linalg.norm(raw))
        if mag > GAZE_MAGNITUDE_MAX:
            raw = raw / mag * GAZE_MAGNITUDE_MAX
        return raw.astype(np.float32)

    def update(self, landmarks, t, eyes_closed, sensitivity_scale=1.0):
        """Return (x, y) cursor target, or None when no move should occur.

        None is returned while calibrating or while both eyes are closed (freeze).
        ``sensitivity_scale`` multiplies the per-axis sensitivity at runtime (used
        by the hand-gesture control to modulate cursor speed; 1.0 = no change).
        """
        raw = self._raw_offset(landmarks)

        if not self.calibrated:
            if not eyes_closed:
                self._calib_accum += raw
                self._calib_count += 1
                self.neutral = self._calib_accum / self._calib_count
            return None

        if eyes_closed:
            return None

        if GAZE_NEUTRAL_ADAPT > 0.0:
            self.neutral = self.neutral * (1.0 - GAZE_NEUTRAL_ADAPT) + raw * GAZE_NEUTRAL_ADAPT

        offset = raw - self.neutral
        sx = -offset[0] if GAZE_INVERT_X else offset[0]
        sy = -offset[1] if GAZE_INVERT_Y else offset[1]

        target_x = self.center_x + sx * (self.screen_w * 0.5) * GAZE_SENSITIVITY_X * sensitivity_scale
        target_y = self.center_y + sy * (self.screen_h * 0.5) * GAZE_SENSITIVITY_Y * sensitivity_scale
        target = np.array([target_x, target_y], dtype=np.float32)

        smoothed = np.asarray(self.filter(target, t), dtype=np.float32)
        fx = float(np.clip(smoothed[0], 0, self.screen_w - 1))
        fy = float(np.clip(smoothed[1], 0, self.screen_h - 1))
        self.last_target = np.array([fx, fy], dtype=np.float32)
        return fx, fy

# =========================================================================
# VOSK LISTENER THREAD
# =========================================================================
def vosk_listener_thread(transcribing_flag, text_queue, model):
    """VOSK thread using preloaded model."""
    if not vosk_available or model is None:
        log_action("VOSK unavailable - cannot start transcription")
        return

    try:
        recognizer = KaldiRecognizer(model, SAMPLE_RATE)
        log_action("VOSK recognizer created from preloaded model")

        accumulated_text = []

        def callback(indata, frames, time_obj, status):
            if status:
                print(f"[VOSK] Stream status: {status}")

            if transcribing_flag[0]:
                if recognizer.AcceptWaveform(indata.tobytes()):
                    result = json.loads(recognizer.Result())
                    text = result.get('text', '').strip()
                    if text:
                        accumulated_text.append(text)
                        print(f"[{time.strftime('%H:%M:%S')}] VOSK: '{text}'")

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                            blocksize=8000, callback=callback):
            log_action("Audio stream active - speak now!")

            while transcribing_flag[0]:
                sd.sleep(100)

        final_result = json.loads(recognizer.FinalResult())
        final_text = final_result.get('text', '').strip()
        if final_text:
            accumulated_text.append(final_text)
            log_action(f"Final phrase: '{final_text}'")

        if accumulated_text:
            full_text = ' '.join(accumulated_text)
            text_queue.put(full_text)
            log_action(f"Queued for typing: '{full_text}'")

        log_action("Transcription complete")

    except sd.PortAudioError as e:
        log_action(f"Audio device error: {e}")
    except Exception as e:
        log_action(f"VOSK error: {e}")
    finally:
        transcribing_flag[0] = False

# =========================================================================
# MAIN CONTROL FUNCTION
# =========================================================================
def start_unified_control():
    """Main control loop with hand gesture controls."""
    global vosk_model

    # CAP_DSHOW (DirectShow) opens faster and delivers higher, more stable FPS on
    # Windows than the default MSMF backend. Lower resolution = faster face mesh =
    # higher loop rate = smoother cursor.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fall back to the default backend if DirectShow is unavailable.
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimise latency: always grab the freshest frame

    # FPS instrumentation (printed every ~2s so we can confirm the real loop rate).
    fps_counter = 0
    fps_timer = time.time()

    # Latest hand detection (persists between throttled detections; may be None).
    hand_results = None

    # State variables
    prev_x, prev_y = screen_center_x, screen_center_y
    gaze_estimator = GazeEstimator(screen_width, screen_height)
    avg_iris_y = 0.5  # latest vertical iris position (for right-wink scroll); defaulted so it is always defined
    last_click_time = 0
    last_scroll_time = 0
    blink_counter = 0
    last_blink_time = 0

    # Hand control state
    current_sensitivity = BASE_SENSITIVITY
    is_dragging = False
    drag_start_pos = None
    last_hand_cursor_pos = None

    # Prayer hands toggle state
    is_paused = False
    was_in_prayer_position = False
    last_prayer_toggle_time = 0
    PRAYER_COOLDOWN = 1.0  # Prevent rapid toggling

    # Adaptive EAR baseline
    ear_open_baseline = 0.35

    # Control flags
    is_eye_closed = False
    is_left_winking = False
    is_right_winking = False
    right_wink_start_time = 0
    is_mouth_open = False
    mouth_open_start_time = 0

    # Transcription state
    is_transcribing = [False]
    is_typing_active = False
    typing_start_time = 0
    MAX_TYPING_TIME = 10.0
    vosk_thread = None

    # Toast notifier
    toaster = None
    if toaster_available:
        try:
            toaster = ToastNotifier()
        except:
            toaster = None

    # Global enable/disable + quit via system-wide hotkeys. The toggle drives the
    # SAME is_paused flag used by the prayer-hands gesture, so both stay in sync.
    should_quit = [False]

    def _toggle_app():
        nonlocal is_paused
        is_paused = not is_paused
        state = "PAUSED " if is_paused else "RESUMED "
        log_action(f"Tracking {state}  (hotkey {TOGGLE_HOTKEY})")
        if toaster:
            try:
                toaster.show_toast("Eye Tracking", f"Controls {state}", duration=1, threaded=True)
            except Exception:
                pass

    def _quit_app():
        should_quit[0] = True
        log_action(f"Quit requested  (hotkey {QUIT_HOTKEY})")

    if keyboard_available:
        try:
            keyboard.add_hotkey(TOGGLE_HOTKEY, _toggle_app)
            keyboard.add_hotkey(QUIT_HOTKEY, _quit_app)
        except Exception as e:
            log_action(f"Could not register global hotkeys: {e}")

    # Initialize cursor
    pyautogui.moveTo(prev_x, prev_y)

    print("\n" + "="*60)
    print("UNIFIED EYE TRACKING WITH HAND GESTURE CONTROL")
    print("="*60)
    print("\nControl Scheme:")
    print("  - PRAYER HANDS: Toggle ALL controls ON/OFF")
    print("    - Bring palms together = Pause/Resume everything")
    print("    - Visual indicator shows current state")
    print("  - Eye Gaze: Move cursor (pauses during drag)")
    print("  - LEFT HAND: Pinch to click & drag")
    print("    - Pinch = Start dragging (eye tracking pauses)")
    print("    - Move hand while pinched = Drag items")
    print("    - Release = Stop dragging (eye tracking resumes)")
    print("  - RIGHT HAND: Adjust cursor speed")
    print("    - Small gap (thumb-index) = Precise control")
    print("    - Large gap = Fast movement")
    print("  - Double Blink: Left click")
    print("  - Triple Blink: Right click")
    print("  - Left Wink: Alt+Tab")
    print("  - Right Wink (at screen edges): Scroll")
    print("  - Hold Mouth Open 1.5s: Toggle speech")
    if keyboard_available:
        print(f"  - {TOGGLE_HOTKEY.upper()}: Enable/Disable all controls (works anywhere)")
        print(f"  - {QUIT_HOTKEY.upper()}: Quit the app")
    else:
        print("  - (global on/off hotkey disabled - run: pip install keyboard)")
    print(f"\nScreen Info: {screen_width}x{screen_height}")
    print(f"Speech: {'READY (preloaded)' if vosk_preload_success else 'UNAVAILABLE'}")
    print("="*60 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Global quit gate (system-wide hotkey). Enable/disable is handled by
            # the is_paused flag, which both this hotkey and prayer-hands toggle.
            if should_quit[0]:
                print("\nQuit hotkey pressed")
                break

            fps_counter += 1

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            # Process face mesh (every frame - this drives the gaze cursor)
            face_results = face_mesh.process(frame_rgb)

            # Process hands on a slower cadence (or not at all) so they don't halve
            # the gaze frame rate. hand_results persists between detections.
            if HAND_TRACKING_ENABLED and (fps_counter % HAND_DETECT_EVERY_N_FRAMES == 0):
                hand_results = hands.process(frame_rgb)

            current_time = time.time()

            # Report the real loop rate every ~2s so smoothness is measurable.
            if current_time - fps_timer >= 2.0:
                print(f"[perf] ~{fps_counter / (current_time - fps_timer):.0f} FPS")
                fps_counter = 0
                fps_timer = current_time

            # Determine lock state
            INPUT_LOCKED = is_transcribing[0] or is_typing_active or not transcribed_text_queue.empty() or is_paused

            # Process typing queue
            if not transcribed_text_queue.empty() and not is_typing_active:
                is_typing_active = True
                typing_start_time = current_time
                text = transcribed_text_queue.get()

                log_action(f"TYPING START - All inputs LOCKED")

                try:
                    for char in text:
                        pyautogui.write(char)
                        time.sleep(0.005)

                    pyautogui.write(" ")
                    log_action(f"Successfully typed: '{text}'")
                except Exception as e:
                    log_action(f"Typing error: {e}")

                is_typing_active = False
                typing_start_time = 0

                if transcribed_text_queue.empty():
                    if toaster:
                        toaster.show_toast("Eye Tracking", "Typing complete - Inputs UNLOCKED", duration=1, threaded=True)
                    log_action("ALL INPUTS UNLOCKED")

            # Safety check
            if is_typing_active and (current_time - typing_start_time > MAX_TYPING_TIME):
                log_action("Typing timeout - force unlocking")
                is_typing_active = False
                typing_start_time = 0

            # ========================================================
            # PRAYER HANDS DETECTION (Always check first)
            # ========================================================
            if hand_results is not None and hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) == 2:
                # Get both hands
                left_hand = None
                right_hand = None

                for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    hand_label = hand_results.multi_handedness[idx].classification[0].label
                    landmarks = get_hand_landmarks(hand_landmarks, w, h)

                    if hand_label == "Left":  # Mirrored - user's right hand
                        left_hand = landmarks
                    else:  # User's left hand
                        right_hand = landmarks

                # Check for prayer position
                if left_hand and right_hand:
                    in_prayer_position = are_hands_in_prayer_position(left_hand, right_hand, w)

                    # Toggle pause state on prayer gesture (with cooldown)
                    if in_prayer_position and not was_in_prayer_position:
                        if current_time - last_prayer_toggle_time > PRAYER_COOLDOWN:
                            is_paused = not is_paused
                            last_prayer_toggle_time = current_time

                            if is_paused:
                                log_action("PRAYER HANDS - ALL CONTROLS PAUSED")
                                print("\n" + "="*60)
                                print("SYSTEM PAUSED - All inputs disabled")
                                print("   Put hands together again to resume")
                                print("="*60 + "\n")

                                # Release any held mouse button
                                if is_dragging:
                                    pyautogui.mouseUp()
                                    is_dragging = False

                                if toaster:
                                    try:
                                        toaster.show_toast("Eye Tracking", "PAUSED - All controls disabled", duration=3, threaded=True)
                                    except:
                                        pass
                            else:
                                log_action("PRAYER HANDS - CONTROLS RESUMED")
                                print("\n" + "="*60)
                                print("SYSTEM RESUMED - All controls active")
                                print("="*60 + "\n")

                                if toaster:
                                    try:
                                        toaster.show_toast("Eye Tracking", "RESUMED - All controls active", duration=3, threaded=True)
                                    except:
                                        pass

                    was_in_prayer_position = in_prayer_position
                else:
                    was_in_prayer_position = False
            else:
                was_in_prayer_position = False

            # Show pause status periodically
            if is_paused and int(current_time * 2) % 2 == 0 and int(current_time * 10) % 10 == 0:
                print(f"\rPAUSED - Prayer hands to resume | Time: {time.strftime('%H:%M:%S')}", end='')

            # ========================================================
            # HAND GESTURE PROCESSING (Only when not paused)
            # ========================================================
            if not INPUT_LOCKED and not is_paused and hand_results is not None and hand_results.multi_hand_landmarks:
                left_hand = None
                right_hand = None

                # Identify which hand is which (SWAPPED)
                for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    hand_label = hand_results.multi_handedness[idx].classification[0].label
                    landmarks = get_hand_landmarks(hand_landmarks, w, h)

                    if hand_label == "Left":  # This is mirrored, so user's right hand
                        left_hand = landmarks  # SWAPPED: Right hand is now left_hand variable
                    else:  # User's left hand
                        right_hand = landmarks  # SWAPPED: Left hand is now right_hand variable

                # RIGHT HAND (user's actual right): Control sensitivity
                if left_hand:  # This is actually the user's right hand
                    gap_distance = get_pinch_distance(left_hand)
                    current_sensitivity = calculate_sensitivity_from_gap(gap_distance, w)

                    # Visual feedback
                    gap_percent = int((current_sensitivity - MIN_SENSITIVITY) / (MAX_SENSITIVITY - MIN_SENSITIVITY) * 100)
                    print(f"\rSpeed: {'#' * (gap_percent // 10)}{'.' * (10 - gap_percent // 10)} {gap_percent}%", end='')

                # LEFT HAND (user's actual left): Click and drag
                if right_hand:  # This is actually the user's left hand
                    pinching = is_pinching(right_hand, w)

                    # Get hand position (use index finger tip for cursor tracking)
                    hand_x = right_hand[8][0]  # Index tip x
                    hand_y = right_hand[8][1]  # Index tip y

                    # Map hand position to screen
                    screen_x = int(np.interp(hand_x, [0, w], [0, screen_width]))
                    screen_y = int(np.interp(hand_y, [0, h], [0, screen_height]))

                    if pinching and not is_dragging:
                        # Start dragging
                        is_dragging = True
                        pyautogui.mouseDown()
                        drag_start_pos = (screen_x, screen_y)
                        last_hand_cursor_pos = (screen_x, screen_y)
                        log_action("DRAG START (Left hand pinch)")

                        # Show toast notification
                        if toaster:
                            try:
                                toaster.show_toast("Eye Tracking", "DRAGGING - Eye tracking paused", duration=2, threaded=True)
                            except:
                                pass

                    elif pinching and is_dragging:
                        # Continue dragging - move cursor with hand
                        if last_hand_cursor_pos:
                            # Calculate delta movement
                            delta_x = screen_x - last_hand_cursor_pos[0]
                            delta_y = screen_y - last_hand_cursor_pos[1]

                            # Apply movement with smoothing
                            current_pos = pyautogui.position()
                            new_x = current_pos.x + delta_x * 0.5  # Smooth movement
                            new_y = current_pos.y + delta_y * 0.5

                            # Constrain to screen
                            new_x = max(0, min(screen_width - 1, new_x))
                            new_y = max(0, min(screen_height - 1, new_y))

                            pyautogui.moveTo(new_x, new_y)
                            prev_x, prev_y = new_x, new_y

                        last_hand_cursor_pos = (screen_x, screen_y)

                    elif not pinching and is_dragging:
                        # Stop dragging
                        is_dragging = False
                        pyautogui.mouseUp()
                        log_action("DRAG END (Released pinch)")
                        drag_start_pos = None
                        last_hand_cursor_pos = None

                        # Show toast notification
                        if toaster:
                            try:
                                toaster.show_toast("Eye Tracking", "Drag complete - Eye tracking resumed", duration=2, threaded=True)
                            except:
                                pass

            # Reset drag state if hands disappear
            elif is_dragging and not INPUT_LOCKED:
                is_dragging = False
                pyautogui.mouseUp()
                log_action("DRAG END (Hand lost)")

                # Show toast notification
                if toaster:
                    try:
                        toaster.show_toast("Eye Tracking", "Drag ended - Eye tracking resumed", duration=2, threaded=True)
                    except:
                        pass

            # ========================================================
            # FACE/EYE TRACKING (PAUSED when dragging with hand)
            # ========================================================
            if face_results.multi_face_landmarks and not is_dragging:  # PAUSE eye tracking during drag
                landmarks = face_results.multi_face_landmarks[0].landmark

                # Calculate EAR for adaptive thresholds
                left_eye_pts = extract_landmarks_points(landmarks, LEFT_EYE_INDICES, w, h)
                right_eye_pts = extract_landmarks_points(landmarks, RIGHT_EYE_INDICES, w, h)

                left_ear = eye_aspect_ratio(left_eye_pts)
                right_ear = eye_aspect_ratio(right_eye_pts)
                avg_ear = (left_ear + right_ear) / 2

                # Update baseline when eyes are open
                if avg_ear > WINK_OPEN_MIN_EAR:
                    ear_open_baseline = ear_open_baseline * (1 - EAR_CALIBRATION_ALPHA) + avg_ear * EAR_CALIBRATION_ALPHA

                # Dynamic thresholds
                blink_threshold = ear_open_baseline * EAR_CLOSE_FACTOR
                wink_threshold = ear_open_baseline * WINK_CLOSE_RATIO

                # Both-eyes-closed state - freezes the gaze cursor during a blink
                # and is reused below for blink-click detection.
                both_eyes_closed_now = (left_ear < blink_threshold) and (right_ear < blink_threshold)

                # MOUTH DETECTION
                if vosk_available and vosk_model:
                    mouth_pts = extract_landmarks_points(landmarks, MOUTH_INDICES, w, h)
                    mar = mouth_aspect_ratio(mouth_pts)

                    mouth_open_now = mar > MOUTH_OPEN_THRESHOLD

                    if mouth_open_now and not is_mouth_open:
                        is_mouth_open = True
                        mouth_open_start_time = current_time
                        log_action(f"Mouth opened (MAR={mar:.3f}), hold {MOUTH_HOLD_TIME}s...")

                    elif mouth_open_now and is_mouth_open:
                        if current_time - mouth_open_start_time >= MOUTH_HOLD_TIME:
                            if not is_transcribing[0]:
                                # START TRANSCRIPTION
                                is_transcribing[0] = True

                                while not transcribed_text_queue.empty():
                                    transcribed_text_queue.get()

                                vosk_thread = threading.Thread(
                                    target=vosk_listener_thread,
                                    args=(is_transcribing, transcribed_text_queue, vosk_model)
                                )
                                vosk_thread.daemon = True
                                vosk_thread.start()

                                if toaster:
                                    toaster.show_toast("Eye Tracking", "RECORDING - All inputs LOCKED", duration=2, threaded=True)
                                log_action("RECORDING STARTED - ALL INPUTS LOCKED")
                            else:
                                # STOP TRANSCRIPTION
                                is_transcribing[0] = False
                                log_action("RECORDING STOPPED")

                                if vosk_thread and vosk_thread.is_alive():
                                    vosk_thread.join(timeout=3.0)

                            is_mouth_open = False
                            last_click_time = current_time

                    elif not mouth_open_now:
                        is_mouth_open = False

                if not INPUT_LOCKED:

                    click_available = (current_time - last_click_time) > CLICK_COOLDOWN
                    scroll_available = (current_time - last_scroll_time) > SCROLL_COOLDOWN

                    # WINK DETECTION
                    ear_difference = abs(left_ear - right_ear)

                    # LEFT WINK (Alt+Tab)
                    left_wink_now = (left_ear < wink_threshold) and (right_ear > WINK_OPEN_MIN_EAR)
                    left_wink_asymmetry = (left_ear < right_ear - WINK_ASYMMETRY_THRESHOLD) and (right_ear > WINK_OPEN_MIN_EAR)
                    left_wink_now = left_wink_now or left_wink_asymmetry

                    if left_wink_now and not is_left_winking and click_available:
                        pyautogui.hotkey('alt', 'tab')
                        log_action(f"ALT+TAB (Left Wink)")
                        last_click_time = current_time
                        is_left_winking = True
                        blink_counter = 0

                    if not left_wink_now:
                        is_left_winking = False

                    # RIGHT WINK (Scroll)
                    right_wink_now = (right_ear < wink_threshold) and (left_ear > WINK_OPEN_MIN_EAR)
                    right_wink_asymmetry = (right_ear < left_ear - WINK_ASYMMETRY_THRESHOLD) and (left_ear > WINK_OPEN_MIN_EAR)
                    right_wink_relative = (right_ear < ear_open_baseline * 0.7) and (left_ear > right_ear + 0.1)

                    right_wink_now = right_wink_now or right_wink_asymmetry or right_wink_relative

                    if right_wink_now and not is_right_winking:
                        is_right_winking = True
                        right_wink_start_time = current_time
                        blink_counter = 0

                    if right_wink_now and is_right_winking and scroll_available:
                        cursor_y = pyautogui.position().y
                        scroll_zone = screen_height * SCROLL_REGION_PERCENTAGE

                        if cursor_y < scroll_zone:
                            pyautogui.scroll(SCROLL_AMOUNT)
                            log_action(f"SCROLL UP")
                            last_scroll_time = current_time
                        elif cursor_y > screen_height - scroll_zone:
                            pyautogui.scroll(-SCROLL_AMOUNT)
                            log_action(f"SCROLL DOWN")
                            last_scroll_time = current_time

                    if not right_wink_now and is_right_winking:
                        is_right_winking = False

                    # GAZE CONTROL (distance-invariant head+iris
                    # fusion, magnitude-clamped, One Euro smoothed, blink-frozen).
                    # The left-hand gap still modulates cursor speed via sensitivity_scale.
                    #
                    # avg_iris_y is still derived here for the right-wink-hold scroll feature.
                    left_iris = get_iris_center(landmarks, LEFT_IRIS, w, h)
                    right_iris = get_iris_center(landmarks, RIGHT_IRIS, w, h)
                    avg_iris_y = ((left_iris[1] + right_iris[1]) / 2) / h

                    sensitivity_scale = current_sensitivity / BASE_SENSITIVITY
                    gaze_target = gaze_estimator.update(
                        landmarks, current_time, both_eyes_closed_now, sensitivity_scale
                    )

                    if gaze_target is not None:
                        final_x, final_y = gaze_target
                        move_distance = np.sqrt((final_x - prev_x)**2 + (final_y - prev_y)**2)
                        # Dead zone of 0 = move every frame for fluid motion; the One Euro
                        # filter (not a dead zone) is what suppresses jitter while still.
                        if move_distance > MOVEMENT_DEAD_ZONE_PIXELS:
                            move_cursor(final_x, final_y)
                            prev_x, prev_y = final_x, final_y

                    # BLINK DETECTION
                    eyes_closed = both_eyes_closed_now

                    if not is_left_winking and not is_right_winking:
                        if eyes_closed and not is_eye_closed:
                            blink_counter += 1
                            last_blink_time = current_time
                            is_eye_closed = True
                            log_action(f"Blink {blink_counter} detected")

                        if not eyes_closed and is_eye_closed:
                            is_eye_closed = False

                    # Process blink sequences
                    if blink_counter > 0:
                        if current_time - last_blink_time > BLINK_TIMEOUT:
                            if click_available:
                                if blink_counter == 2:
                                    pyautogui.click(button='left')
                                    log_action("LEFT CLICK (Double blink)")
                                elif blink_counter >= 3:
                                    pyautogui.click(button='right')
                                    log_action("RIGHT CLICK (Triple blink)")

                            blink_counter = 0
                            last_click_time = current_time

                else:
                    # LOCKED - Reset all control states
                    is_left_winking = False
                    is_right_winking = False
                    is_eye_closed = False
                    blink_counter = 0

            # Small sleep for CPU efficiency
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if is_dragging:
            pyautogui.mouseUp()
            log_action("Released mouse on exit")

        if is_transcribing[0]:
            is_transcribing[0] = False
            if vosk_thread and vosk_thread.is_alive():
                vosk_thread.join(timeout=2.0)

        # Type any remaining text
        while not transcribed_text_queue.empty():
            text = transcribed_text_queue.get()
            pyautogui.write(text + " ")
            log_action(f"Final type: '{text}'")

        if keyboard_available:
            try:
                keyboard.remove_hotkey(TOGGLE_HOTKEY)
                keyboard.remove_hotkey(QUIT_HOTKEY)
            except Exception:
                pass

        cap.release()
        face_mesh.close()
        hands.close()
        print("\nCleanup complete")
        print("="*60)

# =========================================================================
# MAIN EXECUTION
# =========================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ENHANCED EYE TRACKING WITH HAND GESTURE CONTROLS")
    print("="*60)
    print("\nHand Gesture Controls:")
    print("  - LEFT HAND: Pinch to drag")
    print("    - Pinch fingers = Click & hold (eye tracking pauses)")
    print("    - Move hand while pinched = Drag items")
    print("    - Release pinch = Release mouse (eye tracking resumes)")
    print("  - RIGHT HAND: Control cursor speed")
    print("    - Bring thumb & index close = Slow/precise")
    print("    - Spread them apart = Fast movement")
    print("\nEye Controls:")
    print("  - Cursor movement (speed controlled by right hand)")
    print("  - Automatically pauses during left hand drag")
    print("  - Blink clicking")
    print("  - Wink commands")
    print("  - Speech control")

    if vosk_preload_success:
        print("\nSpeech model preloaded - transcription ready!")
    else:
        print("\nSpeech recognition unavailable")

    print("\nStarting in 3 seconds...")
    print("   Show your hands to the camera for gesture control!")
    time.sleep(3)

    start_unified_control()
