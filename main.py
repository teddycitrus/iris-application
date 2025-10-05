"""
Enhanced Eye Tracking System with Speech Recognition and Console Feedback
========================================================================
Features: Gaze control, blink clicks, speech transcription.
HEADLESS MODE: No OpenCV window is displayed. All feedback is printed to the console.
OPTIMIZED: VOSK model preloaded at startup for faster transcription start.
LOCKED: ALL inputs completely frozen during transcription AND typing.

Requirements:
    pip install opencv-python mediapipe pyautogui numpy scipy sounddevice vosk
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from scipy.spatial import distance as dist
import threading
import json
from pathlib import Path
import queue
import sys

# Note: win10toast is OS-specific (Windows). It may not work in all environments.
try:
    from win10toast import ToastNotifier
    toaster_available = True
except ImportError:
    toaster_available = False

# --- PyAutoGUI Setup ---
pyautogui.FAILSAFE = False

# Get screen dimensions for cursor mapping
screen_width, screen_height = pyautogui.size()
screen_center_x, screen_center_y = screen_width // 2, screen_height // 2

# --- MediaPipe setup (Only initialized once) ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- A. GAZE CONTROL PARAMETERS ---
H_RANGE_MIN = 0.20
H_RANGE_MAX = 0.80
V_RANGE_MIN = 0.40
V_RANGE_MAX = 0.60
ALPHA = 0.15      # Smoothing factor
SENSITIVITY = 1.5     # Cursor speed multiplier
YAW_STRENGTH = 0.5    # Head stabilization strength
MOVEMENT_DEAD_ZONE_PIXELS = 48 # Distance threshold to prevent jitter

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
# ADAPTIVE BLINK DETECTION PARAMETERS
EAR_CALIBRATION_ALPHA = 0.05  # Smoothing factor for open EAR baseline
EAR_CLOSE_FACTOR = 0.70       # Blink threshold is 70% of open baseline

# ADAPTIVE WINK PARAMETERS
WINK_CLOSE_RATIO = 0.55       # Tighter close ratio (55% of open EAR)
WINK_OPEN_MIN_EAR = 0.30      # Safe minimum threshold

CLICK_COOLDOWN = 1.0
BLINK_TIMEOUT = 1.25

# --- Click/Wink Landmark Indices ---
RIGHT_EYE_INDICES = [362, 263, 385, 386, 374, 380] 
LEFT_EYE_INDICES = [33, 133, 160, 159, 145, 163]

# --- C. VOSK/SPEECH CONTROL PARAMETERS ---
MOUTH_OPEN_THRESHOLD = 0.09   # Strict MAR threshold
MOUTH_HOLD_TIME = 1.5         # Seconds mouth must be held open

# --- D. SCROLL CONTROL PARAMETERS ---
SCROLL_REGION_PERCENTAGE = 0.30 # Top/bottom 30% of screen (increased from 20%)
SCROLL_AMOUNT = 100             # Increased scroll amount for better visibility
SCROLL_COOLDOWN = 0.3           # Slightly longer cooldown
# Alternative: hold right wink for continuous scroll
RIGHT_WINK_HOLD_TIME = 0.5     # Hold for half second to trigger scroll           

# --- MOUTH LANDMARKS ---
MOUTH_INDICES = [13, 14, 61, 291]

# =========================================================================
# PRELOAD VOSK MODEL AT STARTUP
# =========================================================================
print("\n" + "="*60)
print("📦 PRELOADING VOSK MODEL FOR FASTER TRANSCRIPTION")
print("="*60)

vosk_model = None
vosk_available = False
vosk_preload_success = False
transcribed_text_queue = queue.Queue()  # Thread-safe queue

try:
    from vosk import Model, KaldiRecognizer
    import sounddevice as sd
    
    MODEL_PATH = "vosk-model-en-us-0.22"
    SAMPLE_RATE = 16000
    
    if not Path(MODEL_PATH).exists():
        print(f"❌ VOSK model folder '{MODEL_PATH}' not found!")
        print("   Download from: https://alphacephei.com/vosk/models")
        vosk_available = False
    else:
        print(f"📁 Found VOSK model at: {MODEL_PATH}")
        print("⏳ Loading model (this may take 15-30 seconds)...")
        
        start_load_time = time.time()
        vosk_model = Model(MODEL_PATH)
        load_time = time.time() - start_load_time
        
        print(f"✅ VOSK model loaded successfully in {load_time:.1f} seconds!")
        vosk_available = True
        vosk_preload_success = True
        
except ImportError:
    print("❌ VOSK not installed. Speech transcription disabled.")
    print("   To enable: pip install vosk sounddevice")
except Exception as e:
    print(f"❌ VOSK initialization failed: {e}")

print("="*60 + "\n")

# --- HELPER FUNCTIONS ---

def log_action(message):
    """Prints a message to the console with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] ⚙️ ACTION: {message}")

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
# OPTIMIZED VOSK LISTENER THREAD
# =========================================================================
def vosk_listener_thread(transcribing_flag, text_queue, model):
    """
    Optimized VOSK thread using preloaded model.
    """
    if not vosk_available or model is None:
        log_action("VOSK unavailable - cannot start transcription")
        return
    
    try:
        # Create recognizer from preloaded model (FAST!)
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
                        print(f"[{time.strftime('%H:%M:%S')}] 🔊 VOSK: '{text}'")
        
        # Start audio stream
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', 
                            blocksize=8000, callback=callback):
            log_action("Audio stream active - speak now!")
            
            while transcribing_flag[0]:
                sd.sleep(100)
        
        # Get final partial result
        final_result = json.loads(recognizer.FinalResult())
        final_text = final_result.get('text', '').strip()
        if final_text:
            accumulated_text.append(final_text)
            log_action(f"Final phrase: '{final_text}'")
        
        # Queue all accumulated text as one block
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
    """Main control loop with complete input locking."""
    global vosk_model
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    # State variables
    prev_x, prev_y = screen_center_x, screen_center_y
    last_click_time = 0
    last_scroll_time = 0
    blink_counter = 0
    last_blink_time = 0
    
    # Adaptive EAR baseline
    ear_open_baseline = 0.35
    
    # Control flags
    is_eye_closed = False
    is_left_winking = False
    is_right_winking = False
    right_wink_start_time = 0  # Track when right wink started
    is_mouth_open = False
    mouth_open_start_time = 0
    
    # Transcription state
    is_transcribing = [False]  # Shared with thread
    is_typing_active = False   # Tracks active typing
    vosk_thread = None
    
    # Toast notifier
    toaster = None
    if toaster_available:
        try:
            toaster = ToastNotifier()
        except:
            toaster = None
    
    # Initialize cursor
    pyautogui.moveTo(prev_x, prev_y)
    
    print("\n" + "="*60)
    print("👁️ UNIFIED EYE TRACKING CONTROL (OPTIMIZED)")
    print("="*60)
    print("\n📋 Control Scheme:")
    print("  • Eye Gaze: Move cursor")
    print("  • Double Blink: Left click")
    print("  • Triple Blink: Right click") 
    print("  • Left Wink: Alt+Tab")
    print("  • Right Wink (cursor at top/bottom 30%): Scroll")
    print("  • Right Wink HOLD + look up/down: Scroll anywhere")
    print("  • Hold Mouth Open 1.5s: Toggle speech")
    print(f"\n📊 Screen Info: {screen_width}x{screen_height}")
    print(f"   Scroll zones: Top {int(screen_height * 0.3)}px, Bottom {int(screen_height * 0.7)}px")
    print("\n🔒 LOCKING: All inputs frozen during transcription & typing")
    print(f"🎤 Speech: {'READY (preloaded)' if vosk_preload_success else 'UNAVAILABLE'}")
    print("="*60 + "\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            results = face_mesh.process(frame_rgb)
            
            current_time = time.time()
            
            # ============================================================
            # CRITICAL: DETERMINE LOCK STATE
            # ============================================================
            # Lock if transcribing OR if there's text waiting to be typed
            INPUT_LOCKED = is_transcribing[0] or is_typing_active or not transcribed_text_queue.empty()
            
            # Process typing queue (always runs, even when locked)
            if not transcribed_text_queue.empty() and not is_typing_active:
                is_typing_active = True
                text = transcribed_text_queue.get()
                
                # Type the entire phrase
                pyautogui.write(text + " ")
                log_action(f"💬 Typed: '{text}'")
                
                # Unlock after typing
                is_typing_active = False
                if toaster:
                    toaster.show_toast("Eye Tracking", "Inputs UNLOCKED", duration=1, threaded=True)
                log_action("🔓 INPUTS UNLOCKED - Ready for control")
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
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
                
                # ========================================================
                # MOUTH DETECTION (Always active for start/stop)
                # ========================================================
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
                                
                                vosk_thread = threading.Thread(
                                    target=vosk_listener_thread,
                                    args=(is_transcribing, transcribed_text_queue, vosk_model)
                                )
                                vosk_thread.daemon = True
                                vosk_thread.start()
                                
                                if toaster:
                                    toaster.show_toast("Eye Tracking", "RECORDING - Inputs LOCKED", duration=2, threaded=True)
                                log_action("🎤 RECORDING STARTED - ALL INPUTS LOCKED 🔒")
                            else:
                                # STOP TRANSCRIPTION
                                is_transcribing[0] = False
                                log_action("🛑 RECORDING STOPPED - Processing speech...")
                                
                                if toaster:
                                    toaster.show_toast("Eye Tracking", "Processing speech...", duration=1, threaded=True)
                                
                                # Wait for thread to finish
                                if vosk_thread and vosk_thread.is_alive():
                                    vosk_thread.join(timeout=2.0)
                            
                            is_mouth_open = False
                            last_click_time = current_time
                    
                    elif not mouth_open_now:
                        is_mouth_open = False
                
                # ========================================================
                # ALL OTHER CONTROLS - ONLY IF NOT LOCKED
                # ========================================================
                if not INPUT_LOCKED:
                    
                    click_available = (current_time - last_click_time) > CLICK_COOLDOWN
                    scroll_available = (current_time - last_scroll_time) > SCROLL_COOLDOWN
                    
                    # LEFT WINK (Alt+Tab)
                    left_wink_now = (left_ear < wink_threshold) and (right_ear > WINK_OPEN_MIN_EAR)
                    
                    if left_wink_now and not is_left_winking and click_available:
                        pyautogui.hotkey('alt', 'tab')
                        log_action("ALT+TAB (Left Wink)")
                        last_click_time = current_time
                        is_left_winking = True
                        blink_counter = 0
                    
                    if not left_wink_now:
                        is_left_winking = False
                    
                    # RIGHT WINK (Scroll) - IMPROVED DETECTION
                    right_wink_now = (right_ear < wink_threshold) and (left_ear > WINK_OPEN_MIN_EAR)
                    
                    # Track right wink start
                    if right_wink_now and not is_right_winking:
                        is_right_winking = True
                        right_wink_start_time = current_time
                        log_action(f"Right wink START! R_EAR={right_ear:.3f} < {wink_threshold:.3f}, L_EAR={left_ear:.3f}")
                        blink_counter = 0  # Reset blink counter
                    
                    # Handle right wink hold
                    if right_wink_now and is_right_winking:
                        wink_duration = current_time - right_wink_start_time
                        
                        # Option 1: Quick wink in scroll zones
                        if wink_duration < RIGHT_WINK_HOLD_TIME and scroll_available:
                            cursor_y = pyautogui.position().y
                            scroll_zone = screen_height * SCROLL_REGION_PERCENTAGE
                            
                            if cursor_y < scroll_zone:
                                pyautogui.scroll(SCROLL_AMOUNT)
                                log_action(f"⬆️ SCROLL UP - Cursor at top ({cursor_y:.0f}px)")
                                last_scroll_time = current_time
                            elif cursor_y > screen_height - scroll_zone:
                                pyautogui.scroll(-SCROLL_AMOUNT)
                                log_action(f"⬇️ SCROLL DOWN - Cursor at bottom ({cursor_y:.0f}px)")
                                last_scroll_time = current_time
                        
                        # Option 2: Hold wink for scroll anywhere
                        elif wink_duration >= RIGHT_WINK_HOLD_TIME and scroll_available:
                            # Scroll based on vertical gaze position
                            if avg_iris_y < 0.45:  # Looking up
                                pyautogui.scroll(SCROLL_AMOUNT)
                                log_action(f"⬆️ SCROLL UP (Held wink + looking up)")
                            elif avg_iris_y > 0.55:  # Looking down
                                pyautogui.scroll(-SCROLL_AMOUNT)
                                log_action(f"⬇️ SCROLL DOWN (Held wink + looking down)")
                            last_scroll_time = current_time
                    
                    # Release right wink
                    if not right_wink_now and is_right_winking:
                        wink_duration = current_time - right_wink_start_time
                        log_action(f"Right wink END (duration: {wink_duration:.2f}s)")
                        is_right_winking = False
                    
                    # GAZE CONTROL
                    left_iris = get_iris_center(landmarks, LEFT_IRIS, w, h)
                    right_iris = get_iris_center(landmarks, RIGHT_IRIS, w, h)
                    
                    l_eye_l = (landmarks[LEFT_EYE_CORNERS[0]].x * w, landmarks[LEFT_EYE_CORNERS[0]].y * h)
                    l_eye_r = (landmarks[LEFT_EYE_CORNERS[1]].x * w, landmarks[LEFT_EYE_CORNERS[1]].y * h)
                    r_eye_l = (landmarks[RIGHT_EYE_CORNERS[0]].x * w, landmarks[RIGHT_EYE_CORNERS[0]].y * h)
                    r_eye_r = (landmarks[RIGHT_EYE_CORNERS[1]].x * w, landmarks[RIGHT_EYE_CORNERS[1]].y * h)
                    
                    epsilon = 1e-6
                    left_gaze = (left_iris[0] - l_eye_l[0]) / (l_eye_r[0] - l_eye_l[0] + epsilon)
                    right_gaze = (right_iris[0] - r_eye_l[0]) / (r_eye_r[0] - r_eye_l[0] + epsilon)
                    avg_gaze = (left_gaze + right_gaze) / 2
                    
                    avg_iris_y = ((left_iris[1] + right_iris[1]) / 2) / h
                    
                    # Apply head correction
                    yaw_correction = get_head_yaw_correction(landmarks, w, h)
                    avg_gaze -= yaw_correction * YAW_STRENGTH
                    
                    # Map to screen
                    mapped_x = np.interp(avg_gaze, [H_RANGE_MIN, H_RANGE_MAX], [0, screen_width])
                    mapped_y = np.interp(avg_iris_y, [V_RANGE_MIN, V_RANGE_MAX], [0, screen_height])
                    mapped_x = screen_width - mapped_x
                    
                    # Apply sensitivity and smoothing
                    delta_x = (mapped_x - screen_center_x) * SENSITIVITY
                    delta_y = (mapped_y - screen_center_y) * SENSITIVITY * 1.15
                    
                    target_x = screen_center_x + delta_x
                    target_y = screen_center_y + delta_y
                    
                    smoothed_x = int(prev_x + ALPHA * (target_x - prev_x))
                    smoothed_y = int(prev_y + ALPHA * (target_y - prev_y))
                    
                    # Constrain to screen
                    final_x = max(0, min(screen_width - 1, smoothed_x))
                    final_y = max(0, min(screen_height - 1, smoothed_y))
                    
                    # Apply dead zone
                    move_distance = np.sqrt((final_x - prev_x)**2 + (final_y - prev_y)**2)
                    
                    if move_distance > MOVEMENT_DEAD_ZONE_PIXELS:
                        pyautogui.moveTo(final_x, final_y)
                        prev_x, prev_y = final_x, final_y
                    
                    # BLINK DETECTION
                    eyes_closed = (left_ear < blink_threshold) and (right_ear < blink_threshold)
                    
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
                                    log_action("🖱️ LEFT CLICK (Double blink)")
                                elif blink_counter >= 3:
                                    pyautogui.click(button='right')
                                    log_action("🖱️ RIGHT CLICK (Triple blink)")
                            
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
        print("\n⛔ Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if is_transcribing[0]:
            is_transcribing[0] = False
            if vosk_thread and vosk_thread.is_alive():
                vosk_thread.join(timeout=2.0)
        
        # Type any remaining text
        while not transcribed_text_queue.empty():
            text = transcribed_text_queue.get()
            pyautogui.write(text + " ")
            log_action(f"Final type: '{text}'")
        
        cap.release()
        face_mesh.close()
        print("\n✅ Cleanup complete")
        print("="*60)

# =========================================================================
# MAIN EXECUTION
# =========================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("👁️ ENHANCED EYE TRACKING WITH VOSK PRELOADING")
    print("="*60)
    
    if vosk_preload_success:
        print("✅ Model preloaded - transcription will start instantly!")
    else:
        print("⚠️  Speech recognition unavailable")
    
    print("\n🚀 Starting in 3 seconds...")
    time.sleep(3)
    
    start_unified_control()
