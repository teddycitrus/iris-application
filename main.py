"""
Enhanced Eye Tracking System with Speech Recognition and Console Feedback
========================================================================
Features: Gaze control, blink clicks, speech transcription.
HEADLESS MODE: No OpenCV window is displayed. All feedback is printed to the console.

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

# --- Gaze Landmark Indices ---
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

# --- Head Pose Estimation Setup ---
MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),            # Nose tip (1)
    (-225.0, 170.0, -135.0),    # Left eye left corner (226)
    (225.0, 170.0, -135.0),     # Right eye right corner (446)
    (-150.0, -150.0, -125.0),   # Left mouth corner (57)
    (150.0, -150.0, -125.0),    # Right mouth corner (287)
    (0.0, -330.0, -65.0)        # Chin (152)
], dtype=np.float32)

LANDMARK_POSE_INDICES = [1, 226, 446, 57, 287, 152]

# --- B. CLICK/WINK CONTROL PARAMETERS ---
EAR_WINK_THRESHOLD = 0.18     # Max EAR for a closed eye during a wink
WINK_OPEN_THRESHOLD = 0.30    # Min EAR for the non-winking eye to be considered open
CLICK_COOLDOWN = 1.0          
BLINK_TIMEOUT = 1.25          # Timeout to interpret a sequence of blinks

# --- ADAPTIVE BLINK DETECTION PARAMETERS ---
# New Parameters for dynamic thresholding based on average eye open state.
EAR_CALIBRATION_ALPHA = 0.05  # Smoothing factor for open EAR baseline (0.05 = slow adaptation)
EAR_CLOSE_FACTOR = 0.70       # Blink threshold is calculated as 70% of the open baseline EAR.

# --- Click/Wink Landmark Indices ---
RIGHT_EYE_INDICES = [362, 263, 385, 386, 374, 380] 
LEFT_EYE_INDICES = [33, 133, 160, 159, 145, 163]

# --- C. VOSK/SPEECH CONTROL PARAMETERS ---
MOUTH_OPEN_THRESHOLD = 0.1
MOUTH_HOLD_TIME = 1.5         # Seconds mouth must be held open

# --- D. SCROLL CONTROL PARAMETERS (Right Wink based) ---
SCROLL_REGION_PERCENTAGE = 0.20 # Top/bottom 20% of the screen triggers scroll
SCROLL_AMOUNT = 50              # Units to scroll 
SCROLL_COOLDOWN = 0.5           # Cooldown in seconds between scrolls

# --- E. CONSOLE LOGGING (Replaces OSD) ---
# OSD parameters removed.

# --- VOSK Setup ---
MOUTH_INDICES = [13, 14, 61, 291]  # Mouth landmarks for MAR calculation
transcribed_text_queue = [] 
vosk_thread = None
vosk_available = False
vosk_model_path_ok = True # Assume OK until checked

try:
    from vosk import Model, KaldiRecognizer
    import sounddevice as sd
    
    # Check if the model path is valid immediately
    MODEL_PATH = "vosk-model-en-us-0.22" 
    if not Path(MODEL_PATH).exists():
         vosk_available = False
         vosk_model_path_ok = False
         
    else:
        vosk_available = True
        print("✓ Vosk speech recognition available (Model path verified)")
        
except ImportError:
    vosk_available = False
    print("⚠ Vosk not installed. Speech transcription will be disabled.")
    print("  To enable: pip install vosk sounddevice")

# --- CONSOLE LOGGING HELPER FUNCTION (Replaces OSD) ---

def log_action(message):
    """Prints a message to the console with a timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] ⚙️ ACTION: {message}")

# --- ORIGINAL HELPER FUNCTIONS (Retained and simplified) ---

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
    
    # Assuming standard camera parameters for simplicity
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
        # Use yaw for horizontal stabilization
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
            np.hstack((rotation_matrix, translation_vector))
        )
        yaw = euler_angles[2][0]
        return (yaw / 40.0) * 0.05
    return 0.0

def eye_aspect_ratio(eye_landmarks):
    """Calculate Eye Aspect Ratio for blink and wink detection."""
    # Vertical distances between inner and outer eye points
    A = dist.euclidean(eye_landmarks[2], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[3], eye_landmarks[4])
    # Horizontal distance between corner points
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[1])
    
    if C == 0:
        return 0.5 # Default open value to avoid division by zero
        
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
    """Extract 2D coordinates for any set of landmarks."""
    points = []
    for i in indices:
        x = int(landmarks[i].x * w)
        y = int(landmarks[i].y * h)
        points.append((x, y))
    return np.array(points)

def vosk_listener_thread(transcribing_flag, transcribed_text_queue):
    """
    Background thread for VOSK speech recognition.
    """
    if not vosk_available:
        return
    
    try:
        # --- Vosk Setup ---
        MODEL_PATH = "vosk-model-en-us-0.22" 
        SAMPLE_RATE = 16000
        BLOCK_SIZE = 8000 
        
        # Load model (must be loaded inside the thread that uses it)
        model = Model(MODEL_PATH)
        recognizer = KaldiRecognizer(model, SAMPLE_RATE)

        # Audio stream callback
        def callback(indata, frames, time_obj, status):
            if status:
                pass
            
            if transcribing_flag[0]:
                if recognizer.AcceptWaveform(indata.tobytes()):
                    result = json.loads(recognizer.Result())
                    text = result.get('text', '').strip()
                    if text:
                        # CRITICAL DEBUG: Log transcribed text before queuing
                        print(f"[{time.strftime('%H:%M:%S')}] 🔊 VOSK RAW: '{text}'")
                        transcribed_text_queue.append(text)

        # Start audio stream
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', 
                             blocksize=BLOCK_SIZE, callback=callback):
            # Loop while the main thread flag is true
            while transcribing_flag[0]:
                sd.sleep(100)
                
        # Get final transcription result if the stream ended gracefully
        final_result = json.loads(recognizer.FinalResult())
        text = final_result.get('text', '').strip()
        if text:
            transcribed_text_queue.append(text)

        log_action("Listening stopped.")

    except Exception as e:
        print(f"[VOSK ERROR] An exception occurred in the listening thread: {e}")
        # If an error happens here (e.g., sound device failure), stop the transcription loop
        transcribing_flag[0] = False 

# --- MAIN CONTROL FUNCTION ---

def start_unified_control():
    """Main loop for unified gaze, click, and speech control."""
    global vosk_thread
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # --- State Variables ---
    prev_x, prev_y = screen_center_x, screen_center_y
    last_click_time = time.time()
    last_scroll_time = time.time()
    blink_counter = 0
    last_blink_time = time.time()
    is_eye_closed = False
    
    # Adaptive Blink State (Initialized to a safe value)
    ear_open_baseline = 0.35 

    # Wink/Speech State
    is_left_winking = False
    is_right_winking = False 
    is_mouth_open = False
    mouth_open_start_time = 0.0
    is_transcribing = [False] # Flag shared with VOSK thread
    
    # Initialize cursor position
    pyautogui.moveTo(prev_x, prev_y)

    print("\n" + "="*60)
    print("👁️ UNIFIED EYE TRACKING CONTROL SYSTEM (HEADLESS)")
    print("="*60)
    print("\n📋 Features:")
    print("  • Gaze Control: Move eyes to control cursor")
    print("  • Double Blink: Left Click (Adaptive)")
    print("  • Triple Blink: Right Click (Adaptive)")
    print("  • Left Wink: Alt + Tab (Window Switch)")
    print("  • Right Wink near Top/Bottom: Scroll Up/Down")
    print("  • Hold Mouth Open 1.5s: Toggle Speech Transcription (Stricter)")
    print("\n🎤 Speech Recognition:", "AVAILABLE" if vosk_available else "NOT AVAILABLE")
    if not vosk_model_path_ok:
        print("🚨 VOSK WARNING: Model folder 'vosk-model-en-us-0.22' not found! Typing will not work.")
    print("💻 Visual Feedback: DISABLED (Check console logs)")
    print(f"⏱️ Blink Timeout: {BLINK_TIMEOUT}s")
    print("\nPress Ctrl+C to exit")
    print("="*60 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01) # Small pause if frame read fails
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            results = face_mesh.process(frame_rgb)

            current_time = time.time()
            click_available = (current_time - last_click_time) > CLICK_COOLDOWN
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # --- 1. EAR and WINK DETECTION ---
                left_eye_pts = extract_landmarks_points(landmarks, LEFT_EYE_INDICES, w, h)
                right_eye_pts = extract_landmarks_points(landmarks, RIGHT_EYE_INDICES, w, h)
                
                left_ear = eye_aspect_ratio(left_eye_pts)
                right_ear = eye_aspect_ratio(right_eye_pts)
                
                avg_ear = (left_ear + right_ear) / 2

                # ADAPTIVE BLINK BASELINE UPDATE
                # Only update the baseline if the eyes are reasonably open (above the absolute wink threshold)
                if avg_ear > EAR_WINK_THRESHOLD:
                    ear_open_baseline = ear_open_baseline * (1 - EAR_CALIBRATION_ALPHA) + avg_ear * EAR_CALIBRATION_ALPHA
                
                # Calculate the dynamic blink threshold
                current_blink_threshold = ear_open_baseline * EAR_CLOSE_FACTOR

                # Check for Left Wink (Alt+Tab)
                is_left_wink_now = (left_ear < EAR_WINK_THRESHOLD) and (right_ear > WINK_OPEN_THRESHOLD)
                
                if is_left_wink_now and not is_left_winking and click_available:
                    pyautogui.hotkey('alt', 'tab')
                    log_action("ALT + TAB (Left Wink)")
                    last_click_time = current_time
                    is_left_winking = True
                    blink_counter = 0 # Prevent click during wink
                
                if not is_left_wink_now:
                    is_left_winking = False

                # --- 2. RIGHT WINK SCROLL CONTROL ---
                scroll_available = (current_time - last_scroll_time) > SCROLL_COOLDOWN
                is_right_wink_now = (right_ear < EAR_WINK_THRESHOLD) and (left_ear > WINK_OPEN_THRESHOLD)

                if is_right_wink_now and not is_right_winking and scroll_available:
                    
                    # Calculate scroll region based on current cursor position
                    scroll_threshold_y = screen_height * SCROLL_REGION_PERCENTAGE
                    current_cursor_y = pyautogui.position().y # Use actual cursor Y for region check

                    if current_cursor_y < scroll_threshold_y:
                        pyautogui.scroll(SCROLL_AMOUNT)
                        log_action("⬆️ SCROLL UP (Right Wink)")
                        last_scroll_time = current_time
                    
                    elif current_cursor_y > screen_height - scroll_threshold_y:
                        pyautogui.scroll(-SCROLL_AMOUNT)
                        log_action("⬇️ SCROLL DOWN (Right Wink)")
                        last_scroll_time = current_time
                    
                    is_right_winking = True
                    blink_counter = 0 # Prevent click during wink

                if not is_right_wink_now:
                    is_right_winking = False
                
                # --- 3. MOUTH OPEN DETECTION (Speech Toggle) ---
                if vosk_available:
                    mouth_pts = extract_landmarks_points(landmarks, MOUTH_INDICES, w, h)
                    mar = mouth_aspect_ratio(mouth_pts)
                    
                    is_mouth_open_now = mar > MOUTH_OPEN_THRESHOLD
                    
                    # Log when mouth open is initially detected (for debugging sensitivity)
                    if is_mouth_open_now and not is_mouth_open:
                        is_mouth_open = True
                        mouth_open_start_time = current_time
                        log_action(f"Mouth opened (MAR: {mar:.4f}). Hold for {MOUTH_HOLD_TIME}s to toggle transcription...")
                    
                    elif is_mouth_open_now and is_mouth_open and click_available:
                        # Mouth held open for required time
                        if current_time - mouth_open_start_time >= MOUTH_HOLD_TIME:
                            
                            if not is_transcribing[0]:
                                # ACTIVATE VOSK
                                is_transcribing[0] = True
                                vosk_thread = threading.Thread(target=vosk_listener_thread, 
                                                               args=(is_transcribing, transcribed_text_queue))
                                vosk_thread.daemon = True
                                vosk_thread.start()
                                log_action("🎤 Transcription ENABLED")
                                
                            else:
                                # DEACTIVATE VOSK
                                is_transcribing[0] = False
                                # Wait briefly for the thread to process final input before joining
                                if vosk_thread and vosk_thread.is_alive():
                                    vosk_thread.join(timeout=0.5)
                                log_action("🔇 Transcription DISABLED")

                            last_click_time = current_time
                            is_mouth_open = False
                            
                    elif not is_mouth_open_now and is_mouth_open:
                        # Mouth closed before hold time
                        is_mouth_open = False

                # --- 4. GAZE CONTROL & CURSOR MOVEMENT ---
                left_iris_center = get_iris_center(landmarks, LEFT_IRIS, w, h)
                right_iris_center = get_iris_center(landmarks, RIGHT_IRIS, w, h)
                l_eye_l_corner = (landmarks[LEFT_EYE_CORNERS[0]].x * w, landmarks[LEFT_EYE_CORNERS[0]].y * h)
                l_eye_r_corner = (landmarks[LEFT_EYE_CORNERS[1]].x * w, landmarks[LEFT_EYE_CORNERS[1]].y * h)
                r_eye_l_corner = (landmarks[RIGHT_EYE_CORNERS[0]].x * w, landmarks[RIGHT_EYE_CORNERS[0]].y * h)
                r_eye_r_corner = (landmarks[RIGHT_EYE_CORNERS[1]].x * w, landmarks[RIGHT_EYE_CORNERS[1]].y * h)
                
                epsilon = 1e-6 # Small value to prevent division by zero
                
                # Calculate horizontal gaze ratio for both eyes
                left_eye_gaze_ratio = (left_iris_center[0] - l_eye_l_corner[0]) / (l_eye_r_corner[0] - l_eye_l_corner[0] + epsilon)
                right_eye_gaze_ratio = (right_iris_center[0] - r_eye_l_corner[0]) / (r_eye_r_corner[0] - r_eye_l_corner[0] + epsilon)
                avg_gaze_ratio = (left_eye_gaze_ratio + right_eye_gaze_ratio) / 2
                
                # Calculate vertical gaze ratio (using iris center Y divided by frame height)
                avg_iris_y_ratio = ((left_iris_center[1] + right_iris_center[1]) / 2) / h
                
                # Apply head pose correction for stability
                yaw_correction_base = get_head_yaw_correction(landmarks, w, h)
                avg_gaze_ratio -= yaw_correction_base * YAW_STRENGTH
                
                # Map gaze ratios to screen coordinates
                mapped_x = np.interp(avg_gaze_ratio, [H_RANGE_MIN, H_RANGE_MAX], [0, screen_width])
                mapped_y = np.interp(avg_iris_y_ratio, [V_RANGE_MIN, V_RANGE_MAX], [0, screen_height])
                mapped_x = screen_width - mapped_x # Invert X for natural screen mapping
                
                # Calculate displacement from screen center
                delta_x = mapped_x - screen_center_x
                delta_y = mapped_y - screen_center_y
                
                # Apply sensitivity
                screen_x = screen_center_x + delta_x * SENSITIVITY
                screen_y = screen_center_y + delta_y * SENSITIVITY
                
                # Apply smoothing
                smoothed_x = int(prev_x + ALPHA * (screen_x - prev_x))
                smoothed_y = int(prev_y + ALPHA * (screen_y - prev_y))
                
                # Apply slight vertical scaling (adjust as needed for monitor size/curve)
                vertical_scale = 1.15
                final_y = int((smoothed_y - screen_center_y) * vertical_scale + screen_center_y)
                
                # Clamp coordinates to screen boundaries
                final_y = max(0, min(screen_height - 1, final_y))
                final_x = max(0, min(screen_width - 1, smoothed_x))
                
                pyautogui.moveTo(final_x, final_y)
                prev_x, prev_y = final_x, final_y

                # --- 5. VOSK OUTPUT & TYPING ---
                if transcribed_text_queue:
                    text_to_type = transcribed_text_queue.pop(0)
                    pyautogui.typewrite(text_to_type + " ")
                    print(f"[{time.strftime('%H:%M:%S')}] 💬 Typed: '{text_to_type}'")

                # --- 6. BLINK SEQUENCE CHECK (Click) ---
                eyes_closed = (left_ear < current_blink_threshold) and (right_ear < current_blink_threshold)
                
                # Only count blinks if no wink is actively being held.
                if not is_left_winking and not is_right_winking:
                    if eyes_closed and not is_eye_closed:
                        blink_counter += 1
                        last_blink_time = current_time
                        is_eye_closed = True
                        # Log adaptive threshold for debugging
                        log_action(f"Blink registered (Count: {blink_counter}, Threshold: {current_blink_threshold:.3f})") 
                    
                    if not eyes_closed and is_eye_closed:
                        is_eye_closed = False
                
                # Check for click commands after blink timeout
                if blink_counter > 0:
                    time_since_last_blink = current_time - last_blink_time
                    
                    if time_since_last_blink > BLINK_TIMEOUT:
                        if click_available:
                            if blink_counter == 2:
                                pyautogui.click(button='left')
                                log_action("🖱️ LEFT CLICK (Double Blink) - Success")
                                last_click_time = current_time
                            elif blink_counter >= 3:
                                pyautogui.click(button='right')
                                log_action("🖱️ RIGHT CLICK (Triple Blink) - Success")
                                last_click_time = current_time
                            else:
                                # This handles the single-blink timeout, logging the reset explicitly.
                                log_action(f"Sequence Timeout: {blink_counter} blink(s) detected, resetting counter.")
                        
                        # Always reset the counter after the timeout check, regardless of action
                        blink_counter = 0

            # Since the window is gone, we yield CPU time with a small sleep
            time.sleep(0.001) 

    except KeyboardInterrupt:
        print("\n\n⛔ Script stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ A critical error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if is_transcribing[0]:
            is_transcribing[0] = False
            print("🔇 Stopping transcription thread...")
            if vosk_thread and vosk_thread.is_alive():
                vosk_thread.join(timeout=1.0)
        
        cap.release()
        face_mesh.close()
        print("✅ Webcam released. Unified control terminated.")
        print("="*60)

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" "*10 + "👁️ ENHANCED EYE TRACKING CONTROL SYSTEM")
    print(" "*12 + "Running Headless with Console Feedback")
    print("="*60)
    
    print("\n🚀 Starting in 3 seconds...")
    time.sleep(3)
    
    start_unified_control()
