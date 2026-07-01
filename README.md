# Iris - Eye Tracking Accessibility Tool

> Hands-free computer control using eye gaze, blinks, winks, and voice, with optional hand-gesture enhancements. Built at Hack The Valley X at UTSC.
> Winner: Future Impact Category, 4th Place Overall.

---

## About The Project

Iris is a Python accessibility tool that lets you control your computer without a mouse. Using a standard webcam it tracks your eye gaze to move the cursor, detects blinks and winks as clicks, and activates speech-to-text by mouth detection, all in real time with no OpenCV window required.

The gaze engine was reworked to be smooth and accurate. It uses a distance-invariant signal that fuses head pose and iris position (both divided by the inter-eye width so it does not matter how close you sit), a One Euro Filter for velocity-adaptive smoothing (steady when you hold still, responsive when you move fast), a magnitude clamp so a single bad frame cannot fling the cursor, and a blink freeze so blinking to click does not jerk the aim.

Everything runs from a single `main.py`. The eye-tracking path is fully hands-free; the hand-gesture controls are optional extras that only activate when your hands are visible.

**Core capabilities:**

- Gaze-driven cursor movement with head-pose fusion and One Euro smoothing
- Double blink for left click, triple blink for right click
- Left wink for Alt+Tab, right wink at screen edges to scroll
- Hold mouth open 1.5s to toggle speech recognition (VOSK, preloaded)
- Global hotkeys to enable/disable and quit from anywhere
- Optional hand gestures: pinch to click and drag, thumb-index gap to set cursor speed, prayer hands to pause everything
- All inputs are locked automatically during transcription and typing

---

## Built With

[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![PyAutoGUI](https://img.shields.io/badge/PyAutoGUI-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://pyautogui.readthedocs.io/)
[![VOSK](https://img.shields.io/badge/VOSK-FF6F00?style=for-the-badge&logo=audio&logoColor=white)](https://alphacephei.com/vosk/)
[![SoundDevice](https://img.shields.io/badge/SoundDevice-4CAF50?style=for-the-badge&logo=python&logoColor=white)](https://python-sounddevice.readthedocs.io/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

---

## Getting Started

### Prerequisites

- A working webcam.
- **Python 3.12.** MediaPipe does not ship wheels for Python 3.13/3.14 yet, so a 3.12 environment is required. Your system default can stay on a newer version; the steps below pin this project to 3.12 in a virtual environment.

### Installation

From a terminal in the project folder:

```sh
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install "mediapipe==0.10.18" pyautogui scipy sounddevice vosk keyboard
```

Notes on the pinned install:

- `mediapipe==0.10.18` is used because newer builds have dropped the legacy `solutions` face-mesh API this project relies on. It automatically pulls in a compatible `numpy<2` and `opencv-contrib-python`.
- Do NOT also install `opencv-python`. It conflicts with the `opencv-contrib-python` that MediaPipe provides; having both breaks `cv2`.
- `keyboard` is optional and only enables the global on/off hotkeys. Without it the app still runs and prints a note.

### VOSK Speech Model (optional)

Speech dictation needs a VOSK model folder in the project root. Without it, everything else runs and speech simply reports as unavailable.

Small model (about 40 MB, good for testing). From the project folder:

```sh
curl -L -o vosk.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
tar -xf vosk.zip
ren vosk-model-small-en-us-0.15 vosk-model-en-us-0.22
del vosk.zip
```

For full accuracy, download `vosk-model-en-us-0.22` from https://alphacephei.com/vosk/models and extract it into the project root (no rename needed).

### Run

```sh
python main.py
```

It prints system info, waits 3 seconds, then starts. A `[perf] ~NN FPS` line prints every couple of seconds so you can confirm the loop rate. Press `Ctrl+C` in the terminal to stop (or `Ctrl+Alt+Q` if `keyboard` is installed).

---

## Usage

| Input | Action |
|---|---|
| Eye gaze | Move cursor |
| Double blink | Left click |
| Triple blink | Right click |
| Left wink | Alt+Tab |
| Right wink (cursor at top/bottom 30%) | Scroll |
| Hold mouth open 1.5s | Toggle speech recognition |
| Hold mouth open again | Stop recording and type |
| Ctrl+Alt+E | Enable/disable all control (needs `keyboard`) |
| Ctrl+Alt+Q | Quit the app (needs `keyboard`) |

Optional hand gestures (require `HAND_TRACKING_ENABLED = True`):

| Input | Action |
|---|---|
| Prayer hands (palms together) | Pause/resume everything |
| Left hand pinch | Click and drag (eye tracking pauses while dragging) |
| Right hand thumb-index gap | Adjust cursor speed (small gap = precise, large = fast) |

All inputs are locked while speech is being transcribed and while text is being typed. Console output gives real-time feedback on every detected action.

---

## Tuning

All knobs are constants near the top of `main.py`:

- `ONE_EURO_MIN_CUTOFF` (default 0.4): lower is steadier when holding still (less jitter); try 0.25 if it is still twitchy.
- `ONE_EURO_BETA` (default 0.06): raise if the cursor feels laggy catching up to fast looks.
- `GAZE_SENSITIVITY_X` / `GAZE_SENSITIVITY_Y` (default 2.0): cursor travel per unit of gaze; raise to reach screen edges with less movement.
- `GAZE_INVERT_X` / `GAZE_INVERT_Y`: flip if the cursor moves the wrong way.
- `HAND_TRACKING_ENABLED` (default True): set to False for the smoothest possible gaze (disables all hand gestures and gives the face mesh the full frame rate).

---

## Optional: run at login

The `scripts/` folder has launchers for an always-on setup:

- `scripts/run_iris.bat` runs the app with a visible console (uses the `.venv`, falls back to `py -3.12`).
- `scripts/start_iris_hidden.vbs` runs it hidden, for the Startup folder. Add a shortcut to it in `shell:startup` (press Win+R, type `shell:startup`) so Iris launches at login and the Ctrl+Alt+E hotkey is always available.

---

## Roadmap

- [x] Distance-invariant head + iris gaze with One Euro smoothing and blink freeze
- [x] Adaptive EAR baseline calibration so blink/wink thresholds self-adjust at runtime
- [x] Input locking during transcription and typing
- [x] Global enable/disable and quit hotkeys
- [ ] Per-user calibration flow to tune gaze range and sensitivity for different faces and lighting
- [ ] macOS / Linux support by replacing the Windows-only notification and cursor calls

---

## Contributing

Contributions are welcome:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Contact

[@teddycitrus](https://github.com/teddycitrus)

Project: [https://github.com/teddycitrus/iris-application](https://github.com/teddycitrus/iris-application)

---

## Acknowledgments

- [Hack The Valley X](https://hackthevalley.io/) for hosting and the Future Impact award
- [MediaPipe Face Mesh](https://ai.google.dev/edge/mediapipe/solutions/guide) for landmark detection
- [VOSK](https://alphacephei.com/vosk/) for offline speech recognition
- [Img Shields](https://shields.io) for badges
