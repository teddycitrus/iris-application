# Iris — Eye Tracking Accessibility Tool

> Hands-free computer control using eye gaze, blinks, winks, and voice — built at Hack The Valley X at UTSC.
> 🏆 Winner: *Future Impact* Category · 4th Place Overall

---

## About The Project

Iris is a Python-based accessibility tool that lets you control your computer entirely without hands. Using a standard webcam, it tracks your eye gaze to move the cursor, detects blinks and winks as click inputs, and activates speech-to-text via mouth detection — all in real time with no OpenCV display window required.

Designed for people with motor impairments, Iris demonstrates that meaningful accessibility tooling can be built from commodity hardware and open-source libraries.

**Core capabilities:**

- Gaze-driven cursor movement with head-pose stabilization and smoothing
- Double blink → left click; triple blink → right click
- Left wink → Alt+Tab (switch windows)
- Right wink → scroll (position or gaze direction)
- Hold mouth open 1.5s → toggle speech recognition (VOSK, preloaded)
- All inputs automatically locked during transcription and typing

---

## Built With

- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- [PyAutoGUI](https://pyautogui.readthedocs.io/)
- [VOSK](https://alphacephei.com/vosk/)
- [SoundDevice](https://python-sounddevice.readthedocs.io/)
- [SciPy](https://scipy.org/)
- [NumPy](https://numpy.org/)

---

## Getting Started

### Prerequisites

Python 3.8+ and a working webcam. Install dependencies:

```sh
pip install opencv-python mediapipe pyautogui numpy scipy sounddevice vosk
```

For Windows toast notifications (optional):

```sh
pip install win10toast
```

### VOSK Speech Model

Speech recognition requires a VOSK language model placed in your project root:

1. Download the English model from [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)
   (recommended: `vosk-model-en-us-0.22`)
2. Extract it so the folder is at `./vosk-model-en-us-0.22/`

### Installation

```sh
git clone https://github.com/teddycitrus/iris-application.git
cd iris-application
pip install opencv-python mediapipe pyautogui numpy scipy sounddevice vosk
# Place the VOSK model folder here
python main.py
```

Iris will start in 3 seconds after printing system info to the console.

---

## Usage

| Input | Action |
|---|---|
| Eye gaze | Move cursor |
| Double blink | Left click |
| Triple blink | Right click |
| Left wink | Alt+Tab |
| Right wink (cursor at top/bottom 30%) | Scroll |
| Right wink + hold + look up/down | Scroll anywhere |
| Hold mouth open 1.5s | Toggle speech recognition |
| Hold mouth open again | Stop recording and type |

All inputs are fully locked while speech is being transcribed and while text is being typed. Console output provides real-time feedback on every detected action.

Press `Ctrl+C` to stop.

---

## Roadmap

- [x] GUI configuration panel for tuning thresholds
- [x] Multi-language VOSK support
- [ ] macOS / Linux compatibility (remove `win10toast` dependency)
- [ ] Packaging as a standalone executable

---

## Contributing

Contributions are welcome. To contribute:

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

- [Hack The Valley X](https://hackthevalley.io/) — for hosting and the Future Impact award
- [MediaPipe Face Mesh](https://mediapipe.dev/) — landmark detection
- [VOSK](https://alphacephei.com/vosk/) — offline speech recognition
- [Img Shields](https://shields.io) — badges
