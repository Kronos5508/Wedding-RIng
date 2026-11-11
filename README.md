# Spinning Wedding Ring

A Python script that renders a polished gold wedding band slowly rotating as an MP4 animation.

## Requirements

- Python 3.11 or newer
- NumPy and Matplotlib
- [FFmpeg](https://ffmpeg.org/) installed and available on your `PATH` (required for MP4 encoding)

## Installation

Install the Python dependencies from the project folder:

```bash
python -m pip install -r requirements.txt
```

## Usage

Render the default 8-second, 720 × 720 animation at 24 frames per second:

```bash
python wedding_ring.py --output wedding_ring.mp4
```

The video is saved to the current working directory. Choose a different output path or adjust the render settings with these options:

```bash
python wedding_ring.py --output wedding_ring.mp4 --duration 8 --fps 24 --size 720
```

Run `python wedding_ring.py --help` to see all available options.

## Files

- `wedding_ring.py` — animation renderer
- `requirements.txt` — Python package dependencies
- `README.md` — project instructions
