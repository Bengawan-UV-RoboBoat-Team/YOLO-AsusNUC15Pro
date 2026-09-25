"""Open a camera and show the live feed, cropped to an adjustable aspect ratio.

    .venv/bin/python scripts/open_camera.py --source 0 --ratio 16:9
    .venv/bin/python scripts/open_camera.py --source 2 --width 1280 --height 720 --fourcc MJPG --ratio 1:1

Keys: r = next ratio, f = toggle fill (crop) / fit (black bars), s = save snapshot, q / Esc = quit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

RATIOS = ["native", "16:9", "4:3", "3:2", "1:1", "9:16"]


def parse_ratio(text: str) -> float | None:
    if text == "native":
        return None
    w, h = text.split(":")
    return float(w) / float(h)


def apply_ratio(frame: np.ndarray, ratio: float | None, fill: bool) -> np.ndarray:
    """Center-crop (fill) or letterbox (fit) the frame to the given width/height ratio."""
    if ratio is None:
        return frame
    h, w = frame.shape[:2]
    if fill:
        if w / h > ratio:
            new_w = int(round(h * ratio))
            x0 = (w - new_w) // 2
            return frame[:, x0 : x0 + new_w]
        new_h = int(round(w / ratio))
        y0 = (h - new_h) // 2
        return frame[y0 : y0 + new_h, :]
    if w / h > ratio:
        out_w, out_h = w, int(round(w / ratio))
    else:
        out_w, out_h = int(round(h * ratio)), h
    canvas = np.zeros((out_h, out_w, 3), dtype=frame.dtype)
    x0, y0 = (out_w - w) // 2, (out_h - h) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = frame
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a camera with an adjustable aspect ratio")
    parser.add_argument("--source", default="0", help="Camera index (0, 2, ...) or video file path")
    parser.add_argument("--ratio", default="native", help=f"Aspect ratio W:H, e.g. {', '.join(RATIOS)}")
    parser.add_argument("--fit", action="store_true", help="Letterbox with black bars instead of cropping")
    parser.add_argument("--width", type=int, default=None, help="Camera frame width to request")
    parser.add_argument("--height", type=int, default=None, help="Camera frame height to request")
    parser.add_argument("--fps", type=float, default=None, help="Camera frame rate to request")
    parser.add_argument("--fourcc", type=str, default=None, help="Camera pixel format to request, e.g. MJPG")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale the preview window, e.g. 0.5")
    args = parser.parse_args()

    ratio_names = RATIOS if args.ratio in RATIOS else [args.ratio, *RATIOS]
    ratio_idx = ratio_names.index(args.ratio)
    ratio = parse_ratio(args.ratio)
    fill = not args.fit

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"can't open source '{args.source}'")
    if args.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera {args.source}: {cam_w}x{cam_h} @ {cap.get(cv2.CAP_PROP_FPS):.1f} fps requested/reported")

    window = f"camera {args.source}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("no frame (camera disconnected or video ended)")
                break

            view = apply_ratio(frame, ratio, fill)
            if args.scale != 1.0:
                view = cv2.resize(view, None, fx=args.scale, fy=args.scale)
            cv2.imshow(window, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                ratio_idx = (ratio_idx + 1) % len(ratio_names)
                ratio = parse_ratio(ratio_names[ratio_idx])
            elif key == ord("f"):
                fill = not fill
            elif key == ord("s"):
                out = Path("results/snapshots") / f"cam{args.source}_{time.strftime('%Y%m%d_%H%M%S')}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), apply_ratio(frame, ratio, fill))
                print(f"saved {out}")
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
