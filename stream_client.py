"""
Webcam stream client for Person Detection API.

Run this on any machine with a webcam on the same LAN.
Captures frames, sends to the API, displays annotated results.

Requirements (on the client machine):
  pip install opencv-python requests

Usage:
  python stream_client.py --api http://192.168.1.100:8000
  python stream_client.py --api http://192.168.1.100:8000 --camera 1
  python stream_client.py --api http://192.168.1.100:8000 --delay 2
"""

import argparse
import sys
import time

import cv2
import numpy as np
import requests


def main():
    parser = argparse.ArgumentParser(description="Webcam stream client for Person Detection API")
    parser.add_argument("--api", required=True, help="API base URL (e.g. http://192.168.1.100:8000)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between detections (default: 1.0)")
    args = parser.parse_args()

    api_url = args.api.rstrip("/")

    # Check API health
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        health = r.json()
        if not health.get("model_loaded"):
            print("ERROR: API model not loaded")
            sys.exit(1)
        print(f"Connected to API: {api_url}")
    except Exception as e:
        print(f"ERROR: Cannot reach API at {api_url}: {e}")
        sys.exit(1)

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        sys.exit(1)

    print(f"Camera {args.camera} opened. Press 'q' to quit.")
    print(f"Detection interval: {args.delay}s")

    last_annotated = None
    last_result = None
    last_detect_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        now = time.time()

        # Send frame for detection at the configured interval
        if now - last_detect_time >= args.delay:
            last_detect_time = now

            # Encode frame as JPEG
            _, buf = cv2.imencode(".jpg", frame)
            files = {"file": ("frame.jpg", buf.tobytes(), "image/jpeg")}

            try:
                # Get annotated image
                img_resp = requests.post(f"{api_url}/detect/image", files=files, timeout=10)
                if img_resp.status_code == 200:
                    img_arr = np.frombuffer(img_resp.content, np.uint8)
                    last_annotated = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

                # Get JSON result (re-encode since files was consumed)
                _, buf2 = cv2.imencode(".jpg", frame)
                files2 = {"file": ("frame.jpg", buf2.tobytes(), "image/jpeg")}
                json_resp = requests.post(f"{api_url}/detect", files=files2, timeout=10)
                if json_resp.status_code == 200:
                    last_result = json_resp.json()
                    label = last_result["class"]
                    conf = last_result["confidence"]
                    detected = last_result["detected"]
                    status = "DETECTED" if detected else "---"
                    print(f"  [{status}] {label} ({conf:.0%})")

            except requests.exceptions.RequestException as e:
                print(f"  API error: {e}")

        # Show the latest annotated frame (or raw frame if no result yet)
        display = last_annotated if last_annotated is not None else frame
        cv2.imshow("Person Detection", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
