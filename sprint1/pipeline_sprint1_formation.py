"""
Sprint 1 Formation Pipeline: Process dashboard video through the geometry pipeline
and output a metric top-down video plus per-frame images for analysis.
"""

import os
import pickle
import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
VIDEO_PATH = "road_test.mp4"
PIPELINE_PATH = "geometry_pipeline_video.pkl"
OUTPUT_FRAMES_DIR = "sprint1_frames"

# Square canvas so exported frames show the full bird's-eye view (not cropped)
CANVAS_SIZE = 2000
CANVAS_WIDTH = CANVAS_SIZE
CANVAS_HEIGHT = CANVAS_SIZE
# Translation scaled from original 4000px canvas (1500 * 2000/4000)
SHIFT_X = 750
SHIFT_Y = 0

# Export a frame image only every Nth video frame (1 = every frame, 2 = every 2nd, etc.)
FRAME_EXPORT_EVERY = 2

# Scale overlay: 100 px line represents 10 cm
SCALE_LINE_LENGTH_PX = 100
SCALE_LABEL = "10 cm (Scale)"


def main():
    # -------------------------------------------------------------------------
    # Load geometry pipeline
    # -------------------------------------------------------------------------
    if not os.path.isfile(PIPELINE_PATH):
        print(f"Error: '{PIPELINE_PATH}' not found.")
        return

    with open(PIPELINE_PATH, "rb") as f:
        data = pickle.load(f)

    camera_matrix = data["camera_matrix"]
    dist_coeff = data["dist_coeff"]
    homography_matrix = data["homography_matrix"]
    print(f"Loaded geometry pipeline from '{PIPELINE_PATH}'.")

    # -------------------------------------------------------------------------
    # Build final warp matrix: translation then homography
    # -------------------------------------------------------------------------
    translation = np.array([
        [1, 0, SHIFT_X],
        [0, 1, SHIFT_Y],
        [0, 0, 1],
    ], dtype=np.float64)
    H_final = translation @ homography_matrix

    # -------------------------------------------------------------------------
    # Open video and prepare output
    # -------------------------------------------------------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video '{VIDEO_PATH}'.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None  # Unknown length

    # Create output folder for frames if it does not exist
    os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)
    print(f"Output frames will be saved to '{OUTPUT_FRAMES_DIR}/' (no video export).")

    # No video export: frames only
    out = None

    # -------------------------------------------------------------------------
    # Process each frame
    # -------------------------------------------------------------------------
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Undistort using camera matrix and distortion coefficients
        undistorted = cv2.undistort(
            frame, camera_matrix, dist_coeff, None, camera_matrix
        )

        # Warp to top-down view on the large canvas
        warped = cv2.warpPerspective(
            undistorted,
            H_final,
            (CANVAS_WIDTH, CANVAS_HEIGHT),
        )

        # Draw scale reference: 100 px red line = 10 cm
        scale_x1, scale_y = 100, 100
        scale_x2 = scale_x1 + SCALE_LINE_LENGTH_PX
        cv2.line(
            warped,
            (scale_x1, scale_y),
            (scale_x2, scale_y),
            (0, 0, 255),
            10,
        )
        cv2.putText(
            warped,
            SCALE_LABEL,
            (scale_x1, scale_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (0, 0, 255),
            5,
        )

        # Draw frame ID on bottom left
        frame_text = f"Frame {frame_id}"
        cv2.putText(
            warped,
            frame_text,
            (50, CANVAS_HEIGHT - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (255, 255, 255),
            5,
        )

        # Video export disabled
        if out is not None:
            out.write(warped)

        # Save frame as JPEG only every Nth frame to reduce disk usage
        if frame_id % FRAME_EXPORT_EVERY == 1:
            export_count = (frame_id - 1) // FRAME_EXPORT_EVERY + 1
            frame_filename = os.path.join(
                OUTPUT_FRAMES_DIR,
                f"frame_{export_count:03d}.jpg",
            )
            cv2.imwrite(frame_filename, warped)

        # Progress message
        if total_frames is not None:
            print(f"Processing frame {frame_id}/{total_frames}...")
        else:
            print(f"Processing frame {frame_id}...")

    cap.release()
    if out is not None:
        out.release()
    exported_count = (frame_id + FRAME_EXPORT_EVERY - 1) // FRAME_EXPORT_EVERY
    print(f"Done. Frames saved to '{OUTPUT_FRAMES_DIR}/' ({exported_count} images, {CANVAS_SIZE}x{CANVAS_SIZE}, every {FRAME_EXPORT_EVERY}th frame).")


if __name__ == "__main__":
    main()
