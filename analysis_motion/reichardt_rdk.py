import sys
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from dotenv import dotenv_values

# ---------------------- Configuration ----------------------
try:
    env_path = sys.argv[1]
except IndexError:
    env_path = "config.env"
env = dotenv_values(env_path)
PATH_VIDEO_DIR = Path(env["PATH_DIR"]) / "data" / "videos"
VIDEO_FILENAME = "arena0_repeat00_scene_overview.mp4"
PATH_SAVE = PATH_VIDEO_DIR / "results"

# Degrees counterclockwise: 0 right, 90 up, 180 left, 270 down.
THETA_DEG = 90.0
N_OFF_TARGET_ANGLES = 7
# The target plus N off-target angles form a uniform circular grid.
# With N=7 the offsets from the target are 45, 90, ..., 315 degrees.

SPACING_PX = 5.0
TAU_S = 0.050
BLUR_SIGMA_PX = 1.0
WARMUP_S = 5 * TAU_S
ROI = None                       # (left, top, right, bottom), or None.
FPS_OVERRIDE = None
SHOW_PLOT = False
SAVE_PLOTS = True

# REQUIRED: pure target-direction motion with matched speed, density, contrast,
# dot size, resolution, ROI. Use an independent clip  to check the correction. 
# Do not calibrate on the mixed-motion videos.
TARGET_REFERENCE_VIDEO = Path(r"C:\Users\Roberto\Desktop\highlights\rdk_variants\data\videos\CoherentDotsFrameBasedRedraw_coh100_left\arena0_repeat00_scene_overview.mp4")
# Set True to process only this reference. Do this when checking calibration residuals.
REFERENCE_CHECK_ONLY = False

# ------------------------- Helpers -------------------------
def save_csv(path, header, rows):
    # Existing files with these names are overwritten.
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def positive_activity(local):
    # Rectify BEFORE spatial averaging: opposing local motions cannot cancel.
    return float(np.maximum(local, 0).mean())


def analyze_video(video_path, angles_deg):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(FPS_OVERRIDE if FPS_OVERRIDE is not None
                else cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Invalid FPS; set FPS_OVERRIDE.")

    decay = np.exp(-1.0 / (fps * TAU_S))
    angles = np.deg2rad(angles_deg)
    delayed = None
    frame_index = 0
    records = []
    print(f"\nVideo: {video_path}\nFPS: {fps:g}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if ROI is not None:
                left, top, right, bottom = ROI
                if not (0 <= left < right <= frame.shape[1]
                        and 0 <= top < bottom <= frame.shape[0]):
                    raise ValueError("ROI is outside the video or has invalid bounds.")
                frame = frame[top:bottom, left:right]

            current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current = current.astype(np.float32) / 255.0
            if BLUR_SIGMA_PX > 0:
                current = cv2.GaussianBlur(current, (0, 0), BLUR_SIGMA_PX)
            current -= current.mean()

            if delayed is None:
                height, width = current.shape
                yy, xx = np.mgrid[:height, :width].astype(np.float32)
                maps = []
                valid = np.ones(current.shape, dtype=bool)
                for angle in angles:
                    map_x = (xx + SPACING_PX * np.cos(angle)).astype(np.float32)
                    map_y = (yy - SPACING_PX * np.sin(angle)).astype(np.float32)
                    maps.append((map_x, map_y))
                    # All directions use exactly the same spatial support.
                    valid &= ((map_x >= 0) & (map_x <= width - 1) 
                              & (map_y >= 0) & (map_y <= height - 1))
                if not valid.any():
                    raise ValueError("Detector spacing is too large for this ROI.")
                delayed = current.copy()
            else:
                if current.shape != delayed.shape:
                    raise ValueError("Video frame dimensions changed.")
                delayed *= decay
                delayed += (1.0 - decay) * current

            time_s = frame_index / fps
            if time_s >= WARMUP_S:
                off_target_activity = []
                for j, (map_x, map_y) in enumerate(maps):
                    shifted = cv2.remap(current, map_x, map_y,
                                        interpolation=cv2.INTER_LINEAR)
                    delayed_shifted = cv2.remap(delayed, map_x, map_y,
                                                interpolation=cv2.INTER_LINEAR)
                    local = (delayed * shifted - current * delayed_shifted)[valid]
                    if j == 0:
                        # Target strength remains SIGNED; SD is unchanged in meaning.
                        strength = float(local.mean())
                        variability = float(local.std(ddof=0))
                    else:
                        off_target_activity.append(positive_activity(local))

                # SUM across directions, not an average across directions.
                noise_metrics = float(np.sum(off_target_activity))
                records.append([time_s, strength, variability, noise_metrics,
                                *off_target_activity])
            frame_index += 1
            if frame_index % max(1, round(10 * fps)) == 0:
                print(f"  Processed {frame_index / fps:.1f} s")
    finally:
        cap.release()

    if not records:
        raise RuntimeError(f"No analyzed frames in {video_path}; check WARMUP_S.")
    return np.asarray(records, dtype=float), fps



def fit_leakage(reference):
    # Fit A_off(phi,t) ~= coefficient(phi) * max(signed_target(t), 0).
    # A_off is already rectified locally and then spatially averaged.
    target = np.maximum(reference[:, 1], 0)
    energy = float(target @ target)
    if energy <= 1e-20:
        raise ValueError("Reference has no usable positive target response. "
                         "Check THETA_DEG and reference motion direction.")
    off = reference[:, 4:4 + N_OFF_TARGET_ANGLES]
    return np.maximum((target @ off) / energy, 0)


def correct_leakage(data, coefficients):
    # Preserve raw activity, then remove the calibrated target contribution.
    raw_off = data[:, 4:4 + N_OFF_TARGET_ANGLES].copy()
    raw_noise = data[:, 3].copy()
    target = np.maximum(data[:, 1], 0)
    expected = target[:, None] * coefficients[None, :]
    corrected = np.maximum(raw_off - expected, 0)
    output = data.copy()
    output[:, 3] = corrected.sum(axis=1)  # Still a SUM, never an angle average.
    output[:, 4:4 + N_OFF_TARGET_ANGLES] = corrected
    # Append raw counterparts for auditing; original column layout is retained.
    return np.column_stack([output, raw_noise, raw_off])


def main():
    if not np.isfinite(THETA_DEG):
        raise ValueError("THETA_DEG must be finite; 0 is a valid direction.")
    if (not isinstance(N_OFF_TARGET_ANGLES, int)
            or isinstance(N_OFF_TARGET_ANGLES, bool)
            or N_OFF_TARGET_ANGLES < 1):
        raise ValueError("N_OFF_TARGET_ANGLES must be a positive integer.")
    if SPACING_PX <= 0 or TAU_S <= 0 or BLUR_SIGMA_PX < 0 or WARMUP_S < 0:
        raise ValueError("Invalid detector settings.")

    angles_deg = (THETA_DEG + np.arange(N_OFF_TARGET_ANGLES + 1)
                  * 360.0 / (N_OFF_TARGET_ANGLES + 1)) % 360.0
    print("Target direction:", angles_deg[0])
    print("Off-target directions:", angles_deg[1:])
    if not TARGET_REFERENCE_VIDEO.is_file():
        raise FileNotFoundError("Set TARGET_REFERENCE_VIDEO to a matched, "
                                "100%-coherent target-motion MP4.")
    reference, reference_fps = analyze_video(TARGET_REFERENCE_VIDEO, angles_deg)
    coefficients = fit_leakage(reference)
    PATH_SAVE.mkdir(parents=True, exist_ok=True)
    save_csv(PATH_SAVE / "leakage_calibration.csv",
             ["off_target_angle_deg", "leakage_coefficient"],
             zip(angles_deg[1:], coefficients))
    reference_corrected = correct_leakage(reference, coefficients)
    print("Reference median raw/corrected noise:",
          np.median(reference[:, 3]), np.median(reference_corrected[:, 3]))
    print("Reference residual need not be zero: this is an approximate correction.")
    if REFERENCE_CHECK_ONLY:
        candidates = [TARGET_REFERENCE_VIDEO.parent]
    else:
        candidates = sorted(PATH_VIDEO_DIR.iterdir())
    runs = []
    max_strength = 0.0
    max_noise = 0.0

    # First pass: process every video and find GLOBAL normalization maxima.
    for folder in candidates:
        if not folder.is_dir() or folder.resolve() == PATH_SAVE.resolve():
            continue
        video_path = TARGET_REFERENCE_VIDEO if REFERENCE_CHECK_ONLY else folder / VIDEO_FILENAME
        if not video_path.is_file():
            continue
        parts = folder.name.split("_")
        label = parts[0]
        try:
            coherence = float(parts[1].replace("coh", ""))
        except (IndexError, ValueError):
            coherence = float("nan")
        if video_path.resolve() == TARGET_REFERENCE_VIDEO.resolve():
            data, fps = reference.copy(), reference_fps
        else:
            data, fps = analyze_video(video_path, angles_deg)
        if not np.isclose(fps, reference_fps):
            raise ValueError("Reference and analysis FPS differ. Use matched FPS.")
        data = correct_leakage(data, coefficients)
        max_strength = max(max_strength, float(np.max(np.abs(data[:, 1]))))
        max_noise = max(max_noise, float(np.max(data[:, 3])))
        runs.append((folder, label, coherence, data, fps))
    if not runs:
        raise RuntimeError("No matching videos found. Check directory and filename.")

    # A zero maximum means an identically zero signal; leave it zero.
    strength_denominator = max_strength if max_strength > 0 else 1.0
    noise_denominator = max_noise if max_noise > 0 else 1.0
    PATH_SAVE.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    off_headers = [f"off_target_{j}_angle_{angle:g}_activity"
                   for j, angle in enumerate(angles_deg[1:], start=1)]

    raw_headers = ["noise_metrics_raw"] + [
        f"off_target_{j}_angle_{angle:g}_activity_raw"
        for j, angle in enumerate(angles_deg[1:], start=1)]
    for folder, label, coherence, data, fps in runs:
        normalized_strength = data[:, 1] / strength_denominator
        normalized_noise = data[:, 3] / noise_denominator
        header = ["time_s", "theta_deg", "strength", "spatial_sd",
                  "noise_metrics", "strength_normalized",
                  "noise_metrics_normalized", *off_headers, *raw_headers]
        rows = ([row[0], THETA_DEG, row[1], row[2], row[3],
                 normalized_strength[k], normalized_noise[k], *row[4:]]
                for k, row in enumerate(data))
        save_csv(folder / "motion_timeseries.csv", header, rows)
        summary_rows.append([
            label, coherence, THETA_DEG,
            float(np.median(data[:, 1])), float(np.median(data[:, 2])),
            float(np.median(data[:, 3])),
            float(np.median(normalized_strength)), float(np.median(normalized_noise)),
            N_OFF_TARGET_ANGLES, max_strength, max_noise, fps, len(data),
            float(np.median(data[:, 4 + N_OFF_TARGET_ANGLES])),
            str(TARGET_REFERENCE_VIDEO),
        ])
        print(f"{folder.name}: median strength={np.median(data[:, 1]):+.6g}, "
              f"median noise_metrics={np.median(data[:, 3]):.6g}")

        if SAVE_PLOTS or SHOW_PLOT:
            fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), sharex=True)
            plots = [(normalized_strength, "Signed strength (normalized)"),
                     (data[:, 2], "Spatial response SD (raw)"),
                     (normalized_noise, "Corrected off-target SUM (normalized)")]
            for ax, (values, title) in zip(axes, plots):
                ax.plot(data[:, 0], values, linewidth=0.9)
                ax.set_title(title)
                ax.set_xlabel("Video time (s)")
                ax.grid(alpha=0.25)
            axes[0].axhline(0, color="black", linewidth=0.6)
            fig.tight_layout()
            if SAVE_PLOTS:
                fig.savefig(folder / "motion_metrics.png", dpi=180)
            if SHOW_PLOT:
                plt.show()
            plt.close(fig)

    save_csv(PATH_SAVE / "motion_summary.csv", [
        "stimulus_label", "stimulus_strength", "theta_deg",
        "median_strength", "median_spatial_sd", "noise_metrics",
        "median_strength_normalized", "noise_metrics_normalized",
        "n_off_target_angles", "global_max_abs_strength", "global_max_noise_metrics",
        "fps", "analyzed_frames", "noise_metrics_raw", "target_reference_video",
    ], summary_rows)
    save_csv(PATH_SAVE / "detector_angles.csv", ["role", "angle_deg"],
             [["target" if j == 0 else "off_target", angle]
              for j, angle in enumerate(angles_deg)])
    print(f"\nSummary saved to: {PATH_SAVE / 'motion_summary.csv'}")


if __name__ == "__main__":
    main()
