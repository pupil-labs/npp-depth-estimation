# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "awesome-depth-anything-3",
# ]
# ///
import os
from pathlib import Path
import logging
import csv
import time
from collections.abc import Generator
from enum import Enum

import cv2
import numpy as np
import torch

# Pupil Labs imports
from pupil_labs.neon_player import Plugin, ProgressUpdate, action
from pupil_labs.neon_recording import NeonRecording
from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon, QImage, QPainter
from PySide6.QtWidgets import QMessageBox, QFileDialog
from qt_property_widgets.utilities import (
    action_params,
    property_params,
)
from qt_property_widgets.widgets import DynamicComboWidget

# ENVIRONMENT OVERRIDES
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from depth_anything_3.api import DepthAnything3

METRIC_DEPTH_SCALE_DENOMINATOR = 300.0  # calibration constant used when converting DA3 metric model output into estimated meters.

# Model selection
class DA3ModelType(Enum):
    METRIC_LARGE = "DA3Metric-Large"
    SMALL = "DA3-Small"
    BASE = "DA3-Base"

class DepthEstimationPlugin(Plugin):
    label = "Depth Estimation (DA3)"
    icon = QIcon.fromTheme("camera-depth")
    model_options_changed = Signal()

    def __init__(self):
        super().__init__()
        self._depth_map_alpha = 0.6
        self._model_selection = DA3ModelType.METRIC_LARGE
        self._available_model_versions: list[str] = [model.value for model in DA3ModelType]
        self.estimation_job = None
        self._timeline_plot_name = "Depth Estimation"
        self._timeline_metric_series = "Gaze Depth (m)"
        self._timeline_relative_series = "Relative Inverse Depth"
        self._depth_cache: np.ndarray | None = None   # shape: (N, H/4, W/4) uint8
        self._depth_values_cache: np.ndarray | None = None  # shape: (N, H/4, W/4) float32

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        logging.info("Depth Estimation plugin initialized for recording.")
        self._load_depth_cache()
        # Timeline is added only after depth generation completes, not on load.

    def _depth_cache_stem(self, model_type_str: str | None = None) -> str:
        model = model_type_str or self._model_selection.value
        return model.replace(" ", "_")

    def _load_depth_cache(self, model_type_str: str | None = None) -> None:
        stem = self._depth_cache_stem(model_type_str)
        cache_dir = self.get_cache_path()

        norms_path = cache_dir / f"depth_norms_{stem}.npy"
        if norms_path.exists():
            try:
                self._depth_cache = np.load(str(norms_path))
                logging.info(f"Loaded depth norm cache ({stem}): {self._depth_cache.shape}")
            except Exception:
                logging.exception("Failed to load depth norm cache.")
                self._depth_cache = None
        else:
            self._depth_cache = None

        values_path = cache_dir / f"depth_values_{stem}.npy"
        if values_path.exists():
            try:
                self._depth_values_cache = np.load(str(values_path))
                logging.info(f"Loaded depth values cache ({stem}): {self._depth_values_cache.shape}")
            except Exception:
                logging.exception("Failed to load depth values cache.")
                self._depth_values_cache = None
        else:
            self._depth_values_cache = None

    def _remove_timeline_line(self):
        timeline = self.get_timeline()
        if timeline is None:
            return
        remove_plot = getattr(timeline, "remove_timeline_plot", None)
        if callable(remove_plot):
            try:
                remove_plot(self._timeline_plot_name)
            except Exception:
                logging.debug("Failed to remove timeline plot: %s", self._timeline_plot_name, exc_info=True)
            
        get_series = getattr(timeline, "get_timeline_series", None)
        remove_series = getattr(timeline, "remove_timeline_series", None)
        if callable(get_series) and callable(remove_series):
            for series_name in (self._timeline_metric_series, self._timeline_relative_series):
                try:
                    existing = get_series(self._timeline_plot_name, series_name)
                except Exception:
                    logging.debug("Failed to query timeline series: %s", series_name, exc_info=True)
                    existing = None
                if existing is not None:
                    try:
                        remove_series(self._timeline_plot_name, series_name)
                    except Exception:
                        logging.debug("Failed to remove timeline series: %s", series_name, exc_info=True)

    def _clear_timeline_for_new_run(self):
        timeline = self.get_timeline()
        if timeline is None:
            return

        # Clear only this plugin's timeline plot at new run start.
        self._remove_timeline_line()

    def on_disabled(self) -> None:
        self._remove_timeline_line()

    def render(self, painter: QPainter, time_in_recording: int) -> None:
        if self._depth_cache is None or self.recording is None:
            return

        scene_idx = self.get_scene_idx_for_time(time_in_recording)
        if not (0 <= scene_idx < len(self._depth_cache)):
            return

        depth_norm_small = self._depth_cache[scene_idx]  # uint8, 1/4 res

        # Draw in video pixel coordinates — maps as used by SapiensPlugin.
        w_orig = self.recording.scene.width
        h_orig = self.recording.scene.height

        # Apply INFERNO colormap and resize from 1/4-res cache to full video dimensions
        heatmap_bgr = cv2.applyColorMap(depth_norm_small, cv2.COLORMAP_INFERNO)
        heatmap_full = cv2.resize(heatmap_bgr, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        # Convert BGR -> BGRA and bake opacity into the alpha channel.
        alpha = int(255 * self._depth_map_alpha)
        heatmap_rgba = cv2.cvtColor(heatmap_full, cv2.COLOR_BGR2BGRA)
        heatmap_rgba[:, :, 3] = alpha

        qimg = QImage(
            heatmap_rgba.data, w_orig, h_orig, w_orig * 4, QImage.Format_ARGB32
        ).copy()

        painter.drawImage(0, 0, qimg)

    def _get_gaze_offsets(self) -> tuple[float, float]:
        offset_x, offset_y = 0.0, 0.0
        gaze_plugin = Plugin.get_instance_by_name("GazeDataPlugin") or Plugin.get_instance_by_name("Gaze")
        if gaze_plugin:
            offset_x = getattr(gaze_plugin, "offset_x", getattr(gaze_plugin, "x_offset", 0.0))
            offset_y = getattr(gaze_plugin, "offset_y", getattr(gaze_plugin, "y_offset", 0.0))
        return float(offset_x), float(offset_y)

    @staticmethod
    def _to_pixel_gaze_coords(
        raw_gx: float,
        raw_gy: float,
        width: int,
        height: int,
        offset_x: float,
        offset_y: float,
    ) -> tuple[float, float]:
        return raw_gx + (offset_x * width), raw_gy + (offset_y * height)

    def _resolve_valid_gaze_pixel(
        self,
        gaze_sample,
        width: int,
        height: int,
        offset_x: float,
        offset_y: float,
    ) -> tuple[float, float, int, int] | None:
        if gaze_sample is None:
            return None

        raw_gx = float(gaze_sample.point[0])
        raw_gy = float(gaze_sample.point[1])
        if np.isnan(raw_gx) or np.isnan(raw_gy):
            return None

        gx_val, gy_val = self._to_pixel_gaze_coords(raw_gx, raw_gy, width, height, offset_x, offset_y)
        gx_int, gy_int = int(gx_val), int(gy_val)
        if not (0 <= gx_int < width and 0 <= gy_int < height):
            return None

        return gx_val, gy_val, gx_int, gy_int

    def _cache_paths_for_model(self, model_type_str: str | None = None) -> tuple[Path, Path]:
        stem = self._depth_cache_stem(model_type_str)
        cache_dir = self.get_cache_path()
        return (
            cache_dir / f"depth_norms_{stem}.npy",
            cache_dir / f"depth_values_{stem}.npy",
        )

    def _cache_files_are_fresh(self, model_type_str: str, started_at_ns: int) -> bool:
        norms_path, values_path = self._cache_paths_for_model(model_type_str)
        if not norms_path.exists() or not values_path.exists():
            return False

        try:
            return norms_path.stat().st_mtime_ns >= started_at_ns and values_path.stat().st_mtime_ns >= started_at_ns
        except Exception:
            logging.exception("Failed checking cache timestamps for model: %s", model_type_str)
            return False

    def _infer_scene_fps(self, timestamps: np.ndarray) -> float:
        if len(timestamps) < 2:
            raise ValueError("Need at least two timestamps to calculate FPS.")

        diffs = np.diff(timestamps.astype(np.float64))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if len(diffs) == 0:
            raise ValueError("No positive timestamp deltas available to calculate FPS.")

        median_delta = float(np.median(diffs))
        if median_delta > 1e6:
            # nanoseconds
            fps = 1e9 / median_delta
        elif median_delta > 1e3:
            # microseconds
            fps = 1e6 / median_delta
        elif median_delta > 1.0:
            # milliseconds
            fps = 1e3 / median_delta
        else:
            # seconds
            fps = 1.0 / median_delta

        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"Calculated invalid FPS value: {fps}")

        logging.info(
            "Calculated export FPS from scene timestamps: %.6f (median delta %.6f)",
            fps,
            median_delta,
        )
        return fps

    def _add_depth_timeline_line(self, model_type_str: str | None = None):
        timeline = self.get_timeline()
        if timeline is None or self.recording is None:
            return

        if self._depth_cache is None or self._depth_values_cache is None:
            logging.warning("No depth cache loaded for timeline.")
            return

        total_frames = len(self.recording.scene)
        if total_frames == 0:
            return

        timestamps = np.asarray(self.recording.scene.time)
        if len(timestamps) != total_frames:
            return

        metric_depth = np.full(total_frames, np.nan, dtype=np.float64)
        inverse_depth = np.full(total_frames, np.nan, dtype=np.float64)

        width = int(self.recording.scene.width)
        height = int(self.recording.scene.height)
        offset_x, offset_y = self._get_gaze_offsets()
        matched_gaze_seq = self.recording.gaze.sample(timestamps)

        frame_count = min(total_frames, len(self._depth_cache), len(self._depth_values_cache))
        for frame_idx, gaze_sample in zip(range(frame_count), matched_gaze_seq, strict=False):
            gaze_coords = self._resolve_valid_gaze_pixel(gaze_sample, width, height, offset_x, offset_y)
            if gaze_coords is None:
                continue

            _, _, gx_int, gy_int = gaze_coords

            depth_norm_small = self._depth_cache[frame_idx]
            depth_vals_small = self._depth_values_cache[frame_idx]
            sh, sw = depth_norm_small.shape[:2]

            sx = min(sw - 1, max(0, int(gx_int * sw / width)))
            sy = min(sh - 1, max(0, int(gy_int * sh / height)))

            inv_depth = 255.0 - float(depth_norm_small[sy, sx])
            inverse_depth[frame_idx] = inv_depth

            depth_m = float(depth_vals_small[sy, sx])
            if np.isfinite(depth_m) and depth_m > 0.0:
                metric_depth[frame_idx] = depth_m

        model_type = model_type_str or self._model_selection.value
        is_metric_model = model_type == DA3ModelType.METRIC_LARGE.value

        if is_metric_model:
            valid_metric = np.isfinite(metric_depth)
            if not np.any(valid_metric):
                logging.warning("No metric gaze depth values found for timeline.")
                return
            y = metric_depth.copy()
            series_name = self._timeline_metric_series
            logging.info("Timeline source: metric depth (m)")
        else:
            valid_inverse = np.isfinite(inverse_depth)
            if not np.any(valid_inverse):
                logging.warning("No relative inverse depth values found for timeline.")
                return
            y = inverse_depth.copy()
            series_name = self._timeline_relative_series
            logging.info("Timeline source: relative inverse depth")

        # Remove nans for plotting, but keep time alignment
        tss = timestamps.copy()
        y[~np.isfinite(y)] = np.nan

        # Add the timeline line
        try:
            # Eyestate-style grouped plot + series label.
            timeline.add_timeline_line(self._timeline_plot_name, np.column_stack((tss, y)), series_name)
        except Exception:
            logging.exception("Failed to add depth timeline line.")
            return

    # --- UI PROPERTIES ---
    @property
    @property_params(min=0.0, max=1.0, step=0.1)
    def depth_map_alpha(self) -> float:
        return self._depth_map_alpha

    @depth_map_alpha.setter
    def depth_map_alpha(self, value: float) -> None:
        self._depth_map_alpha = value

    @property
    @property_params(
        label="Model Version",
        widget=DynamicComboWidget,
        options_source="_available_model_versions",
        options_changed_signal="model_options_changed",
    )
    def model_version(self) -> str:
        return self._model_selection.value

    @model_version.setter
    def model_version(self, value: str | DA3ModelType) -> None:
        if isinstance(value, DA3ModelType):
            selected_model = value
        else:
            selected_value = str(value or "").strip()
            selected_model = next(
                (model for model in DA3ModelType if model.value == selected_value),
                self._model_selection,
            )

        if self._model_selection != selected_model:
            self._model_selection = selected_model
            self._load_depth_cache()
            self.changed.emit()

    # --- ACTIONS ---
    @action
    @action_params(compact=True, icon=QIcon.fromTheme("media-playback-start"))
    def run_depth_estimation(self) -> None:
        if self.estimation_job is not None:
            QMessageBox.warning(None, "Job Running", "Depth estimation is already in progress.")
            return

        if not self.recording:
            return

        self._clear_timeline_for_new_run()

        selected_model_type = self._model_selection.value
        run_started_ns = time.time_ns()
        
        job = self.job_manager.run_background_action(
            "Generate Depth",
            "DepthEstimationPlugin.bg_process_depth",
            selected_model_type  
        )

        if job is None:
            return

        self.estimation_job = job

        def on_finished():
            self.estimation_job = None
            self._load_depth_cache(selected_model_type)

            if (
                self._depth_cache is None
                or self._depth_values_cache is None
                or not self._cache_files_are_fresh(selected_model_type, run_started_ns)
            ):
                logging.warning("Depth estimation finished but did not produce fresh cache files for %s.", selected_model_type)
                QMessageBox.warning(
                    None,
                    "Depth Estimation",
                    f"Depth estimation for {selected_model_type} did not complete successfully. Check logs and retry.",
                )
                return

            self._add_depth_timeline_line(selected_model_type)
            QMessageBox.information(
                None,
                "Depth Estimation",
                f"Generation complete using {selected_model_type}! You can now export the files.",
            )

        job.finished.connect(on_finished)

    # --- BACKGROUND WORKER ---
    def bg_process_depth(
        self,
        model_type_str: str = "DA3Metric-Large",
    ) -> Generator[ProgressUpdate, None, None]:
        # Map UI choice to BOTH the repo link AND the internal encoder/name
        repo_map = {
            "DA3Metric-Large": {"repo": "depth-anything/DA3METRIC-LARGE", "model_name": "da3metric-large"},
            "DA3-Small": {"repo": "depth-anything/DA3-SMALL", "model_name": "da3-small"},
            "DA3-Base": {"repo": "depth-anything/DA3-BASE", "model_name": "da3-base"},
        }
        
        config = repo_map.get(model_type_str, repo_map["DA3Metric-Large"])
        model_repo = config["repo"]
        internal_name = config["model_name"]
        
        is_metric = (model_type_str == "DA3Metric-Large")

        logging.info(f"Loading {model_type_str} onto Apple Silicon...")
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        try:
            model = DepthAnything3.from_pretrained(
                model_repo, 
                model_name=internal_name  
            ).to(device).eval()
        except Exception as e:
            logging.exception(f"Failed to load DA3 model: {e}")
            return

        cache_dir = self.get_cache_path()
        cache_dir.mkdir(parents=True, exist_ok=True)

        for legacy_name in ("gaze_depth_output.csv", "gaze_depth.mp4"):
            legacy_path = cache_dir / legacy_name
            if legacy_path.exists():
                try:
                    legacy_path.unlink()
                except Exception:
                    logging.exception("Failed to remove legacy cache artifact: %s", legacy_path)

        focal_length_px: float | None = None
        if is_metric:
            try:
                camera_matrix = self.recording.calibration.scene_camera_matrix
                fx = float(camera_matrix[0][0])
                fy = float(camera_matrix[1][1])
                focal_length_px = (fx + fy) / 2.0
            except Exception:
                logging.exception("Failed to read scene_camera_matrix from recording calibration.")
                return

            if not np.isfinite(focal_length_px) or focal_length_px <= 0.0:
                logging.error("Calculated focal length is invalid: %s", focal_length_px)
                return

            logging.info("Calculated focal length from recording: %.6f px", focal_length_px)

        total_frames = len(self.recording.scene)

        depth_norms_cache = []   # per-frame uint8 depth maps at 1/4 resolution
        depth_values_cache = []  # per-frame float32 depth values at 1/4 resolution

        logging.info(f"Processing {total_frames} frames...")

        with torch.no_grad():
            for frame_idx, frame in enumerate(self.recording.scene):
                pixels = frame.bgr
                height, width = pixels.shape[:2]

                rgb_frame = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
                prediction = model.inference([rgb_frame])
                depth_array = prediction.depth[0]

                if hasattr(depth_array, 'cpu'):
                    depth_array = depth_array.cpu().numpy()

                # --- Core Branching Logic ---
                if is_metric:
                    _, net_w = depth_array.shape
                    scale_factor = net_w / width 
                    if focal_length_px is None:
                        logging.error("Metric depth scaling requires focal length, but none was calculated.")
                        return
                    network_focal_length = focal_length_px * scale_factor
                    depth_scaled = (network_focal_length * depth_array) / METRIC_DEPTH_SCALE_DENOMINATOR
                else:
                    depth_scaled = depth_array  # Leave as raw relative output
                
                depth_resized = cv2.resize(depth_scaled, (width, height))

                depth_norm = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # Cache at 1/4 resolution to keep file size manageable
                small_h, small_w = max(1, height // 4), max(1, width // 4)
                depth_norms_cache.append(cv2.resize(depth_norm, (small_w, small_h), interpolation=cv2.INTER_AREA))
                depth_values_cache.append(cv2.resize(depth_resized.astype(np.float32), (small_w, small_h), interpolation=cv2.INTER_AREA))

                if frame_idx % 10 == 0:
                    yield ProgressUpdate((frame_idx + 1) / total_frames)

        # Save per-frame depth caches (model-specific filenames)
        stem = model_type_str.replace(" ", "_")
        norms_path = cache_dir / f"depth_norms_{stem}.npy"
        values_path = cache_dir / f"depth_values_{stem}.npy"
        try:
            np.save(str(norms_path), np.array(depth_norms_cache, dtype=np.uint8))
            logging.info(f"Saved depth norm cache ({stem}): {len(depth_norms_cache)} frames -> {norms_path}")
        except Exception:
            logging.exception("Failed to save depth norm cache.")
        try:
            np.save(str(values_path), np.array(depth_values_cache, dtype=np.float32))
            logging.info(f"Saved depth values cache ({stem}): {len(depth_values_cache)} frames -> {values_path}")
        except Exception:
            logging.exception("Failed to save depth values cache.")

        yield ProgressUpdate(1.0)

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("document-save"), label="Export")
    def export(self) -> None:
        folder = QFileDialog.getExistingDirectory(None, "Select Export Folder")
        if not folder:
            return
        self._do_export(Path(folder))

    def _do_export(self, destination_path: Path) -> None:
        export_dir = destination_path / "depth_estimation"

        if self.recording is None:
            logging.warning("Cannot export: no recording loaded.")
            return

        self._load_depth_cache()
        if self._depth_cache is None or self._depth_values_cache is None:
            logging.warning("No depth cache found. Run depth estimation first.")
            return

        export_dir.mkdir(parents=True, exist_ok=True)
        csv_output_path = export_dir / "gaze_depth_output.csv"
        video_output_path = export_dir / "gaze_depth.mp4"

        is_metric = self._model_selection.value == DA3ModelType.METRIC_LARGE.value
        timestamps = np.asarray(self.recording.scene.time)
        try:
            export_fps = self._infer_scene_fps(timestamps)
        except ValueError as exc:
            logging.error("Cannot export depth video: %s", exc)
            return
        matched_gaze_seq = self.recording.gaze.sample(timestamps)
        offset_x, offset_y = self._get_gaze_offsets()

        out_video = None
        rows: list[dict[str, object]] = []

        try:
            for frame_idx, (frame, gaze_sample, depth_norm_small, depth_vals_small) in enumerate(
                zip(self.recording.scene, matched_gaze_seq, self._depth_cache, self._depth_values_cache, strict=False)
            ):
                pixels = frame.bgr
                height, width = pixels.shape[:2]

                if out_video is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out_video = cv2.VideoWriter(str(video_output_path), fourcc, export_fps, (width, height))

                depth_norm = cv2.resize(depth_norm_small, (width, height), interpolation=cv2.INTER_LINEAR)
                depth_vals = cv2.resize(depth_vals_small.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)

                render_frame = pixels.copy()
                depth_heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
                cv2.addWeighted(
                    src1=depth_heatmap,
                    alpha=self._depth_map_alpha,
                    src2=render_frame,
                    beta=1 - self._depth_map_alpha,
                    gamma=0,
                    dst=render_frame,
                )

                gx_val, gy_val = (None, None)
                gaze_depth_meters = None
                gaze_diopters = None
                gaze_relative_inv_depth = None

                gaze_coords = self._resolve_valid_gaze_pixel(gaze_sample, width, height, offset_x, offset_y)
                if gaze_coords is not None:
                    gx_val, gy_val, gx_int, gy_int = gaze_coords
                    if 0 <= gx_int < width and 0 <= gy_int < height:
                        gaze_relative_inv_depth = 255.0 - float(depth_norm[gy_int, gx_int])
                        if is_metric:
                            gaze_depth_meters = float(depth_vals[gy_int, gx_int])
                            gaze_diopters = 1.0 / gaze_depth_meters if gaze_depth_meters > 0.0 else 0.0
                            text = f"{gaze_depth_meters:.2f}m | {gaze_diopters:.2f}D"
                        else:
                            text = f"Rel Inv: {gaze_relative_inv_depth:.1f}"

                        cv2.circle(render_frame, (gx_int, gy_int), radius=12, color=(0, 255, 0), thickness=-1)
                        text_pos = (gx_int + 20, gy_int - 20)
                        cv2.putText(render_frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
                        cv2.putText(render_frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                out_video.write(render_frame)

                rows.append({
                    "frame_index": frame_idx,
                    "timestamp": timestamps[frame_idx] if frame_idx < len(timestamps) else "",
                    "gaze_x_px": round(gx_val, 2) if gx_val is not None else "",
                    "gaze_y_px": round(gy_val, 2) if gy_val is not None else "",
                    "depth_meters": round(gaze_depth_meters, 4) if gaze_depth_meters is not None else "",
                    "depth_diopters": round(gaze_diopters, 4) if gaze_diopters is not None else "",
                    "relative_inverse_depth": round(gaze_relative_inv_depth, 2) if gaze_relative_inv_depth is not None else "",
                })
        except Exception:
            logging.exception("Failed to generate export artifacts.")
            if out_video is not None:
                out_video.release()
            return

        if out_video is not None:
            out_video.release()

        with open(csv_output_path, mode="w", newline="") as f:
            fieldnames = ["frame_index", "timestamp", "gaze_x_px", "gaze_y_px", "depth_meters", "depth_diopters", "relative_inverse_depth"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logging.info("Exported depth estimation files to %s", export_dir)

    def on_export(self, destination_path: Path = Path()) -> None:
        self._do_export(destination_path)