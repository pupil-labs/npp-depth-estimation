# Depth Estimation Plugin (DA3)

A [Neon Player](https://docs.pupil-labs.com/neon/neon-player/) plugin that estimates per-frame depth from scene camera video using [Depth Anything 3](https://huggingface.co/depth-anything) (DA3) models, and specifically the optimized implementation provided by the [Awesome Depth Anything 3](https://github.com/Aedelon/awesome-depth-anything-3) repository. It overlays a depth heatmap on the scene video, plots gaze depth over time in the timeline, and exports depth data as a video and CSV file.

More details can be found on the relevant [Alpha Lab article](https://docs.pupil-labs.com/alpha-lab/depth-estimation).

## Installation

For instructions on how to install and manage Neon Player plugins, please refer to the [Neon Player Plugin Documentation](https://docs.pupil-labs.com/neon/neon-player/plugin-api/#adding-a-plugin).

**Requirements:** It carries [PEP 723](https://peps.python.org/pep-0723/) inline dependencies. 

## Models

Three DA3 model variants are supported, selectable via the **Model Version** dropdown in the plugin UI:

| Model | Type | Output |
|---|---|---|
| `DA3Metric-Large` (default) | Metric depth | Estimated distance in **metres** |
| `DA3-Base` | Relative depth | Relative inverse depth (0–255 scale) |
| `DA3-Small` | Relative depth | Relative inverse depth (0–255 scale) |

The metric model reads the scene camera's focal length from the recording calibration to convert raw network output into physical distances.

Inference runs on **Apple Silicon MPS** when available, falling back to CPU.

## Usage

### 1. Run Depth Estimation

Click **Run Depth Estimation** (▶) in the plugin panel. The plugin processes every scene camera frame through the selected DA3 model and saves two cache files per model to the recording's cache directory:

- `depth_norms_<model>.npy` — per-frame normalised depth maps (uint8, ¼ resolution)
- `depth_values_<model>.npy` — per-frame raw depth values (float32, ¼ resolution)

Once complete, a depth timeline line is added showing either **Gaze Depth (m)** (metric model) or **Relative Inverse Depth** (relative models) at the gaze position for each frame.

### 2. Visualise

The plugin overlays an INFERNO colormap heatmap on the scene video in real time. Use the **Depth Map Alpha** slider (0.0–1.0) to control the heatmap opacity.

### 3. Export

Click **Export** to choose a destination folder. The plugin writes to `<folder>/depth_estimation/`:

- `gaze_depth.mp4` — scene video with depth heatmap overlay and gaze annotation (depth label + circle)
- `gaze_depth_output.csv` — per-frame table with the following columns:

| Column | Description |
|---|---|
| `frame_index` | 0-based frame number |
| `timestamp` | Scene camera timestamp (ns) |
| `gaze_x_px` | Gaze x position in pixels |
| `gaze_y_px` | Gaze y position in pixels |
| `depth_meters` | Estimated gaze depth (metric model only) |
| `depth_diopters` | 1 / depth_meters (metric model only) |
| `relative_inverse_depth` | 255 − normalised depth at gaze point |

Export is also triggered automatically by Neon Player's batch export via `on_export`.
