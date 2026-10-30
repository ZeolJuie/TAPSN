# TAPSN: A Multimodal Simulation Dataset for Transferring Autonomous Driving Paradigms to Social Navigation

---

## Overview

**DATASET:** TAPSN is a high-fidelity, CARLA-based simulation dataset for **egocentric social navigation**. It transfers the multimodal 3D-supervision paradigm commonly used in autonomous driving to pedestrian-level navigation, providing synchronized sensor streams and dense annotations in a unified perception-to-navigation data stack.

**PIPELINE:** A CARLA-based pipeline (CARLA v0.9.15) with a controllable scenario specification interface, Social Force Model (SFM) based pedestrian motion, and a pedestrian-level sensor layout with motion-dependent jitter, for generating multimodal pedestrian social-navigation data from a natural-language request.


## Dataset at a Glance

| Item | Value |
| --- | --- |
| Curated cases | 212 (from 11 maps) |
| Annotated keyframes | ~15.7k (@ 2 Hz) |
| Simulated duration | ~2.15 hours |
| 3D object-box labels | ~1.35M |
| Persistent object instances | ~40k |
| Case length | 40–100 keyframes |

## Data Structure

Each **case** is a complete navigation sequence generated from a structured scenario specification; each **keyframe** is a synchronized frame with aligned sensor streams and annotations.

**Sensor suite**
- Surround-view RGB cameras
- 64-beam LiDAR
- Radar
- Depth and semantic cameras
- IMU/GNSS
- Ego-state records


## Download

The TAPSN dataset can be downloaded from:

- **Link:** https://yunpan.ustb.edu.cn/link/AA989AD974B9A84663B5D89F804B98E342
- **Extraction code:** `yptw`


---

## Pipeline generation

### 1. Install the Python environment

```bash
conda create -n carla python=3.9
conda activate carla
pip install -r requirements.txt
```

The CARLA Python API is installed separately from the CARLA package in step 4.

### 2. Install CARLA 0.9.15

Download and extract CARLA 0.9.15. Confirm these paths exist:

```text
CarlaUE4.sh
PythonAPI/carla
PythonAPI/carla/dist/carla-0.9.15-*.whl
```

### 3. Set environment variables

```bash
export CARLA_ROOT=/path/to/CARLA_0.9.15
export PYTHONPATH=$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$PYTHONPATH
```

### 4. Install the CARLA Python API

```bash
conda activate carla
pip install "$CARLA_ROOT"/PythonAPI/carla/dist/carla-0.9.15-*.whl
```

### 5. Start CARLA

```bash
cd "$CARLA_ROOT"
./CarlaUE4.sh -RenderOffScreen
```

Keep this terminal running. CARLA listens on port `2000` by default.

### 6. Prepare map topology (optional)

The repository already provides:

```text
topology/Town10HD_Opt/walkable_cells.json
```

Skip this step when using that map.

For another map, run while CARLA is running:

```bash
cd /path/to/carla-simulation-data

python utils/generate_walkable_cells.py \
  --map-name Town10HD_Opt \
  --timeout 240 \
  --output-dir topology/Town10HD_Opt
```

### 7. Set the LLM API key

DeepSeek is the default provider:

```bash
export DEEPSEEK_API_KEY="..."
```

For OpenAI:

```bash
export OPENAI_API_KEY="..."
```

### 8. Generate and run one scenario

```bash
cd /path/to/carla-simulation-data

python -m scenariogen \
  "Generate 1 rainy urban crosswalk scenario with medium traffic, grouped pedestrians, and a cautious ego pedestrian." \
  --catalog topology/Town10HD_Opt/walkable_cells.json \
  --output-root data \
  --count 1 \
  --execute \
  --duration 60 \
  --wall-time-limit 300 \
  --conda-env carla
```

If the current shell is already inside the `carla` environment, replace `--conda-env carla` with `--no-conda`.

To use OpenAI:

```bash
python -m scenariogen \
  "Generate 1 rainy urban crosswalk scenario with medium traffic, grouped pedestrians, and a cautious ego pedestrian." \
  --catalog topology/Town10HD_Opt/walkable_cells.json \
  --output-root data \
  --count 1 \
  --provider openai \
  --model gpt-4o-mini \
  --execute \
  --duration 60 \
  --wall-time-limit 300 \
  --conda-env carla
```

To inspect the generated plan without running the simulator, add `--dry-run` and remove `--execute`.

### 9. Check the collected data

```bash
ls data/Town10HD_Opt/001
```

Expected layout:

```text
data/Town10HD_Opt/001/
  config/
    scenario.toml
    scenario_spec.json
    case.json
    scene_description.txt
    sampling_policy.toml
    sensor_rig.toml
  simulation/
  sampling/
  visualization/
  logs/
    run.log
    run_result.json
  manifest.json
```

| Path | Content |
|---|---|
| `config/scenario.toml` | Scenario passed to the simulator |
| `config/scenario_spec.json` | Original prompt, scene plan, and resolved case |
| `sampling/` | RGB, LiDAR, radar, depth, semantic, IMU/GNSS, ego state |
| `logs/run.log` | Simulator output |
| `logs/run_result.json` | Return code, elapsed time, timeout flag |

---

### Optional: post-processing and export

These steps require additional dependencies: torch, open3d, mmcv, scipy, chamfer, nuscenes-devkit, pyquaternion, Pillow, and matplotlib. Run them after collection, in this order.

```bash
export TAPSN_ROOT=./data

# 1. Mark valid frames
python utils/postprocess_dataset/check_validation.py "$TAPSN_ROOT"/Town10HD_Opt

# 2. Repair static instance ids and compute LiDAR occlusion
python utils/postprocess_dataset/run_postprocess.py "$TAPSN_ROOT"/Town10HD_Opt

# 3. Generate occupancy ground truth
bash tools/generate_occ_labels_by_town.sh Town10HD_Opt

# 4. Rename occupancy ground truth to the nuScenes scene layout
python utils/postprocess_dataset/rename_occ_gt_to_standard_scene.py \
  --src-root gts \
  --dst-root gts_standard

# 5. Export the dataset to nuScenes format
DATA_ROOT="$TAPSN_ROOT" bash scripts/export_tapsn_nuscenes.sh

# 6. Validate the export with nuscenes-devkit
python utils/nuscenes_converter/validate_with_devkit.py ./data_nuscenes
```


---

## Citation

If you use TAPSN in your research, please cite:

```bibtex
@inproceedings{zhou2026tapsn,
  title     = {TAPSN: A Multimodal Simulation Dataset for Transferring
               Autonomous Driving Paradigms to Social Navigation},
  author    = {Zhou, Mohan and Zhi, Ruicong and Liang, Xiaoyuan},
  booktitle = {Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

## License

See [LICENSE.md](LICENSE.md).
