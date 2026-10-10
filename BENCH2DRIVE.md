# UniMM Anchor-Based on Bench2Drive Base

This adapter keeps source code on the local machine. Raw clips, maps, converted
scenarios, anchors, checkpoints, and all training jobs belong on the server.
Sync source changes through Git: edit locally, push locally, then pull on the
server. Paths below are server paths.

Bench2Drive changes the training data and evaluation domain. A model trained by
this recipe is an Anchor-Based UniMM variant on Bench2Drive, not a reproduction
of the paper's Waymo WOSAC result. CARLA closed-loop evaluation is a separate
integration task.

## Local push and server pull

The writable repository is `https://github.com/Hmx1031688052/UniMM1.git`.
The author's repository is retained as `upstream`. To configure another local
clone that still points to the author's repository, run once:

```bash
git remote rename origin upstream
git remote add origin https://github.com/Hmx1031688052/UniMM1.git
```

For the current adapter changes, run locally:

```bash
git add .gitignore BENCH2DRIVE.md bench2drive_preprocess.py bench2drive_anchors.py train.py datamodules/waymo_datamodule.py modules/qcnet_agent_encoder.py modules/qcnet_map_encoder.py tests/test_bench2drive_adapter.py
git commit -m "Add Bench2Drive Base adapter for Anchor-Based UniMM"
git push -u origin main
```

Clone your fork on the server once:

```bash
git clone https://github.com/Hmx1031688052/UniMM1.git
cd UniMM1
```

For later updates, run in the server's code checkout:

```bash
git pull --ff-only origin main
```

Keep server datasets and run outputs in the absolute paths shown below. Git
transfers source changes; conversion, anchor fitting, and training are executed
on the server after pulling. Keep code edits in the local checkout so the
server checkout can receive updates without merge conflicts.

## Expected server layout

```text
/data/bench2drive/v1/<clip>/anno/00000.json.gz
/data/bench2drive/maps/Town01_HD_map.npz
/data/bench2drive/unimm_processed/       # generated
/data/bench2drive/anchors/               # generated
/runs/unimm_b2d/                          # generated
```

The [official Base split](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/uniad/vad/data/splits/bench2drive_base_train_val_split.json)
is included at `configs/bench2drive_base_train_val_split.json`; pulling this
repository supplies it without a separate download from raw.githubusercontent.com.
The converter accepts the Base root or its `v1` directory. It reads only `anno`
and the town HD maps; cameras, LiDAR, radar, and expert assessments are not
needed by UniMM's trajectory model. No dataset is copied to the local machine.

## Convert on the server

```bash
python bench2drive_preprocess.py \
  --raw-root /data/bench2drive \
  --map-root /data/bench2drive/maps \
  --split-file configs/bench2drive_base_train_val_split.json \
  --output-root /data/bench2drive/unimm_processed \
  --stride 10
```

For an initial format check, add `--max-scenarios 8`. Remove it for the full
conversion. Each 91-frame window contains 1 second of history and 8 seconds of
future at 10 Hz. Windows never cross clip boundaries or missing frame indices.
The map and actors are both converted from CARLA left-hand coordinates to a
right-hand XY frame. The converter writes the same `processed_scenarios_*`
layout read by the existing UniMM dataset class. It uses the official Base
validation clip list and treats all remaining clips as training clips.

## Fit Base-specific anchors on the server

Before a full conversion, inspect a converted training scene on the server:

```bash
python bench2drive_inspect.py \
  --root /data/bench2drive/unimm_smoke \
  --raw-root /data/bench2drive/v1 \
  --output /data/bench2drive/unimm_smoke/inspection.png
```

This checks trajectory shapes, finite values and ego dimensions against the raw
annotations, then draws the map, history, future, and current vehicle boxes.
Confirm the trajectories follow the roads and vehicle boxes face the correct
direction before a full conversion. Data converted before the box dimension
fix must be regenerated with the updated converter.

Install the dependencies with the repository's `install.sh` in a Python 3.10
environment on the server. The conversion step itself only needs NumPy.

```bash
python bench2drive_anchors.py \
  --root /data/bench2drive/unimm_processed \
  --output /data/bench2drive/anchors/2048_anchor-trajs_bench2drive_10Hz-8s.npy \
  --num-modes 2048
```

The script uses training scenarios only. If a rare agent type has too few
complete 8-second trajectories, it repeats available trajectories or uses a
documented fallback and prints a warning. Inspect these counts before a full
run. Never use the bundled Waymo anchor file for Bench2Drive training.

## Train Anchor-Based-4s on the server

```bash
python train.py \
  --dataset bench2drive \
  --root /data/bench2drive/unimm_processed \
  --experiment unimm_anchor_based_b2d_4s \
  --output_dir /runs/unimm_b2d \
  --anchor_file /data/bench2drive/anchors/2048_anchor-trajs_bench2drive_10Hz-8s.npy \
  --reg_decoder mlp --approx_posterior \
  --num_map_layers 1 --num_agent_layers 2 \
  --num_historical_steps 11 --num_prediction_steps 40 \
  --key_frame_interval 5 --num_execution_steps 5 --train_sim_steps 16 \
  --match_execution --align_match \
  --train_batch_size 4 --val_batch_size 4 --test_batch_size 4 \
  --devices 1 --max_epochs 30
```

Adjust `--devices` and batch sizes to the server's GPUs. A single GPU uses
Lightning's automatic strategy; multiple GPUs use DDP. Run a one-epoch small
conversion first, then inspect trajectory and map alignment before training the
full Base set. The converted Base data has train and validation splits only;
the training path no longer requires a testing split.

## Current limits

- The adapter has a local synthetic smoke check but has not yet been checked
  against an actual server clip or map file. Validate actor box dimensions,
  headings, and map alignment on real examples before the full run.
- UniMM's existing validation loss works. Waymo-specific WOSAC metrics and
  submissions are not Bench2Drive metrics.
- The script builds full windows in memory and keeps per-town maps cached.
  Server RAM and output space need to be measured during the small run.
