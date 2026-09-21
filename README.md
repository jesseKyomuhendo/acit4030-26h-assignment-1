# ACIT4030 Assignment 1: 3D Shape Classification

Comparing a 3D CNN (voxels), a Graph CNN (mesh) and PointNet (point cloud) on a
5-class subset of ShapeNetCore.

## Project structure

```
app/
  ConfigManager.py     loads config.yaml and exposes settings
  data.py               splits, binvox reader, voxel/mesh/point cloud loaders
  check_classes.py      verifies class names against the PyTorch3D synset dictionary
  models.py              3D CNN, Graph CNN, PointNet
  engine.py              training loop, evaluation, metrics
  train.py                entry point for training one model
  plots.py                figures and table images
  make_report_assets.py  builds all report tables and figures from saved results
colab_notebook/
  ACIT4030-Colab-Book.ipynb   training notebook (run on Google Colab)
data/                   ShapeNetCore subset (not included, see below)
output/                 generated tables, figures, and result CSVs
config.yaml             all settings (paths, hyperparameters, architecture)
splits.csv              train/val/test split (committed for reproducibility)
requirements.txt        Python packages (excluding torch and pytorch3d)
```

## Dataset

Not included in this submission. The Colab notebook expects a
`ShapeNetCore.zip` file in the project's Google Drive folder, containing one
folder per synset ID (e.g. `02808440/`) at the top level of the zip. The
notebook unzips it into `data/` at the start of each session, matching the
structure in `config.yaml`.

For local use, place the unzipped ShapeNetCore subset directly under `data/`,
with one folder per synset ID (e.g. `data/02808440/`).

## Setup on PC (development only, no training)

Training and PyTorch3D-dependent code (graph CNN, figures using meshes) were
run on Google Colab. On a local PC you can still edit and run the
non-PyTorch3D parts (e.g. `app/data.py`, `app/models.py` for voxel and
PointNet).

1. Create and activate a virtual environment:

   ```
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS/Linux
   ```

2. Install the dependencies:

   ```
   pip install -r requirements.txt
   pip install torch
   ```

   `torch` is installed separately since Colab already has it preinstalled,
   and pinning it here could conflict with Colab's version.

3. Run any module from the project root, for example:

   ```
   python -m app.data
   python -m app.models
   ```

## Training (Google Colab)

All training and PyTorch3D-dependent steps were run in
`colab_notebook/ACIT4030-Colab-Book.ipynb`. Open it in Google Colab, run the
setup cells (mount Drive, install PyTorch3D, clone the repo, install
requirements, get the dataset), then run the training and report-asset cells
in order.

Environment used: Python 3.13.15, PyTorch 2.11.0 (CUDA 12.8), PyTorch3D
0.7.8, NVIDIA A100 40GB GPU.

## Reproducing the report assets

After training (`app/train.py` for each model and seed), run:

```
python -m app.make_report_assets
```

This regenerates all tables and figures in `output/` from the saved result
CSVs in `output/results/csv/`.