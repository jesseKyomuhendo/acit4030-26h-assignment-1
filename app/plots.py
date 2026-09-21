#Imports
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from app.ConfigManager import (DATA_PATH, SYNSETS, CLASS_NAMES, NUM_CLASSES, MESH_FILE, VOXEL_FILE,
                               VOXEL_RESOLUTION, CSV_PATH)
from app.data import load_splits, load_voxel, load_vertices, sample_points, load_mesh

NAMES = ["voxel", "mesh", "pointnet"]             # names used in the CSV files
LABELS = ["3D CNN", "Graph CNN", "PointNet"]      # names shown in the report


#Table as an image
def save_table(df, path):
    fig, ax = plt.subplots(figsize=(1.8 * len(df.columns), 0.5 * (len(df) + 2)))
    ax.axis("off")    # only the table, no plot axes
    table = ax.table(cellText=df.astype(str).values, colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(list(range(len(df.columns))))    # widen the columns to fit the text
    table.scale(1, 1.5)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


#Figure 1: the same object as voxels, mesh and points
def figure1(path):
    row = load_splits().iloc[0]    # first model of the first class
    print("Figure 1 shows:", CLASS_NAMES[SYNSETS.index(row["synset"])], row["model_id"])
    folder = os.path.join(DATA_PATH, row["synset"], row["model_id"], "models")
    verts, faces = load_mesh(os.path.join(folder, MESH_FILE))
    points = sample_points(load_vertices(os.path.join(folder, MESH_FILE)))    # what PointNet sees
    voxels = load_voxel(os.path.join(folder, VOXEL_FILE))[0] > 0              # [0] removes the channel
    limit = verts.abs().max().item()
    fig = plt.figure(figsize=(15, 5))

    # ShapeNet has y pointing up, matplotlib has z pointing up, so y and z are swapped below
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.voxels(voxels.numpy(), facecolors="tab:blue")
    ax.set_title("(a) Voxel grid " + str(VOXEL_RESOLUTION) + "³")

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.plot_trisurf(verts[:, 0].numpy(), verts[:, 2].numpy(), verts[:, 1].numpy(),
                    triangles=faces.numpy(), color="tab:blue", edgecolor="gray", linewidth=0.1)
    ax.set_title("(b) Triangle mesh")

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax.scatter(points[:, 0].numpy(), points[:, 2].numpy(), points[:, 1].numpy(), s=2)
    ax.set_title("(c) Point cloud (" + str(len(points)) + " points)")

    for ax in fig.axes:
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
    # same axis range in the mesh and point panels, so the object is not stretched
    fig.axes[1].set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    fig.axes[2].set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


#Figure 2: training curves
def figure2(path):
    history = pd.read_csv(CSV_PATH / "training_history.csv")
    # top row: loss, bottom row: validation accuracy, one column per model
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey="row")
    for i in range(len(NAMES)):
        part = history[history["model"] == NAMES[i]]
        if len(part) == 0:
            continue    # this model has no results yet
        mean = part.groupby("epoch")[["train_loss", "val_loss", "val_acc"]].mean()    # average over the seeds
        axes[0][i].plot(mean.index, mean["train_loss"], label="train")
        axes[0][i].plot(mean.index, mean["val_loss"], label="validation")
        axes[0][i].set_title(LABELS[i])
        axes[1][i].plot(mean.index, mean["val_acc"])
        axes[1][i].set_xlabel("Epoch")
    axes[0][0].set_ylabel("Loss")
    axes[0][0].legend()
    axes[1][0].set_ylabel("Validation accuracy")
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


#Figure 3: confusion matrices
def figure3(path):
    # constrained_layout spaces the panels so the labels do not overlap
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for i in range(len(NAMES)):
        files = glob.glob(str(CSV_PATH / ("confusion_matrix_" + NAMES[i] + "_seed*.csv")))
        if len(files) == 0:
            continue
        total = 0
        for file in files:
            total = total + pd.read_csv(file, index_col=0).values    # add up the matrices of all seeds
        percent = total / total.sum(axis=1, keepdims=True)           # each row: share of one true class
        axes[i].imshow(percent, cmap="Blues", vmin=0, vmax=1)
        axes[i].set_title(LABELS[i])
        axes[i].set_xticks(range(NUM_CLASSES))
        axes[i].set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        axes[i].set_yticks(range(NUM_CLASSES))
        axes[i].set_yticklabels(CLASS_NAMES)
        axes[i].set_xlabel("Predicted class")
        axes[i].set_ylabel("True class")
        for r in range(NUM_CLASSES):
            for c in range(NUM_CLASSES):
                color = "black"
                if percent[r][c] > 0.5:
                    color = "white"    # white text is easier to read on dark cells
                axes[i].text(c, r, format(percent[r][c] * 100, ".0f"), ha="center", va="center", color=color)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)