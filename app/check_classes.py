#Imports
import os
import matplotlib.pyplot as plt
from app.ConfigManager import DATA_PATH, SYNSETS, CLASS_NAMES, MESH_FILE, NON_RESULTS_PATH
from app.data import load_splits, load_vertices


def check_names():
    from pytorch3d.datasets import ShapeNetCore    # imported here so the rest also works without PyTorch3D
    dataset = ShapeNetCore(str(DATA_PATH), synsets=SYNSETS, version=2, load_textures=False)
    for i in range(len(SYNSETS)):
        # the loader carries a dictionary from synset ID to class name
        official = dataset.synset_dict[SYNSETS[i]]
        if official == CLASS_NAMES[i]:
            result = "OK"
        else:
            result = "DIFFERENT"
        print(SYNSETS[i], "| PyTorch3D name:", official, "| config name:", CLASS_NAMES[i], "|", result)


def show_shapes():
    table = load_splits()
    fig = plt.figure(figsize=(15, 3))
    for i in range(len(SYNSETS)):
        row = table[table["synset"] == SYNSETS[i]].iloc[0]    # first model of the class
        path = os.path.join(DATA_PATH, SYNSETS[i], row["model_id"], "models", MESH_FILE)
        verts = load_vertices(path)
        ax = fig.add_subplot(1, 5, i + 1, projection="3d")
        ax.scatter(verts[:, 0].numpy(), verts[:, 2].numpy(), verts[:, 1].numpy(), s=1)    # y is up in ShapeNet
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(CLASS_NAMES[i])
        ax.set_axis_off()
    os.makedirs(NON_RESULTS_PATH, exist_ok=True)
    fig.savefig(NON_RESULTS_PATH / "class_check.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    check_names()
    show_shapes()