#Imports
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import max_pool3d
from torch.utils.data import Dataset, DataLoader
from app.ConfigManager import (DATA_PATH, SYNSETS, MESH_FILE, VOXEL_FILE, SPLITS_FILE,
                               TRAIN_RATIO, VAL_RATIO, SPLIT_SEED,
                               BATCH_SIZE, NUM_WORKERS, VOXEL_RESOLUTION, NUM_POINTS)


#Splits
def find_models(synset):
    ids = []
    # sorted() gives the same order on every computer
    for model_id in sorted(os.listdir(os.path.join(DATA_PATH, synset))):
        folder = os.path.join(DATA_PATH, synset, model_id, "models")
        # keep only models that have both the mesh and the voxel file
        if os.path.exists(os.path.join(folder, MESH_FILE)) and os.path.exists(os.path.join(folder, VOXEL_FILE)):
            ids.append(model_id)
    return ids


def make_splits():
    random.seed(SPLIT_SEED)
    rows = []
    for synset in SYNSETS:
        ids = find_models(synset)
        random.shuffle(ids)
        n_train = int(len(ids) * TRAIN_RATIO)
        n_val = int(len(ids) * VAL_RATIO)
        # splitting inside the class loop keeps the class balance in every split
        for i in range(len(ids)):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            rows.append([synset, ids[i], split])
    table = pd.DataFrame(rows, columns=["synset", "model_id", "split"])
    table.to_csv(SPLITS_FILE, index=False)
    return table


def load_splits():
    # create the file only the first time
    if not os.path.exists(SPLITS_FILE):
        make_splits()
    # dtype=str keeps the leading zero in ids like 02808440
    return pd.read_csv(SPLITS_FILE, dtype=str)


#Checks
def check_models():
    table = load_splits()
    total_folders = 0
    total_used = 0

    for synset in SYNSETS:
        folders = sorted(os.listdir(os.path.join(DATA_PATH, synset)))
        used = table[table["synset"] == synset]["model_id"].tolist()
        print(synset, "| folders:", len(folders), "| in splits:", len(used))

        for model_id in folders:
            # a model missing from splits.csv was dropped, so find out which file is missing
            if model_id not in used:
                folder = os.path.join(DATA_PATH, synset, model_id, "models")
                print("   dropped:", model_id,
                      "| mesh exists:", os.path.exists(os.path.join(folder, MESH_FILE)),
                      "| voxel exists:", os.path.exists(os.path.join(folder, VOXEL_FILE)))

        total_folders = total_folders + len(folders)
        total_used = total_used + len(used)

    print("Total folders:", total_folders, "| used:", total_used, "| dropped:", total_folders - total_used)


#Voxels
def read_binvox(path):
    with open(path, "rb") as f:
        f.readline()                                    # skip the "#binvox 1" line
        size = int(f.readline().decode().split()[1])    # second line is "dim 128 128 128", the grid is a cube
        line = ""
        while line != "data":                           # skip the translate and scale lines
            line = f.readline().decode().strip()
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    # the data is run-length encoded as (value, count) pairs
    voxels = np.repeat(raw[0::2], raw[1::2])
    return voxels.reshape(size, size, size).astype(np.float32)


def load_voxel(path):
    voxels = torch.tensor(read_binvox(path))
    n = voxels.shape[0]
    block = n // VOXEL_RESOLUTION    # for example 128 -> 32 merges blocks of 4 voxels
    # max pooling: a block counts as filled if any voxel inside it is filled
    voxels = max_pool3d(voxels.reshape(1, 1, n, n, n), block)
    return voxels.reshape(1, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION)   # 1 = channel


#Point clouds
def load_vertices(path):
    verts = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):    # vertex lines look like "v x y z"
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return torch.tensor(verts)


def sample_points(verts):
    n = len(verts)
    if n >= NUM_POINTS:
        indices = torch.randperm(n)[:NUM_POINTS]    # random subset of the vertices
        return verts[indices]
    # too few vertices: fill up with zero points
    padding = torch.zeros(NUM_POINTS - n, 3)
    return torch.cat([verts, padding])


#Meshes
def load_mesh(path):
    from pytorch3d.io import load_obj    # imported here so this file also works without PyTorch3D
    verts, faces, aux = load_obj(path, load_textures=False)    # aux (normals, textures) is not needed
    return verts, faces.verts_idx


def mesh_collate(batch):
    from pytorch3d.structures import Meshes
    verts = []
    faces = []
    labels = []
    for item in batch:
        verts.append(item[0])
        faces.append(item[1])
        labels.append(item[2])
    # Meshes holds several meshes of different sizes in one object
    return Meshes(verts=verts, faces=faces), torch.tensor(labels)


#Dataset and loader
class ShapeDataset(Dataset):    # PyTorch needs a class here, with __len__ and __getitem__
    def __init__(self, split, kind):
        table = load_splits()
        table = table[table["split"] == split]
        self.synsets = table["synset"].tolist()
        self.ids = table["model_id"].tolist()
        self.kind = kind    # "voxel", "mesh" or "points"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        synset = self.synsets[index]
        label = SYNSETS.index(synset)    # class number = position in the config list
        folder = os.path.join(DATA_PATH, synset, self.ids[index], "models")
        if self.kind == "voxel":
            return load_voxel(os.path.join(folder, VOXEL_FILE)), label
        mesh_path = os.path.join(folder, MESH_FILE)
        if self.kind == "points":
            return sample_points(load_vertices(mesh_path)), label
        mesh = load_mesh(mesh_path)
        return mesh[0], mesh[1], label


def get_loader(split, kind):
    collate = None    # None = PyTorch's default way of stacking samples
    if kind == "mesh":
        collate = mesh_collate    # meshes have different sizes, so they need their own collate
    # shuffle only the training data
    return DataLoader(ShapeDataset(split, kind), batch_size=BATCH_SIZE, shuffle=(split == "train"),
                      num_workers=NUM_WORKERS, collate_fn=collate)


if __name__ == "__main__":
    table = load_splits()
    print(pd.crosstab(table["synset"], table["split"]))    # models per class per split
    for kind in ["voxel", "points"]:
        for batch in get_loader("train", kind):
            print(kind, "batch:", batch[0].shape, "labels:", batch[1].shape)
            break    # one batch is enough for the test
    check_models()    # last, so it runs when everything else has finished