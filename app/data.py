#Imports
import os
import random
import pandas as pd
from app.ConfigManager import (DATA_PATH, SYNSETS, MESH_FILE, VOXEL_FILE,
                               SPLITS_FILE, TRAIN_RATIO, VAL_RATIO, SPLIT_SEED)


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


if __name__ == "__main__":
    table = load_splits()
    print(pd.crosstab(table["synset"], table["split"]))   # models per class per split