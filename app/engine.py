#Imports
import os
import time
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from app.ConfigManager import CLASS_NAMES, NUM_CLASSES, EPOCHS, LEARNING_RATE, CSV_PATH
from app.data import get_loader
from app.models import get_model


#Helpers
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync():
    # the GPU works in the background, so wait for it before reading the clock
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def save_rows(new, path):
    if os.path.exists(path):
        old = pd.read_csv(path)
        # a rerun of the same model and seed replaces its old rows
        same = (old["model"] == new["model"].iloc[0]) & (old["seed"] == new["seed"].iloc[0])
        new = pd.concat([old[~same], new])
    new.to_csv(path, index=False)


#Training and evaluation
def train_one_epoch(model, loader, optimizer, loss_fn, device, limit):
    model.train()
    total_loss = 0
    count = 0
    batches = 0
    for inputs, labels in loader:
        if limit > 0 and batches >= limit:    # limit is only used in debug runs
            break
        outputs = model(inputs.to(device))
        loss = loss_fn(outputs, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item() * len(labels)    # weight by batch size
        count = count + len(labels)
        batches = batches + 1
    return total_loss / count


def evaluate(model, loader, loss_fn, device, limit):
    model.eval()
    total_loss = 0
    model_time = 0
    preds = []
    true = []
    batches = 0
    with torch.no_grad():    # no gradients needed, saves memory
        for inputs, labels in loader:
            if limit > 0 and batches >= limit:
                break
            inputs = inputs.to(device)
            sync()
            start = time.time()
            outputs = model(inputs)
            sync()
            model_time = model_time + time.time() - start    # model only, not the file loading
            total_loss = total_loss + loss_fn(outputs, labels.to(device)).item() * len(labels)
            preds = preds + outputs.argmax(dim=1).cpu().tolist()    # class with the highest score
            true = true + labels.tolist()
            batches = batches + 1
    return {"loss": total_loss / len(true), "acc": accuracy_score(true, preds),
            "preds": preds, "true": true, "time": model_time}


#One full experiment
def train_and_test(name, seed, debug=False):
    limit = 0
    epochs = EPOCHS
    if debug:
        limit = 3       # only 3 batches per epoch
        epochs = 2
    set_seed(seed)
    device = get_device()
    model = get_model(name).to(device)
    kind = name
    if name == "pointnet":
        kind = "points"    # the data loader calls it "points"
    train_loader = get_loader("train", kind)
    val_loader = get_loader("val", kind)
    test_loader = get_loader("test", kind)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    history = []
    best_acc = -1
    best_epoch = 0
    best_state = None
    train_time = 0
    for epoch in range(1, epochs + 1):
        sync()
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, limit)
        sync()
        train_time = train_time + time.time() - start    # includes file loading, as in a real run
        val = evaluate(model, val_loader, loss_fn, device, limit)
        history.append([name, seed, epoch, train_loss, val["loss"], val["acc"]])
        print(name, "epoch", epoch, "| train loss", round(train_loss, 4), "| val acc", round(val["acc"], 4))
        # keep the weights of the best epoch, chosen by validation accuracy only
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    # the test set is used once, with the best weights
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, loss_fn, device, limit)
    labels = list(range(NUM_CLASSES))
    p, r, f, support = precision_recall_fscore_support(test["true"], test["preds"], labels=labels, zero_division=0)
    matrix = confusion_matrix(test["true"], test["preds"], labels=labels)

    params = 0
    for weights in model.parameters():
        params = params + weights.numel()
    peak_mb = 0
    gpu = "cpu"
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20
        gpu = torch.cuda.get_device_name(0)
    print(name, "test accuracy:", round(test["acc"], 4), "| macro F1:", round(f.mean(), 4))
    if debug:
        print("debug run, nothing saved")
        return

    #Save the results as CSV files
    os.makedirs(CSV_PATH, exist_ok=True)
    columns = ["model", "seed", "epoch", "train_loss", "val_loss", "val_acc"]
    save_rows(pd.DataFrame(history, columns=columns), CSV_PATH / "training_history.csv")
    summary = pd.DataFrame([[name, seed, test["acc"], f.mean(), params, train_time / epochs,
                             test["time"] / len(test["true"]) * 1000, peak_mb, best_epoch, gpu]],
                           columns=["model", "seed", "test_acc", "macro_f1", "params", "train_time_per_epoch_s",
                                    "infer_time_ms_per_shape", "peak_gpu_mem_mb", "best_epoch", "gpu"])
    save_rows(summary, CSV_PATH / "summary.csv")
    per_class = pd.DataFrame({"model": name, "seed": seed, "class": CLASS_NAMES,
                              "precision": p, "recall": r, "f1": f, "support": support})
    save_rows(per_class, CSV_PATH / "per_class_metrics.csv")
    matrix_file = "confusion_matrix_" + name + "_seed" + str(seed) + ".csv"
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(CSV_PATH / matrix_file)