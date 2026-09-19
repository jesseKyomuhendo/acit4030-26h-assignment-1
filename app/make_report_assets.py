#Imports
import os
import pandas as pd
from torchinfo import summary
from app.ConfigManager import (CLASS_NAMES, SYNSETS, NUM_CLASSES, MESH_HIDDEN, MESH_FC, VOXEL_RESOLUTION,
                               NUM_POINTS, CSV_PATH, RESULTS_PATH, NON_RESULTS_PATH)
from app.data import load_splits
from app.models import get_model
from app.plots import NAMES, LABELS, save_table, figure1, figure2, figure3


#Helpers
def mean_std(values, digits):
    text = format(values.mean(), "." + str(digits) + "f")
    if len(values) > 1:
        text = text + " ± " + format(values.std(), "." + str(digits) + "f")    # only with several seeds
    return text


def count_params(module):
    total = 0
    for weights in module.parameters():
        total = total + weights.numel()
    return total


#Table 1: dataset overview
def table1():
    splits = load_splits()
    rows = []
    for i in range(NUM_CLASSES):
        part = splits[splits["synset"] == SYNSETS[i]]
        n_train = len(part[part["split"] == "train"])
        n_val = len(part[part["split"] == "val"])
        n_test = len(part[part["split"] == "test"])
        rows.append([CLASS_NAMES[i], SYNSETS[i], n_train, n_val, n_test, len(part)])
    df = pd.DataFrame(rows, columns=["Class", "Synset", "Train", "Val", "Test", "Total"])
    df.loc[len(df)] = ["Total", ""] + df[["Train", "Val", "Test", "Total"]].sum().tolist()    # add a total row
    return df


#Tables 2 and 4: architecture from torchinfo
def architecture_table(model, input_size):
    info = summary(model, input_size=input_size, verbose=0, mode="eval")
    rows = []
    for layer in info.summary_list:
        # keep only layers with weights, so ReLU, pooling and dropout are left out
        if layer.is_leaf_layer and layer.num_params > 0:
            rows.append([layer.class_name, str(layer.output_size[1:]), layer.num_params])    # [1:] drops the batch size
    rows.append(["Total", "", info.total_params])
    return pd.DataFrame(rows, columns=["Layer", "Output shape", "Parameters"])


#Table 3: mesh model by hand, torchinfo does not work with Meshes
def mesh_table():
    model = get_model("mesh")    # needs PyTorch3D
    rows = []
    for i in range(len(MESH_HIDDEN)):
        rows.append(["GraphConv " + str(i + 1), "(V, " + str(MESH_HIDDEN[i]) + ")", count_params(model.convs[i])])
    rows.append(["Mean over vertices", "(" + str(MESH_HIDDEN[-1]) + ")", 0])
    rows.append(["Linear", "(" + str(MESH_FC) + ")", count_params(model.classifier[0])])
    rows.append(["Linear", "(" + str(NUM_CLASSES) + ")", count_params(model.classifier[3])])
    rows.append(["Total", "", count_params(model)])
    return pd.DataFrame(rows, columns=["Layer", "Output shape (V = vertices)", "Parameters"])


#Table 6: main results
def table6():
    summary_csv = pd.read_csv(CSV_PATH / "summary.csv")
    rows = []
    for i in range(len(NAMES)):
        part = summary_csv[summary_csv["model"] == NAMES[i]]
        if len(part) == 0:
            continue
        rows.append([LABELS[i], mean_std(part["test_acc"] * 100, 1), mean_std(part["macro_f1"], 3),
                     format(int(part["params"].iloc[0]), ","),    # thousands separator
                     mean_std(part["train_time_per_epoch_s"], 1), mean_std(part["infer_time_ms_per_shape"], 2),
                     mean_std(part["peak_gpu_mem_mb"], 0), len(part)])
    columns = ["Model", "Accuracy (%)", "Macro-F1", "Parameters", "Train time / epoch (s)",
               "Inference (ms / shape)", "Peak GPU memory (MB)", "Runs"]
    return pd.DataFrame(rows, columns=columns)


#Table 7: per-class precision / recall / F1
def table7():
    per_class = pd.read_csv(CSV_PATH / "per_class_metrics.csv")
    columns = ["Class"]
    for label in LABELS:
        columns.append(label + " (P / R / F1)")
    rows = []
    for name in CLASS_NAMES:
        row = [name]
        for model in NAMES:
            part = per_class[(per_class["model"] == model) & (per_class["class"] == name)]
            if len(part) == 0:
                row.append("-")
                continue
            text = ""
            for metric in ["precision", "recall", "f1"]:
                text = text + format(part[metric].mean(), ".2f") + " / "
            row.append(text[:-3])    # remove the last " / "
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


if __name__ == "__main__":
    os.makedirs(CSV_PATH, exist_ok=True)    # also creates output/results
    os.makedirs(NON_RESULTS_PATH, exist_ok=True)

    #Things that need no training results
    save_table(table1(), NON_RESULTS_PATH / "table1_dataset_overview.png")
    voxel_input = (1, 1, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION)    # batch of 1
    save_table(architecture_table(get_model("voxel"), voxel_input), NON_RESULTS_PATH / "table2_voxel_cnn_architecture.png")
    save_table(architecture_table(get_model("pointnet"), (1, NUM_POINTS, 3)), NON_RESULTS_PATH / "table4_pointnet_architecture.png")
    try:
        save_table(mesh_table(), NON_RESULTS_PATH / "table3_mesh_gcn_architecture.png")
        figure1(NON_RESULTS_PATH / "fig1_representations.png")
    except ImportError:
        print("Table 3 and Figure 1 skipped, PyTorch3D is not installed")

    #Things made from the training results
    if os.path.exists(CSV_PATH / "summary.csv"):
        save_table(table6(), RESULTS_PATH / "table6_main_results.png")
        save_table(table7(), RESULTS_PATH / "table7_per_class_metrics.png")
        figure2(RESULTS_PATH / "fig2_training_curves.png")
        figure3(RESULTS_PATH / "fig3_confusion_matrices.png")
    else:
        print("No results yet, skipped Tables 6, 7 and Figures 2, 3")
    print("Done")