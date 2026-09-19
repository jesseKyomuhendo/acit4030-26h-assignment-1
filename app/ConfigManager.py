#Imports
from pathlib import Path
import yaml

#Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # app/ -> project root
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

#Other configurations
DATA_PATH = PROJECT_ROOT / config["data"]["path"]
SYNSETS = config["data"]["synsets"]
CLASS_NAMES = config["data"]["class_names"]
NUM_CLASSES = len(SYNSETS)
MESH_FILE = config["data"]["mesh_file"]
VOXEL_FILE = config["data"]["voxel_file"]
NUM_WORKERS = config["data"]["num_workers"]
SPLITS_FILE = PROJECT_ROOT / config["split"]["file"]
TRAIN_RATIO = config["split"]["train"]
VAL_RATIO = config["split"]["val"]
SPLIT_SEED = config["split"]["seed"]
BATCH_SIZE = config["training"]["batch_size"]
DROPOUT = config["training"]["dropout"]
EPOCHS = config["training"]["epochs"]
LEARNING_RATE = config["training"]["lr"]
SEED = config["training"]["seed"]
OUTPUT_PATH = PROJECT_ROOT / config["output"]["path"]
RESULTS_PATH = OUTPUT_PATH / "results"
NON_RESULTS_PATH = OUTPUT_PATH / "non_results"
CSV_PATH = RESULTS_PATH / "csv"
VOXEL_RESOLUTION = config["voxel_cnn"]["resolution"]
VOXEL_CHANNELS = config["voxel_cnn"]["channels"]
VOXEL_FC = config["voxel_cnn"]["fc_dim"]
MESH_HIDDEN = config["mesh_gcn"]["hidden_dims"]
MESH_FC = config["mesh_gcn"]["fc_dim"]
NUM_POINTS = config["pointnet"]["num_points"]
POINT_MLP = config["pointnet"]["mlp_dims"]
POINT_HEAD = config["pointnet"]["head_dims"]


if __name__ == "__main__":
    #Retriving
    testMessage = config["test-block"]["message"]
    print(testMessage)