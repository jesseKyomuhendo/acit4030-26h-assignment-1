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
MESH_FILE = config["data"]["mesh_file"]
VOXEL_FILE = config["data"]["voxel_file"]
SPLITS_FILE = PROJECT_ROOT / config["split"]["file"]
TRAIN_RATIO = config["split"]["train"]
VAL_RATIO = config["split"]["val"]
SPLIT_SEED = config["split"]["seed"]


if __name__ == "__main__":
    #Retriving
    testMessage = config["test-block"]["message"]
    print(testMessage)