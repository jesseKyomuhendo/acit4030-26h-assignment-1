#Imports
from pathlib import Path
import yaml

#Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # util/ -> project root
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

#Other configurations



if __name__ == "__main__":
    #Retriving
    testMessage = config["test-block"]["message"]
    print(testMessage)