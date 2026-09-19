#Imports
import argparse
from app.ConfigManager import SEED
from app.engine import train_and_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pointnet")          # voxel, mesh or pointnet
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--debug", action="store_true")         # 2 epochs of 3 batches, nothing saved
    args = parser.parse_args()
    train_and_test(args.model, args.seed, args.debug)