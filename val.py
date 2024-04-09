import os, sys
from ultralytics import YOLO

import argparse

# create an argparser for training
parser = argparse.ArgumentParser(description="Validate YOLOv8 on custom dataset")
parser.add_argument("--data", type=str, help="Dataset yaml")
parser.add_argument("--imgsz", type=int, default=416, help="Training size")
parser.add_argument("--ckpt", type=str, help="Checkpoint path")
args = parser.parse_args()

def main(argv):
    model = YOLO(args.ckpt)
    metrics = model.val(data=args.data, imgsz=args.imgsz)

if __name__ == '__main__':
    main(sys.argv)
