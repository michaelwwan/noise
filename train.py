import os, sys
from ultralytics import YOLO
import argparse

# create an argparser for training
parser = argparse.ArgumentParser(description="Train YOLOv8 on custom dataset")
parser.add_argument("--data", type=str, default="./osteo.yaml", help="path to data.yaml")
parser.add_argument("--name", type=str, default="noise", help="experiment name")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
parser.add_argument("--imgsz", type=int, default=416, help="Training size")
parser.add_argument("--ckpt", type=str, default="yolov8l-seg.pt", help="Previous checkpoint -- defaults to COCO seg")
args = parser.parse_args()

def main(argv):
    model = YOLO('yolov8l-seg.pt').load(args.ckpt)
    model.train(exist_ok=True, name=args.name, data=args.data, epochs=args.epochs, imgsz=args.imgsz)

if __name__ == '__main__':
    main(sys.argv)
