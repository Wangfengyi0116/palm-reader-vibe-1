import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train palm line models.")
    parser.add_argument("--task", choices=["seg", "pose", "all"], default="all")
    parser.add_argument("--seg-data", default="datasets/palm_lines_seg.yaml")
    parser.add_argument("--pose-data", default="datasets/palm_hegu_pose.yaml")
    parser.add_argument("--seg-model", default="yolo11s-seg.pt")
    parser.add_argument("--pose-model", default="yolo11s-pose.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="cuda id or cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs")
    return parser.parse_args()


def train_seg(args):
    model = YOLO(args.seg_model)
    return model.train(
        data=args.seg_data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name="palm_seg",
    )


def train_pose(args):
    model = YOLO(args.pose_model)
    return model.train(
        data=args.pose_data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name="palm_pose",
    )


def main():
    args = parse_args()
    Path(args.project).mkdir(parents=True, exist_ok=True)

    if args.task in ("seg", "all"):
        print("[Train] Start segmentation model.")
        train_seg(args)
        print("[Train] Segmentation completed.")

    if args.task in ("pose", "all"):
        print("[Train] Start pose model.")
        train_pose(args)
        print("[Train] Pose completed.")

    print("[Train] All done.")


if __name__ == "__main__":
    main()
