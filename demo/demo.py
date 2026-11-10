# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm
import json

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from cat_seg import add_cat_seg_config
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "MaskFormer demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/ade20k-150/maskformer_R50_bs16_160k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--gt",
        help="Path to the ground-truth semantic mask for reporting per-class IoU.",
    )
    parser.add_argument(
        "--class-json",
        help="Path to class-name json for reporting per-class IoU.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def load_class_names(class_json, num_classes):
    if class_json:
        with open(class_json, "r") as f:
            names = json.load(f)
        if len(names) >= num_classes:
            return names[:num_classes]
    return [f"class_{idx}" for idx in range(num_classes)]


def report_per_class_iou(pred, gt_path, class_json=None):
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    if gt is None:
        raise FileNotFoundError(f"Ground-truth mask not found: {gt_path}")
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    if gt.shape != pred.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

    num_classes = int(max(pred.max(), gt[gt != 255].max() if np.any(gt != 255) else 0)) + 1
    class_names = load_class_names(class_json, num_classes)
    valid = gt != 255
    ious = []

    print("Per-class IoU:")
    for class_id in range(num_classes):
        pred_mask = (pred == class_id) & valid
        gt_mask = (gt == class_id) & valid
        union = np.logical_or(pred_mask, gt_mask).sum()
        if union == 0:
            continue
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        iou = intersection / union * 100.0
        ious.append(iou)
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        print(f"  {class_id:02d} {class_name}: {iou:.2f}")

    if ious:
        print(f"Mean IoU: {np.mean(ious):.2f}")


def save_prediction_gt_comparison(image_bgr, pred_vis, pred_mask, gt_path, metadata, out_dir):
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    if gt is None:
        raise FileNotFoundError(f"Ground-truth mask not found: {gt_path}")
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    if gt.shape[:2] != image_bgr.shape[:2]:
        gt = cv2.resize(gt, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    image_rgb = image_bgr[:, :, ::-1]
    gt_visualizer = Visualizer(image_rgb, metadata)
    gt_visualizer._default_font_size = max(gt_visualizer._default_font_size, 35)
    gt_vis = gt_visualizer.draw_sem_seg(gt, alpha=0.4).get_image()
    pred_rgb = pred_vis.get_image()

    if pred_mask.shape != gt.shape:
        pred_mask = cv2.resize(pred_mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    valid = gt != 255
    wrong = (pred_mask != gt) & valid
    error_vis = image_rgb.copy()
    error_vis[wrong] = (0.35 * error_vis[wrong] + 0.65 * np.array([255, 0, 0])).astype(np.uint8)

    if pred_rgb.shape[:2] != gt_vis.shape[:2]:
        pred_rgb = cv2.resize(pred_rgb, (gt_vis.shape[1], gt_vis.shape[0]), interpolation=cv2.INTER_LINEAR)

    title_h = 32
    panel_w = gt_vis.shape[1]
    canvas = np.full((gt_vis.shape[0] + title_h, panel_w * 3, 3), 255, dtype=np.uint8)
    canvas[title_h:, :gt_vis.shape[1]] = pred_rgb
    canvas[title_h:, panel_w:panel_w * 2] = gt_vis
    canvas[title_h:, panel_w * 2:] = error_vis
    cv2.putText(canvas, "Prediction(baseline)", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "GT", (panel_w + 12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Error", (panel_w * 2 + 12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    os.makedirs(out_dir, exist_ok=True)
    outputs = {
        "prediction.png": pred_rgb,
        "gt.png": gt_vis,
        "error.png": error_vis,
        "comparison.png": canvas,
    }
    for name, image in outputs.items():
        out_path = os.path.join(out_dir, name)
        ok = cv2.imwrite(out_path, image[:, :, ::-1])
        if not ok:
            raise RuntimeError(f"Failed to write output image: {out_path}")
    print(f"Saved outputs: {out_dir}")


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            if args.gt and "sem_seg" in predictions:
                pred = predictions["sem_seg"].argmax(dim=0).to("cpu").numpy().astype(np.uint8)
                report_per_class_iou(pred, args.gt, args.class_json)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                if args.gt and "sem_seg" in predictions:
                    result_dir = os.path.join(os.path.dirname(out_filename), os.path.splitext(os.path.basename(path))[0])
                    save_prediction_gt_comparison(img, visualized_output, pred, args.gt, demo.metadata, result_dir)
                else:
                    result_dir = os.path.join(os.path.dirname(out_filename), os.path.splitext(os.path.basename(path))[0])
                    os.makedirs(result_dir, exist_ok=True)
                    out_filename = os.path.join(result_dir, "prediction.png")
                    visualized_output.save(out_filename)
                    print(f"Saved prediction: {out_filename}")
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
