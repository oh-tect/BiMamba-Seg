#!/usr/bin/env sh

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir" || exit 1

img="$1"
gt="$2"
time=$(date +"%Y-%m-%d")
output="$script_dir"
out_dir="$script_dir/experiment/visualization/$time"

if [ -z "$img" ]; then
  echo "No input image found! Run with: sh inference.sh [PATH2IMG] [PATH2GT]"
    exit 1
fi

mkdir -p "$out_dir"

run_infer() {
    class_json="$1"
    num_classes="$2"
    dataset_name="$3"

    if [ -n "$gt" ]; then
      python demo/demo.py --config-file configs/vitb_384_DLRSD.yaml \
        --input "$img" \
        --output "$out_dir" \
        --gt "$gt" \
        --class-json "$class_json" \
        --opts \
        MODEL.WEIGHTS "$output/model_final2.pth" \
        MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "$class_json" \
        MODEL.SEM_SEG_HEAD.NUM_CLASSES "$num_classes" \
        TEST.SLIDING_WINDOW False \
        DATASETS.TEST "('$dataset_name',)"
    else
      python demo/demo.py --config-file configs/vitb_384_DLRSD.yaml \
        --input "$img" \
        --output "$out_dir" \
        --opts \
        MODEL.WEIGHTS "$output/model_final2.pth" \
        MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "$class_json" \
        MODEL.SEM_SEG_HEAD.NUM_CLASSES "$num_classes" \
        TEST.SLIDING_WINDOW False \
        DATASETS.TEST "('$dataset_name',)"
    fi
}

case "$img" in
  *iSAID*)
    echo "iSAID dataset image detected, using iSAID config and model weights for inference."
    run_infer "/home/ubuntu/YuXiaoTong_workspace/CLIP/RSKT-Seg/datasets/iSAID.json" 16 "iSAID_all_sem_seg"
    ;;
  *UDD5*)
    echo "UDD5 dataset image detected, using UDD5 config and model weights for inference."
    run_infer "/home/ubuntu/YuXiaoTong_workspace/CLIP/RSKT-Seg/datasets/UDD5.json" 5 "UDD5_all_sem_seg"
    ;;
  *LoveDA*)
    echo "LoveDA dataset image detected, using LoveDA config and model weights for inference."
    run_infer "/home/ubuntu/YuXiaoTong_workspace/CLIP/RSKT-Seg/datasets/LoveDA.json" 8 "LoveDA_all_sem_seg"
    ;;
  *uavid*|*UAVid*)
    echo "UAVid dataset image detected, using UAVid config and model weights for inference."
    run_infer "/home/ubuntu/YuXiaoTong_workspace/CLIP/RSKT-Seg/datasets/uavid.json" 8 "uavid_all_sem_seg"
    ;;
  *VDD*)
    echo "VDD dataset image detected, using VDD config and model weights for inference."
    run_infer "/home/ubuntu/YuXiaoTong_workspace/CLIP/RSKT-Seg/datasets/VDD.json" 7 "VDD_all_sem_seg"
    ;;
  *Vaihingen*)
    echo "Vaihingen dataset image detected, using Vaihingen config and model weights for inference."
    run_infer "/home/ubuntu/YuXiaoTong_workspace/CLIP/RSKT-Seg/datasets/Vaihingen.json" 6 "Vaihingen_all_sem_seg"
    ;;
  *Potsdam*|*Postdam*)
    echo "Potsdam dataset image detected, using Potsdam config and model weights for inference."
    run_infer "/home/ubuntu/YuXiaoTong_workspace/CLIP/RSKT-Seg/datasets/Potsdam.json" 6 "Potsdam_all_sem_seg"
    ;;
  *DLRSD*)
    echo "DLRSD dataset image detected, using DLRSD config and model weights for inference."
    run_infer "/home/ubuntu/YuXiaoTong_workspace/CLIP/RSKT-Seg/datasets/DLRSD.json" 17 "DLRSD_all_sem_seg"
    ;;
  *Road_Extraction*)
    echo "Road_Extraction dataset image detected, using Road_Extraction config and model weights for inference."
    run_infer "/home/ubuntu/YuXiaoTong_workspace/CLIP/RSKT-Seg/datasets/road_extraction.json" 2 "Road_Extraction_CHN6_CUG_sem_seg"
    ;;
  *)
    echo "No dataset keyword matched in path: $img"
    echo "Supported keywords: UDD5, LoveDA, uavid/UAVid, VDD, Vaihingen, Potsdam, DLRSD, Road_Extraction"
    exit 1
    ;;
esac

# python demo/demo.py --config-file configs/vitb_384_DLRSD.yaml \
#  --input  $img \
#  --output  vis_output/$time \
#  --opts MODEL.WEIGHTS  $output/model_final.pth  MODEL.SEM_SEG_HEAD.NUM_LAYERS 5 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/UDD5.json" MODEL.SEM_SEG_HEAD.NUM_CLASSES 5 \
#   TEST.SLIDING_WINDOW "False"  DATASETS.TEST \(\"UDD5_all_sem_seg\"\,\)