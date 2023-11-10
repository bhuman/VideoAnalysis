# Training the YOLOv5 Network

## Downloading the Pre-trained Network

Download [YOLOv5n6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n6.pt)
and store it in `weights`.


## Preparing the Data

Download [the labeled data](https://b-human.informatik.uni-bremen.de/public/datasets/video_analysis/video_analysis.zip)
and unpack it. Edit the first row of the file `merged.yaml` replacing the
placeholder path that is given there by the absolute path to the directory that
contains the file `merged.yaml`


## Training

From the main directory of this repository, run (replace `/path/to/` and adapt
`--batch` if necessary):

    yolov5 train --weights weights/yolov5n6.pt --data /path/to/merged.yaml --img 1920 --batch 8 --epochs 400

The results will be written to `runs/train/exp*`. Further instructions assume that
the network was written to `runs/train/exp`.


## Detection

Run:

    yolov5 detect --weights runs/train/exp/weights/best.pt --data /path/to/merged.yaml --img 1088,1920 --line-thickness 2 --hide-labels --conf 0.3 --source /path/to/image_dir_or_video

The results will be written to `runs/detect/exp*`.


## Validation

To get some statistics about the training results, run:

    yolov5 val --weights runs/train/exp/weights/best.pt --data /path/to/merged.yaml --img 1920 --task test

The results will be written to `runs/val/exp*`.


## Committing a Trained Model

To finally commit a trained model, copy `runs/train/exp/weights/best.pt` to
`weights` in the main directory of this repository.
