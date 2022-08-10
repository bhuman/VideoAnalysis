# Training the YOLOv5 Network

## Downloading the Pre-trained Network

Download [YOLOv5n6](https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n6.pt)
and store it in `weights`.


## Preparing the Data

Download [the labeled data](https://tu-dortmund.sciebo.de/s/1akh5dJbuNcss0i) and
unpack it. In the terminal, go into the main directory of the unpacked archive and
create a joined csv file that contains all data except for the one by SPQR (which is
inconsistent with the others). Also note that the extension of the images names is
wrong in one of the dataset. Therefore, `.png` has to be replaced by `.jpg`:

    for i in */*.csv; do
      sed  <$i 's%\.png%.jpg%' \
      | grep '\.jpg' $i \
      | sed -e "s%^%$(dirname $i)/images/%" \
      | grep -v SPQR
    done >VideoAnalysisChallenge.csv

Go to the main directory of this repository and run:

    bin/csv2yolov5.py /path/to/VideoAnalysisChallenge.csv

This creates `data/VideoAnalysisChallenge.yaml` and
`data/(images|labels)/(train|val|test)/*.(jpg|txt)`.


## Training

Run (adapt `--batch` if necessary):

    yolov5 train --weights weights/yolov5n6.pt --data data/VideoAnalysisChallenge.yaml --img 1920 --batch 8 --epochs 200

The results will be written to `runs/train/exp*`. Further instructions assume that
the network was written to `runs/train/exp`.


## Detection

Run:

    yolov5 detect --weights runs/train/exp/weights/best.pt --data data/VideoAnalysisChallenge.yaml --img 1088,1920 --line-thickness 2 --hide-labels --conf 0.3 --source /path/to/image_dir_or_video

The results will be written to `runs/detect/exp*`.


## Validation

To get some statistics about the training results, run:

    yolov5 val --weights runs/train/exp/weights/best.pt --data data/VideoAnalysisChallenge.yaml --img 1920 --task test

The results will be written to `runs/val/exp*`.


## Committing a Trained Model

To finally commit a trained model, copy `runs/train/exp/weights/best.pt` to
`weights` in the main directory of this repository.
