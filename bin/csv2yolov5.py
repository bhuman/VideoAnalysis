#!/usr/bin/env python3
"""Create annotation files for yolov5 from a csv file.
This script creates the directories 'DATA_ROOT/images/(train|val|test)' and
'DATA_ROOT/labels/(train|val|test)'. It copies all image files mentioned in
the csv file to subdirectories of 'DATA_ROOT/images'. It also creates a
.txt file for each image under a matching path under 'DATA_ROOT/labels'.
It also creates a yaml file in 'DATA_ROOT' with the same base name
as the csv file that describes the dataset.
"""
from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Use every 'USE_RATIO' image.
USE_RATIO = 50

# of these, use every 'TEST_RATIO' as test images
TEST_RATIO = 10

# or use every 'VAL_RATIO' as validation images, the others for training.
VAL_RATIO = 5

# Hard code image size.
WIDTH = 1920
HEIGHT = 1080


# The classes the network should detect.
CLASSES = ["ball", "blue", "red", "yellow", "black", "green", "gray"]

# map color numbers to classes (-1 -> currently not present in data)
#                   blue, red, yellow, black, white, green, orange, purple, brown, gray, ball
COLOR_TO_CLASSES = [1, 2, 3, 4, -1, 5, -1, -1, -1, 6, 0]


DATA_ROOT = Path(__file__).parents[1].resolve() / "data"  # The project root.


def main(file: Path, images: Path) -> None:
    # Create the output directories if they do not already exist.
    for path_ in ("images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"):
        path = DATA_ROOT / path_
        path.mkdir(parents=True, exist_ok=True)

    # Create yaml file for training data
    content = {
        "train": DATA_ROOT.joinpath("images/train/").as_posix(),
        "val": DATA_ROOT.joinpath("images/val/").as_posix(),
        "test": DATA_ROOT.joinpath("images/test/").as_posix(),
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    with DATA_ROOT.joinpath(f"{file.stem}.yaml").open(mode="w", encoding="UTF-8", newline="\n") as file_:
        yaml.safe_dump(content, file_, sort_keys=False)

    # Read all rows and sort them (this might move a title row to the end)
    with file.open(encoding="UTF-8", newline="") as file_:
        rows = sorted(list(csv.reader(file_)), key=lambda row: row[0])

    # Init counters
    use_count: int = 0
    image_count: int = 0
    current_file: str = ""
    lines: list[str] = []
    # Go through all rows
    for row in rows:
        # Skip processing if filename is invalid, i.e. title row.
        if not row[0].endswith(".jpg"):
            continue
        # Initialize first file.
        if not current_file:
            current_file = row[0]

        # Next image?
        if row[0] != current_file:
            current_file = row[0]

            use_count += 1
            if use_count % USE_RATIO == 0:
                image_count += 1
                dir_ = "test" if image_count % TEST_RATIO == 0 else "val" if image_count % VAL_RATIO == 1 else "train"
                shutil.copyfile(images / current_file, DATA_ROOT / f"images/{dir_}/{image_count}.jpg")
                DATA_ROOT.joinpath(f"labels/{dir_}/{image_count}.txt").write_text(
                    "\n".join(lines) + "\n", encoding="UTF-8"
                )
            # Reset for next image.
            lines = []

        # Interpret csv rows, assuming that only the ball has color -1
        left = float(row[2]) / WIDTH
        right = float(row[4]) / WIDTH
        top = float(row[3]) / HEIGHT
        bottom = float(row[5]) / HEIGHT
        color = int(row[6])
        lines.append(
            f"{COLOR_TO_CLASSES[color]} {(left + right) / 2} {(top + bottom) / 2} {right - left} {bottom - top}"
        )

    # Enable exporting the last image.
    use_count += 1
    if use_count % USE_RATIO == 0:
        image_count += 1
        dir_ = "test" if image_count % TEST_RATIO == 0 else "val" if image_count % VAL_RATIO == 1 else "train"
        shutil.copyfile(images / current_file, DATA_ROOT / f"images/{dir_}/{image_count}.jpg")
        DATA_ROOT.joinpath(f"labels/{dir_}/{image_count}.txt").write_text("\n".join(lines) + "\n", encoding="UTF-8")


if __name__ == "__main__":
    import click

    @click.command(context_settings=dict(help_option_names=["-h", "--help"]))
    @click.argument(
        "csv-file",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        metavar="<csv-file>",
    )
    @click.argument(
        "images",
        required=False,
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        metavar="<images>",
    )
    def cli(csv_file: Path, images: Path) -> None:
        """Show labels in the csv file.

        \b
        <csv-file> - The csv file so show.
        <images>   - The folder containing the images. Defaults to folder "images" next to <csv-file>.
        """
        main(csv_file, csv_file.parent / "images" if images is None else images)

    logging.basicConfig(level=logging.INFO, format="{levelname:>8}: {message}", style="{")
    cli()  # pylint: disable=no-value-for-parameter
