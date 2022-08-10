#!/usr/bin/env python3
"""Show labels in the csv file."""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import cv2 as cv

logger = logging.getLogger(__name__)

COLORS: list[tuple[int, int, int]] = [
    (255, 128, 0),  # 0: blue
    (0, 0, 255),  # 1: red
    (0, 255, 255),  # 2: yellow
    (0, 0, 0),  # 3: black
    (255, 255, 255),  # 4: white
    (0, 255, 0),  # 5: green
    (0, 128, 255),  # 6: orange
    (255, 0, 255),  # 7: purple
    (0, 64, 128),  # 8: brown
    (192, 192, 192),  # 9: gray
    (128, 128, 128),  # -1: unknown
]


def main(file: Path, images: Path) -> None:
    with file.open(encoding="UTF-8", newline="") as file_:
        rows = sorted(list(csv.reader(file_)), key=lambda row: row[0])

    # Open as front most window
    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.setWindowProperty("image", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.setWindowProperty("image", cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)

    # Start with continuous playback
    single_step: bool = False

    numbers: set[int] = set()
    current_file: str = ""
    image: cv.Mat | None = None
    for row in rows:
        # Skip processing if filename is invalid, i.e. title row.
        if not row[0].endswith(".jpg"):
            continue
        # Initialize first file.
        if image is None:
            current_file = row[0]
            image = cv.imread(str(images / row[0]))
            cv.putText(image, current_file, (0, 24), cv.FONT_HERSHEY_SIMPLEX, 1, COLORS[2], 2, 2)

        # Next image?
        if row[0] != current_file:
            # Show the previously filled image.
            cv.imshow("image", image)

            # Wait for key press or just check.
            key = cv.waitKey(0 if single_step else 1)

            # Quit if Escape was pressed.
            if key == 27:  # Escape
                break
            # Toggle between play and single step.
            if key == 32:  # Space
                single_step = not single_step

            current_file = row[0]

            # Load next image.
            image = cv.imread(str(images / current_file))
            cv.putText(image, current_file, (0, 24), cv.FONT_HERSHEY_SIMPLEX, 1, COLORS[2], 2, 2)
            numbers = set()

        # Interpret csv rows
        is_ball = row[1] == "1"
        left = int(round(float(row[2])))
        top = int(round(float(row[3])))
        right = int(round(float(row[4])))
        bottom = int(round(float(row[5])))
        color = int(row[6])
        number = int(row[7])

        # Sanity checks
        if is_ball and color != -1:
            logger.warning("Ball with color %s in %s", color, current_file)
        if not is_ball and (color < 0 or color >= len(COLORS)):
            logger.warning("Wrong color %s in %s", color, current_file)
            continue  # prevent out of bounds error
        if is_ball and number != -1:
            logger.warning("Ball with player number %s in %s", number, current_file)
        if not is_ball and (number < 1 or (5 < number < 100)):
            logger.warning("Wrong player number %s in %s", number, current_file)
        if number + 1000 * color in numbers:
            if is_ball:
                logger.warning("Duplicate ball in %s", current_file)
            else:
                logger.warning("Duplicate player color %s number %s in %s", color, number, current_file)
        numbers.add(number + 1000 * color)

        # draw rectangle and number
        cv.rectangle(image, (left, top), (right, bottom), COLORS[6] if is_ball else COLORS[color], 1)
        if not is_ball and number != -1:
            cv.putText(image, str(number), (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, COLORS[color], 2, 2)

    if image is not None:
        # Show the previously filled image.
        cv.imshow("image", image)
    cv.destroyAllWindows()


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

        Expects the images mentioned in the file in a subdirectory <images>.

        \b
        <csv-file> - The csv file so show.
        <images>   - The folder containing the images. Defaults to folder "images" next to <csv-file>.
        """
        main(csv_file, csv_file.parent / "images" if images is None else images)

    logging.basicConfig(level=logging.INFO, format="{levelname:>8}: {message}", style="{")
    cli()  # pylint: disable=no-value-for-parameter
