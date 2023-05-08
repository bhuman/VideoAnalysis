#!/usr/bin/env python3

from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]  # 1 folder up.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Add ROOT to PATH.

from video_analysis import VideoAnalysis  # noqa: E402


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(version="1.0.0")
@click.argument("video", type=str)
@click.option(
    "-l",
    "--log",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The GameController log path.",
)
@click.option(
    "-f",
    "--field",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The field_dimensions.json path.  [default: Guess the dimensions.]",
)
@click.option(
    "-c",
    "--calibration",
    type=click.Path(dir_okay=False, path_type=Path),
    help="The camera calibration.json path.  [default: Use name based on game.]",
)
@click.option(
    "--half", type=click.IntRange(0, 2), default=0, show_default=True, help="Half the video shows. 0 means guess."
)
@click.option("-n", "--every-nth-frame", type=int, default=10, show_default=True, help="Only process every nth frame.")
@click.option("--headless", is_flag=True, help="Run without window.")
@click.option("-f", "--force", is_flag=True, help="Force new camera calibration.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output during camera calibration.")
def cli(
    video: str,
    log: Path,
    field: Path | None,
    half: int,
    every_nth_frame: int,
    headless: bool,
    calibration: Path | None,
    force: bool,
    verbose: bool,
) -> None:
    """Start the video analysis tool.

    VIDEO - The video file path or URL.
    """
    # Check whether the video exists if it is not an URL.
    is_url: bool = video.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    if not is_url and not Path(video).is_file():
        raise click.BadParameter(f"File '{video}' does not exist.", param_hint="'VIDEO'")

    analysis = VideoAnalysis(
        video=video,
        log=log,
        field=field,
        half=half,
        every_nth_frame=every_nth_frame,
        calibration=calibration,
        force=force,
        verbose=verbose,
        weights=ROOT / "weights" / ("best.mlmodel" if platform.system() == "Darwin" else "best.pt"),
        headless=headless,
    )
    analysis.run()


if __name__ == "__main__":
    # force=True because yolov5 instantiates its own logger without respecting the user.
    logging.basicConfig(level=logging.INFO, format="{levelname:>8}: {message}", style="{", force=True)
    cli()  # pylint: disable=no-value-for-parameter
