#!/usr/bin/env python3
"""Concatenate a number of csv files, but only keep the header of the first one.

Can be used to concatenate a number of statistics files.
"""
from __future__ import annotations

from pathlib import Path


def main(paths: list[Path]) -> None:
    """Print all files, but skip each first line, except for the first file.

    :param paths: The paths of the files to print.
    """
    firstLine: int = 0
    for path in paths:
        with path.open(encoding="UTF-8", newline="") as file_:
            for line in file_.read().splitlines()[firstLine:]:
                print(line)
        firstLine = 1


if __name__ == "__main__":
    import click

    @click.command(context_settings=dict(help_option_names=["-h", "--help"]))
    @click.argument(
        "csv-files",
        nargs=-1,
        required=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        metavar="<csv-files>",
    )
    def cli(csv_files: list[Path]) -> None:
        """Prints the csv files, but only prints the header of the first one.

        \b
        <csv-files> - The csv files to print.
        """
        main(csv_files)

    cli()  # pylint: disable=no-value-for-parameter
