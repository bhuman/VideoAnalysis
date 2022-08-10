from __future__ import annotations

import os
from datetime import datetime, timedelta
from enum import Enum


class Team:
    """Color and name of a team."""

    def __init__(self, color: str, name: str) -> None:
        self.color: str = color
        self.name: str = name


class GameState:
    """The game state.

    Reads a GameController log and plays back the events from that log while
    time progresses.
    """

    class State(Enum):
        """The different game states."""

        INITIAL = 0
        READY = 1
        SET = 2
        PLAYING = 3
        FINISHED = 4

    _video_delay = 1.5
    """The number of seconds the video starts after leaving state Initial."""

    def __init__(self, log: os.PathLike | str, snd_half: bool) -> None:
        """Initialize this instance by reading a GameController log.

        First, all undos are resolved. Then, the events from the correct half are picked.
        The two teams playing and their colors are determined and stored in a way that
        the team playing on the left in the video is stored at index 0 and the one shown
        on the right is stored at index 1. In addition, a `basename` is computed that
        can be used to name additional files that belong to the current game and half.
        """
        self._timestamps: list[datetime] = []
        self._events: list[str] = []
        self.current_event: str = ""
        self.state = self.State.INITIAL
        self.changed: bool = False
        self.playing_time: float = 0.0

        # Read whole log file.
        with open(log, encoding="UTF-8", newline="\n") as logfile:
            lines = logfile.read().splitlines()

        # Handle undos.
        filtered_lines: list[str] = []
        for line in lines:
            event = line.split(": ")[1]
            if event.startswith("Undo "):
                undos = int(event.split(" ")[1])
                while len(filtered_lines) > 0 and undos > 0:
                    line = filtered_lines.pop()
                    if len(filtered_lines) > 0:
                        prev = filtered_lines[-1]
                        if not (
                            (line.endswith("Ready") or line.endswith("2nd Half") or line.endswith("Finished"))
                            and "Goal for" in prev
                            or line.endswith("Ready")
                            and (
                                "End of Timeout" in prev
                                or "End of Referee Timeout" in prev
                                or "Global Game Stuck" in prev
                                or "Penalty Kick for" in prev
                            )
                            or line.endswith("Initial")
                            and "Timeout" in prev
                            or "Message Budget Exceeded by" in line
                        ):
                            undos -= 1
            else:
                filtered_lines.append(line)
            lines = filtered_lines

        # Determine team names.
        teams = next(x.split(": ")[1] for x in lines if " vs " in x).split(" vs ")
        team0 = Team(color=teams[0][teams[0].find("(") + 1 : teams[0].find(")")], name=teams[0].split(" (")[0])
        team1 = Team(color=teams[1][teams[1].find("(") + 1 : teams[1].find(")")], name=teams[1].split(" (")[0])

        # Determine start time of the game (not half!)
        start_time = next(line for line in lines if " Ready" in line).split(":")[0]
        start_time = datetime.strptime(start_time, "%Y.%m.%d-%H.%M.%S").strftime("%Y-%m-%d_%H-%M")

        # Define unique basename for this half.
        self.basename = (
            f"{start_time}_{team0.name.replace(' ', '_')}_"
            + f"{team1.name.replace(' ', '_')}_{'2nd' if snd_half else '1st'}_Half"
        )

        # Start at first ready state, store teams.
        if snd_half:
            index = next((x for x in range(len(lines)) if " 2nd Half" in lines[x]), -2) + 1
            self.teams = (team1, team0)
        else:
            index = next((x for x in range(len(lines)) if " Ready" in lines[x]), -1)
            self.teams = (team0, team1)
        assert index != -1
        lines = lines[index:]

        # Initialize list of events / timestamps.
        for line in lines:
            self._timestamps.append(datetime.strptime(line.split(":")[0], "%Y.%m.%d-%H.%M.%S"))
            self._events.append(line.split(": ")[1])
        self.start_time = self._timestamps[0] + timedelta(0, self._video_delay)
        self._handle_next_event()

    def update(self, timestamp: float, fps: float) -> None:
        """Update current GameController event.

        :param timestamp: The current time in seconds relative to the start of the video.
        :param fps: The frames per second the video is played back at.
        """
        if not self._timestamps or not self._events:
            return

        next_event = (self._timestamps[0] - self.start_time).seconds
        if timestamp >= next_event:
            self._handle_next_event()
            self.changed = True
        else:
            self.changed = False

        if self.state == self.State.PLAYING:
            self.playing_time += 1 / fps

    def _handle_next_event(self) -> None:
        """Extract the next event and update the game state."""
        self._timestamps.pop(0)
        self.current_event = self._events.pop(0)
        if self.current_event == "Ready":
            self.state = self.State.READY
        elif self.current_event == "Set":
            self.state = self.State.SET
        elif self.current_event == "Playing":
            self.state = self.State.PLAYING
        # TODO: This doesn't work in a lot of situations (e.g. timeouts)
