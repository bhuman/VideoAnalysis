from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .log_parser import LogParser

if TYPE_CHECKING:
    import os

ROOT = Path(__file__).resolve().parents[2]


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
        STANDBY = 5

    def __init__(self, log: os.PathLike | str, snd_half: bool, settings: dict[str, Any]) -> None:
        """Initialize this instance by reading a GameController log.

        First, all undos are resolved. Then, the events from the correct half are picked.
        The two teams playing and their colors are determined and stored in a way that
        the team playing on the left in the video is stored at index 0 and the one shown
        on the right is stored at index 1. In addition, a `basename` is computed that
        can be used to name additional files that belong to the current game and half.
        """
        self.current_game_state: dict[str, Any] = {}
        self.current_status_messages: list[dict[str, Any]] = []
        self.state = self.State.INITIAL
        self.changed: bool = False
        self.playing_time: float = 0.0
        self.has_player_data: bool = False

        self.current_actions: list[dict[str, Any]] = []
        self.current_game_states: list[dict[str, Any]] = []

        self._log_parser: LogParser = LogParser(log, snd_half, settings)

        # The old log is not yaml and do not contain any player information except penalties
        self.has_player_data = self._log_parser.is_yaml
        self.basename = self._log_parser.basename
        self.teams = self._log_parser.teams

        self._handle_next_event()

    def update(self, timestamp: float, fps: float) -> None:
        """Update current GameController event.

        :param timestamp: The current time in seconds relative to the start of the video.
        :param fps: The frames per second the video is played back at.
        """
        self.current_status_messages.clear()
        self.current_actions.clear()
        self.current_game_states.clear()

        self.changed = False

        # Add all actions that happened since the last frame to the current list
        while (
            self._log_parser.action_timestamps
            and timestamp >= (self._log_parser.action_timestamps[0] - self._log_parser.start_time).total_seconds()
        ):
            self._log_parser.action_timestamps.pop(0)
            self.current_actions.append(self._log_parser.action_events.pop(0))

        # Add all game states that happened since the last frame to the current list
        while (
            self._log_parser.game_state_timestamps
            and timestamp >= (self._log_parser.game_state_timestamps[0] - self._log_parser.start_time).total_seconds()
        ):
            self._handle_next_event()
            self.changed = True

        while (
            self._log_parser.status_messages_timestamps
            and timestamp
            >= (self._log_parser.status_messages_timestamps[0] - self._log_parser.start_time).total_seconds()
        ):
            self._log_parser.status_messages_timestamps.pop(0)
            self.current_status_messages.append(self._log_parser.status_messages.pop(0))

        if self.state == self.State.PLAYING:
            self.playing_time += 1 / fps

    def _handle_next_event(self) -> None:
        self._log_parser.game_state_timestamps.pop(0)
        self.current_game_state = self._log_parser.game_state_events.pop(0)

        if self.current_game_state["state"] == "ready":
            self.state = self.State.READY
        elif self.current_game_state["state"] == "set":
            self.state = self.State.SET
        elif self.current_game_state["state"] == "playing":
            self.state = self.State.PLAYING
        elif self.current_game_state["state"] == "standby":
            self.state = self.State.STANDBY
        # TODO: This doesn't work in a lot of situations (e.g. timeouts)
