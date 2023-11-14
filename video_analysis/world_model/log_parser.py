import base64
import os
import struct
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from yaml import CLoader as Loader
from yaml import load

from .team import Team

ROOT = Path(__file__).resolve().parents[2]


def generic_constructor(loader, tag, node):
    classname = node.__class__.__name__
    if classname == "SequenceNode":
        return {tag: loader.construct_sequence(node)}
    elif classname == "MappingNode":
        return {tag: loader.construct_mapping(node)}
    else:
        return {tag: loader.construct_scalar(node)}


class LogParser:
    def __init__(self, log: os.PathLike | str, snd_half: bool, settings: dict[str, Any]) -> None:
        self._log_file = log
        self.action_timestamps: list[datetime] = []
        self.action_events: list[dict[str, Any]] = []
        self.game_state_timestamps: list[datetime] = []
        self.game_state_events: list[dict[str, Any]] = []
        self.status_messages_timestamps: list[datetime] = []
        self.status_messages: list[dict[str, Any]] = []

        self.is_yaml: bool = False
        self._snd_half: bool = snd_half
        self._settings: dict[str, Any] = settings

        if ".yaml" in Path(log).name:
            self.is_yaml = True
            # Do yaml stuff
            loader = Loader
            loader.add_multi_constructor("", generic_constructor)

            document = open(log, encoding="UTF-8", newline="\n")
            log_entries = load(document, Loader=Loader)

            # Undo filtering
            filtered_entries: list[Any] = []
            is_timeout: bool = False
            begin_timeout: timedelta = timedelta()
            # The current_timedelta is the time which is removed due to timeouts
            current_timedelta: timedelta = timedelta()
            for entry in log_entries:
                event = entry["entry"]
                if (
                    "!action" in event
                    and event["!action"]["source"] == "user"
                    and event["!action"]["action"]["type"] == "undo"
                ):
                    undos = event["!action"]["action"]["args"]["states"]
                    counter: int = 1
                    while len(filtered_entries) > 0 and undos > 0:
                        entry = filtered_entries[-counter]
                        if "!action" in entry["entry"] and entry["entry"]["!action"]["source"] == "user":
                            filtered_entries.pop(-counter)
                            undos -= 1
                        else:
                            counter += 1
                elif (
                    "!action" in event
                    and event["!action"]["source"] == "user"
                    and event["!action"]["action"]["type"] == "timeout"
                ):
                    secs = entry["timestamp"]["secs"]
                    nanos = entry["timestamp"]["nanos"]
                    begin_timeout = timedelta(seconds=secs, microseconds=nanos / 1000)
                    is_timeout = True
                else:
                    if is_timeout:
                        if (
                            "!action" in event
                            and event["!action"]["source"] == "user"
                            and event["!action"]["action"]["type"] == "startSetPlay"
                            and event["!action"]["action"]["args"]["setPlay"] == "kickOff"
                        ):
                            # The gameState-entry is the one starting the video again
                            # The time difference needs to be calculated here
                            secs = entry["timestamp"]["secs"]
                            nanos = entry["timestamp"]["nanos"]
                            current_timedelta += timedelta(seconds=secs, microseconds=nanos / 1000) - begin_timeout

                            # There is a delay for stopping the video and one for starting the video
                            # Both must be considered
                            current_timedelta += timedelta(
                                seconds=self._settings["log_parser"]["video_start_delay"]
                                - self._settings["log_parser"]["video_stop_delay"]
                            )
                            is_timeout = False
                    if not is_timeout:
                        if "!statusMessage" in event:
                            decoded_msg = base64.b64decode(event["!statusMessage"]["data"])
                            unpacked = struct.unpack("<4s3B?6f", decoded_msg)
                            entry["entry"]["!statusMessage"]["data"] = unpacked
                        if (
                            "!metadata" in event
                            or "!action" in event
                            or "!gameState" in event
                            or "!statusMessage" in event
                        ):
                            # The timestamps need to be changed due to removed timeout
                            secs = entry["timestamp"]["secs"]
                            nanos = entry["timestamp"]["nanos"]
                            time = timedelta(seconds=secs, microseconds=nanos / 1000) - current_timedelta
                            entry["timestamp"]["secs"] = time.seconds
                            entry["timestamp"]["nanos"] = time.microseconds * 1000
                            filtered_entries.append(entry)

            log_entries = filtered_entries

            metadata = log_entries[0]["entry"]["!metadata"]
            teamdata = metadata["params"]["game"]["teams"]
            home_team_left = metadata["params"]["game"]["sideMapping"] == "homeDefendsLeftGoal"

            home = Team.from_teamdata(teamdata=teamdata["home"], side="home")
            away = Team.from_teamdata(teamdata=teamdata["away"], side="away")

            log_start_time = metadata["timestamp"]

            self.teams = (away, home) if snd_half == home_team_left else (home, away)

            # Find the beginning of the half.
            if snd_half:
                index = next(
                    x
                    for x in range(len(log_entries))
                    if "!action" in log_entries[x]["entry"]
                    and log_entries[x]["entry"]["!action"]["action"]["type"] == "switchHalf"
                )
            else:
                index = 0

            # Find the first kick-off in the half.
            index = next(
                x
                for x in range(index, len(log_entries))
                if "!action" in log_entries[x]["entry"]
                and log_entries[x]["entry"]["!action"]["source"] == "user"
                and log_entries[x]["entry"]["!action"]["action"]["type"] == "startSetPlay"
                and log_entries[x]["entry"]["!action"]["action"]["args"]["setPlay"] == "kickOff"
            )
            assert index != -1
            log_entries = log_entries[index:]

            self.start_time = log_start_time + timedelta(
                seconds=log_entries[0]["timestamp"]["secs"],
                microseconds=log_entries[0]["timestamp"]["nanos"] / 1000,
            )

            self.basename = (
                f"{self.start_time.strftime('%Y-%m-%d_%H-%M')}_{home.name}_"
                + f"{away.name}_{'2nd' if snd_half else '1st'}_Half"
            )

            self.start_time += timedelta(0, self._settings["log_parser"]["video_start_delay"])

            for log_entry in log_entries:
                # Only include changes in the game state or actions from the game controller
                if "!action" in log_entry["entry"]:
                    self.action_timestamps.append(
                        log_start_time
                        + timedelta(
                            seconds=log_entry["timestamp"]["secs"],
                            microseconds=log_entry["timestamp"]["nanos"] / 1000,
                        )
                    )
                    self.action_events.append(log_entry["entry"]["!action"])
                elif "!gameState" in log_entry["entry"]:
                    self.game_state_timestamps.append(
                        log_start_time
                        + timedelta(
                            seconds=log_entry["timestamp"]["secs"],
                            microseconds=log_entry["timestamp"]["nanos"] / 1000,
                        )
                    )
                    self.game_state_events.append(log_entry["entry"]["!gameState"])
                if "!statusMessage" in log_entry["entry"]:
                    self.status_messages_timestamps.append(
                        log_start_time
                        + timedelta(
                            seconds=log_entry["timestamp"]["secs"],
                            microseconds=log_entry["timestamp"]["nanos"] / 1000,
                        )
                    )
                    self.status_messages.append(log_entry["entry"]["!statusMessage"])
        else:
            self._is_timeout: bool = False
            self._begin_timeout: datetime
            self._current_timedelta: timedelta = timedelta()

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
            home = Team.from_legacy_log(
                color=teams[0][teams[0].find("(") + 1 : teams[0].find(")")], name=teams[0].split(" (")[0], side="home"
            )
            away = Team.from_legacy_log(
                color=teams[1][teams[1].find("(") + 1 : teams[1].find(")")], name=teams[1].split(" (")[0], side="away"
            )

            # Determine start time of the game (not half!)
            start_time = next(line for line in lines if " Ready" in line).split(":")[0]
            start_time = datetime.strptime(start_time, "%Y.%m.%d-%H.%M.%S").strftime("%Y-%m-%d_%H-%M")

            # Define unique basename for this half.
            self.basename = (
                f"{start_time}_{home.name.replace(' ', '_')}_"
                + f"{away.name.replace(' ', '_')}_{'2nd' if snd_half else '1st'}_Half"
            )

            # Start at first ready state, store teams.
            if snd_half:
                index = next((x for x in range(len(lines)) if " 2nd Half" in lines[x]), -2) + 1
                self.teams = (away, home)
            else:
                index = next((x for x in range(len(lines)) if " Ready" in lines[x]), -1)
                self.teams = (home, away)
            assert index != -1
            lines = lines[index:]

            # Initialize list of events / timestamps.
            for line in lines:
                self._from_logentry_to_dict(line)
            self.start_time = self.game_state_timestamps[0] + timedelta(
                0, self._settings["log_parser"]["video_start_delay"]
            )

    def _from_logentry_to_dict(self, log_entry: str) -> None:
        timestamp = datetime.strptime(log_entry.split(":")[0], "%Y.%m.%d-%H.%M.%S")
        event = log_entry.split(":")[1]
        team = 0 if self.teams[0].field_player_color in event else 1
        side = self.teams[team].side

        if not self._is_timeout and "Initial" in event:
            self._begin_timeout = timestamp
            self._is_timeout = True
        elif self._is_timeout and "Ready" in event:
            self._current_timedelta += timestamp - self._begin_timeout
            self._current_timedelta += timedelta(
                seconds=self._settings["log_parser"]["video_start_delay"]
                - self._settings["log_parser"]["video_stop_delay"]
            )
            self._is_timeout = False
        timestamp -= self._current_timedelta

        if "Ready" in event or "Set" in event or "Playing" in event:
            self.game_state_timestamps.append(timestamp)
            self.game_state_events.append(
                {
                    "phase": "firstHalf" if not self._snd_half else "secondHalf",
                    "state": event.lower().strip(" "),
                }
            )
        elif "Goal for" in event:
            self.action_timestamps.append(timestamp)
            self.action_events.append(
                {
                    "source": "user",
                    "action": {"type": "goal", "args": {"side": side}},
                }
            )
        elif (
            "Kick In" in event or "Corner Kick" in event or "Goal Kick" in event or "Pushing Free Kick" in event
        ) and "Complete" not in event:
            set_play: str = "none"
            if "Kick In" in event:
                set_play = "kickIn"
            elif "Goal Kick" in event:
                set_play = "goalKick"
            elif "Corner Kick" in event:
                set_play = "cornerKick"
            elif "Pushing Free Kick" in event:
                set_play = "pushingFreeKick"

            self.action_timestamps.append(timestamp)
            self.action_events.append(
                {"source": "user", "action": {"type": "startSetPlay", "args": {"side": side, "setPlay": set_play}}}
            )
        elif (
            "Illegal Position" in event
            or "Illegal Defender" in event
            or "Leaving the Field" in event
            or "Illegal Motion in Set" in event
            or "Player Pushing" in event
            or "Inactive Player" in event
            or "Illegal Ball Contact" in event
            or "Request for PickUp" in event
            or ("Pushing Free Kick" in event or "Penalty Kick" in event)
            and "Complete" not in event
        ):
            penalty: str = "none"

            if "Illegal Position" in event:
                penalty = "illegalPosition"
            elif "Illegal Defender" in event:
                penalty = "illegalPosition"
            elif "Leaving the Field" in event:
                penalty = "leavingTheField"
            elif "Illegal Motion in Set" in event:
                penalty = "motionInSet"
            elif "Player Pushing" in event:
                penalty = "pushing"
            elif "Inactive Player" in event:
                penalty = "fallenInactive"
            elif "Illegal Ball Contact" in event:
                penalty = "ballHolding"
            elif "Request for PickUp" in event:
                penalty = "requestForPickUp"
            elif "Penalty Kick" in event:
                penalty = "penaltyKick"
            elif "Pushing Free Kick" in event:
                penalty = "foul"

            self.action_timestamps.append(timestamp)
            self.action_events.append(
                {"source": "user", "action": {"type": "penalize", "args": {"side": side, "call": penalty}}}
            )
