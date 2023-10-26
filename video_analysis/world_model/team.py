import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Team:
    """Color and name of a team."""

    def __init__(
        self, field_player_color: str, goalkeeper_color: str, name: str, number: int = -1, side: str = "none"
    ) -> None:
        self.field_player_color: str = field_player_color
        self.goalkeeper_color: str = goalkeeper_color
        self.name: str = name
        self.number: int = number
        self.side: str = side

    _config_path = ROOT / "config" / "teams.json"

    @classmethod
    def from_teamdata(cls, teamdata: dict, side: str):
        field_player_color = teamdata["fieldPlayerColor"]
        goalkeeper_color = teamdata["goalkeeperColor"]
        number = teamdata["number"]
        with cls._config_path.open() as f:
            config = json.load(f)
            name = next(team["name"][0] for team in config if team["number"] == number)
        return cls(
            field_player_color=field_player_color,
            goalkeeper_color=goalkeeper_color,
            name=name,
            number=number,
            side=side,
        )

    @classmethod
    def from_legacy_log(cls, color: str, name: str, side: str):
        with cls._config_path.open() as f:
            config = json.load(f)
            number = next(team["number"] for team in config if name in team["name"])
        return cls(field_player_color=color, goalkeeper_color=color, name=name, number=number, side=side)
