from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory


SCENARIO_IDS = {
    "familiarization": 1.0,
    "train1": 2.0,
    "train2": 3.0,
    "train3": 4.0,
    "train4": 5.0,
    "test1": 6.0,
    "train5": 7.0,
    "train6": 8.0,
    "train7": 9.0,
    "train8": 10.0,
    "test2": 11.0,
    "final_test": 12.0,
}

SCENARIO_NAMES = tuple(SCENARIO_IDS.keys())


def get_share_dir() -> Path:
    return Path(get_package_share_directory("reachy_bomi"))


def load_scenarios_config():
    config_path = get_share_dir() / "config" / "scenarios.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    return config.get("scenarios", {})


def resolve_world_for_scenario(scenario_name: str) -> str:
    scenarios = load_scenarios_config()
    scenario_cfg = scenarios.get(scenario_name)
    if scenario_cfg is None:
        raise KeyError("Invalid scenario")

    world_value = scenario_cfg.get("world")
    if not world_value:
        raise ValueError(f"No world configured for the scenario {scenario_name}")

    world_path = Path(world_value)
    if not world_path.is_absolute():
        world_path = get_share_dir() / world_path

    return str(world_path.resolve())