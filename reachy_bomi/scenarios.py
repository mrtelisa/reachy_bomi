from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory


def get_share_dir() -> Path:
    return Path(get_package_share_directory("reachy_bomi"))


def load_scenarios_config():
    config_path = get_share_dir() / "config" / "scenarios.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    return config.get("scenarios", {})


_scenarios = load_scenarios_config()

# Derived from scenarios.yaml — map_id is the authoritative source
SCENARIO_IDS = {name: cfg["map_id"] for name, cfg in _scenarios.items()}
SCENARIO_NAMES = tuple(_scenarios.keys())


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


def resolve_bag_prefix_for_scenario(scenario_name: str) -> str:
    scenarios = load_scenarios_config()
    scenario_cfg = scenarios.get(scenario_name)
    if scenario_cfg is None:
        raise KeyError("Invalid scenario")
    return scenario_cfg.get("bag_prefix", scenario_name)