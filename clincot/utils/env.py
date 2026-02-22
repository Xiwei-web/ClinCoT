from __future__ import annotations

import os


def get_env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return int(default)


def get_env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def is_distributed() -> bool:
    return get_env_int("WORLD_SIZE", 1) > 1


def get_rank() -> int:
    return get_env_int("RANK", 0)


def get_world_size() -> int:
    return get_env_int("WORLD_SIZE", 1)


def is_main_process() -> bool:
    return get_rank() == 0
