# -*- coding: utf-8 -*-

import json
import os

_PILL_STATE_FILE = "pill_state.json"
_PILL_IDS = ("rider", "motor", "cadence", "heartrate", "battery", "elevation")

def _project_root():
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(module_dir)

def load_pill_state():
    root_dir = _project_root()
    path = os.path.join(root_dir, _PILL_STATE_FILE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        state = {}
        for key in _PILL_IDS:
            value = data.get(key)
            if isinstance(value, bool):
                state[key] = value
        return state
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def save_pill_state(state):
    if not isinstance(state, dict):
        return
    data = {key: bool(state.get(key, True)) for key in _PILL_IDS}
    root_dir = _project_root()
    path = os.path.join(root_dir, _PILL_STATE_FILE)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=True)
    except Exception:
        pass
