# -*- coding: utf-8 -*-

import json
import os

def _project_root():
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(module_dir)

def load_translations(lang_code):
    root_dir = _project_root()
    path = os.path.join(root_dir, f"{lang_code}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def discover_languages():
    """Return [(code, label)] by scanning *.json files that contain a language label key."""
    root_dir = _project_root()
    langs = []
    for fname in os.listdir(root_dir):
        if not fname.lower().endswith(".json"):
            continue
        code = os.path.splitext(fname)[0]
        data = load_translations(code)
        label = data.get("menu.language_label") or data.get("menu.language_menu") or data.get("menu.language")
        if label:
            langs.append((code, label))
    langs.sort(key=lambda x: x[0])
    return langs

def load_last_language_selection():
    root_dir = _project_root()
    path = os.path.join(root_dir, "last_language.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip() or None
    except FileNotFoundError:
        return None
    except Exception:
        return None

def save_last_language_selection(code):
    root_dir = _project_root()
    path = os.path.join(root_dir, "last_language.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception:
        # Ignore write errors
        pass
