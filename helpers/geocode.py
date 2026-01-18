# -*- coding: utf-8 -*-

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np

_GEOCODE_LAST_CALL = 0.0
_EXCLUDED_LOCALITIES = {"mollerussa"}

def _project_root():
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(module_dir)

def normalize_locality_name(name):
    if not name:
        return None
    raw = str(name).split(",")[0].strip()
    if not raw:
        return None
    tokens = raw.split()
    if not tokens:
        return None
    articles = {"el", "els", "la", "les", "l'", "lo", "los"}
    first = tokens[0]
    first_norm = first.replace("\u2019", "'").casefold()
    if first_norm in articles:
        tokens = tokens[1:]
    elif first_norm.startswith("l'") and len(first_norm) > 2:
        rest = first[2:]
        tokens = [rest] + tokens[1:]
    if not tokens:
        return None
    return tokens[0]

def is_excluded_locality(name):
    if not name:
        return False
    return normalize_locality_name(name).casefold() in _EXCLUDED_LOCALITIES

def load_geocode_cache(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def save_geocode_cache(path, cache):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=True, indent=2, sort_keys=True)
    except Exception:
        pass

def reverse_geocode_locality(lat, lon, lang="ca", timeout=6.0):
    global _GEOCODE_LAST_CALL
    wait = 1.0 - (time.time() - _GEOCODE_LAST_CALL)
    if wait > 0:
        time.sleep(wait)

    params = {
        "format": "jsonv2",
        "lat": f"{float(lat):.6f}",
        "lon": f"{float(lon):.6f}",
        "zoom": 10,
        "addressdetails": 1,
        "accept-language": lang,
    }
    url = "https://nominatim.openstreetmap.org/reverse?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TrekRideSnapshot/1.0 (local-use)",
            "Accept-Language": lang,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except Exception:
        _GEOCODE_LAST_CALL = time.time()
        return None
    _GEOCODE_LAST_CALL = time.time()

    address = data.get("address", {}) if isinstance(data, dict) else {}
    for key in ("town", "city", "village", "municipality", "hamlet", "locality", "county"):
        name = address.get(key)
        if name:
            return name
    return None

def reverse_geocode_cached(lat, lon, cache, lang="ca", precision=3):
    key = f"{float(lat):.{precision}f},{float(lon):.{precision}f},{lang}"
    if key in cache:
        cached = cache.get(key)
        return cached or None
    name = reverse_geocode_locality(lat, lon, lang=lang)
    cache[key] = name or ""
    return name

def identify_localities_online(
    binned_t,
    binned_dist,
    binned_lat,
    binned_lon,
    binned_dt=None,
    lang="ca",
    min_interval_s=180.0,
    min_interval_km=1.0,
    max_queries=80,
    min_stay_s=60.0,
    min_stay_km=0.3,
):
    if binned_t is None or binned_dist is None or binned_lat is None or binned_lon is None:
        return []
    if len(binned_t) == 0:
        return []

    root_dir = _project_root()
    cache_path = os.path.join(root_dir, "geocode_cache.json")
    cache = load_geocode_cache(cache_path)

    candidate_indices = []
    last_t = None
    last_d = None
    for i in range(len(binned_t)):
        t = float(binned_t[i])
        d = float(binned_dist[i])
        if not np.isfinite(t) or not np.isfinite(d):
            continue
        if last_t is None or (t - last_t) >= min_interval_s or (d - last_d) >= min_interval_km:
            candidate_indices.append(i)
            last_t = t
            last_d = d

    if candidate_indices and candidate_indices[-1] != (len(binned_t) - 1):
        candidate_indices.append(len(binned_t) - 1)

    if len(candidate_indices) > max_queries:
        step = int(np.ceil(len(candidate_indices) / max_queries))
        candidate_indices = candidate_indices[::step]

    samples = []
    for idx in candidate_indices:
        lat = float(binned_lat[idx])
        lon = float(binned_lon[idx])
        if not np.isfinite(lat) or not np.isfinite(lon):
            continue
        dt_value = None
        if binned_dt is not None and idx < len(binned_dt):
            dt_value = binned_dt[idx]
        timestamp = dt_value.timestamp() if isinstance(dt_value, datetime) else None
        name = reverse_geocode_cached(lat, lon, cache, lang=lang)
        if not name or is_excluded_locality(name):
            continue
        name = normalize_locality_name(name)
        if not name:
            continue
        samples.append(
            {
                "name": name,
                "time": float(binned_t[idx]),
                "distance_km": float(binned_dist[idx]),
                "lat": lat,
                "lon": lon,
                "timestamp": timestamp,
            }
        )

    save_geocode_cache(cache_path, cache)

    if not samples:
        return []

    segments = []
    for sample in samples:
        if segments and sample["name"] == segments[-1]["name"]:
            seg = segments[-1]
            seg["end_time"] = sample["time"]
            seg["end_dist"] = sample["distance_km"]
            seg["count"] += 1
            seg["lat_sum"] += sample["lat"]
            seg["lon_sum"] += sample["lon"]
            if sample.get("timestamp") is not None:
                seg["dt_sum"] += sample["timestamp"]
                seg["dt_count"] += 1
        else:
            segments.append(
                {
                    "name": sample["name"],
                    "start_time": sample["time"],
                    "end_time": sample["time"],
                    "start_dist": sample["distance_km"],
                    "end_dist": sample["distance_km"],
                    "count": 1,
                    "lat_sum": sample["lat"],
                    "lon_sum": sample["lon"],
                    "dt_sum": sample["timestamp"] if sample.get("timestamp") is not None else 0.0,
                    "dt_count": 1 if sample.get("timestamp") is not None else 0,
                }
            )

    filtered = []
    for i, seg in enumerate(segments):
        duration = seg["end_time"] - seg["start_time"]
        distance = seg["end_dist"] - seg["start_dist"]
        if seg["count"] == 1 and duration < min_stay_s and distance < min_stay_km:
            prev_seg = segments[i - 1] if i > 0 else None
            next_seg = segments[i + 1] if (i + 1) < len(segments) else None
            if prev_seg and next_seg and prev_seg["name"] == next_seg["name"]:
                continue
        filtered.append(seg)

    merged = []
    for seg in filtered:
        if merged and seg["name"] == merged[-1]["name"]:
            merged[-1]["end_time"] = seg["end_time"]
            merged[-1]["end_dist"] = seg["end_dist"]
            merged[-1]["count"] += seg["count"]
            merged[-1]["lat_sum"] += seg["lat_sum"]
            merged[-1]["lon_sum"] += seg["lon_sum"]
            merged[-1]["dt_sum"] += seg.get("dt_sum", 0.0)
            merged[-1]["dt_count"] += seg.get("dt_count", 0)
        else:
            merged.append(seg)

    stays = []
    for seg in merged:
        mid_time = (seg["start_time"] + seg["end_time"]) / 2.0
        mid_dist = (seg["start_dist"] + seg["end_dist"]) / 2.0
        lat = None
        lon = None
        if seg.get("count"):
            lat = seg.get("lat_sum", 0.0) / seg["count"]
            lon = seg.get("lon_sum", 0.0) / seg["count"]
        dt_value = None
        if seg.get("dt_count"):
            dt_value = datetime.fromtimestamp(seg["dt_sum"] / seg["dt_count"], tz=timezone.utc)
        stays.append(
            {
                "name": seg["name"],
                "time": mid_time,
                "distance_km": mid_dist,
                "lat": lat,
                "lon": lon,
                "datetime": dt_value,
            }
        )

    return stays
