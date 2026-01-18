# -*- coding: utf-8 -*-

import bisect
import math
import os
import re
from datetime import datetime, timezone
from tkinter import messagebox


def _extract_date_token(path):
    base_name = os.path.basename(path)
    match = re.search(r"(\d{8})", base_name)
    return match.group(1) if match else None


def _find_log_paths(gpx_path):
    date_token = _extract_date_token(gpx_path)
    if not date_token:
        return []

    directory = os.path.dirname(os.path.abspath(gpx_path))
    try:
        names = os.listdir(directory)
    except Exception:
        return []

    candidates = []
    for name in names:
        if not name.lower().endswith(".log"):
            continue
        if date_token not in name:
            continue
        candidates.append(os.path.join(directory, name))

    if not candidates:
        return []

    preferred = [path for path in candidates if "car_events" in os.path.basename(path).lower()]
    return sorted(preferred or candidates)


def _parse_timestamp(raw_value):
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    # Convert epoch milliseconds to seconds when needed.
    if value > 100000000000:
        value = value / 1000.0
    return float(value)


def _read_event_entries(log_paths):
    entries = []
    for path in log_paths:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if not line:
                        continue
                    parts = line.strip().split(",", 1)
                    if len(parts) != 2:
                        continue
                    event = parts[0].strip().upper()
                    if event not in ("DETECTAT", "DESAPAREGUT"):
                        continue
                    timestamp = _parse_timestamp(parts[1].strip())
                    if timestamp is not None:
                        entries.append((event, timestamp))
        except Exception:
            continue
    return entries


def _build_intervals(entries):
    if not entries:
        return []
    ordered = sorted(entries, key=lambda item: item[1])
    intervals = []
    active_start = None
    for event, timestamp in ordered:
        if event == "DETECTAT":
            active_start = timestamp
        elif event == "DESAPAREGUT":
            if active_start is None:
                continue
            if timestamp >= active_start:
                intervals.append((active_start, timestamp))
                active_start = None
    return intervals


def load_car_events(gpx_path):
    log_paths = _find_log_paths(gpx_path)
    if not log_paths:
        return None
    entries = _read_event_entries(log_paths)
    if not entries:
        return None
    timestamps = [timestamp for event, timestamp in entries if event == "DETECTAT"]
    intervals = _build_intervals(entries)
    return {
        "count": len(timestamps),
        "timestamps": sorted(timestamps),
        "intervals": intervals,
    }


def _build_time_mapping(binned_dt, binned_t):
    timeline = []
    active_seconds = []
    for dt_value, t_value in zip(binned_dt, binned_t):
        if not isinstance(dt_value, datetime):
            continue
        if dt_value.tzinfo is None:
            dt_value = dt_value.replace(tzinfo=timezone.utc)
        try:
            timeline.append(dt_value.timestamp())
            active_seconds.append(float(t_value))
        except Exception:
            continue

    if not timeline:
        return None

    order = sorted(range(len(timeline)), key=timeline.__getitem__)
    timeline = [timeline[idx] for idx in order]
    active_seconds = [active_seconds[idx] for idx in order]
    return timeline, active_seconds


def _map_timestamp_to_active_seconds(event_ts, timeline, active_seconds):
    if event_ts < timeline[0] or event_ts > timeline[-1]:
        return None
    idx = bisect.bisect_left(timeline, event_ts)
    if idx <= 0:
        return active_seconds[0]
    if idx >= len(timeline):
        return active_seconds[-1]
    t0 = timeline[idx - 1]
    t1 = timeline[idx]
    if t1 == t0:
        return active_seconds[idx - 1]
    a0 = active_seconds[idx - 1]
    a1 = active_seconds[idx]
    frac = (event_ts - t0) / (t1 - t0)
    return a0 + frac * (a1 - a0)


def _interpolate_value(target, xs, ys):
    if xs is None or ys is None:
        return None
    if len(xs) == 0:
        return None
    if target <= xs[0]:
        return float(ys[0])
    if target >= xs[-1]:
        return float(ys[-1])
    idx = bisect.bisect_left(xs, target)
    if idx <= 0:
        return float(ys[0])
    if idx >= len(xs):
        return float(ys[-1])
    x0 = float(xs[idx - 1])
    x1 = float(xs[idx])
    y0 = float(ys[idx - 1])
    y1 = float(ys[idx])
    if x1 == x0:
        return y0
    frac = (target - x0) / (x1 - x0)
    return y0 + frac * (y1 - y0)


def _is_below_speed_threshold(event_ts, speed_times, speed_values, min_speed_kmh):
    if speed_values is None or min_speed_kmh is None:
        return False
    speed = _interpolate_value(event_ts, speed_times, speed_values)
    if speed is None:
        return False
    try:
        speed = float(speed)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(speed):
        return False
    return speed < min_speed_kmh


def map_events_to_active_minutes(
    event_seconds,
    binned_dt,
    binned_t,
    speed_times=None,
    speed_values=None,
    min_speed_kmh=None,
):
    if event_seconds is None or binned_dt is None or binned_t is None:
        return []
    if len(event_seconds) == 0 or len(binned_dt) == 0 or len(binned_t) == 0:
        return []

    mapping = _build_time_mapping(binned_dt, binned_t)
    if not mapping:
        return []
    timeline, active_seconds = mapping

    event_minutes = []
    for event_value in event_seconds:
        try:
            event_ts = float(event_value)
        except (TypeError, ValueError):
            continue
        if _is_below_speed_threshold(event_ts, speed_times, speed_values, min_speed_kmh):
            continue
        mapped = _map_timestamp_to_active_seconds(event_ts, timeline, active_seconds)
        if mapped is None:
            continue
        event_minutes.append(mapped / 60.0)

    return sorted(event_minutes)


def map_intervals_to_active_minutes(
    intervals,
    binned_dt,
    binned_t,
    speed_times=None,
    speed_values=None,
    min_speed_kmh=None,
):
    if intervals is None or binned_dt is None or binned_t is None:
        return []
    if len(intervals) == 0 or len(binned_dt) == 0 or len(binned_t) == 0:
        return []

    mapping = _build_time_mapping(binned_dt, binned_t)
    if not mapping:
        return []
    timeline, active_seconds = mapping

    spans = []
    for start_ts, end_ts in intervals:
        try:
            start_value = float(start_ts)
            end_value = float(end_ts)
        except (TypeError, ValueError):
            continue
        if _is_below_speed_threshold(start_value, speed_times, speed_values, min_speed_kmh):
            continue
        mapped_start = _map_timestamp_to_active_seconds(start_value, timeline, active_seconds)
        mapped_end = _map_timestamp_to_active_seconds(end_value, timeline, active_seconds)
        if mapped_start is None or mapped_end is None:
            continue
        start_min = mapped_start / 60.0
        end_min = mapped_end / 60.0
        if end_min < start_min:
            start_min, end_min = end_min, start_min
        spans.append((start_min, end_min))

    return sorted(spans, key=lambda pair: pair[0])


def show_car_events_popup(gpx_path, t, parent=None, car_events=None):
    if car_events is None:
        car_events = load_car_events(gpx_path)
    if not car_events:
        return

    count = int(car_events.get("count", 0))
    title = t("message.radar.title")
    body = t("message.radar.body", count=count)
    messagebox.showinfo(title, body, parent=parent)
