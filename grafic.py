#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tkinter UI to select a GPX file and display:
- Smoothed rider and motor power
- Background elevation profile
- Moving time (minutes) on the lower X axis
- Distance (km) on the upper X axis
- Mean power lines

Requirements:
    pip install matplotlib numpy
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import os
import json

import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
import ctypes
from ctypes import wintypes


class LocalizedError(Exception):
    """Exception carrying a localization key (and optional detail)."""
    def __init__(self, key, detail=None):
        self.key = key
        self.detail = detail
        super().__init__(key)

# Use the Tkinter backend
matplotlib.use("TkAgg")


def load_translations(lang_code):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, f"{lang_code}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def discover_languages():
    """Return [(code, label)] by scanning *.json files that contain a language label key."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    langs = []
    for fname in os.listdir(script_dir):
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "last_language.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip() or None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def save_last_language_selection(code):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "last_language.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception:
        # Ignore write errors
        pass


def parse_gpx(path):
    """
    Read a GPX file and return:
    times, power, motor_power, battery, heartrate, ebike_mode, elevation, lat, lon
    (all as numpy arrays)
    """
    tree = ET.parse(path)
    root = tree.getroot()

    # Namespace (depends on the GPX)
    if root.tag.startswith("{"):
        uri = root.tag.split("}")[0].strip("{")
        ns = {"gpx": uri}
    else:
        ns = {"gpx": ""}

    times = []
    powers = []
    motor_powers = []
    batteries = []
    heartrates = []
    ebike_modes = []
    elevs = []
    lats = []
    lons = []

    for trkpt in root.findall(".//gpx:trkpt", ns):
        time_elem = trkpt.find("gpx:time", ns)
        ele_elem = trkpt.find("gpx:ele", ns)
        if time_elem is None or not time_elem.text or ele_elem is None or not ele_elem.text:
            continue

        try:
            t = datetime.fromisoformat(time_elem.text.replace("Z", "+00:00"))
            ele = float(ele_elem.text)
            lat = float(trkpt.get("lat"))
            lon = float(trkpt.get("lon"))
        except Exception:
            continue

        # power and motor_power are often inside extensions
        power_elem = trkpt.find(".//gpx:power", ns)
        motor_elem = trkpt.find(".//gpx:motor_power", ns)
        if power_elem is None or motor_elem is None:
            continue

        try:
            p = float(power_elem.text)
            mp = float(motor_elem.text)
        except (TypeError, ValueError):
            continue

        # Battery (optional)
        battery_elem = trkpt.find(".//gpx:ebike_battery", ns)
        if battery_elem is not None and battery_elem.text:
            try:
                battery = float(battery_elem.text)
            except (TypeError, ValueError):
                battery = np.nan
        else:
            battery = np.nan

        # Heart rate (optional)
        heartrate_elem = trkpt.find(".//gpx:heartrate", ns)
        if heartrate_elem is not None and heartrate_elem.text:
            try:
                heartrate = float(heartrate_elem.text)
            except (TypeError, ValueError):
                heartrate = np.nan
        else:
            heartrate = np.nan

        # Assist mode (optional)
        mode_elem = trkpt.find(".//gpx:ebike_mode", ns)
        if mode_elem is not None and mode_elem.text:
            try:
                mode = int(float(mode_elem.text))
            except (TypeError, ValueError):
                mode = np.nan
        else:
            mode = np.nan

        times.append(t)
        powers.append(p)
        motor_powers.append(mp)
        batteries.append(battery)
        heartrates.append(heartrate)
        ebike_modes.append(mode)
        elevs.append(ele)
        lats.append(lat)
        lons.append(lon)

    if not times:
        raise LocalizedError("error.gpx.missing_power")

    return (
        np.array(times),
        np.array(powers),
        np.array(motor_powers),
        np.array(batteries),
        np.array(heartrates),
        np.array(ebike_modes),
        np.array(elevs),
        np.array(lats),
        np.array(lons),
    )


def compute_moving_binned_data(times, powers, motor_powers, batteries, heartrates, ebike_modes, elevs, lats, lons):
    """
    - Compute distance, speed, and long stops.
    - Remove long pauses (active moving time).
    - Resample every 10 s of active time.
    - Apply 7+5-point smoothing to power, light smoothing to elevation.
    - Compute means and distance.
    - Returns a dict with all series ready to plot.
    """
    # Cumulative distance (vectorized haversine)
    R = 6371000.0  # metres
    lat1 = np.radians(lats[:-1])
    lat2 = np.radians(lats[1:])
    dlat = lat2 - lat1
    dlon = np.radians(lons[1:] - lons[:-1])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.minimum(1, np.sqrt(a)))
    seg_dist = R * c  # metres per segment

    distances = np.zeros(len(lats))
    distances[1:] = np.cumsum(seg_dist)
    cumdist_km = distances / 1000.0

    # Elapsed seconds from start
    elapsed = np.array([(t - times[0]).total_seconds() for t in times])

    # Instant speed (m/s and km/h)
    dt = np.diff(elapsed, prepend=elapsed[0])
    dt[dt == 0] = 1e-6
    ddist = np.diff(distances, prepend=distances[0])
    speed_ms = ddist / dt
    speed_kmh = speed_ms * 3.6

    # Long stops: power=0, motor=0 and speed very low for >=60s
    mask_stop = (powers == 0) & (motor_powers == 0) & (speed_kmh < 0.5)
    long_pause = np.zeros_like(mask_stop, dtype=bool)

    i = 0
    n = len(mask_stop)
    while i < n:
        if mask_stop[i]:
            j = i + 1
            while j < n and mask_stop[j]:
                j += 1
            duration = elapsed[j - 1] - elapsed[i]
            if duration >= 60:  # at least 60s stopped
                long_pause[i:j] = True
            i = j
        else:
            i += 1

    # Active time (long stops removed)
    active_elapsed = np.zeros_like(elapsed)
    for k in range(1, len(elapsed)):
        if not long_pause[k] and not long_pause[k - 1]:
            active_elapsed[k] = active_elapsed[k - 1] + (elapsed[k] - elapsed[k - 1])
        else:
            active_elapsed[k] = active_elapsed[k - 1]

    keep = ~long_pause
    active_elapsed_kept = active_elapsed[keep]
    powers_kept = powers[keep]
    motor_kept = motor_powers[keep]
    battery_kept = batteries[keep]
    heartrate_kept = heartrates[keep]
    mode_kept = ebike_modes[keep]
    elev_kept = elevs[keep]
    dist_kept = cumdist_km[keep]

    # Sample every 10 s of active time
    bin_size = 10.0
    max_t_active = active_elapsed_kept.max()
    bins = np.arange(0, max_t_active + bin_size, bin_size)
    bin_indices = np.floor(active_elapsed_kept / bin_size).astype(int)

    binned_t = []
    binned_power = []
    binned_motor = []
    binned_battery = []
    binned_heartrate = []
    binned_mode = []
    binned_ele = []
    binned_dist = []

    for idx in range(len(bins)):
        mask = bin_indices == idx
        if not np.any(mask):
            continue
        binned_t.append(bins[idx])
        binned_power.append(powers_kept[mask].mean())
        binned_motor.append(motor_kept[mask].mean())
        # Battery: mean of valid (non-NaN) values only
        battery_mask = ~np.isnan(battery_kept[mask])
        if np.any(battery_mask):
            binned_battery.append(np.nanmean(battery_kept[mask]))
        else:
            binned_battery.append(np.nan)
        # Heart rate: mean of valid (non-NaN) values only
        heartrate_mask = ~np.isnan(heartrate_kept[mask])
        if np.any(heartrate_mask):
            binned_heartrate.append(np.nanmean(heartrate_kept[mask]))
        else:
            binned_heartrate.append(np.nan)
        # Assist mode: keep most frequent / last valid value
        mode_mask = ~np.isnan(mode_kept[mask])
        if np.any(mode_mask):
            mode_values = mode_kept[mask][mode_mask]
            # Take last valid value in bin (representative of change)
            binned_mode.append(int(mode_values[-1]))
        else:
            binned_mode.append(np.nan)
        binned_ele.append(elev_kept[mask].mean())
        binned_dist.append(dist_kept[mask].mean())

    binned_t = np.array(binned_t)
    binned_power = np.array(binned_power)
    binned_motor = np.array(binned_motor)
    binned_battery = np.array(binned_battery)
    binned_heartrate = np.array(binned_heartrate)
    binned_mode = np.array(binned_mode)
    binned_ele = np.array(binned_ele)
    binned_dist = np.array(binned_dist)

    def convolve_edge_preserving(x, kernel):
        pad = (len(kernel) - 1) // 2
        if len(x) >= 2 and pad > 0:
            x_padded = np.pad(x, (pad, pad), mode="edge")
            return np.convolve(x_padded, kernel, mode="valid")
        return np.convolve(x, kernel, mode="same")

    # Double smoothing: 7-point then 5-point moving average (edge-preserving)
    kernel1 = np.ones(7) / 7
    kernel2 = np.ones(5) / 5

    def double_smooth(x):
        s1 = convolve_edge_preserving(x, kernel1)
        s2 = convolve_edge_preserving(s1, kernel2)
        return s2

    smooth_power = double_smooth(binned_power)
    smooth_motor = double_smooth(binned_motor)

    # Smoothed battery (if valid data)
    if not np.all(np.isnan(binned_battery)):
        # To smooth, fill NaNs by interpolation
        valid_mask = ~np.isnan(binned_battery)
        if np.any(valid_mask):
            smooth_battery = np.copy(binned_battery)
            if np.sum(valid_mask) > 1:
                # Interpolate NaN values
                valid_indices = np.where(valid_mask)[0]
                smooth_battery_interp = np.interp(np.arange(len(binned_battery)), 
                                                  valid_indices, 
                                                  binned_battery[valid_indices])
                kernel_battery = np.ones(5) / 5
                smooth_battery = convolve_edge_preserving(smooth_battery_interp, kernel_battery)
            else:
                smooth_battery = binned_battery
        else:
            smooth_battery = binned_battery
    else:
        smooth_battery = binned_battery

    # Smoothed heart rate (if valid data)
    if not np.all(np.isnan(binned_heartrate)):
        # To smooth, fill NaNs by interpolation
        valid_mask = ~np.isnan(binned_heartrate)
        if np.any(valid_mask):
            smooth_heartrate = np.copy(binned_heartrate)
            if np.sum(valid_mask) > 1:
                # Interpolate NaN values
                valid_indices = np.where(valid_mask)[0]
                smooth_heartrate_interp = np.interp(np.arange(len(binned_heartrate)), 
                                                   valid_indices, 
                                                   binned_heartrate[valid_indices])
                kernel_heartrate = np.ones(5) / 5
                smooth_heartrate = convolve_edge_preserving(smooth_heartrate_interp, kernel_heartrate)
            else:
                smooth_heartrate = binned_heartrate
        else:
            smooth_heartrate = binned_heartrate
    else:
        smooth_heartrate = binned_heartrate

    # Lightly smoothed elevation (edge-preserving to avoid start/end dropping)
    kernel_ele = np.ones(11) / 11
    pad = (len(kernel_ele) - 1) // 2
    if len(binned_ele) >= 2 and pad > 0:
        ele_padded = np.pad(binned_ele, (pad, pad), mode="edge")
        smooth_ele = np.convolve(ele_padded, kernel_ele, mode="valid")
    else:
        smooth_ele = binned_ele

    mean_power = np.mean(binned_power)
    mean_motor = np.mean(binned_motor)

    # Detect assist mode changes
    mode_changes = []
    if not np.all(np.isnan(binned_mode)):
        valid_mode_mask = ~np.isnan(binned_mode)
        if np.any(valid_mode_mask):
            valid_modes = binned_mode[valid_mode_mask]
            valid_indices = np.where(valid_mode_mask)[0]
            valid_times = binned_t[valid_indices]
            
            if len(valid_modes) > 0:
                # Detect all mode changes
                all_changes = []
                
                # Detect change points (mode switches)
                for i in range(1, len(valid_modes)):
                    if valid_modes[i] != valid_modes[i-1]:
                        # On change, mark the new mode
                        all_changes.append({
                            'idx': valid_indices[i],
                            'time': valid_times[i],
                            'mode': int(valid_modes[i])
                        })
                
                # Collapse rapid successive changes (<30s) into the last one
                if len(all_changes) > 1:
                    filtered_changes = []
                    i = 0
                    while i < len(all_changes):
                        # Find next change more than 30 seconds away
                        j = i + 1
                        while j < len(all_changes) and (all_changes[j]['time'] - all_changes[i]['time']) < 30:
                            j += 1
                        # Mark last change of the cluster (or the single one)
                        if j > i + 1:
                            # Clustered changes: mark the last one
                            filtered_changes.append(all_changes[j-1])
                        else:
                            # Single change: mark it
                            filtered_changes.append(all_changes[i])
                        i = j
                    
                    mode_changes = filtered_changes
                elif len(all_changes) == 1:
                    # Single change
                    mode_changes = all_changes
                else:
                    # No mode changes
                    mode_changes = []
                
                # Always add first and last mode
                first_mode = {
                    'idx': valid_indices[0],
                    'time': valid_times[0],
                    'mode': int(valid_modes[0])
                }
                last_mode = {
                    'idx': valid_indices[-1],
                    'time': valid_times[-1],
                    'mode': int(valid_modes[-1])
                }
                
                # Add first mode if missing
                if not mode_changes or mode_changes[0]['time'] != first_mode['time']:
                    mode_changes.insert(0, first_mode)
                
                # Add last mode if missing
                if not mode_changes or mode_changes[-1]['time'] != last_mode['time']:
                    mode_changes.append(last_mode)

    total_distance_km = float(dist_kept[-1]) if len(dist_kept) else 0.0
    total_time_min = float(max_t_active / 60.0) if np.isfinite(max_t_active) else 0.0
    max_rider_power = float(np.nanmax(powers_kept)) if len(powers_kept) else float("nan")
    max_motor_power = float(np.nanmax(motor_kept)) if len(motor_kept) else float("nan")
    mean_heartrate = (
        float(np.nanmean(heartrate_kept))
        if len(heartrate_kept) and not np.all(np.isnan(heartrate_kept))
        else float("nan")
    )
    if len(battery_kept) and not np.all(np.isnan(battery_kept)):
        valid_battery = battery_kept[~np.isnan(battery_kept)]
        last_battery = float(valid_battery[-1]) if len(valid_battery) else float("nan")
    else:
        last_battery = float("nan")
    if len(elev_kept):
        elev_diff = np.diff(elev_kept.astype(float), prepend=elev_kept[0].astype(float))
        elevation_gain_m = float(np.nansum(np.clip(elev_diff, 0.0, None)))
    else:
        elevation_gain_m = float("nan")

    return {
        "binned_t": binned_t,
        "smooth_t_min": binned_t / 60.0,
        "smooth_power": smooth_power,
        "smooth_motor": smooth_motor,
        "smooth_battery": smooth_battery,
        "smooth_heartrate": smooth_heartrate,
        "smooth_ele": smooth_ele,
        "binned_ele": binned_ele,
        "binned_dist": binned_dist,
        "mean_power": mean_power,
        "mean_motor": mean_motor,
        "mode_changes": mode_changes,
        "total_distance_km": total_distance_km,
        "total_time_min": total_time_min,
        "max_rider_power": max_rider_power,
        "max_motor_power": max_motor_power,
        "mean_heartrate": mean_heartrate,
        "last_battery": last_battery,
        "elevation_gain_m": elevation_gain_m,
    }


def create_figure(data, t):
    # 1980 px wide, 21:9 aspect ratio
    dpi = 100
    width_in = 1980 / dpi
    # Add some vertical space for header + pills (keep it tight).
    height_in = ((1980 * 9 / 21) + 250) / dpi
    fig_width_px = width_in * dpi
    fig_height_px = height_in * dpi

    fig = Figure(figsize=(width_in, height_in), dpi=dpi)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    # Ensure ax1 artists (lines, tooltip) draw above the elevation background on ax2.
    ax2.set_zorder(0)
    ax1.set_zorder(1)
    ax1.patch.set_alpha(0)

    # Header (title, subtitle, key stats)
    header_title = t("dashboard.title")
    total_distance_km = data.get("total_distance_km")
    total_time_min = data.get("total_time_min")
    max_rider_power = data.get("max_rider_power")
    max_motor_power = data.get("max_motor_power")
    if total_distance_km is not None and total_time_min is not None and max_rider_power is not None and max_motor_power is not None:
        summary = (
            f'{t("dashboard.summary.total_distance")}: {float(total_distance_km):.1f} km     '
            f'{t("dashboard.summary.total_time")}: {int(round(float(total_time_min)))} min     '
            f'{t("dashboard.summary.max_rider_power")}: {int(round(float(max_rider_power)))} W     '
            f'{t("dashboard.summary.max_motor_power")}: {int(round(float(max_motor_power)))} W'
        )
    else:
        summary = ""

    fig.text(0.02, 0.965, header_title, ha="left", va="top", fontsize=14, color="#111827")
    if summary:
        fig.text(0.02, 0.915, summary, ha="left", va="top", fontsize=9, color="#6b7280")

    # Pills (cards) row
    def format_value_or_na(value, fmt):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return t("dashboard.value.na")
        return fmt(v) if np.isfinite(v) else t("dashboard.value.na")

    pill_shift_right_px = 20
    pill_shift_down_px = 35
    pill_x0, pill_y, pill_w, pill_h = (
        0.02 + (pill_shift_right_px / fig_width_px),
        0.845 - (pill_shift_down_px / fig_height_px),
        0.18,
        0.055,
    )

    def draw_pill(x, y, w, h, label, value, face, edge, value_color):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.0,rounding_size=0.003",
            transform=fig.transFigure,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            zorder=10,
        )
        fig.add_artist(patch)
        label_text = fig.text(
            x + 0.012,
            y + h / 2,
            label,
            ha="left",
            va="center",
            fontsize=9,
            color="#374151",
            zorder=11,
        )
        value_text = fig.text(
            x + w - 0.012,
            y + h / 2,
            value,
            ha="right",
            va="center",
            fontsize=9,
            color=value_color,
            zorder=11,
        )
        return patch, label_text, value_text

    avg_rider = format_value_or_na(data.get("mean_power"), lambda v: f"{int(round(v))} W")
    avg_motor = format_value_or_na(data.get("mean_motor"), lambda v: f"{int(round(v))} W")
    avg_hr = format_value_or_na(data.get("mean_heartrate"), lambda v: f"{int(round(v))} bpm")
    battery_now = format_value_or_na(data.get("last_battery"), lambda v: f"{int(round(v))}%")
    elevation_gain = format_value_or_na(data.get("elevation_gain_m"), lambda v: f"{chr(0x2197)} {int(round(v))} m")

    pill_gap_px = 10
    pill_gap = pill_gap_px / fig_width_px
    pills = [
        ("rider", t("dashboard.card.rider_power"), f'{t("dashboard.value.avg")}: {avg_rider}', "#eff6ff", "#bfdbfe", "#2563eb"),
        ("motor", t("dashboard.card.motor_power"), f'{t("dashboard.value.avg")}: {avg_motor}', "#fff7ed", "#fed7aa", "#ea580c"),
        ("heartrate", t("dashboard.card.heartrate"), f'{t("dashboard.value.avg")}: {avg_hr}', "#fdf2f8", "#fbcfe8", "#db2777"),
        ("battery", t("dashboard.card.battery"), f"final: {battery_now}", "#ecfdf5", "#a7f3d0", "#059669"),
        ("elevation", t("dashboard.card.elevation"), f"{elevation_gain}", "#f8fafc", "#e2e8f0", "#475569"),
    ]
    pill_ui = {}
    for idx, (pill_id, label, value, face, edge, value_color) in enumerate(pills):
        x = pill_x0 + idx * (pill_w + pill_gap)
        patch, label_text, value_text = draw_pill(x, pill_y, pill_w, pill_h, label, value, face, edge, value_color)
        pill_ui[pill_id] = {
            "target_key": f"pill.{pill_id}",
            "patch": patch,
            "label_text": label_text,
            "value_text": value_text,
            "active_face": face,
            "active_edge": edge,
            "active_value_color": value_color,
        }

    t_min = data["smooth_t_min"]
    smooth_power = data["smooth_power"]
    smooth_motor = data["smooth_motor"]
    smooth_battery = data["smooth_battery"]
    smooth_heartrate = data["smooth_heartrate"]
    smooth_ele = data["smooth_ele"]
    binned_ele = data["binned_ele"]
    binned_dist = data["binned_dist"]
    mean_power = data["mean_power"]
    mean_motor = data["mean_motor"]
    line_battery = None
    line_heartrate = None
    mode_markers = []
    mode_texts = []

    # --- background elevation (smoothed + gradient) ---
    kernel_ele_plot = np.ones(21) / 21
    pad = (len(kernel_ele_plot) - 1) // 2
    if len(smooth_ele) >= 2 and pad > 0:
        ele_padded = np.pad(smooth_ele, (pad, pad), mode="edge")
        smooth_ele_plot = np.convolve(ele_padded, kernel_ele_plot, mode="valid")
    else:
        smooth_ele_plot = smooth_ele

    # Keep secondary Y axis starting at 0 for proportionality.
    ele_ymin = 0.0
    ele_ymax = max(smooth_ele_plot.max(), binned_ele.max()) + 10

    clip_poly = ax2.fill_between(t_min, smooth_ele_plot, ele_ymin, color="none", alpha=0.0, zorder=0)
    gradient_rows = 256
    gradient_rgba = np.ones((gradient_rows, 2, 4), dtype=float)
    gradient_rgb = np.array(to_rgba("#cbd5e1"))[:3]
    gradient_rgba[..., :3] = gradient_rgb
    # bottom -> top alpha (fade to white at the bottom)
    gradient_rgba[..., 3] = np.linspace(0.0, 0.55, gradient_rows)[:, None]
    elevation_fill = ax2.imshow(
        gradient_rgba,
        extent=[t_min.min(), t_min.max(), ele_ymin, ele_ymax],
        origin="lower",
        aspect="auto",
        zorder=0,
    )
    elevation_fill.set_clip_path(clip_poly.get_paths()[0], transform=ax2.transData)
    elevation_line, = ax2.plot(t_min, smooth_ele_plot, color="#64748b", linewidth=1.5, alpha=0.9, zorder=1)

    ax2.set_ylabel(t("figure.axis.altitude"))
    ax2.set_ylim(ele_ymin, ele_ymax)
    ax2.tick_params(axis="y", labelsize=8, colors="gray")

    # --- power ---
    line_cyclist, = ax1.plot(t_min, smooth_power, label=t("figure.legend.cyclist"), zorder=3, linewidth=2.2)
    line_motor,   = ax1.plot(t_min, smooth_motor, label=t("figure.legend.motor"),   zorder=3, linewidth=2.2)

    # --- battery (green line) on left Y axis ---
    if not np.all(np.isnan(smooth_battery)):
        battery_color = "#047857"
        line_battery, = ax1.plot(t_min, smooth_battery, color=battery_color,
                                 label=t("figure.legend.battery"), zorder=3, linewidth=2.2)
        
        # Assist mode change markers
        mode_changes = data.get("mode_changes", [])
        if mode_changes:
            binned_t = data["binned_t"]
            for change in mode_changes:
                # Time in minutes for the change
                change_time_min = change['time'] / 60.0
                # Battery value at that time (interpolated)
                battery_value = np.interp(change['time'], binned_t, smooth_battery)
                mode_value = change['mode']
                
                # Draw a square with the mode number
                marker_line, = ax1.plot(change_time_min, battery_value, 's', color=battery_color,
                        markersize=12, markeredgecolor=battery_color, markeredgewidth=1.5,
                        zorder=5, clip_on=False)
                mode_markers.append(marker_line)
                mode_text = ax1.text(change_time_min, battery_value, str(mode_value), 
                        ha='center', va='center', fontsize=8, fontweight='bold',
                        color='white', zorder=6)
                mode_texts.append(mode_text)

    # --- heart rate (purple line) on left Y axis ---
    if not np.all(np.isnan(smooth_heartrate)):
        line_heartrate, = ax1.plot(t_min, smooth_heartrate, color="purple", 
                                    label=t("figure.legend.heartrate"), zorder=3, linewidth=2.2)

    mean_line_power = ax1.axhline(mean_power, linestyle="--", linewidth=1.8,
                                  color=line_cyclist.get_color(), alpha=0.5, label=t("figure.legend.mean_cyclist"))
    mean_line_motor = ax1.axhline(mean_motor, linestyle="--", linewidth=1.8,
                                  color=line_motor.get_color(),   alpha=0.5, label=t("figure.legend.mean_motor"))

    pill_targets = {
        "rider": [line_cyclist, mean_line_power],
        "motor": [line_motor, mean_line_motor],
        "heartrate": [line_heartrate] if line_heartrate is not None else [],
        "battery": ([line_battery] + mode_markers + mode_texts) if line_battery is not None else [],
        "elevation": [elevation_fill, elevation_line],
    }

    ax1.set_xlabel(t("figure.axis.time"))
    ax1.set_ylabel(t("figure.axis.power"))
    # --- power ---
    ymax = max(smooth_power.max(), smooth_motor.max())
    if not np.all(np.isnan(smooth_battery)):
        ymax = max(ymax, smooth_battery.max())
    if not np.all(np.isnan(smooth_heartrate)):
        ymax = max(ymax, smooth_heartrate.max())
    ax1.set_ylim(0, ymax * 1.05)

    # --- axis styling (soft gray, fewer ticks, subtle dotted grid) ---
    axis_color = "#64748b"
    spine_color = "#cbd5e1"
    grid_color = "#e2e8f0"

    x_max = float(np.nanmax(t_min)) if len(t_min) else 0.0
    tick_step_min = 10.0
    x_end = x_max if x_max > 0 else tick_step_min
    base_ticks = np.arange(0.0, x_end + 1e-9, tick_step_min)
    minute_ticks = base_ticks.tolist()
    if not minute_ticks or abs(minute_ticks[-1] - x_end) > 1e-6:
        minute_ticks.append(x_end)
    minute_ticks = np.array(minute_ticks, dtype=float)
    ax1.set_xlim(0.0, x_end)
    ax1.set_xticks(minute_ticks)
    ax1.margins(x=0)

    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.set_axisbelow(True)
    ax1.grid(True, which="major", axis="both", linestyle=(0, (1, 4)), linewidth=0.8, color=grid_color, alpha=0.35)

    def style_axes(ax, hide_top=False, hide_right=False, hide_left=False, hide_bottom=False):
        ax.tick_params(axis="both", colors=axis_color, labelsize=9, length=3, width=0.8)
        ax.xaxis.label.set_color(axis_color)
        ax.yaxis.label.set_color(axis_color)
        for name, spine in ax.spines.items():
            spine.set_color(spine_color)
            spine.set_linewidth(1.0)
        if hide_top:
            ax.spines["top"].set_visible(False)
        if hide_right:
            ax.spines["right"].set_visible(False)
        if hide_left:
            ax.spines["left"].set_visible(False)
        if hide_bottom:
            ax.spines["bottom"].set_visible(False)

    style_axes(ax1, hide_top=True, hide_right=True)
    style_axes(ax2, hide_top=True, hide_left=True)
    ax2.margins(x=0)

    # --- top axis in km ---
    ax_top = ax1.twiny()
    ax_top.set_xlim(ax1.get_xlim())
    seconds_ticks = minute_ticks * 60.0
    km_labels = np.interp(seconds_ticks, data["binned_t"], binned_dist)
    ax_top.set_xticks(minute_ticks)
    ax_top.set_xticklabels([f"{k:.1f}" for k in km_labels])
    ax_top.set_xlabel(t("figure.axis.distance"), labelpad=2)
    style_axes(ax_top, hide_bottom=True)
    ax_top.margins(x=0)

    # --- hover tooltip (details) ---
    hover_line = ax1.axvline(
        0.0,
        color=grid_color,
        linestyle=(0, (1, 4)),
        linewidth=1.0,
        alpha=0.5,
        zorder=2,
        visible=False,
    )
    hover_dot_rider, = ax1.plot([], [], marker="o", markersize=4, linestyle="None",
                                color=line_cyclist.get_color(), zorder=6, visible=False)
    hover_dot_motor, = ax1.plot([], [], marker="o", markersize=4, linestyle="None",
                                color=line_motor.get_color(), zorder=6, visible=False)
    hover_dot_battery = None
    if line_battery is not None:
        hover_dot_battery, = ax1.plot([], [], marker="o", markersize=4, linestyle="None",
                                      color=line_battery.get_color(), zorder=6, visible=False)
    hover_dot_heartrate = None
    if line_heartrate is not None:
        hover_dot_heartrate, = ax1.plot([], [], marker="o", markersize=4, linestyle="None",
                                        color=line_heartrate.get_color(), zorder=6, visible=False)
    hover_dot_elevation, = ax2.plot([], [], marker="o", markersize=4, linestyle="None",
                                    color=elevation_line.get_color(), zorder=6, visible=False)

    tooltip_rows = {
        "time": TextArea("", textprops=dict(color="#6b7280", fontsize=9)),
        "distance": TextArea("", textprops=dict(color="#6b7280", fontsize=9)),
        "elevation": TextArea("", textprops=dict(color="#6b7280", fontsize=9)),
        "rider": TextArea("", textprops=dict(color=line_cyclist.get_color(), fontsize=9)),
        "motor": TextArea("", textprops=dict(color=line_motor.get_color(), fontsize=9)),
        "battery": TextArea("", textprops=dict(color=(line_battery.get_color() if line_battery else "#9ca3af"), fontsize=9)),
        "heartrate": TextArea("", textprops=dict(color=(line_heartrate.get_color() if line_heartrate else "#9ca3af"), fontsize=9)),
    }
    tooltip_box = VPacker(
        children=[
            tooltip_rows["time"],
            tooltip_rows["distance"],
            tooltip_rows["elevation"],
            TextArea("", textprops=dict(color="#6b7280", fontsize=3)),
            tooltip_rows["rider"],
            tooltip_rows["motor"],
            tooltip_rows["battery"],
            tooltip_rows["heartrate"],
        ],
        align="left",
        pad=0,
        sep=4,
    )
    tooltip = AnnotationBbox(
        tooltip_box,
        (0.0, 0.0),
        xybox=(18, 18),
        xycoords="data",
        boxcoords=("offset points"),
        box_alignment=(0.0, 0.0),
        bboxprops=dict(
            boxstyle="round,pad=0.5,rounding_size=0.2",
            fc="#ffffff",
            ec="#e2e8f0",
            alpha=0.995,
        ),
        frameon=True,
        zorder=20,
    )
    tooltip.set_visible(False)
    ax1.add_artist(tooltip)

    artist_groups = {
        line_cyclist.get_label(): [line_cyclist],
        line_motor.get_label(): [line_motor],
        mean_line_power.get_label(): [mean_line_power],
        mean_line_motor.get_label(): [mean_line_motor],
    }
    if line_battery is not None:
        artist_groups[line_battery.get_label()] = [line_battery] + mode_markers + mode_texts
    if line_heartrate is not None:
        artist_groups[line_heartrate.get_label()] = [line_heartrate]

    for pill_id, target_artists in pill_targets.items():
        if not target_artists:
            continue
        pill_key = pill_ui.get(pill_id, {}).get("target_key")
        if pill_key:
            artist_groups[pill_key] = target_artists

    all_artists = set()
    for group_items in artist_groups.values():
        all_artists.update(group_items)
    all_artists.add(elevation_fill)

    base_alpha = {artist: (artist.get_alpha() if artist.get_alpha() is not None else 1.0) for artist in all_artists}
    dim_factor = 0.2
    active_target = None

    pill_inactive_face = "#ffffff"
    pill_inactive_edge = "#e5e7eb"
    pill_inactive_text = "#9ca3af"

    def set_pill_active(pill_id, active):
        ui = pill_ui.get(pill_id)
        if ui is None:
            return
        patch = ui["patch"]
        label_text = ui["label_text"]
        value_text = ui["value_text"]
        if active:
            patch.set_facecolor(ui["active_face"])
            patch.set_edgecolor(ui["active_edge"])
            label_text.set_color("#374151")
            value_text.set_color(ui["active_value_color"])
        else:
            patch.set_facecolor(pill_inactive_face)
            patch.set_edgecolor(pill_inactive_edge)
            label_text.set_color(pill_inactive_text)
            value_text.set_color(pill_inactive_text)

    def refresh_pills():
        for pill_id, ui in pill_ui.items():
            key = ui.get("target_key")
            group = artist_groups.get(key) or []
            is_active = bool(group) and group[0].get_visible()
            set_pill_active(pill_id, is_active)

    refresh_pills()

    def is_group_visible(key):
        group = artist_groups.get(key) or []
        return bool(group) and group[0].get_visible()

    def hide_hover_details():
        tooltip.set_visible(False)
        hover_line.set_visible(False)
        hover_dot_rider.set_visible(False)
        hover_dot_motor.set_visible(False)
        if hover_dot_battery is not None:
            hover_dot_battery.set_visible(False)
        if hover_dot_heartrate is not None:
            hover_dot_heartrate.set_visible(False)
        hover_dot_elevation.set_visible(False)

    def update_hover_details(event):
        if event.inaxes not in (ax1, ax2, ax_top):
            hide_hover_details()
            return
        if event.xdata is None or event.ydata is None or len(t_min) == 0:
            hide_hover_details()
            return

        x = float(event.xdata)
        xmin = float(t_min[0])
        xmax = float(t_min[-1])
        if x < xmin or x > xmax:
            hide_hover_details()
            return

        idx = int(np.searchsorted(t_min, x))
        if idx <= 0:
            idx = 0
        elif idx >= len(t_min):
            idx = len(t_min) - 1
        else:
            if abs(t_min[idx] - x) > abs(t_min[idx - 1] - x):
                idx -= 1

        x0 = float(t_min[idx])
        sec0 = x0 * 60.0
        dist0 = float(np.interp(sec0, data["binned_t"], binned_dist))
        ele0 = float(smooth_ele_plot[idx]) if len(smooth_ele_plot) else float("nan")

        def fmt_or_na(v, fmt):
            try:
                vv = float(v)
            except (TypeError, ValueError):
                return t("dashboard.value.na")
            return fmt(vv) if np.isfinite(vv) else t("dashboard.value.na")

        rider0 = fmt_or_na(smooth_power[idx], lambda v: f"{int(round(v))} W")
        motor0 = fmt_or_na(smooth_motor[idx], lambda v: f"{int(round(v))} W")
        battery0 = fmt_or_na(smooth_battery[idx], lambda v: f"{int(round(v))}%")
        hr0 = fmt_or_na(smooth_heartrate[idx], lambda v: f"{int(round(v))} bpm")

        tooltip_rows["time"].set_text(f'{t("tooltip.time")}: {int(round(x0))} min')
        tooltip_rows["distance"].set_text(f'{t("tooltip.distance")}: {dist0:.1f} km')
        tooltip_rows["elevation"].set_text(f'{t("tooltip.elevation")}: {int(round(ele0))} m')

        def set_metric_row(row_key, label, value, color, visible):
            tooltip_rows[row_key].set_text(f"{label}: {value}" if visible else f"{label}: {t('dashboard.value.na')}")
            tooltip_rows[row_key]._text.set_color(color if visible else "#9ca3af")

        set_metric_row("rider", t("dashboard.card.rider_power"), rider0, line_cyclist.get_color(), line_cyclist.get_visible())
        set_metric_row("motor", t("dashboard.card.motor_power"), motor0, line_motor.get_color(), line_motor.get_visible())
        if line_battery is not None:
            set_metric_row("battery", t("dashboard.card.battery"), battery0, line_battery.get_color(), line_battery.get_visible())
        else:
            tooltip_rows["battery"].set_text(f'{t("dashboard.card.battery")}: {t("dashboard.value.na")}')
            tooltip_rows["battery"]._text.set_color("#9ca3af")
        if line_heartrate is not None:
            set_metric_row("heartrate", t("dashboard.card.heartrate"), hr0, line_heartrate.get_color(), line_heartrate.get_visible())
        else:
            tooltip_rows["heartrate"].set_text(f'{t("dashboard.card.heartrate")}: {t("dashboard.value.na")}')
            tooltip_rows["heartrate"]._text.set_color("#9ca3af")

        hover_line.set_xdata([x0, x0])
        hover_line.set_visible(True)

        def set_dot(dot, y, show):
            if dot is None:
                return
            if not show or not np.isfinite(y):
                dot.set_visible(False)
                return
            dot.set_data([x0], [float(y)])
            dot.set_visible(True)

        set_dot(hover_dot_rider, smooth_power[idx], line_cyclist.get_visible())
        set_dot(hover_dot_motor, smooth_motor[idx], line_motor.get_visible())
        if hover_dot_battery is not None and line_battery is not None:
            set_dot(hover_dot_battery, smooth_battery[idx], line_battery.get_visible())
        if hover_dot_heartrate is not None and line_heartrate is not None:
            set_dot(hover_dot_heartrate, smooth_heartrate[idx], line_heartrate.get_visible())
        set_dot(hover_dot_elevation, ele0, elevation_line.get_visible())

        y_min, y_max = ax1.get_ylim()
        if event.x is not None and event.y is not None:
            # Use display->data transform so the tooltip follows cursor height even on ax2/ax_top.
            y_from_display = ax1.transData.inverted().transform((float(event.x), float(event.y)))[1]
            y_anchor = float(np.clip(y_from_display, y_min, y_max))
            y_frac = float((float(event.y) - ax1.bbox.y0) / max(1.0, ax1.bbox.height))
        else:
            y_anchor = float(y_min + 0.6 * (y_max - y_min))
            y_frac = 0.5

        tooltip.xy = (x0, y_anchor)

        # Flip the tooltip placement to keep it away from the pills/top header and chart edges.
        show_below = y_frac > 0.70
        y_offset = -18 if show_below else 18
        align_y = 1.0 if show_below else 0.0
        show_left = x0 > xmin + 0.7 * (xmax - xmin)
        x_offset = -110 if show_left else 18
        align_x = 1.0 if show_left else 0.0
        tooltip.xybox = (x_offset, y_offset)
        tooltip.box_alignment = (align_x, align_y)
        tooltip.set_visible(True)

        # Keep tooltip vertically inside the plotting area.
        renderer = fig.canvas.get_renderer()
        if renderer is not None:
            try:
                margin_px = 6.0
                dpi = float(fig.dpi) if fig.dpi else 100.0
                for _ in range(2):
                    bbox = tooltip.get_window_extent(renderer=renderer)
                    ax_bbox = ax1.bbox
                    y_offset_new = float(y_offset)

                    if bbox.y1 > (ax_bbox.y1 - margin_px):
                        delta_px = float(bbox.y1 - (ax_bbox.y1 - margin_px))
                        y_offset_new -= (delta_px * 72.0 / dpi)
                    if bbox.y0 < (ax_bbox.y0 + margin_px):
                        delta_px = float((ax_bbox.y0 + margin_px) - bbox.y0)
                        y_offset_new += (delta_px * 72.0 / dpi)

                    if abs(y_offset_new - float(y_offset)) < 0.25:
                        break
                    y_offset = y_offset_new
                    tooltip.xybox = (x_offset, y_offset)
            except Exception:
                pass

    def update_alpha(target_label):
        if target_label not in artist_groups:
            for artist in all_artists:
                artist.set_alpha(base_alpha.get(artist, 1.0))
            return
        target_set = set(artist_groups[target_label])
        for artist in all_artists:
            base = base_alpha.get(artist, 1.0)
            artist.set_alpha(base if artist in target_set else base * dim_factor)

    def pill_entry_at(event):
        if event.x is None or event.y is None:
            return None
        renderer = fig.canvas.get_renderer()
        if renderer is None:
            return None
        for ui in pill_ui.values():
            key = ui.get("target_key")
            if key not in artist_groups:
                continue
            bbox = ui["patch"].get_window_extent(renderer=renderer)
            if bbox.contains(event.x, event.y):
                return key
        return None

    def on_move(event):
        nonlocal active_target
        hovered = pill_entry_at(event)
        if hovered:
            hide_hover_details()
            if hovered and not is_group_visible(hovered):
                hovered = None
            if hovered == active_target:
                return
            active_target = hovered
            update_alpha(hovered)
            fig.canvas.draw_idle()
            return

        if active_target is not None:
            active_target = None
            update_alpha(None)

        update_hover_details(event)
        fig.canvas.draw_idle()

    def on_click(event):
        nonlocal active_target
        hide_hover_details()
        hovered = pill_entry_at(event)
        if hovered is None:
            return
        if hovered in artist_groups:
            target_artists = artist_groups[hovered]
            if not target_artists:
                return
            new_vis = not target_artists[0].get_visible()
            for artist in target_artists:
                artist.set_visible(new_vis)
            if hovered == pill_ui.get("elevation", {}).get("target_key"):
                ax2.yaxis.set_visible(new_vis)
                ax2.spines["right"].set_visible(new_vis)
            refresh_pills()
            active_target = hovered if new_vis else None
            update_alpha(hovered if new_vis else None)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)

    # Margins: leave some space top/bottom
    # Leave enough space for the header/pills so they don't overlap the top X axis.
    plot_top = max(0.1, pill_y - 0.09)
    fig.subplots_adjust(left=0.07, right=0.95, top=plot_top, bottom=0.10)

    # IMPORTANT: do not call fig.tight_layout() here

    return fig


def copy_figure_to_windows_clipboard(fig):
    """
    Copy the figure to the Windows clipboard as CF_DIB (bitmap).
    No external dependencies required.
    """
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    height, width, _ = rgba.shape
    # Convert to BGRA as Windows expects
    bgra = rgba[:, :, [2, 1, 0, 3]]
    pixel_bytes = bgra.astype(np.uint8).tobytes()

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", ctypes.c_long),
            ("biHeight", ctypes.c_long),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", ctypes.c_long),
            ("biYPelsPerMeter", ctypes.c_long),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    BI_RGB = 0
    CF_DIB = 8
    GMEM_MOVEABLE = 0x0002

    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    user32.CloseClipboard.restype = wintypes.BOOL

    header = BITMAPINFOHEADER()
    header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    header.biWidth = width
    header.biHeight = -height  # negative = top-down
    header.biPlanes = 1
    header.biBitCount = 32
    header.biCompression = BI_RGB
    header.biSizeImage = len(pixel_bytes)

    total_size = ctypes.sizeof(BITMAPINFOHEADER) + len(pixel_bytes)
    h_global = kernel32.GlobalAlloc(GMEM_MOVEABLE, total_size)
    if not h_global:
        raise LocalizedError("error.clip.alloc")

    try:
        ptr = kernel32.GlobalLock(h_global)
        if not ptr:
            raise LocalizedError("error.clip.lock")
        ptr_value = ctypes.cast(ptr, ctypes.c_void_p).value

        try:
            ctypes.memmove(ptr_value, ctypes.byref(header), ctypes.sizeof(BITMAPINFOHEADER))
            ctypes.memmove(ptr_value + ctypes.sizeof(BITMAPINFOHEADER), pixel_bytes, len(pixel_bytes))
        finally:
            kernel32.GlobalUnlock(h_global)

        if not user32.OpenClipboard(None):
            raise LocalizedError("error.clip.open")

        try:
            if not user32.EmptyClipboard():
                raise LocalizedError("error.clip.clear")
            if not user32.SetClipboardData(CF_DIB, h_global):
                raise LocalizedError("error.clip.set")
            h_global = None
        finally:
            user32.CloseClipboard()
    finally:
        if h_global:
            kernel32.GlobalFree(h_global)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.available_languages = discover_languages()
        self.available_language_codes = {code for code, _ in self.available_languages}

        last_lang = load_last_language_selection()
        if self.available_languages:
            if last_lang in self.available_language_codes:
                default_lang = last_lang
            elif "ca" in self.available_language_codes:
                default_lang = "ca"
            else:
                default_lang = self.available_languages[0][0]
        else:
            default_lang = "ca"
            self.available_languages = [("ca", "Català")]
            self.available_language_codes = {"ca"}

        self.lang = default_lang
        self.translations = load_translations(self.lang)
        self.lang_var = tk.StringVar(value=self.lang)

        self.title(self.t("app.title"))

        # Window size: a bit smaller than the figure
        self.geometry("1200x600")

        # Main menu bar
        self.menubar = tk.Menu(self)
        self.config(menu=self.menubar)
        self.menu_indices = {}

        self.menubar.add_command(label=self.t("menu.open"), command=self.open_and_plot)
        self.menu_indices["open"] = self.menubar.index("end")
        self.menubar.add_command(label=self.t("menu.copy"), command=self.copy_chart, state=tk.DISABLED)
        self.menu_indices["copy"] = self.menubar.index("end")
        self.menubar.add_command(label=self.t("menu.export"), command=self.export_png, state=tk.DISABLED)
        self.menu_indices["export"] = self.menubar.index("end")

        self.language_menu = tk.Menu(self.menubar, tearoff=0)
        self.language_menu_indices = {}
        self.language_labels = {code: label for code, label in self.available_languages}
        for code, label in self.available_languages:
            self.language_menu.add_radiobutton(
                label=label,
                value=code,
                variable=self.lang_var,
                command=lambda c=code: self.set_language(c),
            )
            self.language_menu_indices[code] = self.language_menu.index("end")
        language_menu_label = self.translations.get("menu.language_menu", "Language")
        self.menubar.add_cascade(label=language_menu_label, menu=self.language_menu)
        self.menu_indices["language"] = self.menubar.index("end")

        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_widget = None
        self.figure = None
        self.last_data = None

        # Prompt for file automatically on startup
        self.after(100, self.open_and_plot)

    def t(self, key, **kwargs):
        text = self.translations.get(key, key)
        if kwargs:
            try:
                text = text.format(**kwargs)
            except Exception:
                pass
        return text

    def format_error(self, exc, fallback_key):
        if isinstance(exc, LocalizedError):
            return self.t(exc.key, detail=getattr(exc, "detail", "") or "")
        return self.t(fallback_key, error=exc)

    def set_language(self, lang_code):
        self.lang = lang_code
        self.translations = load_translations(lang_code)
        self.lang_var.set(lang_code)
        save_last_language_selection(lang_code)
        self.update_menu_labels()
        self.title(self.t("app.title"))
        if self.last_data is not None:
            self.render_figure(self.last_data)

    def update_menu_labels(self):
        for key, label_key in [
            ("open", "menu.open"),
            ("copy", "menu.copy"),
            ("export", "menu.export"),
        ]:
            idx = self.menu_indices[key]
            state = self.menubar.entrycget(idx, "state")
            self.menubar.entryconfig(idx, label=self.t(label_key), state=state)
        language_menu_label = self.translations.get("menu.language_menu", "Language")
        self.menubar.entryconfig(self.menu_indices["language"], label=language_menu_label)
        for code, idx in self.language_menu_indices.items():
            label = self.language_labels.get(code, code)
            self.language_menu.entryconfig(idx, label=label)

    def render_figure(self, data):
        fig = create_figure(data, self.t)
        if self.canvas_widget is not None:
            self.canvas_widget.destroy()
        self.figure = fig
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        self.canvas_widget = widget
        self.menubar.entryconfig(self.menu_indices["copy"], state=tk.NORMAL)
        self.menubar.entryconfig(self.menu_indices["export"], state=tk.NORMAL)

    def open_and_plot(self):
        path = filedialog.askopenfilename(
            title=self.t("dialog.open.title"),
            filetypes=[(self.t("dialog.open.gpx"), "*.gpx"), (self.t("dialog.open.all"), "*.*")],
        )
        if not path:
            return

        try:
            (
                times,
                powers,
                motor_powers,
                batteries,
                heartrates,
                ebike_modes,
                elevs,
                lats,
                lons,
            ) = parse_gpx(path)
            data = compute_moving_binned_data(
                times, powers, motor_powers, batteries, heartrates, ebike_modes, elevs, lats, lons
            )
        except Exception as e:
            messagebox.showerror(self.t("message.error.title"), self.format_error(e, "message.process.error"))
            return

        self.last_data = data
        self.render_figure(data)

    def copy_chart(self):
        if self.figure is None:
            messagebox.showinfo(self.t("message.no_chart.title"), self.t("message.no_chart.body"))
            return
        try:
            copy_figure_to_windows_clipboard(self.figure)
        except Exception as e:
            messagebox.showerror(self.t("message.error.title"), self.format_error(e, "message.copy.error"))

    def export_png(self):
        if self.figure is None:
            messagebox.showinfo(self.t("message.no_chart.title"), self.t("message.no_chart.body"))
            return
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = filedialog.asksaveasfilename(
            title=self.t("dialog.save.title"),
            defaultextension=".png",
            filetypes=[(self.t("dialog.save.filter_png"), "*.png")],
            initialdir=script_dir,
            initialfile=self.t("dialog.save.default_name"),
        )
        if not save_path:
            return
        try:
            self.figure.savefig(save_path, dpi=self.figure.dpi)
        except Exception as e:
            messagebox.showerror(self.t("message.error.title"), self.format_error(e, "message.save.error"))

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
