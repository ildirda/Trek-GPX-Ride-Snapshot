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

    # Double smoothing: 7-point then 5-point moving average
    kernel1 = np.ones(7) / 7
    kernel2 = np.ones(5) / 5

    def double_smooth(x):
        s1 = np.convolve(x, kernel1, mode="same")
        s2 = np.convolve(s1, kernel2, mode="same")
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
                smooth_battery_centered = np.convolve(smooth_battery_interp, kernel_battery, mode="same")
                # Keep original (interpolated) edge values to avoid vertical artifacts
                edge_preserve = 3
                if len(smooth_battery_interp) > edge_preserve * 2:
                    # At edges keep original interpolated values (unsmoothed)
                    smooth_battery[:edge_preserve] = smooth_battery_interp[:edge_preserve]
                    smooth_battery[-edge_preserve:] = smooth_battery_interp[-edge_preserve:]
                    # Use smoothing in the middle
                    smooth_battery[edge_preserve:-edge_preserve] = smooth_battery_centered[edge_preserve:-edge_preserve]
                else:
                    smooth_battery = smooth_battery_interp
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
                smooth_heartrate_centered = np.convolve(smooth_heartrate_interp, kernel_heartrate, mode="same")
                # Keep original (interpolated) edge values to avoid vertical artifacts
                edge_preserve = 3
                if len(smooth_heartrate_interp) > edge_preserve * 2:
                    # At edges keep original interpolated values (unsmoothed)
                    smooth_heartrate[:edge_preserve] = smooth_heartrate_interp[:edge_preserve]
                    smooth_heartrate[-edge_preserve:] = smooth_heartrate_interp[-edge_preserve:]
                    # Use smoothing in the middle
                    smooth_heartrate[edge_preserve:-edge_preserve] = smooth_heartrate_centered[edge_preserve:-edge_preserve]
                else:
                    smooth_heartrate = smooth_heartrate_interp
            else:
                smooth_heartrate = binned_heartrate
        else:
            smooth_heartrate = binned_heartrate
    else:
        smooth_heartrate = binned_heartrate

    # Lightly smoothed elevation
    kernel_ele = np.ones(11) / 11
    smooth_ele = np.convolve(binned_ele, kernel_ele, mode="same")

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
    }


def create_figure(data, t):
    # 1980 px wide, 21:9 aspect ratio
    dpi = 100
    width_in = 1980 / dpi
    height_in = (1980 * 9 / 21) / dpi

    fig = Figure(figsize=(width_in, height_in), dpi=dpi)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

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

    # --- background elevation ---
    elevation_fill = ax2.fill_between(t_min, smooth_ele, color="lightgray", alpha=0.4, step="pre", zorder=0)
    ax2.set_ylabel(t("figure.axis.altitude"))
    ax2.set_ylim(
        min(smooth_ele.min(), binned_ele.min()) - 10,
        max(smooth_ele.max(), binned_ele.max()) + 10,
    )
    ax2.tick_params(axis="y", labelsize=8, colors="gray")

    # --- power ---
    line_cyclist, = ax1.plot(t_min, smooth_power, label=t("figure.legend.cyclist"), zorder=3)
    line_motor,   = ax1.plot(t_min, smooth_motor, label=t("figure.legend.motor"),   zorder=3)

    # --- battery (red line) on left Y axis ---
    if not np.all(np.isnan(smooth_battery)):
        line_battery, = ax1.plot(t_min, smooth_battery, color="red", 
                                 label=t("figure.legend.battery"), zorder=3, linewidth=1.5)
        
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
                marker_line, = ax1.plot(change_time_min, battery_value, 's', color='red', 
                        markersize=12, markeredgecolor='red', markeredgewidth=1.5,
                        zorder=5, clip_on=False)
                mode_markers.append(marker_line)
                mode_text = ax1.text(change_time_min, battery_value, str(mode_value), 
                        ha='center', va='center', fontsize=8, fontweight='bold',
                        color='white', zorder=6)
                mode_texts.append(mode_text)

    # --- heart rate (purple line) on left Y axis ---
    if not np.all(np.isnan(smooth_heartrate)):
        line_heartrate, = ax1.plot(t_min, smooth_heartrate, color="purple", 
                                    label=t("figure.legend.heartrate"), zorder=3, linewidth=1.5)

    mean_line_power = ax1.axhline(mean_power, linestyle="--", linewidth=1.2,
                                  color=line_cyclist.get_color(), alpha=0.5, label=t("figure.legend.mean_cyclist"))
    mean_line_motor = ax1.axhline(mean_motor, linestyle="--", linewidth=1.2,
                                  color=line_motor.get_color(),   alpha=0.5, label=t("figure.legend.mean_motor"))

    ax1.set_xlabel(t("figure.axis.time"))
    ax1.set_ylabel(t("figure.axis.power"))
    # --- power ---
    ymax = max(smooth_power.max(), smooth_motor.max())
    if not np.all(np.isnan(smooth_battery)):
        ymax = max(ymax, smooth_battery.max())
    if not np.all(np.isnan(smooth_heartrate)):
        ymax = max(ymax, smooth_heartrate.max())
    ax1.set_ylim(0, ymax * 1.05)

    # --- top axis in km ---
    ax_top = ax1.twiny()
    ax_top.set_xlim(ax1.get_xlim())
    minute_ticks = ax1.get_xticks()
    seconds_ticks = minute_ticks * 60.0
    km_labels = np.interp(seconds_ticks, data["binned_t"], binned_dist)
    ax_top.set_xticks(minute_ticks)
    ax_top.set_xticklabels([f"{k:.1f}" for k in km_labels])
    ax_top.set_xlabel(t("figure.axis.distance"))

    ax1.set_title(t("figure.title"))

    # --- legend at bottom inside figure ---
    handles, labels = ax1.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    # Items per legend row
    n_items = len(unique)
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=n_items,
        bbox_to_anchor=(0.5, 0.02)  # inside the figure
    )
    legend = fig.legends[-1]

    # Legend hover: highlight entry and dim others
    legend_handles = getattr(legend, "legendHandles", None)
    if legend_handles is None:
        legend_handles = getattr(legend, "legend_handles", None)
    if legend_handles is None:
        legend_handles = legend.get_lines()
    legend_items = list(zip(legend_handles, legend.get_texts()))
    legend_labels = list(unique.keys())
    legend_text_map = {lbl: txt for lbl, (_h, txt) in zip(legend_labels, legend_items)}
    legend_base_label = {lbl: lbl for lbl in legend_labels}
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

    all_artists = set()
    for group_items in artist_groups.values():
        all_artists.update(group_items)
    all_artists.add(elevation_fill)

    base_alpha = {artist: (artist.get_alpha() if artist.get_alpha() is not None else 1.0) for artist in all_artists}
    dim_factor = 0.2
    active_label = None

    def update_alpha(target_label):
        if target_label not in artist_groups:
            for artist in all_artists:
                artist.set_alpha(base_alpha.get(artist, 1.0))
            return
        target_set = set(artist_groups[target_label])
        for artist in all_artists:
            base = base_alpha.get(artist, 1.0)
            artist.set_alpha(base if artist in target_set else base * dim_factor)

    def legend_entry_at(event):
        renderer = fig.canvas.get_renderer()
        if renderer is None:
            return None
        for idx, (handle, text) in enumerate(legend_items):
            hb = handle.get_window_extent(renderer=renderer)
            tb = text.get_window_extent(renderer=renderer)
            if hb.contains(event.x, event.y) or tb.contains(event.x, event.y):
                return legend_labels[idx]
        return None

    def on_move(event):
        nonlocal active_label
        hovered = legend_entry_at(event)
        if hovered and hovered in artist_groups and not artist_groups[hovered][0].get_visible():
            hovered = None
        if hovered == active_label:
            return
        active_label = hovered
        update_alpha(hovered)
        fig.canvas.draw_idle()

    def on_click(event):
        hovered = legend_entry_at(event)
        if hovered is None:
            return
        if hovered in artist_groups:
            target_artists = artist_groups[hovered]
            new_vis = not target_artists[0].get_visible()
            for artist in target_artists:
                artist.set_visible(new_vis)
            base_lbl = legend_base_label.get(hovered, hovered)
            legend_text = legend_text_map.get(hovered)
            if legend_text is not None:
                hidden_suffix = t("figure.legend.hidden")
                legend_text.set_text(base_lbl if new_vis else f"{base_lbl} {hidden_suffix}")
                legend_text.set_color("black" if new_vis else "gray")
            update_alpha(hovered if new_vis else None)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)

    # Margins: leave some space top/bottom
    fig.subplots_adjust(left=0.07, right=0.95, top=0.85, bottom=0.20)

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
