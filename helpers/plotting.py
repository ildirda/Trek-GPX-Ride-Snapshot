# -*- coding: utf-8 -*-

import os

import numpy as np
import matplotlib.image as mpimg
from matplotlib.figure import Figure
from matplotlib import transforms
from matplotlib.colors import to_rgba
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea, VPacker
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import RangeSlider

from helpers.state import save_pill_state

def _project_root():
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(module_dir)

def create_figure(data, t):
    # 1980 px wide, 21:9 aspect ratio
    dpi = 100
    width_in = (1500) / dpi
    # Add some vertical space for header + pills (keep it tight).
    height_in = ((1980 * 9 / 21) + 250) / dpi
    fig_width_px = width_in * dpi
    fig_height_px = height_in * dpi

    content_shift_down_px = 0
    content_shift_frac = content_shift_down_px / fig_height_px
    axis_block_shift_px = 58
    axis_block_shift_frac = axis_block_shift_px / fig_height_px

    fig = Figure(figsize=(width_in, height_in), dpi=dpi)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    # Ensure ax1 artists (lines, tooltip) draw above the elevation background on ax2.
    ax2.set_zorder(0)
    ax1.set_zorder(1)
    ax1.patch.set_alpha(0)

    def format_value_or_na(value, fmt):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return t("dashboard.value.na")
        return fmt(v) if np.isfinite(v) else t("dashboard.value.na")

    def clean_label(key):
        label = t(key)
        stripped = label.lstrip()
        while stripped and not stripped[0].isalnum():
            stripped = stripped[1:].lstrip()
        return stripped or label

    # Header (title, subtitle, key stats)
    header_title = data.get("title") or t("dashboard.title")
    total_distance_km = data.get("total_distance_km")
    total_time_min = data.get("total_time_min")
    max_rider_power = data.get("max_rider_power")
    max_motor_power = data.get("max_motor_power")
    towns_visited = data.get("towns_visited")
    car_events_count = data.get("car_events_count")
    weather_summary = data.get("weather_summary")
    has_car_events = data.get("has_car_events")
    if has_car_events is None:
        has_car_events = car_events_count is not None

    assets_dir = os.path.join(_project_root(), "assets")
    icon_cache = {}

    def desaturate_icon(image):
        if image is None:
            return None
        arr = image.astype(float)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.shape[-1] >= 4:
            rgb = arr[..., :3]
            alpha = arr[..., 3:4]
        else:
            rgb = arr[..., :3]
            alpha = None
        gray = (rgb[..., 0] * 0.299) + (rgb[..., 1] * 0.587) + (rgb[..., 2] * 0.114)
        gray_rgb = np.stack([gray, gray, gray], axis=-1)
        if alpha is not None:
            out = np.concatenate([gray_rgb, alpha], axis=-1)
        else:
            out = gray_rgb
        if image.dtype == np.uint8:
            out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        else:
            out = out.astype(image.dtype)
        return out

    def load_icon(name):
        if name in icon_cache:
            return icon_cache[name]
        path = os.path.join(assets_dir, name)
        try:
            icon = mpimg.imread(path)
            icon_cache[name] = desaturate_icon(icon)
        except Exception:
            icon_cache[name] = None
        return icon_cache[name]

    header_items = [
        {
            "icon": "ruta.png",
            "label": clean_label("dashboard.summary.total_distance"),
            "value": format_value_or_na(total_distance_km, lambda v: f"{v:.1f} km"),
            "weight": 1.0,
        },
        {
            "icon": "temps.png",
            "label": clean_label("dashboard.summary.total_time"),
            "value": format_value_or_na(total_time_min, lambda v: f"{int(round(v))} min"),
            "weight": 1.0,
        },
        {
            "icon": "Wciclista.png",
            "label": clean_label("dashboard.summary.max_rider_power"),
            "value": format_value_or_na(max_rider_power, lambda v: f"{int(round(v))} W"),
            "weight": 1.0,
        },
        {
            "icon": "Wmotor.png",
            "label": clean_label("dashboard.summary.max_motor_power"),
            "value": format_value_or_na(max_motor_power, lambda v: f"{int(round(v))} W"),
            "weight": 1.0,
        },
        {
            "icon": "pobles.png",
            "label": clean_label("dashboard.summary.towns_visited"),
            "value": format_value_or_na(towns_visited, lambda v: f"{int(round(v))}"),
            "weight": 1.0,
        },
    ]
    if has_car_events:
        header_items.append(
            {
                "icon": "cotxes.png",
                "label": clean_label("dashboard.summary.car_events"),
                "value": format_value_or_na(car_events_count, lambda v: f"{int(round(v))}"),
                "weight": 1.0,
            }
        )
    header_items.append(
        {
            "icon": "clima.png",
            "label": clean_label("dashboard.summary.initial_weather"),
            "value": weather_summary if weather_summary else t("dashboard.value.na"),
            "weight": 2.0,
        }
    )

    fig.text(0.02, 0.965, header_title, ha="left", va="top", fontsize=14, color="#111827")
    label_y = 0.905 - content_shift_frac
    value_y = 0.875 - content_shift_frac
    x_start = 0.02
    x_end = 0.98
    total_weight = sum(item["weight"] for item in header_items)
    icon_zoom = 0.4
    icon_size_px = 40.0 * icon_zoom
    icon_gap_px = 14.0
    icon_text_offset = (icon_size_px + icon_gap_px) / fig_width_px
    icon_y = value_y + (label_y - value_y) * 0.25
    x_pos = x_start
    for item in header_items:
        icon = load_icon(item["icon"]) if item.get("icon") else None
        if icon is not None:
            image = OffsetImage(icon, zoom=icon_zoom)
            icon_box = AnnotationBbox(
                image,
                (x_pos, icon_y),
                xycoords=fig.transFigure,
                box_alignment=(0.0, 0.5),
                frameon=False,
            )
            fig.add_artist(icon_box)
            text_x = x_pos + icon_text_offset
        else:
            text_x = x_pos
        fig.text(
            text_x,
            label_y,
            item["label"],
            ha="left",
            va="top",
            fontsize=8,
            color="#6b7280",
        )
        fig.text(
            text_x,
            value_y,
            item["value"],
            ha="left",
            va="top",
            fontsize=10,
            color="#111827",
        )
        x_pos += (x_end - x_start) * (item["weight"] / total_weight)

    # Pills (cards) row
    pill_shift_right_px = 0
    pill_shift_down_px = 85
    pill_x0, pill_y, pill_w, pill_h = (
        0.02 + (pill_shift_right_px / fig_width_px),
        0.845 - (pill_shift_down_px / fig_height_px) - content_shift_frac,
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
    avg_cadence = format_value_or_na(data.get("mean_cadence"), lambda v: f"{int(round(v))} rpm")
    battery_start = format_value_or_na(data.get("first_battery"), lambda v: f"{int(round(v))}%")
    battery_end = format_value_or_na(data.get("last_battery"), lambda v: f"{int(round(v))}%")
    battery_range = t("dashboard.value.battery_range", start=battery_start, end=battery_end)
    elevation_gain = format_value_or_na(data.get("elevation_gain_m"), lambda v: f"{chr(0x2197)} {int(round(v))} m")

    pill_gap_px = 10
    pill_gap = pill_gap_px / fig_width_px
    pills = [
        ("rider", t("dashboard.card.rider_power"), f'{t("dashboard.value.avg")}: {avg_rider}', "#eff6ff", "#bfdbfe", "#2563eb"),
        ("motor", t("dashboard.card.motor_power"), f'{t("dashboard.value.avg")}: {avg_motor}', "#fff7ed", "#fed7aa", "#ea580c"),
        ("cadence", t("dashboard.card.cadence"), f'{t("dashboard.value.avg")}: {avg_cadence}', "#ecfeff", "#a5f3fc", "#06b6d4"),
        ("heartrate", t("dashboard.card.heartrate"), f'{t("dashboard.value.avg")}: {avg_hr}', "#fdf2f8", "#fbcfe8", "#db2777"),
        ("battery", t("dashboard.card.battery"), battery_range, "#ecfdf5", "#a7f3d0", "#059669"),
        ("elevation", t("dashboard.card.elevation"), f"{elevation_gain}", "#f8fafc", "#e2e8f0", "#475569"),
    ]

    pill_right = 0.98
    available_width = max(0.1, pill_right - pill_x0)
    pill_w = (available_width - (len(pills) - 1) * pill_gap) / max(1, len(pills))

    pill_ui = {}
    avg_pill_ids = {"rider", "motor", "cadence", "heartrate"}
    for idx, (pill_id, label, value, face, edge, value_color) in enumerate(pills):
        x = pill_x0 + idx * (pill_w + pill_gap)
        patch, label_text, value_text = draw_pill(x, pill_y, pill_w, pill_h, label, value, face, edge, value_color)
        pill_ui[pill_id] = {
            "target_key": f"pill.{pill_id}",
            "patch": patch,
            "label_text": label_text,
            "value_text": value_text,
            "active_value_text": value,
            "hide_value_when_inactive": pill_id in avg_pill_ids,
            "active_face": face,
            "active_edge": edge,
            "active_value_color": value_color,
        }

    t_min = data["smooth_t_min"]
    binned_t = data["binned_t"]
    binned_dist = data["binned_dist"]
    binned_ele = data["binned_ele"]
    binned_power = data["binned_power"]
    binned_motor = data["binned_motor"]
    binned_cadence = data["binned_cadence"]
    smooth_battery = data["smooth_battery"]
    smooth_heartrate = data["smooth_heartrate"]
    smooth_ele = data["smooth_ele"]
    mean_power = data["mean_power"]
    mean_motor = data["mean_motor"]
    mean_cadence = data["mean_cadence"]
    has_heartrate = not np.all(np.isnan(smooth_heartrate))
    has_cadence = not np.all(np.isnan(binned_cadence))

    def convolve_edge_preserving(x, kernel):
        pad = (len(kernel) - 1) // 2
        if len(x) >= 2 and pad > 0:
            x_padded = np.pad(x, (pad, pad), mode="edge")
            return np.convolve(x_padded, kernel, mode="valid")
        return np.convolve(x, kernel, mode="same")

    kernel_light = np.ones(3) / 3
    kernel_medium = np.ones(5) / 5
    kernel_heavy_1 = np.ones(7) / 7
    kernel_heavy_2 = np.ones(5) / 5

    def smooth_series(series, level, interpolate_nans=False):
        if series is None or len(series) == 0:
            return series
        if level == "none":
            return series
        if interpolate_nans:
            valid_mask = ~np.isnan(series)
            if np.sum(valid_mask) <= 1:
                return series
            valid_indices = np.where(valid_mask)[0]
            series = np.interp(np.arange(len(series)), valid_indices, series[valid_indices])
        if level == "light":
            return convolve_edge_preserving(series, kernel_light)
        if level == "medium":
            return convolve_edge_preserving(series, kernel_medium)
        return convolve_edge_preserving(convolve_edge_preserving(series, kernel_heavy_1), kernel_heavy_2)

    def build_series_by_level(series, interpolate_nans=False):
        return {
            "none": smooth_series(series, "none", interpolate_nans),
            "light": smooth_series(series, "light", interpolate_nans),
            "medium": smooth_series(series, "medium", interpolate_nans),
            "heavy": smooth_series(series, "heavy", interpolate_nans),
        }

    series_by_level = {
        "power": build_series_by_level(binned_power),
        "motor": build_series_by_level(binned_motor),
        "cadence": build_series_by_level(binned_cadence, interpolate_nans=True),
    }

    smooth = {
        "power": series_by_level["power"]["heavy"],
        "motor": series_by_level["motor"]["heavy"],
        "cadence": series_by_level["cadence"]["heavy"],
        "heartrate": smooth_heartrate,
        "battery": smooth_battery,
    }
    line_battery = None
    line_heartrate = None
    line_cadence = None
    mean_line_cadence = None
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

    # Use elevation min/max with a small margin so the background isn't anchored at zero.
    ele_values = []
    for series in (smooth_ele_plot, binned_ele):
        if series is None or len(series) == 0:
            continue
        series = np.asarray(series, dtype=float)
        series = series[np.isfinite(series)]
        if series.size:
            ele_values.append((float(series.min()), float(series.max())))
    if ele_values:
        ele_min = min(v[0] for v in ele_values)
        ele_max = max(v[1] for v in ele_values)
        span = ele_max - ele_min
        margin = max(5.0, span * 0.05)
        ele_ymin = ele_min - margin
        ele_ymax = ele_max + margin
    else:
        ele_ymin = 0.0
        ele_ymax = 1.0

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
    line_cyclist, = ax1.plot(t_min, smooth["power"], label=t("figure.legend.cyclist"), zorder=3, linewidth=2.2)
    line_motor,   = ax1.plot(t_min, smooth["motor"], label=t("figure.legend.motor"),   zorder=3, linewidth=2.2)

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
    if has_heartrate:
        line_heartrate, = ax1.plot(t_min, smooth["heartrate"], color="purple", 
                                    label=t("figure.legend.heartrate"), zorder=3, linewidth=2.2)

    # --- cadence (blue line) on left Y axis ---
    if has_cadence:
        line_cadence, = ax1.plot(
            t_min,
            smooth["cadence"],
            color="#06b6d4",
            label=t("figure.legend.cadence"),
            zorder=3,
            linewidth=2.2,
        )
        if mean_cadence is not None and np.isfinite(mean_cadence):
            mean_line_cadence = ax1.axhline(
                mean_cadence,
                linestyle="--",
                linewidth=1.8,
                color=line_cadence.get_color(),
                alpha=0.5,
                label=t("figure.legend.mean_cadence"),
            )

    mean_line_power = ax1.axhline(mean_power, linestyle="--", linewidth=1.8,
                                  color=line_cyclist.get_color(), alpha=0.5, label=t("figure.legend.mean_cyclist"))
    mean_line_motor = ax1.axhline(mean_motor, linestyle="--", linewidth=1.8,
                                  color=line_motor.get_color(),   alpha=0.5, label=t("figure.legend.mean_motor"))

    pill_targets = {
        "rider": [line_cyclist, mean_line_power],
        "motor": [line_motor, mean_line_motor],
        "heartrate": [line_heartrate] if line_heartrate is not None else [],
        "cadence": (
            [line_cadence, mean_line_cadence]
            if mean_line_cadence is not None
            else ([line_cadence] if line_cadence is not None else [])
        ),
        "battery": ([line_battery] + mode_markers + mode_texts) if line_battery is not None else [],
        "elevation": [elevation_fill, elevation_line],
    }

    pill_state = data.get("pill_state") or {}
    for pill_id, target_artists in pill_targets.items():
        if not target_artists:
            continue
        visible = bool(pill_state.get(pill_id, True))
        for artist in target_artists:
            artist.set_visible(visible)
        if pill_id == "elevation":
            ax2.yaxis.set_visible(visible)
            ax2.spines["right"].set_visible(visible)

    def distance_span_km(x0, x1):
        if len(binned_t) == 0:
            return 0.0
        try:
            s0 = float(x0) * 60.0
            s1 = float(x1) * 60.0
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(s0) or not np.isfinite(s1):
            return 0.0
        d0 = float(np.interp(s0, binned_t, binned_dist))
        d1 = float(np.interp(s1, binned_t, binned_dist))
        return abs(d1 - d0)

    def smoothing_level_for_span(span_km):
        if not np.isfinite(span_km):
            return "heavy"
        if span_km < 1.0:
            return "none"
        if span_km < 3.0:
            return "light"
        if span_km < 7.0:
            return "medium"
        return "heavy"

    current_smoothing = {"level": None}

    def safe_max(series):
        if series is None or len(series) == 0:
            return float("nan")
        try:
            return float(np.nanmax(series))
        except (TypeError, ValueError):
            return float("nan")

    def update_y_limits():
        values = [safe_max(smooth["power"]), safe_max(smooth["motor"])]
        if line_battery is not None:
            values.append(safe_max(smooth_battery))
        if has_heartrate:
            values.append(safe_max(smooth["heartrate"]))
        if has_cadence:
            values.append(safe_max(smooth["cadence"]))
        values = [v for v in values if np.isfinite(v)]
        if not values:
            return
        ymax = max(values)
        ax1.set_ylim(0, ymax * 1.05)

    def apply_smoothing_level(level):
        if level == current_smoothing["level"]:
            return
        smooth["power"] = series_by_level["power"][level]
        smooth["motor"] = series_by_level["motor"][level]
        smooth["cadence"] = series_by_level["cadence"][level]
        line_cyclist.set_ydata(smooth["power"])
        line_motor.set_ydata(smooth["motor"])
        if line_heartrate is not None:
            line_heartrate.set_ydata(smooth["heartrate"])
        if line_cadence is not None:
            line_cadence.set_ydata(smooth["cadence"])
        update_y_limits()
        current_smoothing["level"] = level

    def update_smoothing_for_xlim(x0, x1):
        span_km = distance_span_km(x0, x1)
        level = smoothing_level_for_span(span_km)
        apply_smoothing_level(level)

    ax1.set_xlabel(t("figure.axis.time"))
    ax1.set_ylabel(t("figure.axis.power"))

    # --- axis styling (soft gray, fewer ticks, subtle dotted grid) ---
    axis_color = "#64748b"
    spine_color = "#cbd5e1"
    grid_color = "#e2e8f0"

    x_max = float(np.nanmax(t_min)) if len(t_min) else 0.0
    tick_step_min = 10.0
    x_end = x_max if x_max > 0 else tick_step_min
    ax1.set_xlim(0.0, x_end)
    update_smoothing_for_xlim(0.0, x_end)

    def update_time_ticks(x0, x1):
        try:
            x0 = float(x0)
            x1 = float(x1)
        except (TypeError, ValueError):
            return
        if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0:
            return

        span = x1 - x0
        preferred_steps = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0], dtype=float)
        target = span / 6.0
        step = preferred_steps[-1]
        for s in preferred_steps:
            if s >= target:
                step = float(s)
                break

        start = np.ceil(x0 / step) * step
        ticks = np.arange(start, x1 + 1e-9, step, dtype=float)
        if ticks.size < 2:
            ticks = np.array([x0, x1], dtype=float)

        ax1.set_xticks(ticks)

    update_time_ticks(0.0, x_end)
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
    ax_top.set_zorder(3)
    ax_top.patch.set_alpha(0)
    distance_label_y = 1.0
    label_offset_px = 22
    weather_offset_px = 10
    label_transform = ax1.get_xaxis_transform() + transforms.ScaledTranslation(
        0,
        label_offset_px / fig.dpi,
        fig.dpi_scale_trans,
    )
    weather_transform = ax1.get_xaxis_transform() + transforms.ScaledTranslation(
        0,
        weather_offset_px / fig.dpi,
        fig.dpi_scale_trans,
    )

    def set_top_axis_ticks(x0, x1):
        xticks = np.array([float(x0), float(x1)], dtype=float)
        ax_top.set_xticks(xticks)
        ax_top.set_xticklabels(["", ""])
        return xticks

    def update_distance_tick_texts(x0, x1):
        xticks = np.array([float(x0), float(x1)], dtype=float)
        seconds_ticks = xticks * 60.0
        km_labels = np.interp(seconds_ticks, data["binned_t"], binned_dist)
        labels = []
        for value in km_labels:
            try:
                labels.append(f"{float(value):.1f}")
            except (TypeError, ValueError):
                labels.append("")
        if len(distance_tick_texts) >= 2:
            distance_tick_texts[0].set_x(float(xticks[0]))
            distance_tick_texts[0].set_text(labels[0])
            distance_tick_texts[1].set_x(float(xticks[1]))
            distance_tick_texts[1].set_text(labels[1])

    x0_full, x1_full = ax1.get_xlim()
    ax_top.set_xlim((x0_full, x1_full))
    set_top_axis_ticks(x0_full, x1_full)
    ax_top.set_xlabel(t("figure.axis.distance"), labelpad=32)
    style_axes(ax_top, hide_bottom=True)
    ax_top.tick_params(axis="x", pad=2, labeltop=False)
    ax_top.margins(x=0)

    distance_tick_texts = [
        ax1.text(
            x0_full,
            distance_label_y,
            "",
            transform=label_transform,
            ha="left",
            va="bottom",
            fontsize=8,
            color=axis_color,
            zorder=6,
            clip_on=False,
        ),
        ax1.text(
            x1_full,
            distance_label_y,
            "",
            transform=label_transform,
            ha="right",
            va="bottom",
            fontsize=8,
            color=axis_color,
            zorder=6,
            clip_on=False,
        ),
    ]
    update_distance_tick_texts(x0_full, x1_full)

    car_event_markers = []
    car_event_spans_min = data.get("car_event_spans_min") or []
    if car_event_spans_min:
        spans = []
        for span in car_event_spans_min:
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                continue
            try:
                start = float(span[0])
                end = float(span[1])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(start) or not np.isfinite(end):
                continue
            if end < start:
                start, end = end, start
            spans.append((start, end))
        for start, end in spans:
            segment, = ax_top.plot(
                [start, end],
                [distance_label_y, distance_label_y],
                color="#ef4444",
                linewidth=4.0,
                alpha=0.85,
                solid_capstyle="round",
                transform=ax_top.get_xaxis_transform(),
                zorder=12,
                clip_on=False,
            )
            car_event_markers.append(segment)
    else:
        car_event_times_min = data.get("car_event_times_min") or []
        if car_event_times_min:
            car_event_times = []
            for value in car_event_times_min:
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(v):
                    car_event_times.append(v)
            if car_event_times:
                car_event_times = np.array(sorted(car_event_times), dtype=float)
                car_event_line, = ax_top.plot(
                    car_event_times,
                    np.full_like(car_event_times, distance_label_y, dtype=float),
                    linestyle="None",
                    marker="_",
                    markersize=14,
                    markeredgewidth=4,
                    color="#ef4444",
                    alpha=0.85,
                    transform=ax_top.get_xaxis_transform(),
                    zorder=12,
                    clip_on=False,
                )
                car_event_markers.append(car_event_line)

    locality_lines = []
    locality_texts = []
    locality_weather_texts = []
    locality_weather_positions = []
    locality_stays = data.get("locality_stays", [])
    if locality_stays:
        locality_color = "#1f2937"
        for stay in locality_stays:
            time_min = float(stay.get("time", 0.0)) / 60.0
            name = stay.get("name")
            if not name or not np.isfinite(time_min):
                continue
            line = ax1.axvline(
                time_min,
                color=locality_color,
                linewidth=1.1,
                linestyle=(0, (3, 3)),
                alpha=0.7,
                zorder=2,
            )
            locality_lines.append(line)
            dist_km = stay.get("distance_km")
            dist_label = None
            try:
                dist_value = float(dist_km)
                if np.isfinite(dist_value):
                    dist_label = f"{int(round(dist_value))}"
            except (TypeError, ValueError):
                dist_label = None
            if dist_label:
                label = f"{str(name).upper()} ({dist_label})"
            else:
                label = str(name).upper()
            text = ax1.text(
                time_min,
                distance_label_y,
                label,
                transform=label_transform,
                ha="center",
                va="bottom",
                fontsize=7,
                #fontweight="semibold",
                color=locality_color,
                alpha=0.85,
                zorder=6,
                clip_on=False,
            )
            locality_texts.append(text)
    locality_weather_segments = data.get("locality_weather_segments", [])
    for segment in locality_weather_segments:
        time_min = float(segment.get("time", 0.0)) / 60.0
        label = segment.get("label")
        if not label or not np.isfinite(time_min):
            continue
        weather_text = ax1.text(
            time_min,
            distance_label_y,
            str(label),
            transform=weather_transform,
            ha="center",
            va="bottom",
            fontsize=6,
            color="#6b7280",
            alpha=0.85,
            zorder=6,
            clip_on=False,
        )
        locality_weather_texts.append(weather_text)
        locality_weather_positions.append(time_min)

    def update_range_visibility(x0, x1):
        try:
            xmin = float(min(x0, x1))
            xmax = float(max(x0, x1))
        except (TypeError, ValueError):
            return
        if not np.isfinite(xmin) or not np.isfinite(xmax):
            return
        pad = (xmax - xmin) * 0.001

        def in_range(x):
            return (xmin - pad) <= x <= (xmax + pad)

        for idx, line in enumerate(locality_lines):
            try:
                x_line = float(np.array(line.get_xdata()).ravel()[0])
            except Exception:
                x_line = None
            visible = in_range(x_line) if x_line is not None else False
            line.set_visible(visible)
            if idx < len(locality_texts):
                locality_texts[idx].set_visible(visible)
        for weather_text, x_pos in zip(locality_weather_texts, locality_weather_positions):
            visible = in_range(x_pos) if x_pos is not None else False
            weather_text.set_visible(visible)

        battery_visible = bool(line_battery is not None and line_battery.get_visible())
        for marker in mode_markers:
            try:
                x_marker = float(np.array(marker.get_xdata()).ravel()[0])
            except Exception:
                x_marker = None
            marker.set_visible(battery_visible and x_marker is not None and in_range(x_marker))
        for text in mode_texts:
            try:
                x_text = float(text.get_position()[0])
            except Exception:
                x_text = None
            text.set_visible(battery_visible and x_text is not None and in_range(x_text))

    update_range_visibility(x0_full, x1_full)

    def sync_top_axis(_=None):
        x0, x1 = ax1.get_xlim()
        update_smoothing_for_xlim(x0, x1)
        ax_top.set_xlim((x0, x1))
        set_top_axis_ticks(x0, x1)
        update_distance_tick_texts(x0, x1)
        update_range_visibility(x0, x1)

    ax1.callbacks.connect("xlim_changed", sync_top_axis)

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
    hover_dot_cadence = None
    if line_cadence is not None:
        hover_dot_cadence, = ax1.plot([], [], marker="o", markersize=4, linestyle="None",
                                      color=line_cadence.get_color(), zorder=6, visible=False)
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
        "cadence": TextArea("", textprops=dict(color=(line_cadence.get_color() if line_cadence else "#9ca3af"), fontsize=9)),
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
            tooltip_rows["cadence"],
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
    if line_cadence is not None:
        artist_groups[line_cadence.get_label()] = [line_cadence]
    if mean_line_cadence is not None:
        artist_groups[mean_line_cadence.get_label()] = [mean_line_cadence]

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
    all_artists.update(locality_lines)
    all_artists.update(locality_texts)
    all_artists.update(locality_weather_texts)
    all_artists.update(car_event_markers)

    base_alpha = {artist: (artist.get_alpha() if artist.get_alpha() is not None else 1.0) for artist in all_artists}
    dim_factor = 0.2
    active_target = None
    locality_artists = set(locality_lines) | set(locality_texts) | set(locality_weather_texts)

    pill_inactive_face = "#ffffff"
    pill_inactive_edge = "#e5e7eb"
    pill_inactive_text = "#9ca3af"

    def set_pill_active(pill_id, active, has_group=True):
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
            value_text.set_text(ui.get("active_value_text", value_text.get_text()))
            value_text.set_visible(True)
        else:
            patch.set_facecolor(pill_inactive_face)
            patch.set_edgecolor(pill_inactive_edge)
            label_text.set_color(pill_inactive_text)
            value_text.set_color(pill_inactive_text)
            hide_value = bool(has_group) and ui.get("hide_value_when_inactive", False)
            value_text.set_visible(not hide_value)

    def refresh_pills():
        for pill_id, ui in pill_ui.items():
            key = ui.get("target_key")
            group = artist_groups.get(key) or []
            is_active = bool(group) and group[0].get_visible()
            set_pill_active(pill_id, is_active, has_group=bool(group))

    refresh_pills()

    def current_pill_state():
        state = {}
        for pill_id, target_artists in pill_targets.items():
            if not target_artists:
                continue
            state[pill_id] = bool(target_artists[0].get_visible())
        return state

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
        if hover_dot_cadence is not None:
            hover_dot_cadence.set_visible(False)
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

        rider0 = fmt_or_na(smooth["power"][idx], lambda v: f"{int(round(v))} W")
        motor0 = fmt_or_na(smooth["motor"][idx], lambda v: f"{int(round(v))} W")
        battery0 = fmt_or_na(smooth_battery[idx], lambda v: f"{int(round(v))}%")
        hr0 = fmt_or_na(smooth["heartrate"][idx], lambda v: f"{int(round(v))} bpm")
        cadence0 = fmt_or_na(smooth["cadence"][idx], lambda v: f"{int(round(v))} rpm")

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
        if line_cadence is not None:
            set_metric_row("cadence", t("dashboard.card.cadence"), cadence0, line_cadence.get_color(), line_cadence.get_visible())
        else:
            tooltip_rows["cadence"].set_text(f'{t("dashboard.card.cadence")}: {t("dashboard.value.na")}')
            tooltip_rows["cadence"]._text.set_color("#9ca3af")

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

        set_dot(hover_dot_rider, smooth["power"][idx], line_cyclist.get_visible())
        set_dot(hover_dot_motor, smooth["motor"][idx], line_motor.get_visible())
        if hover_dot_battery is not None and line_battery is not None:
            set_dot(hover_dot_battery, smooth_battery[idx], line_battery.get_visible())
        if hover_dot_heartrate is not None and line_heartrate is not None:
            set_dot(hover_dot_heartrate, smooth["heartrate"][idx], line_heartrate.get_visible())
        if hover_dot_cadence is not None and line_cadence is not None:
            set_dot(hover_dot_cadence, smooth["cadence"][idx], line_cadence.get_visible())
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
            if artist in locality_artists:
                artist.set_alpha(base)
            else:
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
            save_pill_state(current_pill_state())
            update_range_visibility(*ax1.get_xlim())
            fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes not in (ax1, ax2, ax_top):
            return
        if event.xdata is None:
            return

        step = 0
        if hasattr(event, "step") and event.step is not None:
            step = int(event.step)
        elif hasattr(event, "button"):
            if event.button == "up":
                step = 1
            elif event.button == "down":
                step = -1
        if step == 0:
            return

        try:
            x = float(event.xdata)
            xmin, xmax = ax1.get_xlim()
            xmin = float(xmin)
            xmax = float(xmax)
        except (TypeError, ValueError):
            return
        if not np.isfinite(x) or not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            return

        zoom_base = 0.85
        scale = zoom_base ** step
        left = x - xmin
        right = xmax - x
        new_xmin = x - left * scale
        new_xmax = x + right * scale

        global_xmin = float(t_min[0]) if len(t_min) else 0.0
        global_xmax = float(t_min[-1]) if len(t_min) else new_xmax
        total_span = global_xmax - global_xmin
        min_span = 0.5

        if total_span > 0 and (new_xmax - new_xmin) > total_span:
            new_xmin, new_xmax = global_xmin, global_xmax
        elif (new_xmax - new_xmin) < min_span:
            return
        else:
            if new_xmin < global_xmin:
                shift = global_xmin - new_xmin
                new_xmin += shift
                new_xmax += shift
            if new_xmax > global_xmax:
                shift = new_xmax - global_xmax
                new_xmin -= shift
                new_xmax -= shift
            new_xmin = max(global_xmin, new_xmin)
            new_xmax = min(global_xmax, new_xmax)

        ax1.set_xlim(new_xmin, new_xmax)

        slider = getattr(fig, "_time_slider", None)
        if slider is not None:
            try:
                slider.set_val((new_xmin, new_xmax))
            except Exception:
                pass

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    # Margins: leave some space top/bottom
    # Leave enough space for the header/pills so they don't overlap the top X axis.
    plot_top = max(0.1, pill_y - 0.09 - axis_block_shift_frac)
    plot_bottom = max(0.06, 0.18 - content_shift_frac - axis_block_shift_frac)
    fig.subplots_adjust(left=0.06, right=0.94, top=plot_top, bottom=plot_bottom)

    # --- bottom range slider for time zoom ---
    if x_end > 0:
        slider_left = 0.06
        slider_right = 0.94
        slider_bottom = max(0.01, 0.04 - content_shift_frac - axis_block_shift_frac)
        slider_height = 0.05
        ax_slider = fig.add_axes([slider_left, slider_bottom, slider_right - slider_left, slider_height])
        ax_slider.set_facecolor("#ffffff")
        ax_slider.set_yticks([])
        ax_slider.xaxis.set_ticks_position("top")

        if x_end <= 60:
            slider_step = 1.0
        elif x_end <= 180:
            slider_step = 5.0
        else:
            slider_step = 10.0
        slider_ticks = np.arange(0.0, x_end + 1e-9, slider_step, dtype=float)
        if len(slider_ticks) == 0 or abs(slider_ticks[-1] - x_end) > 1e-6:
            slider_ticks = np.append(slider_ticks, x_end)
        ax_slider.set_xticks(slider_ticks)
        ax_slider.tick_params(axis="x", colors=axis_color, labelsize=7, length=2, width=0.8, pad=2)
        for spine in ax_slider.spines.values():
            spine.set_color(spine_color)
            spine.set_linewidth(1.0)

        time_slider = RangeSlider(ax_slider, "", 0.0, x_end, valinit=(0.0, x_end))
        time_slider.label.set_visible(False)
        time_slider.valtext.set_visible(False)
        time_slider.track.set_facecolor("#f1f5f9")
        time_slider.poly.set_facecolor("#cbd5e1")
        time_slider.poly.set_alpha(0.75)
        try:
            for handle in getattr(time_slider, "_handles", []) or []:
                handle.set_color("#64748b")
                handle.set_linewidth(6.0)
        except Exception:
            pass

        def on_time_slider(val):
            try:
                x0, x1 = float(val[0]), float(val[1])
            except Exception:
                return
            if not np.isfinite(x0) or not np.isfinite(x1):
                return
            if x1 - x0 <= 1e-6:
                return
            ax1.set_xlim(x0, x1)
            fig.canvas.draw_idle()

        time_slider.on_changed(on_time_slider)

        slider_drag_state = {"active": False}

        def on_slider_press(event):
            if event.inaxes == ax_slider:
                slider_drag_state["active"] = True

        def on_slider_release(event):
            if not slider_drag_state["active"]:
                return
            slider_drag_state["active"] = False
            x0, x1 = ax1.get_xlim()
            update_time_ticks(x0, x1)
            sync_top_axis()
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", on_slider_press)
        fig.canvas.mpl_connect("button_release_event", on_slider_release)
        fig._time_slider = time_slider
        fig._time_slider_ax = ax_slider

    # IMPORTANT: do not call fig.tight_layout() here

    return fig
