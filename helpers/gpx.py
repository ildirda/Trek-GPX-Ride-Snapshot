# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np

from helpers.errors import LocalizedError

def parse_gpx(path):
    """
    Read a GPX file and return:
    times, power, motor_power, battery, heartrate, cadence, ebike_mode, elevation, lat, lon
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
    cadences = []
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

        # Cadence (optional)
        cadence_elem = trkpt.find(".//{*}cadence")
        if cadence_elem is None:
            cadence_elem = trkpt.find(".//{*}cad")
        if cadence_elem is not None and cadence_elem.text:
            try:
                cadence = float(cadence_elem.text)
            except (TypeError, ValueError):
                cadence = np.nan
        else:
            cadence = np.nan

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
        cadences.append(cadence)
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
        np.array(cadences),
        np.array(ebike_modes),
        np.array(elevs),
        np.array(lats),
        np.array(lons),
    )

def compute_moving_binned_data(times, powers, motor_powers, batteries, heartrates, cadences, ebike_modes, elevs, lats, lons):
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
    raw_times_ts = np.array(
        [t.timestamp() if isinstance(t, datetime) else float("nan") for t in times],
        dtype=float,
    )

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
    cadence_kept = cadences[keep]
    mode_kept = ebike_modes[keep]
    elev_kept = elevs[keep]
    dist_kept = cumdist_km[keep]
    lat_kept = lats[keep]
    lon_kept = lons[keep]
    times_kept = times[keep]

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
    binned_cadence = []
    binned_mode = []
    binned_ele = []
    binned_dist = []
    binned_lat = []
    binned_lon = []
    binned_dt = []

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
        # Cadence: mean of valid (non-NaN) values only
        cadence_mask = ~np.isnan(cadence_kept[mask])
        if np.any(cadence_mask):
            binned_cadence.append(np.nanmean(cadence_kept[mask]))
        else:
            binned_cadence.append(np.nan)
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
        binned_lat.append(lat_kept[mask].mean())
        binned_lon.append(lon_kept[mask].mean())
        dt_values = times_kept[mask]
        dt_list = []
        for dt_value in dt_values:
            if isinstance(dt_value, datetime):
                dt_list.append(dt_value.timestamp())
        if dt_list:
            avg_ts = float(np.mean(dt_list))
            binned_dt.append(datetime.fromtimestamp(avg_ts, tz=timezone.utc))
        else:
            binned_dt.append(None)

    binned_t = np.array(binned_t)
    binned_power = np.array(binned_power)
    binned_motor = np.array(binned_motor)
    binned_battery = np.array(binned_battery)
    binned_heartrate = np.array(binned_heartrate)
    binned_cadence = np.array(binned_cadence)
    binned_mode = np.array(binned_mode)
    binned_ele = np.array(binned_ele)
    binned_dist = np.array(binned_dist)
    binned_lat = np.array(binned_lat)
    binned_lon = np.array(binned_lon)
    if len(binned_t) > 1:
        speed_km_per_s = np.gradient(binned_dist, binned_t)
        binned_speed_kmh = speed_km_per_s * 3600.0
    elif len(binned_t) == 1:
        binned_speed_kmh = np.array([0.0])
    else:
        binned_speed_kmh = np.array([])

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

    # Smoothed cadence (if valid data)
    if not np.all(np.isnan(binned_cadence)):
        valid_mask = ~np.isnan(binned_cadence)
        if np.any(valid_mask):
            smooth_cadence = np.copy(binned_cadence)
            if np.sum(valid_mask) > 1:
                valid_indices = np.where(valid_mask)[0]
                smooth_cadence_interp = np.interp(
                    np.arange(len(binned_cadence)), valid_indices, binned_cadence[valid_indices]
                )
                smooth_cadence = double_smooth(smooth_cadence_interp)
            else:
                smooth_cadence = binned_cadence
        else:
            smooth_cadence = binned_cadence
    else:
        smooth_cadence = binned_cadence

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
    mean_cadence = (
        float(np.nanmean(cadence_kept))
        if len(cadence_kept) and not np.all(np.isnan(cadence_kept))
        else float("nan")
    )
    if len(battery_kept) and not np.all(np.isnan(battery_kept)):
        valid_battery = battery_kept[~np.isnan(battery_kept)]
        first_battery = float(valid_battery[0]) if len(valid_battery) else float("nan")
        last_battery = float(valid_battery[-1]) if len(valid_battery) else float("nan")
    else:
        first_battery = float("nan")
        last_battery = float("nan")
    if len(elev_kept):
        elev_diff = np.diff(elev_kept.astype(float), prepend=elev_kept[0].astype(float))
        elevation_gain_m = float(np.nansum(np.clip(elev_diff, 0.0, None)))
    else:
        elevation_gain_m = float("nan")

    return {
        "binned_t": binned_t,
        "smooth_t_min": binned_t / 60.0,
        "binned_power": binned_power,
        "binned_motor": binned_motor,
        "binned_battery": binned_battery,
        "binned_heartrate": binned_heartrate,
        "binned_cadence": binned_cadence,
        "smooth_power": smooth_power,
        "smooth_motor": smooth_motor,
        "smooth_battery": smooth_battery,
        "smooth_heartrate": smooth_heartrate,
        "smooth_cadence": smooth_cadence,
        "smooth_ele": smooth_ele,
        "binned_ele": binned_ele,
        "binned_dist": binned_dist,
        "binned_speed_kmh": binned_speed_kmh,
        "raw_times_ts": raw_times_ts,
        "raw_speed_kmh": speed_kmh,
        "binned_lat": binned_lat,
        "binned_lon": binned_lon,
        "binned_dt": binned_dt,
        "mean_power": mean_power,
        "mean_motor": mean_motor,
        "mode_changes": mode_changes,
        "total_distance_km": total_distance_km,
        "total_time_min": total_time_min,
        "max_rider_power": max_rider_power,
        "max_motor_power": max_motor_power,
        "mean_heartrate": mean_heartrate,
        "mean_cadence": mean_cadence,
        "first_battery": first_battery,
        "last_battery": last_battery,
        "elevation_gain_m": elevation_gain_m,
    }
