# -*- coding: utf-8 -*-

import math
import os
from datetime import datetime, timezone
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from helpers.car_events import load_car_events, map_events_to_active_minutes, map_intervals_to_active_minutes, show_car_events_popup
from helpers.clipboard_utils import copy_figure_to_windows_clipboard
from helpers.errors import LocalizedError
from helpers.geocode import identify_localities_online, reverse_geocode_locality
from helpers.gpx import compute_moving_binned_data, parse_gpx
from helpers.i18n import discover_languages, load_last_language_selection, load_translations, save_last_language_selection
from helpers.plotting import create_figure
from helpers.state import load_pill_state
from helpers.utils import debug_log, format_local_datetime
from helpers.weather import fetch_openweather_snapshot, format_weather_brief, get_weather_brief_data, get_weather_summary

def _project_root():
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(module_dir)

def compute_bearing_deg(lat1, lon1, lat2, lon2):
    try:
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (lat1, lon1, lat2, lon2)):
        return None
    if lat1 == lat2 and lon1 == lon2:
        return None
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    bearing = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
    return bearing


def heading_at_time(binned_t, binned_lat, binned_lon, target_time):
    if binned_t is None or binned_lat is None or binned_lon is None:
        return None
    if len(binned_t) < 2 or len(binned_lat) < 2 or len(binned_lon) < 2:
        return None
    try:
        target_time = float(target_time)
    except (TypeError, ValueError):
        return None
    idx = int(np.searchsorted(binned_t, target_time))
    n = len(binned_t)
    pairs = []
    if idx <= 0:
        pairs.append((0, 1))
    elif idx >= n:
        pairs.append((n - 2, n - 1))
    else:
        pairs.append((idx - 1, idx))
    if idx + 1 < n:
        pairs.append((idx, idx + 1))
    if idx - 2 >= 0:
        pairs.append((idx - 2, idx - 1))
    for i0, i1 in pairs:
        bearing = compute_bearing_deg(binned_lat[i0], binned_lon[i0], binned_lat[i1], binned_lon[i1])
        if bearing is not None:
            return bearing
    return None


def angular_diff_deg(a, b):
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)


def describe_wind_relative(wind_deg, heading_deg, t):
    try:
        wind_deg = float(wind_deg)
        heading_deg = float(heading_deg)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(wind_deg) or not math.isfinite(heading_deg):
        return None
    wind_deg = wind_deg % 360.0
    heading_deg = heading_deg % 360.0
    threshold_deg = 45.0
    if angular_diff_deg(wind_deg, heading_deg) <= threshold_deg:
        return t("wind.relative.headwind")
    if angular_diff_deg(wind_deg, (heading_deg + 180.0) % 360.0) <= threshold_deg:
        return t("wind.relative.tailwind")
    return t("wind.relative.crosswind")


def _mean_number(values):
    items = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        items.append(value)
    if not items:
        return None
    return float(sum(items) / len(items))


def _mean_angle_deg(values):
    sin_sum = 0.0
    cos_sum = 0.0
    count = 0
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        rad = math.radians(value)
        sin_sum += math.sin(rad)
        cos_sum += math.cos(rad)
        count += 1
    if count == 0:
        return None
    if abs(sin_sum) < 1e-8 and abs(cos_sum) < 1e-8:
        return None
    return (math.degrees(math.atan2(sin_sum, cos_sum)) + 360.0) % 360.0


def _dt_at_index(binned_dt, idx):
    if binned_dt is None:
        return None
    if idx < 0 or idx >= len(binned_dt):
        return None
    dt_value = binned_dt[idx]
    return dt_value if isinstance(dt_value, datetime) else None


def _interpolate_dt(dt0, dt1, frac):
    if dt0 is None and dt1 is None:
        return None
    if dt0 is None:
        return dt1
    if dt1 is None:
        return dt0
    try:
        ts0 = float(dt0.timestamp())
        ts1 = float(dt1.timestamp())
    except Exception:
        return dt0
    ts = ts0 + (ts1 - ts0) * frac
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _sample_track_at_distance(binned_dist, binned_lat, binned_lon, binned_dt, target_dist):
    if binned_dist is None or binned_lat is None or binned_lon is None:
        return None, None, None
    if len(binned_dist) == 0:
        return None, None, None
    try:
        target = float(target_dist)
    except (TypeError, ValueError):
        return None, None, None
    if not math.isfinite(target):
        return None, None, None

    idx = int(np.searchsorted(binned_dist, target))
    if idx <= 0:
        i0 = i1 = 0
    elif idx >= len(binned_dist):
        i0 = i1 = len(binned_dist) - 1
    else:
        i0 = idx - 1
        i1 = idx

    try:
        d0 = float(binned_dist[i0])
        d1 = float(binned_dist[i1])
    except (TypeError, ValueError):
        return None, None, None
    if not math.isfinite(d0) or not math.isfinite(d1):
        return None, None, None

    if i0 == i1 or d0 == d1:
        return binned_lat[i1], binned_lon[i1], _dt_at_index(binned_dt, i1)

    frac = (target - d0) / (d1 - d0)
    try:
        lat0 = float(binned_lat[i0])
        lat1 = float(binned_lat[i1])
        lon0 = float(binned_lon[i0])
        lon1 = float(binned_lon[i1])
    except (TypeError, ValueError):
        lat = None
        lon = None
    else:
        if math.isfinite(lat0) and math.isfinite(lat1):
            lat = lat0 + (lat1 - lat0) * frac
        else:
            lat = None
        if math.isfinite(lon0) and math.isfinite(lon1):
            lon = lon0 + (lon1 - lon0) * frac
        else:
            lon = None
    dt0 = _dt_at_index(binned_dt, i0)
    dt1 = _dt_at_index(binned_dt, i1)
    return lat, lon, _interpolate_dt(dt0, dt1, frac)


def _weather_speed_kmh(weather):
    if not isinstance(weather, dict):
        return None
    speed = weather.get("wind_kmh")
    if speed is None:
        speed = weather.get("wind_ms")
        if isinstance(speed, (int, float)) and math.isfinite(speed):
            return float(speed) * 3.6
    if isinstance(speed, (int, float)) and math.isfinite(speed):
        return float(speed)
    return None


def _aggregate_weather_samples(samples):
    if not samples:
        return None
    temps = []
    wind_dirs = []
    wind_speeds = []
    for weather in samples:
        if not isinstance(weather, dict):
            continue
        temp = weather.get("temp_c")
        if isinstance(temp, (int, float)) and math.isfinite(temp):
            temps.append(temp)
        wind_deg = weather.get("wind_deg")
        if isinstance(wind_deg, (int, float)) and math.isfinite(wind_deg):
            wind_dirs.append(wind_deg)
        speed = _weather_speed_kmh(weather)
        if speed is not None:
            wind_speeds.append(speed)
    if not temps and not wind_dirs and not wind_speeds:
        return None
    aggregate = {}
    temp_mean = _mean_number(temps)
    if temp_mean is not None:
        aggregate["temp_c"] = temp_mean
    speed_mean = _mean_number(wind_speeds)
    if speed_mean is not None:
        aggregate["wind_kmh"] = speed_mean
    wind_mean = _mean_angle_deg(wind_dirs)
    if wind_mean is not None:
        aggregate["wind_deg"] = wind_mean
    return aggregate if aggregate else None


def _segment_weather_samples(
    start_dist,
    end_dist,
    binned_dist,
    binned_lat,
    binned_lon,
    binned_dt,
    lang="ca",
):
    try:
        start = float(start_dist)
        end = float(end_dist)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end):
        return None
    if end <= start:
        return None

    samples = []
    for frac in (1.0 / 3.0, 2.0 / 3.0):
        target = start + (end - start) * frac
        lat, lon, when_dt = _sample_track_at_distance(binned_dist, binned_lat, binned_lon, binned_dt, target)
        if lat is None or lon is None or when_dt is None:
            continue
        weather = fetch_openweather_snapshot(lat, lon, when_dt, lang=lang, allow_current_fallback=False)
        if weather:
            samples.append(weather)
    return _aggregate_weather_samples(samples)

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
        self.geometry("1500x600")

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
        self.loading_window = None
        self.loading_label = None

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

    def show_loading(self, message):
        if self.loading_window is not None:
            if self.loading_label is not None:
                self.loading_label.config(text=message)
            self.update_idletasks()
            return
        win = tk.Toplevel(self)
        win.title(self.t("message.loading.title"))
        win.resizable(False, False)
        win.transient(self)
        win.configure(padx=20, pady=12)
        label = tk.Label(win, text=message)
        label.pack()
        win.update_idletasks()
        try:
            x = self.winfo_rootx() + (self.winfo_width() // 2) - (win.winfo_reqwidth() // 2)
            y = self.winfo_rooty() + (self.winfo_height() // 2) - (win.winfo_reqheight() // 2)
            win.geometry(f"+{x}+{y}")
        except Exception:
            pass
        self.loading_window = win
        self.loading_label = label
        self.update_idletasks()
        self.update()

    def hide_loading(self):
        if self.loading_window is None:
            return
        try:
            self.loading_window.destroy()
        except Exception:
            pass
        self.loading_window = None
        self.loading_label = None

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
            self.show_loading(self.t("message.loading.body"))
            (
                times,
                powers,
                motor_powers,
                batteries,
                heartrates,
                cadences,
                ebike_modes,
                elevs,
                lats,
                lons,
            ) = parse_gpx(path)
            debug_log("Fitxer carregat")
            data = compute_moving_binned_data(
                times, powers, motor_powers, batteries, heartrates, cadences, ebike_modes, elevs, lats, lons
            )
            car_events = load_car_events(path)
            if car_events:
                min_speed_kmh = 5.0
                car_event_times_min = map_events_to_active_minutes(
                    car_events.get("timestamps"),
                    data.get("binned_dt"),
                    data.get("binned_t"),
                    data.get("raw_times_ts"),
                    data.get("raw_speed_kmh"),
                    min_speed_kmh=min_speed_kmh,
                )
                data["car_event_times_min"] = car_event_times_min
                data["car_event_spans_min"] = map_intervals_to_active_minutes(
                    car_events.get("intervals"),
                    data.get("binned_dt"),
                    data.get("binned_t"),
                    data.get("raw_times_ts"),
                    data.get("raw_speed_kmh"),
                    min_speed_kmh=min_speed_kmh,
                )
                data["car_events_count"] = len(car_event_times_min)
                car_events["count"] = len(car_event_times_min)
            else:
                car_events = None
            weather_summary = None
            if len(times) and len(lats) and len(lons):
                start_lat = lats[0]
                start_lon = lons[0]
                start_dt = times[0]
                place_label = None
                try:
                    place_label = reverse_geocode_locality(start_lat, start_lon, lang=self.lang)
                except Exception:
                    place_label = None
                if not place_label:
                    place_label = f"{float(start_lat):.5f}, {float(start_lon):.5f}"
                dt_label = format_local_datetime(start_dt) or str(start_dt)
                debug_log(f"Buscant temps a {place_label} el {dt_label}")
                weather_summary = get_weather_summary(
                    start_lat,
                    start_lon,
                    start_dt,
                    lang=self.lang,
                    logger=debug_log,
                )
                if weather_summary:
                    debug_log(f"Dades del temps: {weather_summary}")
            data["pill_state"] = load_pill_state()
            data["locality_stays"] = identify_localities_online(
                data.get("binned_t"),
                data.get("binned_dist"),
                data.get("binned_lat"),
                data.get("binned_lon"),
                data.get("binned_dt"),
                lang="ca",
            )
            locality_stays = data.get("locality_stays", [])
            locality_weather_segments = []
            binned_t = data.get("binned_t")
            binned_dist = data.get("binned_dist")
            binned_lat = data.get("binned_lat")
            binned_lon = data.get("binned_lon")
            if (
                locality_stays
                and binned_t is not None
                and binned_dist is not None
                and binned_lat is not None
                and binned_lon is not None
                and len(binned_t)
                and len(binned_dist)
                and len(binned_lat)
                and len(binned_lon)
            ):
                first_stay = locality_stays[0]
                try:
                    start_time = float(binned_t[0])
                    start_dist = float(binned_dist[0])
                    start_lat = float(binned_lat[0])
                    start_lon = float(binned_lon[0])
                except (TypeError, ValueError):
                    start_time = None
                    start_dist = None
                    start_lat = None
                    start_lon = None
                segment_weather = _segment_weather_samples(
                    start_dist,
                    first_stay.get("distance_km"),
                    binned_dist,
                    binned_lat,
                    binned_lon,
                    data.get("binned_dt"),
                    lang=self.lang,
                )
                if segment_weather:
                    weather_brief = format_weather_brief(segment_weather)
                    if weather_brief:
                        heading_deg = compute_bearing_deg(
                            start_lat,
                            start_lon,
                            first_stay.get("lat"),
                            first_stay.get("lon"),
                        )
                        wind_label = None
                        if heading_deg is not None:
                            wind_label = describe_wind_relative(
                                segment_weather.get("wind_deg"),
                                heading_deg,
                                self.t,
                            )
                        if wind_label:
                            weather_brief = f"{weather_brief} ({wind_label})"
                        if start_time is not None and first_stay.get("time") is not None:
                            segment_time = (start_time + float(first_stay["time"])) / 2.0
                            locality_weather_segments.append({"time": segment_time, "label": weather_brief})

            for idx, stay in enumerate(locality_stays):
                lat = stay.get("lat")
                lon = stay.get("lon")
                when_dt = stay.get("datetime")
                if lat is None or lon is None or when_dt is None:
                    continue
                weather_brief = None
                wind_deg = None
                heading_deg = None
                prev_stay = locality_stays[idx - 1] if idx > 0 else None
                if prev_stay:
                    heading_deg = compute_bearing_deg(
                        prev_stay.get("lat"),
                        prev_stay.get("lon"),
                        lat,
                        lon,
                    )
                    segment_weather = _segment_weather_samples(
                        prev_stay.get("distance_km"),
                        stay.get("distance_km"),
                        data.get("binned_dist"),
                        data.get("binned_lat"),
                        data.get("binned_lon"),
                        data.get("binned_dt"),
                        lang=self.lang,
                    )
                    if segment_weather:
                        weather_brief = format_weather_brief(segment_weather)
                        wind_deg = segment_weather.get("wind_deg")

                if weather_brief is None:
                    weather_brief, wind_deg = get_weather_brief_data(lat, lon, when_dt, lang=self.lang)
                if weather_brief:
                    if heading_deg is None:
                        heading_deg = heading_at_time(
                            data.get("binned_t"),
                            data.get("binned_lat"),
                            data.get("binned_lon"),
                            stay.get("time"),
                        )
                    wind_label = None
                    if wind_deg is not None and heading_deg is not None:
                        wind_label = describe_wind_relative(wind_deg, heading_deg, self.t)
                    if wind_label:
                        weather_brief = f"{weather_brief} ({wind_label})"
                    stay["weather_brief"] = weather_brief
                    if prev_stay and prev_stay.get("time") is not None and stay.get("time") is not None:
                        segment_time = (float(prev_stay["time"]) + float(stay["time"])) / 2.0
                        locality_weather_segments.append({"time": segment_time, "label": weather_brief})

            if locality_stays:
                last_stay = locality_stays[-1]
                if (
                    binned_t is not None
                    and binned_dist is not None
                    and binned_lat is not None
                    and binned_lon is not None
                    and len(binned_t)
                    and len(binned_dist)
                    and len(binned_lat)
                    and len(binned_lon)
                ):
                    end_time = float(binned_t[-1])
                    end_dist = float(binned_dist[-1])
                    end_lat = float(binned_lat[-1])
                    end_lon = float(binned_lon[-1])
                    segment_weather = _segment_weather_samples(
                        last_stay.get("distance_km"),
                        end_dist,
                        binned_dist,
                        binned_lat,
                        binned_lon,
                        data.get("binned_dt"),
                        lang=self.lang,
                    )
                    if segment_weather:
                        weather_brief = format_weather_brief(segment_weather)
                        if weather_brief:
                            heading_deg = compute_bearing_deg(
                                last_stay.get("lat"),
                                last_stay.get("lon"),
                                end_lat,
                                end_lon,
                            )
                            wind_label = None
                            if heading_deg is not None:
                                wind_label = describe_wind_relative(
                                    segment_weather.get("wind_deg"),
                                    heading_deg,
                                    self.t,
                                )
                            if wind_label:
                                weather_brief = f"{weather_brief} ({wind_label})"
                            if last_stay.get("time") is not None:
                                segment_time = (float(last_stay["time"]) + end_time) / 2.0
                                locality_weather_segments.append({"time": segment_time, "label": weather_brief})

            data["locality_weather_segments"] = locality_weather_segments
            locality_names = {stay.get("name") for stay in data.get("locality_stays", []) if stay.get("name")}
            data["towns_visited"] = len(locality_names)
            data["weather_summary"] = weather_summary
        except Exception as e:
            self.hide_loading()
            messagebox.showerror(self.t("message.error.title"), self.format_error(e, "message.process.error"))
            return
        finally:
            self.hide_loading()

        self.last_data = data
        self.render_figure(data)
        # Car events remain in the chart, but no popup is shown.

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
        script_dir = _project_root()
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
