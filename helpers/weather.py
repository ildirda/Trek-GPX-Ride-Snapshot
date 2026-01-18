# -*- coding: utf-8 -*-

import json
import os
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone

import numpy as np

_WEATHER_API_ENV_VARS = ("OPENWEATHER_API_KEY", "OPENWEATHERMAP_API_KEY")
_WEATHER_API_KEY_FILE = "openweather_api_key.json"

def _project_root():
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(module_dir)

def load_openweather_api_key():
    for env_name in _WEATHER_API_ENV_VARS:
        value = os.environ.get(env_name)
        if value:
            return value.strip()

    root_dir = _project_root()
    path = os.path.join(root_dir, _WEATHER_API_KEY_FILE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = f.read().strip()
        return value or None
    except FileNotFoundError:
        return None
    except Exception:
        return None

def fetch_json(url, timeout=6.0, logger=None, label=None):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "TrekRideSnapshot/1.0 (local-use)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        body = None
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = None
        if logger:
            prefix = f"{label}: " if label else ""
            if body:
                logger(f"{prefix}HTTP {exc.code} {exc.reason}: {body}")
            else:
                logger(f"{prefix}HTTP {exc.code} {exc.reason}")
    except urllib.error.URLError as exc:
        if logger:
            prefix = f"{label}: " if label else ""
            logger(f"{prefix}Error de xarxa: {exc.reason}")
    except Exception as exc:
        if logger:
            prefix = f"{label}: " if label else ""
            logger(f"{prefix}Error inesperat: {exc}")
        return None

def summarize_payload(payload, max_len=700):
    try:
        text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        text = str(payload)
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text

def redact_url_query_param(url, param_name):
    try:
        parts = urllib.parse.urlsplit(url)
        query = urllib.parse.parse_qs(parts.query, keep_blank_values=True)
        if param_name in query:
            query[param_name] = ["***"]
        redacted_query = urllib.parse.urlencode(query, doseq=True)
        return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, redacted_query, parts.fragment))
    except Exception:
        return url

def extract_openweather_entry(payload):
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if isinstance(data, list) and data:
        return data[0]
    current = payload.get("current")
    if isinstance(current, dict):
        return current
    hourly = payload.get("hourly")
    if isinstance(hourly, list) and hourly:
        return hourly[0]
    if "weather" in payload and ("main" in payload or "temp" in payload):
        return payload
    return None

def parse_openweather_entry(entry):
    if not isinstance(entry, dict):
        return None
    description = None
    weather_items = entry.get("weather")
    if isinstance(weather_items, list) and weather_items:
        first = weather_items[0]
        if isinstance(first, dict):
            description = first.get("description") or first.get("main")

    temp = None
    if isinstance(entry.get("temp"), (int, float)):
        temp = float(entry["temp"])
    elif isinstance(entry.get("main"), dict):
        temp_value = entry["main"].get("temp")
        if isinstance(temp_value, (int, float)):
            temp = float(temp_value)

    wind_speed = None
    wind_deg = None
    if isinstance(entry.get("wind_speed"), (int, float)):
        wind_speed = float(entry["wind_speed"])
    if isinstance(entry.get("wind_deg"), (int, float)):
        wind_deg = float(entry["wind_deg"])
    wind = entry.get("wind")
    if isinstance(wind, dict):
        if wind_speed is None:
            speed = wind.get("speed")
            if isinstance(speed, (int, float)):
                wind_speed = float(speed)
        if wind_deg is None:
            deg = wind.get("deg")
            if isinstance(deg, (int, float)):
                wind_deg = float(deg)

    if description is None and temp is None and wind_speed is None:
        return None

    wind_kmh = wind_speed * 3.6 if wind_speed is not None else None
    return {
        "description": description,
        "temp_c": temp,
        "wind_ms": wind_speed,
        "wind_kmh": wind_kmh,
        "wind_deg": wind_deg,
    }

def fetch_openweather_snapshot(
    lat,
    lon,
    when_dt,
    lang="ca",
    units="metric",
    timeout=6.0,
    allow_current_fallback=False,
    logger=None,
):
    api_key = load_openweather_api_key()
    if not api_key:
        if logger:
            logger("API key d'OpenWeather no trobada")
        return None

    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return None

    if not isinstance(when_dt, datetime):
        return None
    if when_dt.tzinfo is None:
        when_dt = when_dt.replace(tzinfo=timezone.utc)
    timestamp = int(when_dt.timestamp())

    params = {
        "lat": f"{lat_f:.5f}",
        "lon": f"{lon_f:.5f}",
        "dt": timestamp,
        "appid": api_key,
        "units": units,
        "lang": lang,
    }
    url = "https://api.openweathermap.org/data/3.0/onecall/timemachine?" + urllib.parse.urlencode(params)
    if logger:
        redacted_url = redact_url_query_param(url, "appid")
        logger(f"URL OpenWeather (timemachine): {redacted_url}")
        logger("Connectant a OpenWeather (timemachine)")
    payload = fetch_json(url, timeout=timeout, logger=logger, label="OpenWeather timemachine")
    if logger:
        if payload is None:
            logger("Sense resposta d'OpenWeather (timemachine)")
        elif isinstance(payload, dict):
            keys = ", ".join(sorted(payload.keys()))
            logger(f"Payload OpenWeather (timemachine) claus: {keys}")
            logger(f"Payload OpenWeather (timemachine): {summarize_payload(payload)}")
        else:
            logger(f"Payload OpenWeather (timemachine) tipus: {type(payload).__name__}")
    entry = extract_openweather_entry(payload)
    weather = parse_openweather_entry(entry) if entry else None
    if logger:
        if entry and not weather:
            logger(f"Entrada OpenWeather no parsejada: {entry}")
        elif weather:
            logger(f"Dades parsejades: {weather}")
    if logger and weather:
        logger("Dades rebudes d'OpenWeather (timemachine)")
    if weather or not allow_current_fallback:
        if logger and not weather:
            logger("Resposta buida d'OpenWeather (timemachine)")
        return weather

    params = {
        "lat": f"{lat_f:.5f}",
        "lon": f"{lon_f:.5f}",
        "appid": api_key,
        "units": units,
        "lang": lang,
    }
    url = "https://api.openweathermap.org/data/2.5/weather?" + urllib.parse.urlencode(params)
    if logger:
        redacted_url = redact_url_query_param(url, "appid")
        logger(f"URL OpenWeather (actual): {redacted_url}")
        logger("Intentant OpenWeather actual")
    payload = fetch_json(url, timeout=timeout, logger=logger, label="OpenWeather actual")
    if logger:
        if payload is None:
            logger("Sense resposta d'OpenWeather (actual)")
        elif isinstance(payload, dict):
            keys = ", ".join(sorted(payload.keys()))
            logger(f"Payload OpenWeather (actual) claus: {keys}")
            logger(f"Payload OpenWeather (actual): {summarize_payload(payload)}")
        else:
            logger(f"Payload OpenWeather (actual) tipus: {type(payload).__name__}")
    entry = extract_openweather_entry(payload)
    weather = parse_openweather_entry(entry) if entry else None
    if logger:
        if entry and not weather:
            logger(f"Entrada OpenWeather no parsejada: {entry}")
        elif weather:
            logger(f"Dades parsejades: {weather}")
    if logger and weather:
        logger("Dades rebudes d'OpenWeather (actual)")
    if logger and not weather:
        logger("Resposta buida d'OpenWeather (actual)")
    return weather

def wind_direction_phrase(deg, lang="ca"):
    try:
        deg = float(deg)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(deg):
        return None
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((deg + 22.5) // 45) % 8
    code = directions[idx]

    if lang.startswith("ca"):
        return {
            "N": "del nord",
            "NE": "del nord-est",
            "E": "de l'est",
            "SE": "del sud-est",
            "S": "del sud",
            "SW": "del sud-oest",
            "W": "de l'oest",
            "NW": "del nord-oest",
        }.get(code)
    if lang.startswith("en"):
        return {
            "N": "from the north",
            "NE": "from the north-east",
            "E": "from the east",
            "SE": "from the south-east",
            "S": "from the south",
            "SW": "from the south-west",
            "W": "from the west",
            "NW": "from the north-west",
        }.get(code)
    return code

def wind_direction_cardinal(deg):
    try:
        deg = float(deg)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(deg):
        return None
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((deg + 22.5) // 45) % 8
    return directions[idx]

def format_weather_summary(weather, lang="ca"):
    if not isinstance(weather, dict):
        return None
    parts = []

    description = weather.get("description")
    if isinstance(description, str):
        description = description.strip()
        if description:
            description = description[0].upper() + description[1:]
            parts.append(description)

    temp = weather.get("temp_c")
    if isinstance(temp, (int, float)) and np.isfinite(temp):
        parts.append(f"{int(round(temp))}\N{DEGREE SIGN}")

    wind_kmh = weather.get("wind_kmh")
    wind_deg = weather.get("wind_deg")
    if isinstance(wind_kmh, (int, float)) and np.isfinite(wind_kmh):
        speed = int(round(wind_kmh))
        wind_label = "vent" if lang.startswith("ca") else "wind"
        direction = wind_direction_phrase(wind_deg, lang=lang)
        if direction:
            parts.append(f"{wind_label} {speed} km/h {direction}")
        else:
            parts.append(f"{wind_label} {speed} km/h")

    if not parts:
        return None
    return ", ".join(parts)

def format_weather_brief(weather):
    if not isinstance(weather, dict):
        return None
    parts = []

    temp = weather.get("temp_c")
    if isinstance(temp, (int, float)) and np.isfinite(temp):
        parts.append(f"{int(round(temp))}\N{DEGREE SIGN}")

    wind_kmh = weather.get("wind_kmh")
    if wind_kmh is None:
        wind_ms = weather.get("wind_ms")
        if isinstance(wind_ms, (int, float)) and np.isfinite(wind_ms):
            wind_kmh = wind_ms * 3.6
    wind_deg = weather.get("wind_deg")
    if isinstance(wind_kmh, (int, float)) and np.isfinite(wind_kmh):
        speed = int(round(wind_kmh))
        direction = wind_direction_cardinal(wind_deg)
        if direction:
            parts.append(f"{speed}km/h {direction}")
        else:
            parts.append(f"{speed}km/h")

    if not parts:
        return None
    return ", ".join(parts)

def get_weather_summary(lat, lon, when_dt, lang="ca", logger=None, allow_current_fallback=False):
    weather = fetch_openweather_snapshot(
        lat,
        lon,
        when_dt,
        lang=lang,
        allow_current_fallback=allow_current_fallback,
        logger=logger,
    )
    return format_weather_summary(weather, lang=lang)

def get_weather_brief_data(lat, lon, when_dt, lang="ca"):
    weather = fetch_openweather_snapshot(lat, lon, when_dt, lang=lang, allow_current_fallback=False)
    if not isinstance(weather, dict):
        return None, None
    return format_weather_brief(weather), weather.get("wind_deg")


def get_weather_brief(lat, lon, when_dt, lang="ca"):
    brief, _ = get_weather_brief_data(lat, lon, when_dt, lang=lang)
    return brief
