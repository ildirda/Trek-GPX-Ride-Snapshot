# -*- coding: utf-8 -*-

from datetime import datetime, timezone

from helpers.i18n import load_translations

def debug_log(message):
    try:
        print(message, flush=True)
    except Exception:
        pass

def format_local_datetime(dt):
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local_dt = dt.astimezone()
    return local_dt.strftime("%Y-%m-%d %H:%M")

def format_ride_title(dt, t=None, lang="ca"):
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local_dt = dt.astimezone()
    time_label = local_dt.strftime("%H:%M")
    weekday_idx = local_dt.weekday()
    month_idx = local_dt.month - 1
    weekdays = None
    months = None
    template = None
    if callable(t):
        weekdays = t("dashboard.weekdays")
        months = t("dashboard.months")
        template = t("dashboard.title.template")
    if not isinstance(weekdays, (list, tuple)) or len(weekdays) != 7:
        weekdays = None
    if not isinstance(months, (list, tuple)) or len(months) != 12:
        months = None
    if not isinstance(template, str) or template == "dashboard.title.template":
        template = None
    if weekdays is None or months is None or template is None:
        lang_code = lang.lower() if isinstance(lang, str) else ""
        translations = load_translations(lang_code) if lang_code else {}
        if not translations and lang_code:
            base_code = lang_code.split("-", 1)[0].split("_", 1)[0]
            if base_code and base_code != lang_code:
                translations = load_translations(base_code)
        if translations:
            weekdays = translations.get("dashboard.weekdays")
            months = translations.get("dashboard.months")
            template = translations.get("dashboard.title.template")
        if not isinstance(weekdays, (list, tuple)) or len(weekdays) != 7:
            weekdays = None
        if not isinstance(months, (list, tuple)) or len(months) != 12:
            months = None
        if not isinstance(template, str) or template == "dashboard.title.template":
            template = None
        if weekdays is None or months is None or template is None:
            return local_dt.strftime("%Y-%m-%d %H:%M")
    weekday = weekdays[weekday_idx]
    month = months[month_idx]
    try:
        return template.format(
            weekday=weekday,
            day=local_dt.day,
            month=month,
            year=local_dt.year,
            time=time_label,
        )
    except Exception:
        return local_dt.strftime("%Y-%m-%d %H:%M")
