# -*- coding: utf-8 -*-

from datetime import datetime, timezone

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
