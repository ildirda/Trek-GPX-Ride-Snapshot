# -*- coding: utf-8 -*-

class LocalizedError(Exception):
    """Exception carrying a localization key (and optional detail)."""
    def __init__(self, key, detail=None):
        self.key = key
        self.detail = detail
        super().__init__(key)
