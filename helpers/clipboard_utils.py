# -*- coding: utf-8 -*-

import ctypes
from ctypes import wintypes

import numpy as np

from helpers.errors import LocalizedError

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
