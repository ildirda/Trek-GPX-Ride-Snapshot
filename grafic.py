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

import matplotlib

matplotlib.use("TkAgg")

from helpers.app import main


if __name__ == "__main__":
    main()
