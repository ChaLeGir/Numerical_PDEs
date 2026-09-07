import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy as sp
from datetime import datetime, date, time, timedelta
import gurobipy
import copy
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

blaster_blue = "#09396C"
dark_blue = "#21314d"
light_blue = "#879EC3"
colorado_red = "#CC4628"
pale_blue = "#CFDCE9"

mines_colors = [
    blaster_blue,
    dark_blue,
    light_blue,
    colorado_red,
    pale_blue
]

def color_name(color):
    if color == blaster_blue:
        return "blaster_blue"
    elif color == dark_blue:
        return "dark_blue"
    elif color == light_blue:
        return "light_blue"
    elif color == colorado_red:
        return "colorado_red"
    elif color == pale_blue:
        return "pale_blue"
    else:
        return "unknown"