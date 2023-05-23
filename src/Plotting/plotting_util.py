import json
from typing import Callable, Optional
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np;
import matplotlib.cm as cm
import matplotlib.scale as mpl_scale
import math
import os
from pathlib import Path

def inv(x):
    return math.sqrt(1 - (1 - x)**2)

def round_scale(x):
    return 1 - math.sqrt(1 - x**2)

def plot_simple_line_2d(ys, function = inv):
    axe = plt.axes()
    axe.plot(range(len(ys)), ys)
    axe.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{function(x):.3f}"))
    plt.show()

def plot_line_2d(xs, y_series, labels, function = inv, x_label = "", y_label = "", save_path = "", y_scale:Optional[mpl_scale.ScaleBase]=None):
    axe = plt.axes()
    for index, ys in enumerate(y_series):
        axe.plot(xs, ys, label = labels[index])

    if y_scale:
        axe.set_yscale(y_scale)
    else:
        axe.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{function(x):.3f}"))
    plt.legend()

    axe.set_xlabel(x_label)
    axe.set_ylabel(y_label)

    if save_path == "":
        plt.show()
    else:
        plt.savefig(os.path.dirname(__file__) + "/../../embeddingData/" + save_path)
        plt.close()

def plot_line_series_2d(xs, ys, labels, x_label = "", y_label = "", save_path = "", legend = False, y_scale:Optional[mpl_scale.ScaleBase]=None, function = None):
    axe = plt.axes()
    for i in range(len(ys)):
        axe.plot(xs[i], ys[i], label = labels[i])
    if legend:
        plt.legend()

    
    if y_scale:
        axe.set_yscale(y_scale)
    elif function:
        axe.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: function(x)))
    axe.set_xlabel(x_label)
    axe.set_ylabel(y_label)

    if save_path == "":
        plt.show()
    else:
        plt.savefig(os.path.dirname(__file__) + "/../../" + save_path)
        plt.close()

def plot_points_2d(xs, ys):
    plt.scatter(xs, ys, marker="o", s = [0.1 for _ in range(len(xs))])
    plt.legend()
    plt.show()

def plot_big_points(xs, ys, x_title, y_title, labels, marker = "o", size = 5, save_path = "", legend = True):

    axe = plt.axes()

    COLORS = get_colors(len(xs))
    #COLORS = [[[v * (1 - (i) / 4) for v in values[:3]] + [1] for i in range(3)] for values in COLORS]

    for i in range(len(xs)):
        plt.scatter(xs[i], ys[i], marker=marker if i < len(xs) -1 else "+", s = [100 + i * 60 for i in range(len(xs[i]))], 
                    color = COLORS[i], label= labels[i])

    ylim = plt.gca().get_ylim()
    xlim = plt.gca().get_xlim()

    start_x = max(ylim[0], xlim[0])
    start_y = max(ylim[0], xlim[0])

    end_x = min(ylim[1], xlim[1])
    end_y = min(ylim[1], xlim[1])

    plt.plot([start_x, end_x], [start_y, end_y], color = [0.5, 0.5, 0.5, 0.5], label = "ln(ced) = ln(med)")

    axe.set_xbound(xlim[0], xlim[1])
    axe.set_ybound(ylim[0], ylim[1])

    axe.set_xlabel(x_title)
    axe.set_ylabel(y_title)

    if legend:
        plt.legend()
    #plt.legend(labels, scatterpoints = 1)
    if save_path == "":
        plt.show()
    else:
        plt.savefig(os.path.dirname(__file__) + "/../../" + save_path)
        plt.close()

def get_colors(num):
    return cm.rainbow(np.linspace(0, 1, num))

def get_list_colors(values:list[int], min_value:int, max_value:int):
    return [cm.rainbow(np.linspace(min_value, max_value, val)) for val in values] 

def plotHeatMap(xs, ys, width, height, label):
    x_max = max(xs)
    x_min = min(xs)
    y_max = max(ys)
    y_min = min(ys)

    heights = [[0 for _ in range(height)] for _ in range(width)]
    for i, x in enumerate(xs):
        y = ys[i]
        xi = int((x - x_min) / (x_max - x_min) * (width - 1))
        yi = int((y - y_min) / (y_max - y_min) * (height - 1))
        heights[xi][yi] += 1

    for i in range(width):
        for ii in range(height):
            heights[i][ii] = np.log(heights[i][ii] + 1)

    plotSurface([heights], "Amount", 
                [i * (x_max - x_min) / (width - 1) + x_min for i in range(width)], "x",
                [i * (y_max - y_min) / (height - 1) + y_min for i in range(height)], "y",
                num_of_surfaces=1, surfaceLabels=[label])

def plotSurface(heights, zTitle, xAxis, xTitle, yAxis, yTitle, num_of_surfaces, surfaceLabels):
    mpl.rcParams['legend.fontsize'] = 10

    xIndices = range(len(xAxis))
    yIndices = range(len(yAxis))

    xIndices, yIndices = np.meshgrid(xIndices, yIndices)
    xAxis, yAxis = np.meshgrid(xAxis, yAxis)

    x_rav = np.ravel(xIndices)
    y_rav = np.ravel(yIndices)

    total_range = range(len(x_rav))

    height_rav = []

    for surface in range(num_of_surfaces):
        height_rav.append(np.array([heights[surface][x_rav[i]][y_rav[i]] for i in total_range]))
        height_rav[surface] = height_rav[surface].reshape(xIndices.shape)
    
    COLOR = get_colors(num_of_surfaces)
    fig = plt.figure()
    axe = plt.axes(projection='3d')

    for surface in range(num_of_surfaces):
        surf = axe.plot_surface(xAxis, yAxis, height_rav[surface], alpha = 1, rstride=1, cstride=1, linewidth=0.0, 
                                antialiased=False, color=COLOR[surface], label = surfaceLabels[surface])
        surf._facecolors2d=surf._facecolor3d
        surf._edgecolors2d=surf._edgecolor3d

    axe.set_xlabel(xTitle)
    axe.set_ylabel(yTitle)
    axe.set_zlabel(zTitle)

    axe.legend()

    plt.show()

def get_min_max(xs): 
    x_max = -math.inf
    x_min = math.inf
    for x in xs:
        if len(x) > 0:
            x_max = max(x_max, max(x))
            x_min = min(x_min, min(x))
    return x_min, x_max

def plotPoints(xs, ys, zs, axis_names = ["", "", ""], legend = True, num_of_series = 1, series_labels=[], function = inv, marker = "o"):
    mpl.rcParams['legend.fontsize'] = 10

    if series_labels == []: 
        series_labels = [axis_names[2] for _ in range(num_of_series)]

    if(not isinstance(xs[0], list)):
        xs = [xs]
        ys = [ys]
        zs = [zs]

    x_min, x_max = get_min_max(xs)
    y_min, y_max = get_min_max(ys)
    z_min, z_max = get_min_max(zs)

    COLOR = get_colors(num_of_series)
    fig = plt.figure()
    axe = plt.axes(projection='3d')
    

    for series in range(num_of_series):
        axe.plot(xs[series], ys[series], zs[series], marker, color=COLOR[series], label=series_labels[series])

    axe.set_xlabel(axis_names[0])
    axe.set_ylabel(axis_names[1])
    axe.set_zlabel(axis_names[2])

    axe.set_xbound(x_min, x_max)
    axe.set_ybound(y_min, y_max)
    axe.set_zbound(z_min, z_max)
    axe.w_zaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{function(x):.3f}"))

    if legend:
        axe.legend()

    plt.show()

class axis():
    label:str
    data:list[list]
    scale:mpl_scale.ScaleBase = mpl_scale.LinearScale(None)

    def __init__(self, label:str, data:list[list] = []) -> None:
        self.label = label
        self.data = data

def plotPoints2d(xs:axis, ys:axis, legend = True, num_of_series = 1, series_labels=[], function = inv, marker = "o"):
    mpl.rcParams['legend.fontsize'] = 10

    if series_labels == []: 
        series_labels = [ys.label[2] for _ in range(num_of_series)]

    x_min, x_max = get_min_max(xs.data)
    y_min, y_max = get_min_max(ys.data)

    COLOR = get_colors(num_of_series)
    fig = plt.figure()
    axe = plt.axes()

    for series in range(num_of_series):
        # axe.plot(xs.data[series], ys.data[series], marker=marker, color=COLOR[series], label=series_labels[series])
        axe.scatter(xs.data[series], ys.data[series], marker=marker, label=series_labels[series])

    axe.set_xlabel(xs.label)
    axe.set_ylabel(ys.label)

    axe.set_xbound(x_min, x_max)
    axe.set_ybound(y_min, y_max)
    # axe.w_zaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{function(x):.3f}"))

    axe.set_xscale(xs.scale)
    axe.set_yscale(ys.scale)

    if legend:
        axe.legend()

    # plt.show()
    return plt

def plot_nested_bars(values, groups, labels, x_label = "", y_label = "", save_path = ""):
    # Set up the figure and axes
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    # Set the width of each bar group
    bar_width = 0.2

    COLORS = get_colors(len(labels))
    x = 0
    x_ticks = []
    values = np.array(values)
    maximum = np.max(values)
    minimum = round((np.where(values > 0, values, np.inf).min() - 0.05) * 100) / 100

    # Loop over the groups and create the bar chart
    for i, group in enumerate(groups):
        # Calculate the x positions of the bars within the group
        x_positions = []
        tick = 0
        n = 0
        for y in values[i][0]:
            if y > 0:
                n += 1
                x += bar_width
                tick += x
            x_positions.append(x)
        x_ticks.append(tick/n)
        l = len(values[i])
        x += bar_width
        for ii in range(l - 1, -1, -1):
            cscale = (l - ii / 2) / l
            ax.bar(x_positions, [max(v - minimum, 0) for v in values[i][ii]], (ii + 1) * bar_width / l, color = [[v * cscale for v in values[:3]] + [1] for values in COLORS], bottom = minimum,label=labels, edgecolor = "black")
        
    # Set the x-axis labels and title
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(groups)
    step = np.round((maximum - minimum) * 10) / 100
    ax.set_yticks(np.arange(minimum, maximum + step, step))
    ax.set_xlabel(x_label)

    # Set the y-axis labels
    ax.set_ylabel(y_label)

    # Add a legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower left')
    plt.set_cmap("magma")

    if save_path == "":
        plt.show()
    else:
        plt.savefig(os.path.dirname(__file__) + "/../../" + save_path)
        plt.close()


#Takes an array of dictionaries that contain keys label, marker, color, points, xs, ys zs, 
# all keys are optional but the dictionaries should contain either points or both xs, ys, and zs
def plotCustomPoints(series, axis_names = ["", "", ""], legend = True, axes=[0,1,2]):
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    axe = plt.axes(projection='3d')

    for i, s in enumerate(series):
        coordinates = np.array(s["points"]).transpose() if "points" in s else [s["xs"], s["ys"], s["zs"]]
        if len(coordinates) > 0:
            label = s["label"] if "label" in s else axis_names[2]
            marker = s["marker"] if "marker" in s else "o"
            color = s["color"] if "color" in s else [0,0,0]
            axe.plot(coordinates[axes[0]], coordinates[axes[1]], coordinates[axes[2]], marker, color=color, label=label)
        else:
            print(f"SERIES {i} IS EMPTY, WHAT THE FUCK IS UP WITH THAT?")

    axe.set_xlabel(axis_names[0])
    axe.set_ylabel(axis_names[1])
    axe.set_zlabel(axis_names[2])

    if legend:
        axe.legend()

    plt.show()