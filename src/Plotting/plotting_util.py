import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np;
import matplotlib.cm as cm
import math
from pathlib import Path

def inv(x):
    return math.sqrt(1 - (1 - x)**2)

def plot_line_2d(ys, function = inv):
    axe = plt.axes()
    axe.plot(range(len(ys)), ys)
    axe.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{function(x):.3f}"))
    plt.show()

def plot_line_2d(xs, y_series, labels, function = inv):
    axe = plt.axes()
    for index, ys in enumerate(y_series):
        axe.plot(xs, ys, label = labels[index])
    axe.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{function(x):.3f}"))
    plt.legend()
    plt.show()

def plot_points_2d(xs, ys):
    plt.scatter(xs, ys, marker="o", s = [0.1 for _ in range(len(xs))])
    plt.legend()
    plt.show()

def plot_points_series_2d(xs, ys):
    for i in range(len(xs)):
        plt.scatter(xs[i], ys[i], marker="o", s = [1 for _ in range(len(xs[i]))])
    plt.legend()
    plt.show()

def get_colors(num):
    return cm.rainbow(np.linspace(0, 1, num))

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

#Takes an array of dictionaries that contain keys label, marker, color, points, xs, ys zs, 
# all keys are optional but the dictionaries should contain either points or both xs, ys, and zs
def plotCustomPoints(series, axis_names = ["", "", ""], legend = True, axes=[0,1,2]):
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    axe = plt.axes(projection='3d')

    for i, s in enumerate(series):
        coordinates = np.array(s["points"]).transpose() if "points" in s else [s["xs"], s["ys"], s["zs"]]
        label = s["label"] if "label" in s else axis_names[2]
        marker = s["marker"] if "marker" in s else "o"
        color = s["color"] if "color" in s else [0,0,0]
        axe.plot(coordinates[axes[0]], coordinates[axes[1]], coordinates[axes[2]], marker, color=color, label=label)

    axe.set_xlabel(axis_names[0])
    axe.set_ylabel(axis_names[1])
    axe.set_zlabel(axis_names[2])

    if legend:
        axe.legend()

    plt.show()
