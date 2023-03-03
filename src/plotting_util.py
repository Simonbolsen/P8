import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np;
import matplotlib.cm as cm
import math
from pathlib import Path

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
    
    COLOR = cm.rainbow(np.linspace(0, 1, num_of_surfaces))
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


def plotPoints(xs, ys, zs, axis_names, num_of_series = 1, series_labels=[]):
    mpl.rcParams['legend.fontsize'] = 10

    if series_labels == []: 
        series_labels = [axis_names[2] for _ in range(num_of_series)]

    if(not isinstance(xs[0], list)):
        xs = [xs]
        ys = [ys]
        zs = [zs]

    COLOR = cm.rainbow(np.linspace(0, 1, num_of_series))
    fig = plt.figure()
    axe = plt.axes(projection='3d')

    for series in range(num_of_series):
        axe.plot(xs[series], ys[series], zs[series], "o", color=COLOR[series], label=series_labels[series])

    axe.set_xlabel(axis_names[0])
    axe.set_ylabel(axis_names[1])
    axe.set_zlabel(axis_names[2])

    axe.legend()

    plt.show()

if __name__ == '__main__':
    plotPoints([[0, 1, 2, 3]], [[2,2,5,2]], [[3,2,3,4]], ["Length", "Width", "Height"])
    plotPoints([0, 1, 2, 3], [2,2,5,2], [3,2,3,4], ["Length", "Width", "Height"]) #simpler version for a single series
    plotPoints([[0, 1], [2, 3]], [[2,2],[5,2]], [[3,2],[3,4]], ["Length", "Width", "Height"], num_of_series=2) #two series, will have different colors