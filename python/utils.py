'''
UTILITY FUNCTIONS FOR COMPRESSIVE SENSING AND SPARSE SIGNAL PROCESSING

AUTHOR: ABIJITH J. KAMATH, INDIAN INSTITUTE OF SCIENCE, BANGALORE
abijithj@iisc.ac.in, kamath-abhijith.gihub.io

'''

# %% IMPORT LIBRARIES
import numpy as np
import seaborn as sb

from matplotlib import style
from matplotlib import rcParams
from matplotlib import pyplot as plt

# %% PLOTTING

def plot_sparse_vector(x, ax=None, plot_colour='blue', xaxis_label=None,
    yaxis_label=None, title_text=None, legend_label=None, legend_show=True,
    legend_loc='upper left', line_style='-', line_width=None,
    show=False, save=None):
    '''
    Plots sparse vector x

    '''

    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()
    
    # tk = np.argwhere(np.abs(x) > 0.1)
    # ak = np.squeeze(x[tk], axis=1)

    # markerline211_1, stemlines211_1, baseline211_1 = ax.stem(np.squeeze(tk, axis=1), ak,
        # line_style, label=legend_label)
    markerline211_1, stemlines211_1, baseline211_1 = ax.stem(np.arange(len(x)), x,
        line_style, label=legend_label)
    plt.setp(stemlines211_1, linewidth=line_width, color=plot_colour)
    plt.setp(markerline211_1, marker='*', linewidth=1.5, markersize=8,
        markerfacecolor=plot_colour, mec=plot_colour)
    plt.setp(baseline211_1, linewidth=0)
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)

    if legend_label and legend_show:
        plt.legend(ncol=2, loc=legend_loc, frameon=True, framealpha=0.8, facecolor='white')
    plt.ylabel(yaxis_label)
    plt.xlabel(xaxis_label)

    if show:
        plt.xlim([0, len(x)])
        plt.title(title_text, y=-.28)
        if save:
            plt.savefig(save + '.pdf', format='pdf')
        plt.show()

    return

def plot_signal(x, y, ax=None, plot_colour='blue', xaxis_label=None,
    yaxis_label=None, title_text=None, legend_label=None, legend_show=True,
    legend_loc='upper left', line_style='-', line_width=None,
    show=False, xlimits=None, ylimits=None, save=None):
    '''
    Plots signal with abscissa in x and ordinates in y 

    '''
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    if plot_colour is not 'random':
        plt.plot(x, y, linestyle=line_style, linewidth=line_width, color=plot_colour, label=legend_label)
    elif plot_colour == 'random':
        plt.plot(x, y, linestyle=line_style, linewidth=line_width, label=legend_label)
    # plt.grid('on')
    if legend_label and legend_show:
        plt.legend(ncol=2, loc=legend_loc, frameon=True, framealpha=0.8, facecolor='white')
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)

    if show:
        plt.xlim(xlimits)
        plt.ylim(ylimits)
        plt.title(title_text, y=-.28)
        if save:
            plt.savefig(save + '.pdf', format='pdf')
        plt.show()

    return

def plot_heatmap(data, ax=None, xaxis_label=None, yaxis_label=None,
    annotation=False, show=True, save=False):
    '''
    Plots 2D heatmap with data entries

    '''
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    ax = sb.heatmap(data, linewidths=0.5, annot=annotation)
    ax.invert_yaxis()
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    if show:
        if save:
            plt.savefig(save + '.pdf', format='pdf')
        plt.show()

    return

# %% OPERATORS

def fftmtx(N):
    '''
    Returns FFT matrix of size N x N

    '''
    return np.fft.fft(np.identity(N,dtype=float))/np.sqrt(N)

# %% THRESHOLDING FUNCTIONS

def hard_thresholding(x, lambd):
    return

def soft_thresholding(x, lambd):
    '''
    Returns soft threholding of x by lambd

    :param x: Input vector
    :param lambd: Soft thresholding parameter

    :returns: Soft thresholding of x

    '''

    return np.maximum(0,x-lambd) - np.maximum(0,-x-lambd)

# %% PERFORMANCE METRICS

def mean_squared_error(a, b):
    '''
    Computes mean-squared-error between a and b

    :param a: input vector
    :param b: input vector

    :return: mse between a and b

    '''

    return np.mean((a-b)**2)