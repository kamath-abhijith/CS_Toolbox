'''

UTILITIES FOR COMPRESSIVE SENSING AND SPARSE SIGNAL PROCESSING

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

'''

# %% IMPORT LIBRARIES

import numpy as np

from matplotlib import pyplot as plt

# %% HELPER FUNCTIONS

def add_noise(data, snr=None, sigma=None):
    '''
    Add white Gaussian noise to data according to given SNR or standard deviation

    :param data: input data vector
    :param snr: desired signal to noise ratio
    :param sigma: desired noise variance

    :returns: noisy data

    '''

    if snr:
        awgn = np.random.randn(data.shape[0], data.shape[1])
        awgn = awgn / np.linalg.norm(awgn) * np.linalg.norm(data) * 10 ** (-1.0*snr / 20.)

    elif sigma:
        awgn = np.random.normal(scale=sigma, loc=0, size=data.shape)

    return data + awgn

# %% PLOTTING TOOLS

def plot_sparsevec(x, ax=None, plot_colour='blue', marker_style='*',
    xaxis_label=None, yaxis_label=None, title_text=None, legend_label=None,
    legend_show=True, legend_loc='upper right', leg_ncol=1, line_style='-',
    line_width=4, show=False, save=None):
    '''
    Plots sparse vector x

    '''

    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()
    
    markerline211_1, stemlines211_1, baseline211_1 = ax.stem(np.arange(len(x)), x,
        line_style, label=legend_label)
    plt.setp(stemlines211_1, linewidth=line_width, color=plot_colour)
    plt.setp(markerline211_1, marker=marker_style, linewidth=line_width, markersize=10,
        markerfacecolor=plot_colour, mec=plot_colour)
    plt.setp(baseline211_1, linewidth=0)
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)

    if legend_label and legend_show:
        plt.legend(ncol=leg_ncol, loc=legend_loc, frameon=True, framealpha=0.8, facecolor='white')
    plt.ylabel(yaxis_label)
    plt.xlabel(xaxis_label)

    plt.xlim([0, len(x)])
    plt.title(title_text, y=-.28)

    if show:
        plt.show()

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    return

def plot_signal(x, y, ax=None, plot_colour='blue', xaxis_label=None,
    yaxis_label=None, title_text=None, legend_label=None, legend_show=True,
    legend_loc='upper right', line_style='-', line_width=2,
    show=False, xlimits=[0,100], ylimits=[0,10], save=None):
    '''
    Plots signal with abscissa in x and ordinates in y 

    '''
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    plt.plot(x, y, linestyle=line_style, linewidth=line_width, color=plot_colour,
        label=legend_label)
    if legend_label and legend_show:
        plt.legend(loc=legend_loc, frameon=True, framealpha=0.8, facecolor='white')
    
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

# %% PERFORMANCE METRICS

def rec_snr(true, estimate):
    ''' Reconstruction SNR between true, estimate '''

    return 20*np.log10(np.linalg.norm(true)/np.linalg.norm(true-estimate))