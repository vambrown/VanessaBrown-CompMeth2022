"""DFT and DCT from Exercises 7.4 and 7.6 for Homework 5."""
import argparse
import numpy as np
import matplotlib.pyplot as plt


# creating functions for dft and dct
def fft_dow_10(filename):
    """
    Filters and smooths first 10% of Dow data with DFT.

    Parameters
    ----------
    filename : string
        Name of Dow data file.

    Returns
    -------
    A plot of filtered, smoothed data compared to
    unfiltered data.
    """
    closing_values = np.loadtxt(filename, float)
    day = np.arange(1, len(closing_values) + 1)
    percent_keep = int(len(day)*0.1)

    coeffs = np.fft.rfft(closing_values)  # get coefficients from DFT
    coeffs_kept = np.copy(coeffs)  # keep only first 10% of coefficients
    coeffs_kept[percent_keep:] = 0
    inv_coeffs = np.fft.irfft(coeffs_kept)  # inverse transform of coefficients

    plt.plot(day, closing_values, 'k', label="All data")
    plt.plot(day, inv_coeffs, 'r', label="Filtered (10%) DFT data")
    plt.title("Dow Closing Values")
    plt.xlabel("Time (Days)")
    plt.ylabel("Daily Closing Value")
    plt.legend()
    plt.show()


def fft_dow_2(filename):
    """
    Filters and smooths first 2% of Dow data with DFT.

    Parameters
    ----------
    filename : string
        Name of Dow data file.

    Returns
    -------
    A plot of filtered, smoothed data compared to
    unfiltered data.
    """
    closing_values = np.loadtxt(filename, float)
    day = np.arange(1, len(closing_values) + 1)
    percent_keep = int(len(day)*0.02)

    coeffs_2 = np.fft.rfft(closing_values)  # get coefficients from DFT
    coeffs_kept_2 = np.copy(coeffs_2)  # keep only first 2% of coefficients
    coeffs_kept_2[percent_keep:] = 0
    inv_coeffs_2 = np.fft.irfft(coeffs_kept_2)

    plt.plot(day, closing_values, 'k', label="All data")
    plt.plot(day, inv_coeffs_2, 'g', label="Filtered (2%) DFT data")
    plt.title("Dow Closing Values")
    plt.xlabel("Time (Days)")
    plt.ylabel("Daily Closing Value")
    plt.legend()
    plt.show()


# dct function from author resources
def dct(rdata):
    """Type-II discrete cosine transform (DCT) of data."""
    length = len(rdata)
    rdata2 = np.empty(2*length, float)
    rdata2[:length] = rdata[:]
    rdata2[length:] = rdata[::-1]

    coeff = np.fft.rfft(rdata2)
    phi = np.exp(-1j*np.pi*np.arange(length)/(2*length))
    return np.real(phi*coeff[:length])


# idct function from author resources
def idct(data):
    """Type-II inverse DCT of data."""
    longth = len(data)
    coeffs = np.empty(longth+1, complex)

    phi = np.exp(1j*np.pi*np.arange(longth)/(2*longth))
    coeffs[:longth] = phi*data
    coeffs[longth] = 0.0
    return np.fft.irfft(coeffs)[:longth]


def dct_dow_10(filename):
    """
    Filters and smooths first 10% of Dow data with DCT.

    Parameters
    ----------
    filename : string
        Name of Dow data file.

    Returns
    -------
    A plot of filtered, smoothed data compared to
    unfiltered data.
    """
    closing_values = np.loadtxt(filename, float)
    day = np.arange(1, len(closing_values) + 1)
    percent_keep = int(len(day)*0.1)

    dct_coeffs = dct(closing_values)   # get coefficients from DCT
    dct_coeffs = np.copy(dct_coeffs)  # keep only first 10% of coefficients
    dct_coeffs[percent_keep:] = 0
    inv_dct_coeffs = idct(dct_coeffs)

    plt.plot(day, closing_values, 'k', label="All data")
    plt.plot(day, inv_dct_coeffs, 'b', label="Filtered (10%) DCT data")
    plt.title("Dow Closing Values")
    plt.xlabel("Time (Days)")
    plt.ylabel("Daily Closing Value")
    plt.legend()
    plt.show()


def dct_dow_2(filename):
    """
    Filters and smooths first 2% of Dow data with DCT.

    Parameters
    ----------
    filename : string
        Name of Dow data file.

    Returns
    -------
    A plot of filtered, smoothed data compared to
    unfiltered data.
    """
    closing_values = np.loadtxt(filename, float)
    day = np.arange(1, len(closing_values) + 1)
    percent_keep = int(len(day)*0.02)

    dct_coeffs = dct(closing_values)   # get coefficients from DCT
    dct_coeffs_2 = np.copy(dct_coeffs)  # keep only first 2% of coefficients
    dct_coeffs_2[percent_keep:] = 0
    inv_dct_coeffs_2 = idct(dct_coeffs_2)

    plt.plot(day, closing_values, 'k', label="All data")
    plt.plot(day, inv_dct_coeffs_2, 'c', label="Filtered (2%) DCT data")
    plt.title("Dow Closing Values")
    plt.xlabel("Time (Days)")
    plt.ylabel("Daily Closing Value")
    plt.legend()
    plt.show()


# setting up the argparse
ARG_DESC = '''\
        Let's plot Fourier and cosine transforms!
        --------------------------------
            This program filters, smooths, and plots
            daily closing values of the Dow.
        '''
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=ARG_DESC)

parser.add_argument("file", type=str, choices=['dow.txt', 'dow2.txt'],
                    help="Enter name of data file.")

FUNCTIONS = {'fft10': fft_dow_10,
             'fft2': fft_dow_2,
             'dct10': dct_dow_10,
             'dct2': dct_dow_2}
parser.add_argument("transform", choices=FUNCTIONS.keys(),
                    help="Choose DFT or DCT for 10%% or 2%% of data.")

args = parser.parse_args()

func = FUNCTIONS[args.transform]
file = args.file
func(file)
