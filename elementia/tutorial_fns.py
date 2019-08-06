import numpy as np
import pandas as pd
import math

def calculate_cross_validation(concs=[], intensities=[]):
# calculates cross-validation statistics [see Tutorial 4]
# concs = list of concentrations
# intensities = list of associated intensities

    rmsep_sum = 0.0
    output_concs = list(concs)
    for i in range(len(concs)):
        new_intensities = []
        new_concs = []

        for j in range(len(concs)):
            if i != j:
                new_intensities.append(intensities[j])
                new_concs.append(concs[j])

        np_new_intensities = np.array(new_intensities)
        np_new_concs = np.array(new_concs)
        this_slope = np.dot(np_new_concs, np_new_intensities)/np.dot(np_new_intensities, np_new_intensities)

        cv_conc = this_slope*intensities[i]
        output_concs[i] = cv_conc

        conc_diff = cv_conc - concs[i]
        rmsep_sum += conc_diff*conc_diff

    rmsecv = math.sqrt(rmsep_sum/float(len(concs)))

    return output_concs, rmsecv

def calculate_intensity_cv(wavelengths=[], intensities=[], concs=[], samples=[], wavelength_lower=None,
                           wavelength_upper=None, wavelength_background=None):
    wl_index = pd.Index(wavelengths)
    index_lower = wl_index.get_loc(wavelength_lower, method='nearest')
    index_upper = wl_index.get_loc(wavelength_upper, method='nearest')
    index_background = wl_index.get_loc(wavelength_background, method='nearest')

    max_intensities = []

    for sample in samples:
        max_intensities.append(max(intensities[sample][index_lower:index_upper])-intensities[sample][index_background])

    return calculate_cross_validation(concs=concs, intensities=max_intensities)

def calculate_sum_cv(wavelengths=[], intensities=[], concs=[], samples=[], wavelength_lower=None,
                           wavelength_upper=None, wavelength_background=None):
    wl_index = pd.Index(wavelengths)
    index_lower = wl_index.get_loc(wavelength_lower, method='nearest')
    index_upper = wl_index.get_loc(wavelength_upper, method='nearest')
    index_background = wl_index.get_loc(wavelength_background, method='nearest')

    max_intensities = []

    for sample in samples:
        max_intensities.append(sum(intensities[sample][index_lower:index_upper])-(index_upper-index_lower)*intensities[sample][index_background])

    return calculate_cross_validation(concs=concs, intensities=max_intensities)

test1, test2 = calculate_cross_validation(intensities=[6017.200000000001, 4323.5, 1906.8, 1107.0, 705.7, 238.70000000000002, 286.5],
                                          concs=[20848, 14543, 6068, 4283, 1515, 594, 30])
