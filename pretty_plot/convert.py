# Functions for the conversion of data and metadata formats
# TODO: we should decide how and if to conglomerate this with stuff in
#  marslab.compat -- michael

import pandas as pd
import numpy as np

from marslab.compat.mertools import (
    MERSPECT_COLOR_MAPPINGS, WAVELENGTH_TO_FILTER,
)
from marslab.compat.xcam import DERIVED_CAM_DICT


def scale_eyes(data, method="scale_to_avg"):
    # Accepts a "marslab format" spectra file pandas DataFrame
    # This translation is done _in place_... which is maybe bad...
    #    but will be fine as long as you always restart + rerun
    # TODO: Remove duplicate functionality with marslab.compat.xcam
    if np.isnan(data["L1"].values).any() or np.isnan(data["R1"].values).any():
        # shared filters don't exist
        return data
    if method == "scale_to_left":
        # Scale the Right eye data to the Left eye at 800nm
        for i in range(len(data.index)):
            scale_factor = data.iloc[i]["L1"] / data.iloc[i]["R1"]
            for k in data.keys():
                if (
                    ("R" in k)
                    and (not "STD" in k)
                    and (not "L0" in k)
                    and (not "RMS" in k)
                    and len(k) <= 3
                ):
                    # Scale R to L in place
                    data.iloc[i][k] = data.iloc[i][k] * scale_factor
    elif method == "scale_to_avg":
        for i in range(len(data.index)):
            left_scale = data.iloc[i][["L1", "R1"]].mean() / data.iloc[i]["L1"]
            right_scale = (
                data.iloc[i][["L1", "R1"]].mean() / data.iloc[i]["R1"]
            )
            for k in data.keys():
                if (
                    ("R" in k)
                    and ("STD" not in k)
                    and ("L0" not in k)
                    and ("RSM" not in k)
                    and ("SOL" not in k)
                    and len(k) <= 3
                ):
                    # Scale R to L in place
                    # data.iloc[i][k] = data.iloc[i][k] * right_scale # bad
                    data.loc[i, k] = data.iloc[i][k] * right_scale
                elif (
                    ("L" in k)
                    and ("STD" not in k)
                    and ("R0" not in k)
                    and ("RSM" not in k)
                    and ("SOL" not in k)
                    and len(k) <= 3
                ):
                    # data.iloc[i][k] = data.iloc[i][k] * left_scale # bad
                    data.loc[i, k] = data.iloc[i][k] * left_scale
    return data


def convert_for_plot(
    spectra_fn,
    instrument="ZCAM",
    color_to_feature={},
    scale_method="scale_to_avg",
):
    if spectra_fn.__class__ == str:
        spectra_fn = [spectra_fn]
    data = pd.DataFrame()
    for fn in spectra_fn:
        marslab_data = pd.read_csv(fn, index_col=None, na_values="-")
        data = pd.concat([data, marslab_data], ignore_index=True)
    data = scale_eyes(data, method=scale_method)
    data.replace(np.nan, "-", inplace=True)
    return data


def convert_to_simple_format(
        spectra_fn,
        instrument="ZCAM",
        scale_method="scale_to_avg",
):
    """does not currently work for CCAM (no L/R eyes)"""
    data = convert_for_plot(spectra_fn, instrument=instrument, scale_method=scale_method)
    available_bands = [
        k for k in data.keys() if k in DERIVED_CAM_DICT[instrument]["filters"]
    ]
    f2w = dict((v, k) for k, v in WAVELENGTH_TO_FILTER[instrument]["L"].items())
    for k, v in WAVELENGTH_TO_FILTER[instrument]["R"].items():
        f2w[v] = k
    wavelengths = [(b, f2w[b], "LEFT") if b[0] == "L" else (b, f2w[b], "RIGHT")
                   for b in available_bands]
    wavelengths.sort(key=lambda x: x[1], reverse=True)
    simple_df = pd.DataFrame.from_records(wavelengths, columns=['Band_name', 'Wavelength (nm)', 'Eye'])
    simple_df.drop("Band_name", axis=1, inplace=True)
    for index, row in data.iterrows():
        roi_val = [row[b[0]] for b in wavelengths]
        roi_std = [row[b[0]+"_STD"] for b in wavelengths]
        row_df = pd.DataFrame({row["COLOR"]+"_mean": roi_val,
                               row["COLOR"]+"_err": roi_std})
        simple_df = pd.concat([simple_df, row_df], axis=1)
    if "marslab" in str(spectra_fn):
        simple_csv_fn = str(spectra_fn).replace(
            "marslab", "simplified"
        )
    else:
        simple_csv_fn = str(spectra_fn).replace('.csv', '_simplified.csv')
    print("Writing " + simple_csv_fn)
    simple_df.to_csv(simple_csv_fn, index=False)
    return simple_df
