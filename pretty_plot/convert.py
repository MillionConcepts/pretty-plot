# Functions for the conversion of data and metadata formats
# TODO: we should decide how and if to conglomerate this with stuff in
#  marslab.compat -- michael

from dustgoggles.structures import listify
import numpy as np
import pandas as pd

from marslab.compat.mertools import WAVELENGTH_TO_FILTER
from marslab.compat.xcam import DERIVED_CAM_DICT

class InstrumentIsMonocular(TypeError):
    """
    This is a single-eye instrument and you cannot do binocular things to it.
    """
    pass


def scale_eyes(data, method="scale_to_avg", instrument="ZCAM"):
    if method is None:
        return data
    if "L1" not in data.columns and "R1" not in data.columns:
        raise InstrumentIsMonocular
    # Accepts a "marslab format" spectra file pandas DataFrame
    # This translation is done _in place_... which is maybe bad...
    #    but will be fine as long as you always restart + rerun
    # TODO: Remove duplicate functionality with marslab.compat.xcam
    if np.isnan(data["L1"].values).any() or np.isnan(data["R1"].values).any():
        # shared filters don't exist
        return data
    # reduce chance of unwanted casts
    fcols = [
        d for d in data.columns if d in DERIVED_CAM_DICT[instrument]['filters']
    ]
    filterframe = data[fcols]
    if method == "scale_to_left":
        # scale right to left eye at 800 (ZCAM) or 527 (MCAM) nm
        for i in range(len(filterframe.index)):
            scale_factor = (
                filterframe.loc[i, "L1"] / filterframe.loc[i, "R1"]
            )
            for k in filterframe.keys():
                if k.startswith('R'):
                    # Scale R to L in place
                    data.loc[i, k] = filterframe.loc[i, k] * scale_factor
    elif method == "scale_to_avg":
        # scale right and left eyes to average at 800 (ZCAM) or 527 (MCAM) nm
        for i in range(len(data.index)):
            eye_mean = np.mean(
                (filterframe.loc[i, "L1"], filterframe.loc[i, "R1"])
            )
            left_scale = eye_mean / filterframe.loc[i, "L1"]
            right_scale = eye_mean / filterframe.loc[i, "R1"]
            for k in filterframe.keys():
                if k.startswith("R"):
                    # Scale R to L in place
                    data.loc[i, k] = filterframe.loc[i, k] * right_scale
                elif k.startswith("L"):
                    # data.iloc[i][k] = data.iloc[i][k] * left_scale # bad
                    data.loc[i, k] = filterframe.loc[i, k] * left_scale
    return data


def load_spectra(spectra_fn):
    data = [
        pd.read_csv(fn, index_col=None, na_values="-")
        for fn in listify(spectra_fn)
    ]
    return pd.concat(data, ignore_index=True)


def convert_to_simple_format(
    spectra_fn,
    instrument="ZCAM",
    scale_method="scale_to_avg",
):
    """WARNING: does not currently work for instruments with no L/R eyes"""
    data = pd.read_csv(spectra_fn, index_col=None, na_values="-")
    if instrument in DERIVED_CAM_DICT.keys():
        scale_eyes(data, scale_method, instrument)
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
