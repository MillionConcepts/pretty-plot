import sys


def looks_like_marslab(fn: str) -> bool:
    return bool(
        ("marslab" in fn) and not ("extended" in fn) and (fn.endswith(".csv"))
    )


def directory_of(path):
    if path.is_dir():
        return path
    return path.parent


def do_pplot(
    path_or_file,
    *,
    offset: "o" = None,
    recursive: "r" = False,
    debug: "d" = False,
    plt_bayer: "b" = True,
    roi_labels: "l" = None,
    annotation: "a" = None,
    width_sf: "w" = 1,
    height_sf: "y" = 1,
    normalize: bool = False
):
    """
    non-interactive CLI to pretty-plot. generates .png files
    from pretty-plot's default settings, much like when pretty-plot
    is called by asdf.
    all marslab files need SOLAR_ELEVATION, SEQ_ID, and SOL or things
    will not work out.
    param path_or_file: marslab file or root_dir containing marslab files
    param recursive: runs pretty_plot on all marslab files in root_dir tree,
        regardless of what specific file you passed it
    """
    from pathlib import Path
    import warnings

    from fs.osfs import OSFS
    import numpy as np
    import pandas as pd

    import pretty_plot.pplot_utils as pplot_utils
    from pretty_plot.convert import load_spectra

    # TODO, maybe: merge or something with handle_pretty_plot()
    path = Path(path_or_file)
    if recursive:
        tree = OSFS(directory_of(path))
        marslab_files = map(
            tree.getsyspath,
            filter(looks_like_marslab, tree.walk.files())
        )
    elif path.is_dir():
        marslab_files = filter(looks_like_marslab, map(str, path.iterdir()))
    else:
        marslab_files = [path]
    for marslab_file in marslab_files:
        try:
            # NOTE: there was a bunch of code here about like 'titular title'
            #  and manipulating the 'NAME' field but it wasn't actually getting
            #  passed to anything, so I deleted it. But it's possible that the
            #  fact it wasn't doing anything was a regression and we should
            #  replace the _intended_ functionality, whatever that was...

            # TODO: I don't think 'marslab' and 'marslab_spectra' should be
            #  loaded separately
            marslab = pd.read_csv(marslab_file)
            plot_fn = str(marslab_file).replace(".csv", ".png")
            print("Writing " + plot_fn)
            marslab_spectra = load_spectra(str(marslab_file))
            if "SOLAR_ELEVATION" not in marslab.columns:
                solar_elevation = None
            else:
                solar_elevation = marslab["SOLAR_ELEVATION"].iloc[0]
            if "UNITS" in marslab.columns:
                units = marslab["UNITS"].iloc[0]
            else:
                units = None
            pplot_utils.pretty_plot(
                marslab_spectra,
                solar_elevation=solar_elevation,
                units=units,
                plot_fn=plot_fn,
                underplot=None,
                offset=offset,
                plt_bayer=plt_bayer,
                roi_labels=roi_labels,
                annotation=annotation,
                width_sf=width_sf,
                height_sf=height_sf,
                normalize=normalize
            )
        except (KeyError, ValueError) as error:
            if debug is True:
                raise
            print(
                "couldn't plot "
                + str(marslab_file)
                + ": "
                + str(type(error))
                + " "
                + str(error)
            )


def pplot_run_hook():
    try:
        import fire

        return fire.Fire(do_pplot)
    except ImportError:
        print(
            "'fire' package not found. Did you "
            "forget to activate a virtual environment?"
        )
        sys.exit(1)

