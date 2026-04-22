# signle fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from diffpy.srfit.fitbase import FitResults
from fit_functions import make_recipe
from procedure import save_fit_results
import os

data_folder = r"C:/Users/plexa/OneDrive/Bayreuth/LSS5-LSS20/data/"
LSS_cif_path = r"C:\Users\plexa\OneDrive\Bayreuth\LSS5-LSS20\diffPy\CIFs/LaSr10ScO3_mp-31116_symmetrized.cif"
RP_cif_path = r"C:\Users\plexa\OneDrive\Bayreuth\LSS5-LSS20\diffPy\CIFs/SrLa2Sc2O7_Fmmm.cif"

LSS10_cif_path = r"C:\Users\plexa\OneDrive\Bayreuth\LSS5-LSS20\diffPy\CIFs/LaSr10ScO3_vesta.cif"
LSS15_cif_path = r"C:\Users\plexa\OneDrive\Bayreuth\LSS5-LSS20\diffPy\CIFs/LaSr10ScO3_vesta.cif"
LSS20_cif_path = r"C:\Users\plexa\OneDrive\Bayreuth\LSS5-LSS20\diffPy\CIFs/LaSr10ScO3_vesta.cif"

LSS10_data_path = data_folder + "LSS10_dry_ext.gr"
LSS15_data_path = data_folder + "LSS15_dry_ext.gr"
LSS20_data_path = data_folder + "LSS20_dry_ext.gr"
# CONFIGS
CONFIG = {
    # Run settings
    "RUN_PARALLEL": True,
    "REFINE_ADP": True,
    "REFINE_COORD": True,

    # Data and Instrument
    "PDF_RMIN": 1.8,
    "PDF_RMAX": 60,
    "PDF_RSTEP": 0.01,
    "QMAX": 17.13,
    "QMIN": 0.1,
    "QDAMP_I": 0.011,
    "QBROAD_I": 0.01,

    # Structure (single phase)
    "SCALE_I": 0.2,
    "LATTICE_A": 5.8,
    "LATTICE_B": 8.2,
    "LATTICE_C": 5.7,
    "BISO_I": 0.005,
    "DELTA2_I": 2,

    # Structure (multi-phase)
    "SCALE": [0.2, 0.05],
    "DELTA2": [2, 2]
}

datasets = {
    "LSS10": LSS10_data_path,
    "LSS15": LSS15_data_path,
    "LSS20": LSS20_data_path
}

cifs = {
    "LSS10": LSS10_cif_path,
    "LSS15": LSS15_cif_path,
    "LSS20": LSS20_cif_path
}
result_path = "C:/Users/plexa/OneDrive/Bayreuth/LSS5-LSS20/diffPy/multiple_datasets/fit_results/"

def run():
    t_subgroups = [
        "Pnma",
        "Pnmm",
        "Pbam",
        "Pmma",
        "Pna21",
        "Pn21a",
        "Pmc21",
        "Pmm2",
        "P2/m",
        "Pc",
        "Pm",
        "P21",
        "P2",
        "P1",
    ]
    k_subgroups = [
        "Pnma",
        "Pnnm",
        "Pbcm",
        "P21/c",
        "P21/m",
        "C2/c",
        "Pc",
        "C2",
        "P21",
        "P1",
    ]
    for name, path in datasets.items():
        recipe = None
        space_groups = t_subgroups
        print(f"Fitting dataset: {name}")
        print("Starting single phase fit...")
        LSS_cif = cifs[name]
        for sg in space_groups:
            recipe = make_recipe(LSS_cif, path, config=CONFIG,space_group=sg)  # Call the function to create the fit recipe

            r = recipe.PDFfit.profile.x
            g = recipe.PDFfit.profile.y
            recipe.fithooks[0].verbose = 0
            recipe.fix("all")
            tags = ["lat", "scale", "all"]
            for tag in tags:
                recipe.free(tag)
                result = least_squares(recipe.residual, recipe.values, x_scale="jac", bounds=recipe.bounds2,verbose=0, max_nfev=200)
                print(f"Refined {tag}: Number of function evaluations: {result.nfev}")
            res = FitResults(recipe)
            if res.rw > 0.17:
                print(f"Space group: {sg} / Rw = {res.rw} - reducing symmetry")
            else:
                print(f"Space group: {sg} / Rw = {res.rw} - threshold reached")
                break

        res = FitResults(recipe)
        result_path_dir = result_path + f"{name}/"
        if not os.path.exists(result_path_dir):
            os.makedirs(result_path_dir)
        res.saveResults(result_path_dir + f"{name}_fit_results.res")
        res.printResults()
        # Write the fitted data to a file.
        if True:
            save_fit_results(recipe, result_path_dir)

