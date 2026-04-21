import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from diffpy.srfit.fitbase import FitResults
from fit_functions import make_recipe
from procedure import save_fit_results

#CONFIGS
CONFIG = {
    # Run settings
    "RUN_PARALLEL": True,
    "REFINE_ADP": True,
    "REFINE_COORD": True,
    
    # Data and Instrument
    "PDF_RMIN": 1.8,
    "PDF_RMAX": 10,
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

def single_phase_fit(data_path, config=CONFIG, plot=True, save_results=True, iterations=100):
    print("Starting single phase fit...")
    RP_cif = r"C:\Users\plexa\OneDrive\Bayreuth\LSS5-LSS20\diffPy\CIFs/SrLa2Sc2O7_Fmmm.cif"
    LSS_cif = r"C:\Users\plexa\OneDrive\Bayreuth\LSS5-LSS20\diffPy\CIFs/LaSr10ScO3_vesta.cif"
    recipe = make_recipe(LSS_cif, data_path, config=config,space_group='P21') # Call the function to create the fit recipe


    r = recipe.PDFfit.profile.x
    g = recipe.PDFfit.profile.y
    recipe.fithooks[0].verbose = 0
    result = least_squares(recipe.residual, recipe.values, x_scale="jac",verbose=0, max_nfev=iterations)
    print(f"Number of function evaluations: {result.nfev}")
    gcalc = recipe.PDFfit.profile.ycalc
    res = FitResults(recipe)
    res.printResults()



    diffzero = -0.95 * max(g) * np.ones_like(g) # offset the difference curve
    diff = g - gcalc + diffzero

    # Write the fitted data to a file.
    if save_results:
        name = 'LSS10'
        result_folder_path = f"C:/Users/plexa/OneDrive/Bayreuth/LSS5-LSS20/diffPy/{name}/fit_results/"
        save_fit_results(recipe, result_folder_path)
        res.saveResults(result_folder_path + f"{name}_fit_results.res")

    if plot:
        plt.figure(figsize=(15, 6))
        plt.scatter(r, g, label="Experimental PDF")
        plt.plot(r, gcalc, label="Fitted PDF", color = "red")
        plt.plot(r, diff, label="Difference")

        plt.xlabel("r (Angstrom)")
        plt.ylabel("G(r)")
        plt.title("LSS10 PDF Fit")
        plt.legend()
        plt.show()

data_folder = r"C:/Users/plexa/OneDrive/Bayreuth/LSS5-LSS20/data/"
data_path = data_folder + "LSS10_dry_ext.gr"
def run(path = data_path):
    print("Running single phase fit...")
    single_phase_fit(path, iterations = 100)