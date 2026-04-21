import matplotlib.pyplot as plt
import numpy as np
from diffpy.srfit.fitbase import FitResults
import os
import spglib
from sqlalchemy import true


def get_crystal_system(number):
    if number <= 2:
        return "triclinic"
    elif number <= 15:
        return "monoclinic"
    elif number <= 74:
        return "orthorhombic"
    elif number <= 142:
        return "tetragonal"
    elif number <= 167:
        return "trigonal"
    elif number <= 194:
        return "hexagonal"
    else:
        return "cubic"


def rewrite_spacegroup(cif_path, sg_name):
    # normalize input (spglib is picky about formatting)
    sg_name = sg_name.replace(" ", "")

    # find matching space group
    sg_number = None
    for num in range(1, 531):
        sg_type = spglib.get_spacegroup_type(num)
        if sg_type["international_short"].replace(" ", "") == sg_name:
            sg_number = sg_type["number"]
            sg_symbol = sg_type["international_short"]
            break

    if sg_number is None:
        print(f"Space group '{sg_name}' not found in spglib")
        sg_symbol = sg_name

    sg_number = 1
    cell_setting = get_crystal_system(sg_number)

    # rewrite CIF
    with open(cif_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if "_symmetry_space_group_name_H-M" in line:
            new_lines.append(f"_symmetry_space_group_name_H-M    '{sg_symbol}'\n")
        elif "_symmetry_Int_Tables_number" in line:
            new_lines.append(f"_symmetry_Int_Tables_number       {sg_number}\n")
        elif "_symmetry_cell_setting" in line:
            new_lines.append(f"_symmetry_cell_setting           {cell_setting}\n")
        else:
            new_lines.append(line)

    with open(cif_path, "w") as f:
        f.writelines(new_lines)


def save_fit_results(recipe,
                     result_folder_path=r"C:/Users/plexa/OneDrive/Bayreuth/LSS5-LSS20/diffPy/LSS20/fit_results/"):
    """Save the fit results to a file"""

    r = recipe.PDFfit.profile.x
    g = recipe.PDFfit.profile.y
    recipe.fithooks[0].verbose = 0
    gcalc = recipe.PDFfit.profile.ycalc
    res = FitResults(recipe)

    diffzero = -0.95 * max(g) * np.ones_like(g)  # offset the difference curve
    diff = g - gcalc + diffzero

    # Write the fitted data to a file.

    profile = recipe.PDFfit.profile
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    profile.savetxt(result_folder_path + f"plot.fit")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.scatter(r, g, label="Experimental PDF")
    ax.plot(r, gcalc, label="Fitted PDF", color="red")
    ax.plot(r, diff, label="Difference")

    ax.set_xlabel("r (Angstrom)")
    ax.set_ylabel("G(r)")
    ax.set_title("PDF Fit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(result_folder_path + f"plot.png")
    plt.close()

    # recipe.contribution_name.generator_name.stru
    for con_name, con in recipe._contributions.items():
        for gen_name, gen in con._generators.items():
            # This accesses every generator in every contribution
            print(f"Contribution: {con_name}, Generator: {gen_name}")
            cif_path = result_folder_path + f"{gen_name}.cif"
            gen.stru.write(cif_path, 'cif')
            if recipe.space_group[gen_name] is not None:
                rewrite_spacegroup(cif_path, recipe.space_group[gen_name])


def create_ranges(start, stop, step, multiplyer = 2):
    length = multiplyer*step
    ranges = []
    n = start
    m = start + length
    while True:
        ranges.append((n, m))
        n += step
        m += step
        if n > stop:
            break

    return ranges

