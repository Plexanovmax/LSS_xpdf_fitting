import os
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_res(filepath):
    data = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            key = parts[0]

            # scalar values like Rw, Chi2 etc.
            if key in ["Rw", "Chi2", "Reduced", "Residual"]:
                try:
                    data[key] = float(parts[-1])
                except:
                    pass

            # variables like Delta2 etc.
            if "+/-" in line:
                try:
                    value = float(parts[1])
                    data[key] = value
                except:
                    pass

    return data


def parse_cif(filepath):
    data = {}
    atoms = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # lattice parameters
        if line.startswith("_cell_length_a"):
            data["a"] = float(line.split()[-1])
        elif line.startswith("_cell_length_b"):
            data["b"] = float(line.split()[-1])
        elif line.startswith("_cell_length_c"):
            data["c"] = float(line.split()[-1])
        elif line.startswith("_cell_angle_alpha"):
            data["alpha"] = float(line.split()[-1])
        elif line.startswith("_cell_angle_beta"):
            data["beta"] = float(line.split()[-1])
        elif line.startswith("_cell_angle_gamma"):
            data["gamma"] = float(line.split()[-1])

        elif line.startswith("_symmetry_space_group_name_H-M"):
            data["space_group"] = line.split()[-1].strip("'")

        # atom loop
        elif line.startswith("loop_"):
            headers = []
            i += 1
            while lines[i].strip().startswith("_"):
                headers.append(lines[i].strip())
                i += 1

            # read atom rows
            while i < len(lines) and lines[i].strip():
                parts = lines[i].split()
                if len(parts) < len(headers):
                    break

                atom_name = parts[0]
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])

                atoms[atom_name] = {"x": x, "y": y, "z": z}
                i += 1
            continue

        i += 1

    # compute volume (general triclinic)
    if all(k in data for k in ["a", "b", "c", "alpha", "beta", "gamma"]):
        a, b, c = data["a"], data["b"], data["c"]
        alpha = np.radians(data["alpha"])
        beta = np.radians(data["beta"])
        gamma = np.radians(data["gamma"])

        V = a * b * c * np.sqrt(
            1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2
            + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
        )
        data["V"] = V

    return data, atoms


def extract_fit_index(folder_name):
    match = re.match(r"fit_([0-9]*\.?[0-9]+)_\d+", folder_name)
    return float(match.group(1)) if match else None


def collect_data(base_path, parameters):
    results = []

    for folder in os.listdir(base_path):
        full_path = os.path.join(base_path, folder)

        if not os.path.isdir(full_path):
            continue

        n = extract_fit_index(folder)
        if n is None:
            continue

        res_data = {}
        cif_data = {}
        atoms = {}

        for file in os.listdir(full_path):
            filepath = os.path.join(full_path, file)

            if file.endswith(".res"):
                res_data = parse_res(filepath)
            elif file.endswith(".cif"):
                cif_data, atoms = parse_cif(filepath)

        combined = {"n": n}

        for param in parameters:
            if "_" in param and param.endswith(("x", "y", "z")):
                # atom coordinate like La1_x
                atom, coord = param.split("_")
                combined[param] = atoms.get(atom, {}).get(coord, np.nan)

            elif param in res_data:
                combined[param] = res_data[param]

            elif param in cif_data:
                combined[param] = cif_data[param]

            else:
                combined[param] = np.nan

        results.append(combined)

    return sorted(results, key=lambda x: x["n"])


def plot_results(data, parameters, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    n_vals = [d["n"] for d in data]

    for param in parameters:
        y_vals = [d[param] for d in data]

        plt.figure()
        plt.plot(n_vals, y_vals, marker='o')
        plt.xlabel("n (from fit_n_m)")
        plt.ylabel(param)
        plt.title(param)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{param}.png"))
        plt.close()


def analyze_fits(base_path, parameters):
    data = collect_data(base_path, parameters)

    output_dir = os.path.join(base_path, "plots")
    plot_results(data, parameters, output_dir)

    print(f"Plots saved in: {output_dir}")