import numpy as np
from scipy.optimize import least_squares
from diffpy.srfit.fitbase import FitContribution, FitRecipe, FitResults, Profile
from inherit import MyFitRecipe
from diffpy.srfit.pdf import PDFParser, PDFGenerator
from diffpy.structure.parsers import get_parser
from diffpy.srfit.structure import constrainAsSpaceGroup
from pathlib import Path


class Metadata:
    EXPECTED_HEADER = "# xPDFsuite Configuration #"

    def __init__(self, file_path):
        self.file_path = Path(file_path)

        # --- PDF section fields ---
        self.wavelength = None
        self.dataformat = None
        self.inputfile = None
        self.backgroundfile = None
        self.mode = None
        self.bgscale = None
        self.composition = None
        self.outputtype = None
        self.qmaxinst = None
        self.qmin = None
        self.qmax = None
        self.rmax = None
        self.rmin = None
        self.rstep = None
        self.rpoly = None

        # --- Misc section fields ---
        self.inputdir = None
        self.savedir = None
        self.backgroundfilefull = None

        self._validate_file()
        self._parse_file()

    def _validate_file(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        with self.file_path.open("r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        if first_line != self.EXPECTED_HEADER:
            raise ValueError(
                f"Invalid header: expected '{self.EXPECTED_HEADER}', got '{first_line}'"
            )

    def _parse_file(self):
        with self.file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line or line == self.EXPECTED_HEADER:
                    continue

                # Skip section headers like [PDF], [Misc]
                if line.startswith("[") and line.endswith("]"):
                    continue

                if "=" in line:
                    key, value = map(str.strip, line.split("=", 1))

                    # Only set attributes that exist
                    if hasattr(self, key):
                        setattr(self, key, self._convert_value(value))

    def _convert_value(self, value):
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def __repr__(self):
        return f"Metadata({self.file_path})"



def make_recipe(cif_paths, dat_path, config, space_group = None):

    # 1. Get structural info
    structures = list()
    SGs = list()
    for i, cif_path in enumerate(cif_paths):
        p_cif = get_parser('cif')
        stru = p_cif.parse_file(cif_path)
        structures.append(stru)
        if space_group is None:
            sg = p_cif.spacegroup.short_name
            SGs.append(sg)
        else:
            sg = space_group[i]
            SGs.append(sg)
        print(f"Space group #{i}: {sg}")

    # 2. Get PDF data
    measurement_info = Metadata(dat_path)
    profile = Profile()
    parser = PDFParser()
    parser.parseFile(dat_path)
    profile.loadParsedData(parser)
    profile.setCalculationRange(
        xmin=config["PDF_RMIN"], xmax=config["PDF_RMAX"], dx=config["PDF_RSTEP"]
    )

    # 3. Create a PDFGenerator to generate a simulated PDF from the structure and contributions
    PDFs = list()
    contribution = FitContribution("PDFfit") # Give the contribution a name
    for i, stru in enumerate(structures):
        genpdf = PDFGenerator(f"PDF_{SGs[i]}") # Give the generator a name
        genpdf.setStructure(stru, periodic=True) # Give the generator the structure
        PDFs.append(genpdf)
        contribution.addProfileGenerator(genpdf) # Add the PDFGenerator to the contribution

    # 4. If you have a multi-core computer (you probably do), run your refinement in parallel!
    if config["RUN_PARALLEL"]:
        try:
            import psutil
            import multiprocessing
            from multiprocessing import Pool
        except ImportError:
            print("\nYou don't appear to have the necessary packages for parallelization")
        syst_cores = multiprocessing.cpu_count()
        cpu_percent = psutil.cpu_percent()
        avail_cores = np.floor((100 - cpu_percent) / (100.0 / syst_cores))
        ncpu = int(np.max([1, avail_cores]))
        pool = Pool(processes=ncpu)
        for generator in PDFs:
            generator.parallel(ncpu=ncpu, mapfunc=pool.map)

    # 5. Add the Profile to the contribution
    contribution.setProfile(profile, xname="r")

    # 6. Set the equation used to combine the simulated PDF with the experimental PDF
    string_equation = ''
    for i, stru in enumerate(structures):
        if len(string_equation)> 0:
            string_equation += '+'
        string_equation += f's{i}*PDF_{SGs[i]}'
    contribution.setEquation(string_equation) # scaling factor for the simulated PDF

    # 7. Create a Fit Recipe which turns our physics model into a mathematical recipe
    recipe = MyFitRecipe()
    recipe.addContribution(contribution)
    for i, stru in enumerate(structures):
        if recipe.space_group is None:
            recipe.space_group = {}
        recipe.space_group[f'PDF_{SGs[i]}'] = SGs[i]

    # 8. Initialize the experimental parameters
    for i in range(len(structures)):
        contribution_obj = getattr(recipe, "PDFfit")
        generator_obj = getattr(contribution_obj, f"PDF_{SGs[i]}")
        generator_obj.qdamp.value = config["QDAMP_I"]
        generator_obj.qbroad.value = config["QBROAD_I"]
        generator_obj.setQmin(measurement_info.qmin)
        generator_obj.setQmax(measurement_info.qmax)


    # 9. Add scaling factor to the recipe
    for i, stru in enumerate(structures):
        contrib_obj = getattr(contribution, f"s{i}")
        recipe.addVar(contrib_obj, value=config["SCALE"][i], tag=f"scale_{i}").boundRange(lb=0.0)

    # 10. Add structural params to recipe, constraining them to the space group
    for i, sg in enumerate(SGs):
        atoms = PDFs[i].phase.getScatterers()
        filtered_scatterers = [a for a in atoms if a.element.title() != "Sr"]
        spacegroupparams = constrainAsSpaceGroup(
            PDFs[i].phase, sg, scatterers=filtered_scatterers
        )
        # Add lattice variables
        for par in spacegroupparams.latpars:
            recipe.addVar(par, name=f"{sg}_{par.name}", fixed=False, tag="lat")
        # Build mappings for coordinate and ADP parameters


        # Add coordinate variables from the space group parameters
        for par in spacegroupparams.xyzpars:
            name = par.name
            if "_" in name:
                coord, idx_str = name.rsplit("_", 1)
                idx = int(idx_str)
                atom = filtered_scatterers[idx]
                if atom.element.title() != "Sr":
                    recipe.addVar(par, name=f"{sg}_{par.name}", fixed=(not config["REFINE_COORD"]), tag="xyz")

        # Add ADP parameters from the space group parameters
        for par in spacegroupparams.adppars:
            recipe.addVar(par, name=f"{sg}_{par.name}", fixed=(not config["REFINE_ADP"]), tag="adp").boundRange(lb=0.0)

        # Constrain Sr atoms to corresponding La atoms for coordinates and ADPs
        for atom in atoms:
            if atom.element.title() == "Sr":
                la_name = atom.name.replace("Sr", "La")
                la_idx = None
                for j, a in enumerate(atoms):
                    if a.name == la_name:
                        la_idx = j
                        break

                if la_idx is not None:
                    recipe.constrain(atom.x, atoms[la_idx].x)
                    recipe.constrain(atom.y, atoms[la_idx].y)
                    recipe.constrain(atom.z, atoms[la_idx].z)
                    recipe.constrain(atom.Biso, atoms[la_idx].Biso)

        recipe.addVar(
            PDFs[i].delta2,
            fixed=True,
            name=f"{sg}_delta2",
            value=config["DELTA2_I"],
            tag="d2",
        )


    return recipe