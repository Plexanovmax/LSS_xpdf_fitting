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



def make_recipe(cif_path, dat_path, config, space_group = None):

    # 1. Get structural info
    p_cif = get_parser('cif')
    measurement_info = Metadata(dat_path)
    stru = p_cif.parse_file(cif_path)
    if space_group is None:
        sg = p_cif.spacegroup.short_name
    else:
        sg = space_group
    print("Space group:", sg)

    # 2. Get PDF data
    profile = Profile()
    parser = PDFParser()
    parser.parseFile(dat_path)
    profile.loadParsedData(parser)
    profile.setCalculationRange(
        xmin=config["PDF_RMIN"], xmax=config["PDF_RMAX"], dx=config["PDF_RSTEP"]
    )

    # 3. Create a PDFGenerator to generate a simulated PDF from the structure
    generator_name = 'generatedPDF'
    genpdf = PDFGenerator(generator_name)
    genpdf.setStructure(stru, periodic=True)

    # 4. Create a Fit Contribution object
    contribution = FitContribution("PDFfit")
    contribution.addProfileGenerator(genpdf)

    # 5. Add the Profile to the contribution
    contribution.setProfile(profile, xname="r")

    # 6. Set the equation used to combine the simulated PDF with the experimental PDF
    contribution.setEquation("s1*generatedPDF")

    # 7. Create a Fit Recipe which turns our physics model into a mathematical recipe
    recipe = MyFitRecipe()
    recipe.addContribution(contribution)
    recipe.space_group = {
        generator_name: sg
    }

    # 8. Initialize the experimental parameters
    recipe.PDFfit.generatedPDF.qdamp.value = config["QDAMP_I"]
    recipe.PDFfit.generatedPDF.qbroad.value = config["QBROAD_I"]
    recipe.PDFfit.generatedPDF.setQmin(measurement_info.qmin)
    recipe.PDFfit.generatedPDF.setQmax(measurement_info.qmax)

    # 9. Add scaling factor to the recipe
    recipe.addVar(contribution.s1, value=config["SCALE_I"], tag="scale")

    # 10. Add structural params to recipe, constraining them to the space group
    atoms = genpdf.phase.getScatterers()

    # Constrain to space group BUT exclude Sr atoms (they will be constrained to La instead)
    scatterers_to_constrain = [a for a in atoms if a.element.title() != "Sr"]
    spacegroupparams = constrainAsSpaceGroup(
        genpdf.phase, sg, scatterers=scatterers_to_constrain
    )

    # Add lattice variables
    for par in spacegroupparams.latpars:
        recipe.addVar(par, fixed=False, tag="lat")

    # Build mappings for coordinate and ADP parameters
    filtered_scatterers = scatterers_to_constrain
    

    # xyzpar_map = {}
    # for par in spacegroupparams.xyzpars:
    #     name = par.name
    #     if "_" in name:
    #         coord, idx_str = name.rsplit("_", 1)
    #         idx = int(idx_str)
    #         actual_atom = filtered_scatterers[idx]
    #         actual_idx = atoms.index(actual_atom)
    #         if actual_idx not in xyzpar_map:
    #             xyzpar_map[actual_idx] = {}
    #         xyzpar_map[actual_idx][coord] = par
    #
    # adp_map = {}
    # for par in spacegroupparams.adppars:
    #     name = par.name
    #     if "_" in name:
    #         _, idx_str = name.rsplit("_", 1)
    #         idx = int(idx_str)
    #         actual_atom = filtered_scatterers[idx]
    #         actual_idx = atoms.index(actual_atom)
    #         adp_map[actual_idx] = par

    # Add coordinate variables from the space group parameters
    for par in spacegroupparams.xyzpars:
        name = par.name
        if "_" in name:
            coord, idx_str = name.rsplit("_", 1)
            idx = int(idx_str)
            atom = filtered_scatterers[idx]
            if atom.element.title() != "Sr":
                recipe.addVar(par, fixed=(not config["REFINE_COORD"]), tag="xyz")

    # Add ADP parameters from the space group parameters
    for par in spacegroupparams.adppars:
        recipe.addVar(par, fixed=(not config["REFINE_ADP"]), tag="adp").boundRange(lb=0.0)

    # Constrain Sr atoms to corresponding La atoms for coordinates and ADPs
    for atom in atoms:
        if atom.element.title() == "Sr":
            la_name = atom.name.replace("Sr", "La")
            la_idx = None
            for i, a in enumerate(atoms):
                if a.name == la_name:
                    la_idx = i
                    break

            if la_idx is not None:
                recipe.constrain(atom.x, atoms[la_idx].x)
                recipe.constrain(atom.y, atoms[la_idx].y)
                recipe.constrain(atom.z, atoms[la_idx].z)
                recipe.constrain(atom.Biso, atoms[la_idx].Biso)

    recipe.addVar(
        genpdf.delta2,
        fixed=True,
        name="Delta2",
        value=config["DELTA2_I"],
        tag="d2",
    )

    # 11. Add instrumental Qdamp and Qbroad parameters to the recipe
    # recipe.addVar(
    #     genpdf.qdamp,
    #     fixed=True,
    #     name="Calib_Qdamp",
    #     value=config["QDAMP_I"],
    #     tag="inst",
    # ).boundRange(lb=0.0, ub=6)
    # recipe.addVar(
    #     genpdf.qbroad,
    #     fixed=True,
    #     name="Calib_Qbroad",
    #     value=config["QBROAD_I"],
    #     tag="inst",
    # )
    return recipe