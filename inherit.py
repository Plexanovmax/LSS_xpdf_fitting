from diffpy.srfit.fitbase import FitRecipe

class MyFitRecipe(FitRecipe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.space_group = None