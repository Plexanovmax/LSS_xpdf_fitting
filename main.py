#from setups import box_car_var_sg as setup
from setups import single_fit as sf
#from setups import multiple_datasets_var_sg as setup
#from setups import multiple_datasets as setup
#from analysis import analyze_fits
#from setups import box_car as setup
from setups import two_sim_fit as tsf



if __name__ == "__main__":
    #tsf.run()
    sf.run()
    # setup.run()
    # path1 = r'C:\Users\plexa\OneDrive\Bayreuth\LSS5-LSS20\diffPy\multiple_datasets\fit_results\boxcar_fits\LSS10'
    # path2 = r'C:\Users\plexa\OneDrive\Bayreuth\LSS5-LSS20\diffPy\multiple_datasets\fit_results\boxcar_fits\LSS15'
    # path3 = r'C:\Users\plexa\OneDrive\Bayreuth\LSS5-LSS20\diffPy\multiple_datasets\fit_results\boxcar_fits\LSS20'
    # folders = [
    #     path1,path2,path3
    # ]
    # params = ["Rw", "La1_x", "Delta2", "space_group", "a", "V"]
    # for path in folders:
    #     analyze_fits(path, params)