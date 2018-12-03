#probably need to update these imports to the new name, PyHAT
import PySAT.libpysat.spectral.spectral_data as spectral_data
import PySAT.libpysat.regression.cv as cv
import pandas as pd
import numpy as np


best_settings = []
models = []

#specify the list of elements that we want to develop models for
elements = ['Na2O','K2O']

#set output path
outpath = r"C:\Users\rbanderson\Documents\Projects\MSL\ChemCam\Database\Alkalis\full_db_mars_corrected_dopedTiO2_highAlk1 - 1\Feb2018\\"

#set source data path
sourcepath = outpath

# set the composition ranges to consider. This allows cross validation to test out all of the different submodels
# that will be used in the eventual calibration
yranges = [[0,100], [0,1.5], [1,6], [5,100]]

#set the list of methods to try
methods = [
            'PLS',
            'Elastic Net',
            'LASSO',
            'Ridge',
            'GP',
            'OLS',
            'BRR',
            'LARS',
            'OMP'
            ]
#create a log-spaced set of alpha values to consider - this is used by a number of different methods
alphas = np.logspace(np.log10(0.000000001), np.log10(0.001),
                                         num=20)

# set the parameters to consider for each method. Each parameter is in a list, so more than one value can be specified.
# The cross validation will evaluate every possible permutation of parameters, so beware of adding too many,
# especially for slower methods

params = {'PLS':{'n_components': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                              'scale': [False]},
            'Elastic Net':{ 'alpha': alphas,
                                'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                                'fit_intercept': [True, False],
                                'normalize': [False],
                                'precompute': [True],
                                'max_iter': [1000],
                                'copy_X': [True],
                                'tol': [0.0001],
                                'warm_start': [True],
                                'positive': [True, False],
                                'selection': ['random']},
           'GP':{   'reduce_dim': ['PCA'],
                    'n_components': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                    'regr': ['linear'],
                    'corr': ['squared_exponential'],
                    'storage_mode': ['light'],
                    'verbose': [True],
                    'theta0': [0.1],
                    'normalize': [True],
                    'optimizer': ['fmin_cobyla'],
                    'random_start': [5]},
           'OLS':{  'fit_intercept': [True, False]},
           'LASSO':{'alpha': list(alphas),
                    'fit_intercept': [True, False],
                    'max_iter': [1000],
                    'tol': [0.0001],
                    'positive': [True, False],
                    'selection': ['random']},
           'BRR':{  'n_iter': [300],
                    'tol': [0.001],
                    'alpha_1': [0.000001],
                    'alpha_2': [0.000001],
                    'lambda_1': [0.000001],
                    'lambda_2': [0.000001],
                    'compute_score': [False],
                    'fit_intercept': [True, False],
                    'normalize': [False],
                    'copy_X': [True],
                    'verbose': [True]},

           'LARS':{ 'fit_intercept': [True, False],
                    'verbose': [True],
                    'normalize': [False],
                    'precompute': ['auto'],
                    'n_nonzero_coefs': [1,5,10,50,100,200,400,800],
                    'copy_X': [True],
                    'fit_path': [False],
                    'positive': [True, False]},
           'OMP':{'n_nonzero_coefs': [1,5,10,50,100,200,400,800],
                  'fit_intercept': [True, False],
                  'normalize': [False],
                  'precompute': ['auto'],
                  },
           'Ridge':{'alpha': list(alphas),
                    'copy_X': [True],
                    'fit_intercept': [True, False],
                    'max_iter': [None],
                    'normalize': [False],
                    'solver': ['auto'],
                    'tol': [0.001],
                    'random_state': [None]}
            }

#Step through each of the elements listed
for element in elements:
    #get the data normalized to 3 (each spectrometer separately) or to 1 (sum of all spectrometers)
    filenames = [sourcepath+"data1_"+element+"_train.csv",sourcepath+"data3_"+element+"_train.csv"]

    #Step through each data file
    for file in filenames:
        # Read the data in, make it a "spectral_data" object
        data = spectral_data.spectral_data(pd.read_csv(file, header=[0, 1], verbose=True))

        #Step through each regression method
        for method in methods:
            paramstemp = params[method] #get the parameters for this method
            cv_obj = cv.cv(paramstemp) #set up cross validation across all permutations of parameters

            #do the cross validation for each composition range
            for yrange in yranges:
                #set up an output file name that specifies the current composition range being considered
                if file == filenames[0]:
                    outfile_root = 'data1_' + element + '_' + str(yrange[0])+'-'+str(yrange[1])+'_'
                if file == filenames[1]:
                    outfile_root = 'data3_' + element + '_' + str(yrange[0])+'-'+str(yrange[1])+'_'

                #apply yrange to filter the compositions used in the regression
                y = np.array(data.df[('comp',element)])
                match = np.squeeze((y > yrange[0]) & (y < yrange[1]))
                datatemp = spectral_data.spectral_data(data.df.ix[match])

                #do the actual cross validation for the current method, element, and yrange
                datatemp.df, cv_results, cvmodels, cvmodelkeys = cv_obj.do_cv(datatemp.df, xcols='wvl',
                                                                              ycol=('comp', element),
                                                                              yrange=yrange, method=method)
                # save cross validation results
                filename = outpath + outfile_root + method + '_CV.csv'
                cv_results['cv'].to_csv(filename)

                # save cross validation predictions
                filename = outpath + outfile_root + method + '_CV_results.csv'
                temp = datatemp.df.drop('wvl', axis=1, level=0)
                temp.to_csv(filename)

            pass