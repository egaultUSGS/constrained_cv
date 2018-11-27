import PySAT.libpysat.spectral.spectral_data as spectral_data
import PySAT.libpysat.regression.cv as cv
import PySAT.libpysat.plotting.plots as plots
import pandas as pd
import numpy as np


best_settings = []
models = []
elements = ['Na2O','K2O']
outpath = r"C:\Users\rbanderson\Documents\Projects\MSL\ChemCam\Database\Alkalis\full_db_mars_corrected_dopedTiO2_highAlk1 - 1\Feb2018\\"
yranges = [[0,100], [0,1.5], [1,6], [5,100]]

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
alphas = np.logspace(np.log10(0.000000001), np.log10(0.001),
                                         num=20)

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

for element in elements:
    filenames = [r"C:\Users\rbanderson\Documents\Projects\MSL\ChemCam\Database\Alkalis\full_db_mars_corrected_dopedTiO2_highAlk1 - 1\Feb2018\data1_"+element+"_train.csv",
    r"C:\Users\rbanderson\Documents\Projects\MSL\ChemCam\Database\Alkalis\full_db_mars_corrected_dopedTiO2_highAlk1 - 1\Feb2018\data3_"+element+"_train.csv"]
    for file in filenames:

        data = spectral_data.spectral_data(pd.read_csv(file, header=[0, 1], verbose=True))

        for method in methods:
            paramstemp = params[method]
            cv_obj = cv.cv(paramstemp)

            for yrange in yranges:
                if file == filenames[0]:
                    outfile_root = 'data1_' + element + '_' + str(yrange[0])+'-'+str(yrange[1])+'_'
                if file == filenames[1]:
                    outfile_root = 'data3_' + element + '_' + str(yrange[0])+'-'+str(yrange[1])+'_'

                #apply yrange
                y = np.array(data.df[('comp',element)])
                match = np.squeeze((y > yrange[0]) & (y < yrange[1]))
                datatemp = spectral_data.spectral_data(data.df.ix[match])

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