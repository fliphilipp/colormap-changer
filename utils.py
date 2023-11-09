import numpy as np
import pandas as pd

# function for robust (iterative) nonparametric regression (to fit surface and bed of lake)
def robust_npreg(df_fit, n_iter=10, poly_degree=1, frac_x=0.15, resolutions=[5,1], stds=[10,6], 
                 range_vweight = 50.0, full=False, init=None):

    h_list = []
    x_list = []
    len_x = df_fit.x.max() - df_fit.x.min()
    resols = np.linspace(resolutions[0], resolutions[1], n_iter)
    n_stds = np.hstack((np.linspace(stds[0], stds[1], n_iter-1), stds[1]))
    minx = df_fit.x.min()
    maxx = df_fit.x.max()
    
    # take into account initial guess, if specified (needs to be dataframe with columns 'x' and 'h')
    if (init is not None) and (len(init) > 0): 
        df_fit['y_fit'] = np.interp(df_fit.x, init.x, init.y, left=np.nan, right=np.nan)
        vert_weight = (1.0 - np.clip((np.abs(df_fit.y-df_fit.y_fit)/range_vweight),0,1)**3 )**3
        vert_weight[np.isnan(vert_weight)] = 0.01
        df_fit['vert_weight'] = vert_weight
    else: 
        df_fit['vert_weight'] = 1.0
    
    for it in range(n_iter):
        
        res = resols[it]
        n_std = n_stds[it]
        evaldf = pd.DataFrame(np.arange(minx,maxx+res/2,step=res),columns=['x'])
        y_arr = np.full_like(evaldf.x,fill_value=np.nan)
        stdev_arr = np.full_like(evaldf.x,fill_value=np.nan)
        df_fit_nnz = df_fit.copy()
    
        # for every point at which to evaluate local fit
        for i,x in enumerate(evaldf.x):
            
            # get the start and end locations to fit
            xstart = x - len_x*frac_x/2
            xend = x + len_x*frac_x/2
            idx_start = int(np.clip(xstart, 0, None))
            idx_end = int(np.clip(xend, None, len(df_fit_nnz)-1))
            # print(x, xstart, xend, idx_start, idx_end)
    
            # make a data frame with the data for the fit
            dfi = df_fit_nnz.iloc[idx_start:idx_end].copy()
    
            # tricube weights for x distance from evaluation point
            maxdist = np.nanmax(np.abs(dfi.x - x))
            dfi['weights'] = (1.0-(np.abs(dfi.x-x)/(1.00001*maxdist))**3)**3
    
            if (init is not None) | (it > 0):  # vertical weights are only available after first iteration or with initial guess
                dfi.weights *= dfi.vert_weight
    
            # do the polynomial fit
            try: 
                reg_model = np.poly1d(np.polyfit(dfi.x, dfi.y, poly_degree, w=dfi.weights))
                y_arr[i] = reg_model(x)
                stdev_arr[i] = np.average(np.abs(dfi.y - reg_model(dfi.x)), weights=dfi.weights) # use weighted mean absolute error
            except:  # if polynomial fit does not converge, use a weighted average
                y_arr[i] = np.average(dfi.x,weights=dfi.weights)
                stdev_arr[i] = np.average(np.abs(dfi.y - y_arr[i]), weights=dfi.weights) # use weighted mean absolute error
            
        evaldf['y_fit'] = y_arr
        evaldf['stdev'] = stdev_arr
        
        # interpolate the fit and residual MAE to the photon-level data
        df_fit['y_fit'] = np.interp(df_fit.x, evaldf.x, evaldf.y_fit, left=-9999, right=-9999)
        df_fit['std_fit'] = np.interp(df_fit.x, evaldf.x, evaldf.stdev)
    
        # compute tricube weights for the vertical distance for the next iteration
        width_vweight = np.clip(n_std*df_fit.std_fit, 5, None)
        df_fit['vert_weight'] = (1.0 - np.clip((np.abs(df_fit.y-df_fit.y_fit)/width_vweight),0,1)**3 )**3
        df_fit.loc[df_fit.y_fit == -9999, 'vert_weight'] = 0.0 # give small non-zero weight for leading and trailing photons
        
        if full:
            h_list.append(y_arr)
            x_list.append(evaldf.x)

    if full:
        return evaldf, df_fit, x_list, h_list
    else:
        return np.array(df_fit.y_fit)
