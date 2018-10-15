


import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def feature_calculation(df):

    print(f"feature_calculation starting on pid={os.getpid()}")

    # create DataFrame and populate with stdDev
    result = pd.DataFrame(df.std(axis=0))
    result.columns = ["stdDev"]
    
    # mean
    result["mean"] = df.mean(axis=0)

    # percentiles
    for i in [0.1, 0.25, 0.5, 0.75, 0.9]:
        result[str(int(i*100)) + "perc"] = df.quantile(q=i)

    # percentile differences / amplitudes
    result["diff_90perc10perc"] = (result["10perc"] - result["90perc"])
    result["diff_75perc25perc"] = (result["75perc"] - result["25perc"])

    # percentiles of lagged time-series
    for lag in [10, 20, 30, 40, 50]:
        for i in [0.1, 0.25, 0.5, 0.75, 0.9]:
            result["lag" + str(lag) + "_" + str(int(i*100)) + "perc"] = (df - df.shift(lag)).quantile(q=i)

    # fft
    df_fft = np.fft.fft(df, axis=0)  # fourier transform only along time axis
    result["fft_angle_mean"] = np.mean(np.angle(df_fft, deg=True), axis=0)
    result["fft_angle_min"] = np.min(np.angle(df_fft, deg=True), axis=0)
    result["fft_angle_max"] = np.max(np.angle(df_fft, deg=True), axis=0)
    
    return result


def parallel_feature_calculation_ppe(df, partitions, processes):
    
    df_split = np.array_split(df, partitions, axis=1)  # split dataframe into partitions column wise
    
    with ProcessPoolExecutor(processes) as pool:        
        df = pd.concat(pool.map(feature_calculation, df_split))
    
    return df








def main():
    isMP = True

    ts_df = pd.DataFrame(np.random.random(size=(305, 30000)))

    if isMP:
        df_res = parallel_feature_calculation_ppe(ts_df, partitions=100, processes=8)
    else:
        df_res = feature_calculation(ts_df)



    print (df_res.head())


if __name__ == '__main__':
    np.show_config()
    main()

    
# import os; os.environ['OMP_NUM_THREADS'] = '1'





