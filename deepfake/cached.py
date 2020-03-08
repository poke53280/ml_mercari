

import pandas as pd


####################################################################################
#
#   Cached
#

class Cached:
    def __init__(self, cache_dir):
        assert cache_dir.is_dir()
        self.cache_dir = cache_dir


    ####################################################################################
    #
    #   _read_cache()
    #

    def _read_cache(self,name):

        file_path = self.cache_dir / f"{name}.pkl"
        if file_path.is_file():
            return pd.read_pickle(file_path)
        else:
            return None


    ####################################################################################
    #
    #   write_cache()
    #

    def _write_cache(self, df, name):

        file_path = self.cache_dir / f"{name}.pkl"
        df.to_pickle(file_path)


    ####################################################################################
    #
    #   cached()
    #

    def cached(self, func):
        def wrapper(*args):
            df_cache = self._read_cache(func.__name__)

            if df_cache is not None:
                print (f"{func.__name__} cached")
                return df_cache

            print (f"{func.__name__} begin")

            df = func(args)

            self._write_cache(df, func.__name__)

            print (f"{func.__name__} end")

            return df
        return wrapper

