from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

def test_packages():

    print("                           _____           ")
    print("                          |_   _|   /\     ")
    print("  _ __ ___  _   _ _ __ ___  | |    /  \    ")
    print(" | '_ ` _ \| | | | '__/ __| | |   / /\ \   ")
    print(" | | | | | | |_| | | | (__ _| |_ / ____ \  ")
    print(" |_| |_| |_|\__ _|_|  \___|_____/_/    \_\ ")
    print("                                           ")
    print(" Adapted from JaviAbellan for Saturdays.AI ")

    ##################################################################### IMPORTS

    # General libraries
    import os
    import gc
    import time
    import platform
    import datetime
    import multiprocessing
    from tqdm import tqdm_notebook as tqdm
    import torch # Import early to get gpu_name below


    print("🕑 Time:  ", datetime.datetime.now().strftime("%H:%M"))
    print("🗓️ Date: ", datetime.datetime.now().strftime("%d/%m/%Y"))
    print("💻 OS:  ", platform.system(), platform.release())
    print("🔥 CPU:   ", multiprocessing.cpu_count(), "cores")
    print("🔥 GPU:   ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No")
    print("🐍 Python:", platform.python_version())
    print("")
    print("Packages:")
    print("")

    # Data libraries
    import numpy             as np;  print("  Numpy (np):           ", np.__version__)
    import pandas            as pd;  print("  Pandas (pd):          ", pd.__version__)
    import pandas_profiling  as pp;  print("  Pandas Profiling (pp):", pp.__version__)
    import missingno         as ms;  print("  Missingno (ms):       ", ms.__version__)
    import seaborn           as sns; print("  Seaborn (sns):        ", sns.__version__); sns.set()
    import altair            as at;  print("  Altair (at):          ", at.__version__)
    import matplotlib        as mp;  print("  Matplotlib (plt):     ", mp.__version__)
    import matplotlib.pyplot as plt
    # ML libraries
    import sklearn           as skl;print("  Sklearn (skl):        ", skl.__version__)
    #import xgboost           as xgb;print("  XGBoost (xgb):        ", xgb.__version__)
    #import lightgbm          as lgb;print("  LightGBM (lgb):       ", lgb.__version__)
    from sklearn import preprocessing
    from sklearn import model_selection
    from sklearn import pipeline
    from sklearn import ensemble
    from sklearn import impute
    from sklearn import compose
    from sklearn import metrics

    # DL libraries
    import tensorflow as tf
    import fastai
    print("  Tensorflow (tf):      ", tf.__version__)
    print("  Keras:                ", tf.keras.__version__)
    print("  Pytorch:              ", torch.__version__)
    print("  Fast.ai:              ", fastai.__version__)
    print("")

    # Set options
    pd.set_option('display.max_rows',    500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width',       1000)

    # Packages still not used
    # import datatable         as dt
    # import catboost          as cgb
    # import h2o.automl        as ml_auto
    # import yellowbrick       as ml_vis
    # import eli5              as ml_exp
    # import arviz             as av
    # import category_encoders as ce
    # from fbprophet import Prophet


def print_imports():
    print("""
    # General libraries
    import os
    import gc
    import time
    import platform
    import datetime
    import multiprocessing
    from tqdm import tqdm_notebook as tqdm

    # Data libraries
    import numpy             as np;  
    import pandas            as pd;  
    import pandas_profiling  as pp;  
    import missingno         as ms;  
    import seaborn           as sns;
    import altair            as at;  
    import matplotlib        as mp;  
    import matplotlib.pyplot as plt

    # ML libraries
    import sklearn
    import xgboost
    import lightgbm
    from sklearn import preprocessing
    from sklearn import model_selection
    from sklearn import pipeline
    from sklearn import ensemble
    from sklearn import impute
    from sklearn import compose
    from sklearn import metrics

    # DL libraries
    import tensorflow as tf
    import fastai
    import torch

    # Set options
    pd.set_option('display.max_rows',    500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width',       1000)

    # Packages still not used
    # import datatable         as dt
    # import catboost          as cgb
    # import h2o.automl        as ml_auto
    # import yellowbrick       as ml_vis
    # import eli5              as ml_exp
    # import arviz             as av
    # import category_encoders as ce
    # from fbprophet import Prophet

    # Jupyter ext
    %matplotlib inline
    %load_ext autoreload
    %autoreload 2
    """)

def test_gpu(advanced=False, pytorch=True):
    """[Verifies GPU support on tf and pytorch]

    Keyword Arguments:
        pytorch {bool} -- [Check also for pytorch gpu support] (default: {True})
    """
    import tensorflow as tf
    print("tf version: ", tf.__version__)
    print("Number of GPUs: ", len(tf.config.list_physical_devices('GPU')) )
    print("Device names: ", tf.config.list_physical_devices('GPU'))
    print("CUDA support: ", tf.test.is_built_with_cuda())

    if advanced:
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())

    if pytorch:
        import torch
        print("\nPytorch version:", torch.__version__)
        print("GPU available:", torch.cuda.is_available())
        print("Cuda version:", torch.version.cuda)
        print("GPU name:     ", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    test_packages()