import os
os.environ['QT_API'] = 'PyQt6'

import pandas as pd
import numpy as np
# import plotnine as pn
from janitor import clean_names
from matplotlib import rcParams
# import seaborn as sns
# import matplotlib.pyplot as plt
from pyhere import here


# Set some options
pd.set_option('display.max_columns', None)
pd.set_option('mode.copy_on_write', True)
rcParams.update({'savefig.bbox': 'tight'}) 

from google.cloud import storage

storage_client = storage.Client()
bucket_name = 'jp-stan-models'
bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob('vascular_dementia_indicators_long.csv')
content = blob.download_as_text()

from io import StringIO
long = pd.read_csv(StringIO(content))

long1 = long.loc[long['.imp'] == 1]

long1.columns.tolist()

long1a = long1.loc[:, 'told_high_bp':'told_stroke']
long1b = long1.loc[:, 'ever_use_coke_heroin_meth':'smoke_100cig_life']
long1c = long1['obese']

long1 = long1a.join(long1b).join(long1c)

long1['dr_told_diabetes_no'] = np.where(long1['dr_told_diabetes'] == 0, 1, 0)
long1['dr_told_diabetes_border'] = np.where(long1['dr_told_diabetes'] == 1, 1, 0)
long1['dr_told_diabetes_yes'] = np.where(long1['dr_told_diabetes'] == 2, 1, 0)

long1 = long1.drop(columns = 'dr_told_diabetes')

x_num = long1.loc[:, 'cerad_score_trial1_recall':'digit_symbol_score']
x_num = x_num.join(long1[['num_ready_eat_30day', 'num_frozen_meal_30day']])

x_bi = long1.drop(columns = x_num.columns)

x_num_scale = (x_num - x_num.mean()) / x_num.std()

stan_dict = {
  'J': long1.shape[0],
  'C': 2,
  'num_feat': x_num_scale.shape[1],
  'bi_feat': x_bi.shape[1],
  'x_num': x_num_scale,
  'x_bi': x_bi
}

from cmdstanpy import CmdStanModel
import joblib
# import arviz

stan_blob = bucket.blob('naive_bayes.stan')
stan_code = stan_blob.download_as_text()

with open('naive_bayes.stan', 'w') as f:
    f.write(stan_code)

model = CmdStanModel(stan_file= 'naive_bayes.stan')

fit = model.sample(
  stan_dict,
  chains = 4,
  adapt_delta = .99,
  iter_warmup = 2000,
  iter_sampling = 2000,
  seed = 101625
  )

(
  joblib.dump([model, fit],
              'naive_bayes_model1.joblib',
              compress = 3)
)

# then I'll use gcloud storage cp /home/cpppedroza/naive_bayes_model1.joblib \
# gs://jp-stan-models/naive_bayes_model1.joblib
