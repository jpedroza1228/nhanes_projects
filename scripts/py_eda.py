import os
os.environ['QT_API'] = 'PyQt6'

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
from cmdstanpy import CmdStanModel
import arviz as az
import matplotlib.pyplot as plt
import plotnine as pn
import joblib
from pyhere import here
from janitor import clean_names
from great_tables import GT as gt

pd.set_option('display.max_columns', None)
pd.options.mode.copy_on_write = True
matplotlib.rcParams.update({'savefig.bbox': 'tight'})

data = pd.read_csv(here("data/vascular_dementia_indicators_long.csv"))

data1 = data.loc[data['.imp'] == 1]

data1.columns.tolist()

# data1['cerad_recall_avg'] = data1[['cerad_score_trial1_recall', 'cerad_score_trial2_recall', 'cerad_score_trial3_recall']].sum(axis = 1)/3

# pn.ggplot.show(
#   pn.ggplot(
#     data1, pn.aes('age', 'cerad_recall_avg')
#   )
#   + pn.geom_jitter()
#   + pn.geom_smooth(method = 'lm', se = False)
#   + pn.theme_light()
# )

pn.ggplot.show(
  pn.ggplot(
    data1, pn.aes('age', 'cerad_score_delay_recall')
  )
  + pn.geom_jitter()
  + pn.geom_smooth(method = 'lm', se = False)
  + pn.theme_light()
)

pn.ggplot.show(
  pn.ggplot(
    data1, pn.aes('age', 'cerad_intrusion_wordcount_recall')
  )
  + pn.geom_jitter()
  + pn.geom_smooth(method = 'lm', se = False)
  + pn.theme_light()
)


data1['cerad_score_delay_recall'].value_counts(normalize = True).reset_index().sort_values('cerad_score_delay_recall')
data1['cerad_intrusion_got_wrong'].value_counts()
data1[['cerad_score_delay_recall', 'animal_fluency_score', 'digit_symbol_score']].describe().transpose()

data1['animal_fluency_score'].mean() - (data1['animal_fluency_score'].std() * 2)

pn.ggplot.show(
  pn.ggplot(data1, pn.aes('animal_fluency_score'))
  + pn.geom_histogram()
  + pn.geom_vline(xintercept = 5.66)
  + pn.geom_vline(xintercept = 26.44)
  + pn.theme_light()
)


# data1['age'] = data1['age'].astype('object')
# data1['cerad_score_delay_recall'] = data1['cerad_score_delay_recall'].astype('object')
data1['cerad_intrusion_got_wrong'] = np.where(data1['cerad_intrusion_wordcount_recall'] > 0, 1, 0)
data1['animal_fluency_score'] = (np.select([data1['animal_fluency_score'] < 12,
           data1['animal_fluency_score'].between(12, 19, 'both'),
           data1['animal_fluency_score'] > 19],
          [1, 2, 3]))
data1['digit_symbol_score'] = (np.select([data1['digit_symbol_score'] < 25,
           data1['digit_symbol_score'].between(25, 51, 'both'),
           data1['digit_symbol_score'] > 51],
          [1, 2, 3]))

data1 = data1[[
      #  'age',
      #  'cerad_score_delay_recall',
       'cerad_intrusion_got_wrong',
       'animal_fluency_score',
       'digit_symbol_score',
       'told_high_bp',
       'dr_told_high_chol',
       'diff_think_remember',
       'dr_told_diabetes',
       'told_heart_fail',
       'told_heart_disease',
       'told_stroke',
       'ever_use_coke_heroin_meth',
       'ever_45_drink_everyday',
       'obese',
       'dr_told_sleep_trouble',
       'dr_told_sleep_disorder',
       'smoke_100cig_life',
       'walk_bike',
       'vig_rec_act',
       'mod_rec_act'
       ]].astype('category')

data1['dr_told_diabetes'].value_counts()

from pgmpy.models import DiscreteBayesianNetwork
# FunctionalBayesianNetwork
# from pgmpy.factors.hybrid import FunctionalCPD
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import ExpectationMaximization
# from pgmpy.utils import get_example_model

edges = [
  # ('age', 'cerad_score_delay_recall'),
  # ('age', 'cerad_intrusion_got_wrong'),
  # ('age', 'animal_fluency_score'),
  # ('age', 'digit_symbol_score'),
  # ('age', 'told_high_bp'),
  # ('age', 'dr_told_high_chol'),
  # ('age', 'diff_think_remember'),
  # ('age', 'dr_told_diabetes'),
  # ('age', 'told_heart_fail'),
  # ('age', 'told_heart_disease'),
  # ('age', 'told_stroke'),
  
  # ('cerad_score_delay_recall', 'vascular_dementia'),
  ('cerad_intrusion_got_wrong', 'vascular_dementia'),
  ('animal_fluency_score', 'vascular_dementia'),
  ('digit_symbol_score', 'vascular_dementia'),
  ('told_high_bp', 'vascular_dementia'),
  ('dr_told_high_chol', 'vascular_dementia'),
  ('diff_think_remember', 'vascular_dementia'),
  ('dr_told_diabetes', 'vascular_dementia'),
  ('told_heart_fail', 'vascular_dementia'),
  ('told_heart_disease', 'vascular_dementia'),
  ('told_stroke', 'vascular_dementia'),
  ('ever_use_coke_heroin_meth', 'vascular_dementia'),
  ('ever_45_drink_everyday', 'vascular_dementia'),
  ('obese', 'vascular_dementia'),
  ('dr_told_sleep_trouble', 'vascular_dementia'),
  ('dr_told_sleep_disorder', 'vascular_dementia'),
  ('smoke_100cig_life', 'vascular_dementia'),
  
  ('walk_bike', 'obese'),
  ('vig_rec_act', 'obese'),
  ('mod_rec_act', 'obese')  
]

cpd_vd = TabularCPD(variable='vascular_dementia',
                    variable_card = 2, values=[[.01], [.99]])


model = DiscreteBayesianNetwork(edges, latents = {'vascular_dementia'})
est = ExpectationMaximization(model, data1)
params = est.get_parameters(seed = 12345
                            # latent_card={'vascular_dementia': 2}
                            )
model.add_cpds(*params)

# model_graphviz = model.to_graphviz()
# model_graphviz.draw("nhanes_bn.png", prog="dot")

len(params)
print(params[16])

params[16].variables
obese_values = params[16].values
print(obese_values.shape)

params[16].variables
obese_values[0, 0, 0, 0] #Obese = 0, mod_act= 0, vig_act = 0, walk_bike = 0
obese_values[1, 0, 0, 0] #Obese = 1, mod_act= 0, vig_act = 0, walk_bike = 0

obese_values[0, 1, 1, 1] #Obese = 0, mod_act= 1, vig_act = 1, walk_bike = 1
obese_values[1, 1, 1, 1] #Obese = 1, mod_act= 1, vig_act = 1, walk_bike = 1

# Vascular Dementia
params[19].variables
vd_values = params[19].values
print(vd_values.shape)

vd_values[0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # best profile
vd_values[1, 0, 1, 1, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # worst profile

# simulate data
np.random.seed(12345)
sim_df = model.simulate(n_samples = 10000, include_latents = True)

# sim_df.columns.tolist()

# sim_df['vascular_dementia'].astype(int).mean()
sim_df['vascular_dementia'].value_counts(normalize = True)
sim_df.groupby('dr_told_diabetes')['vascular_dementia'].value_counts(normalize = True)

# what if one indicator changes
# model.get_cpds('obese').values = np.array([[0.7], [0.3]])  # change P(obese)
# new_samples = model.simulate(5000)
# new_samples['vascular_dementia'].mean()

# inference testing
from pgmpy.inference import VariableElimination

infer = VariableElimination(model)

# Example: what's P(vascular_dementia | obese=1, age=2)?
print(
  infer.query(variables=['vascular_dementia'],
            evidence={'obese': 1, 'dr_told_diabetes': 2})
)

# train downstream model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

x = sim_df[[
  # 'animal_fluency_score',
 'cerad_intrusion_got_wrong',
 'diff_think_remember',
#  'digit_symbol_score',
 'dr_told_diabetes',
 'dr_told_high_chol',
 'dr_told_sleep_disorder',
 'dr_told_sleep_trouble',
 'ever_45_drink_everyday',
 'ever_use_coke_heroin_meth',
 'obese',
 'smoke_100cig_life',
 'told_heart_disease',
 'told_heart_fail',
 'told_high_bp',
 'told_stroke']]
y = sim_df['vascular_dementia']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2, random_state = 123345)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = .2, random_state = 12345)

rf = RandomForestClassifier(random_state = 12345).fit(x_train, y_train)
log_pred = rf.predict(x_val)
roc_auc_score(y_val, log_pred)

