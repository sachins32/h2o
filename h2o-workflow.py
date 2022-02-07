import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
import sweetviz as sv

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, f1_score

import os

import matplotlib.pyplot as plt

filename = "Boston.csv"
#filename = "processed_opportunity_scoring.csv"
# filename = "Is_medicare_suppliment_policy.csv"
# filename = "Is_Medicare_Advantage__c.csv"

model_name = "lead_scoring"
#model_name = "opportunity_scoring"
# model_name = "Medicare_suppliment"
# model_name = 'Medicare_Advantage'

target = "medv"
# target = 'oplabel'
# target = 'Is_MedicareSupplement__c'

report_path = "_data_report.html"

df = pd.read_csv(filename)

list(df.columns)

#path = "/home/jovyan/" + model_name + "/"

#if not os.path.isdir(path):
#    os.mkdir(path)
#    print("Created dir :", path)


df[target].value_counts()

df[target] = df[target].apply(lambda x: True if x==1 else False)

# df[target].value_counts()

df.shape

for col in list(df.columns):
    if 'Unnamed' in col:
        df.drop([col], axis=1, inplace=True)

#from pandas_profiling import ProfileReport

#profile = ProfileReport(df, title="Data Profiling Report")

#profile.to_file(path+report_path)

#profile

#my_report = sv.analyze(df, target_feat=target, pairwise_analysis='on')

#my_report.show_html(filepath= report_path, layout='vertical') # Default arguments will generate to "SWEETVIZ_REPORT.html"

#H2O

import h2o
from h2o.automl import H2OAutoML

def run_h2o(train, target, predictors):
    h2o.init(ip = "h2o-h2o-3.h2o-system.svc.cluster.local", port = 54321)
    data = h2o.H2OFrame(train)
    aml = H2OAutoML(balance_classes=True, max_runtime_secs=200, seed=1)
    aml.train(x=predictors, y=target, training_frame=data)
    # lead_models = h2o.get_model(list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])[:3])
    return h2o, aml

X = df.drop([target], axis=1)
y = df[[target]]

engineered_cols = list(X.columns)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y,random_state=42, shuffle=True)
x_train.reset_index(drop=True, inplace=True)
x_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
o_train = pd.concat([x_train, y_train], axis=1)
o_test = pd.concat([x_test, y_test], axis=1)
print(o_train.shape, o_test.shape)

start_time = time.time()
h2o_aml, h2o_models = run_h2o(df, target, engineered_cols)
print("time took : {}".format(time.time()-start_time))

h2o_models.leaderboard[:3,:]

lead_models = list(h2o_models.leaderboard['model_id'].as_data_frame().iloc[:,0])

lead_models
print(lead_models)

lead_model1 = h2o_aml.get_model(lead_models[0])
lead_model2 = h2o_aml.get_model(lead_models[1])
lead_model3 = h2o_aml.get_model(lead_models[2])

xtest = h2o.H2OFrame(x_test)
h2o_predictions1 = lead_model1.predict(xtest)

h2o_predictions1

h2o_predictions1 = h2o_predictions1.as_data_frame()

h2o_predictions_2 = lead_model2.predict(xtest)
h2o_predictions_2 = h2o_predictions_2.as_data_frame()

h2o_predictions_3 = lead_model3.predict(xtest)
h2o_predictions_3 = h2o_predictions_3.as_data_frame()

print(model_name+" Model Scores\n")
print("________________________Top Model1________________________")
print(classification_report(y_test, h2o_predictions1['predict']))
print("________________________Top Model2________________________")
print(classification_report(y_test, h2o_predictions_2['predict']))
print("________________________Top Model3________________________")
print(classification_report(y_test, h2o_predictions_3['predict']))

test = h2o.H2OFrame(o_test)

h2o_aml.explain(lead_model1, test)

h2o_aml.explain(lead_model2, test)

report = h2o_aml.explain(lead_model2, test)

report.keys()

# confusion_matrix = report['confusion_matrix']['subexplanations']['DRF_1_AutoML_1_20211029_91743']['plots']['DRF_1_AutoML_1_20211029_91743']
# confusion_matrix