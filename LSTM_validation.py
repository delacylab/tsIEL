#Copyright 2023 Nina de Lacy and Michael Ramshaw
   #Licensed under the Apache License, Version 2.0 (the "License");
   #you may not use this file except in compliance with the License.
   #You may obtain a copy of the License at

     #http://www.apache.org/licenses/LICENSE-2.0

   #Unless required by applicable law or agreed to in writing, software
   #distributed under the License is distributed on an "AS IS" BASIS,
   #WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   #See the License for the specific language governing permissions and
   #limitations under the License.


#eli5 permutation routine is released under the MIT license and may be found at https://eli5.readthedocs.io/en/latest/

#import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Adamax, Nadam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from eli5.permutation_importance import get_score_importances
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score, roc_curve, auc, roc_auc_score
import os
import time
import argparse
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import shap
import shutil
from collections import Counter
import copy

parser = argparse.ArgumentParser()
parser.add_argument('target',help='prefix of DLFS file to use, not including .csv')
parser.add_argument('SD',help='Number of standard deviations, e.g. 0,2,4')
parser.add_argument('IDkey',help='DB Subject Identifier header')
parser.add_argument('threshold',help='lasso threshold to use')
parser.add_argument('UseTPs',help='Use TPs or regular directory')
parser.add_argument('tps',help='should look like 0123')
args = parser.parse_args()
target = args.target #Condition such as anxiety or somatic 
SD = args.SD         #standard deviations - see recursion script
IDkey = args.IDkey   #row level subject identification column for dataset - for ABCD V4, it is subjectkey
threshold = args.threshold #Lasso coefficient threshold for columns to be included in dataset, we used 0.0
UseTPs = args.UseTPs  #False = determine important features based on mean of feature importance over time periods 
#True means important features are determined with feature/time point specificity.
if UseTPs == "True":
   print("using time periods")
time_periods = str(args.tps) 

start_time = time.time()

#Load and construct time series dataset - rep=unseen data
nanfill = 1
#time_periods = '0123' #Now a runtime param
time_periods_list = [int(t) for t in time_periods] #e.g. 02 will be [0,2]
allTimeLabels = ['baseline', '1Y','2Y','3Y']
timelabels = [ allTimeLabels[i] for i in time_periods_list ]
base_input_dir = '../DLFS_V2Combined'
suffix = ''
if '3yearonset' in target:
   base_input_dir += '_3yo'
base_input_dir += '/'
print('Using basedir',base_input_dir)
input_dir = base_input_dir + str(threshold) + '/'
targetdf = pd.read_csv('../targets/cbcl_3year_targets_after_methods_rep.csv')
base = pd.read_csv(input_dir + target + '^baseline_rep_combined_DLFS.csv')
one = pd.read_csv(input_dir + target + '^1_year_rep_combined_DLFS.csv')
two = pd.read_csv(input_dir + target + '^2_year_rep_combined_DLFS.csv')
three = pd.read_csv(input_dir + target + '^3_year_rep_combined_DLFS.csv')

allDFs = [base, one, two, three]
dfsWeWant = [ allDFs[i] for i in time_periods_list ] 

#Create target cols and make sure we are using correct targets corresponding to subjects in dataset
for i in range(len(dfsWeWant)):
   if i == 0:
      subsInAll = dfsWeWant[0][['subjectkey']]
   else:
      subsInAll = subsInAll.merge(dfsWeWant[i][['subjectkey']], how='inner',on='subjectkey')
for i in range(len(dfsWeWant)):
   dfsWeWant[i] = subsInAll.merge(dfsWeWant[i], how='left', on='subjectkey')
for i in range(1,len(dfsWeWant)):
   otherDF = dfsWeWant[i]
   if dfsWeWant[0][['subjectkey']].compare(otherDF[['subjectkey']]).empty == False:
      print("subject mismatch. Exiting.")
      exit(1)
y = dfsWeWant[0][['subjectkey']].merge(targetdf, how='inner',on='subjectkey')
if y['subjectkey'].compare(dfsWeWant[0]['subjectkey']).empty == False:
   print('subject/target mismatch')
   exit(1)
y = y[target]
print('target length should match earliest row count',len(y))

#When constructing timeseries dataset, it is important to know whether columns are unique
#(only appearing in one time period) or a true timeseries col (column has data for multiple time periods)
#--count columns both by year and determine whether they are timeseries cols or not
columnsByYear = {}
#want list of unique cols that appear in all dfs
all_cols_list = None
for i in range(len(dfsWeWant)):
   columnsByYear[timelabels[i]] = dfsWeWant[i].columns.tolist()
   #columnsByYear[timeperiods[i]].remove('eventname')
   if i == 0:
      all_cols_list = columnsByYear[timelabels[i]]
   else:
      all_cols_list = all_cols_list + columnsByYear[timelabels[i]]
#all_cols_list has all columns in all time periods. So there will be duplicates for timeseries.
countAllCols = Counter(all_cols_list)
uniques = []
timeseries = []
for k,v in countAllCols.items():
   if v == 0:
      print("ERROR, every variable should be appearing at least once or we got problems")
      exit(1)
   elif v == 1:
      uniques.append(k)
   else:
      timeseries.append(k)
      
unique_columns = np.unique(all_cols_list) #NOTE this still has subjectkey which we will want to remove at the very end.
print("Uniques(one time only) + timeseries = num_unique_columns",  len(uniques), len(timeseries), len(unique_columns) )
print(dfsWeWant[0].shape[0])
#subject list sanity check
for i in range(len(dfsWeWant)):
   if dfsWeWant[0]['subjectkey'].equals(dfsWeWant[i]['subjectkey']) == False:
      print("subjectkey mismatch in final_array assembly. Exiting.")
      exit(1) 
      
unique_columns = unique_columns.tolist()
#Time for array conversion, even subjectkey has to go now
unique_columns.remove('subjectkey')
num_unique_columns = len(unique_columns)
runningColIndexes = np.zeros((num_unique_columns), dtype=int)
coef_lookup = dict.fromkeys(unique_columns, [])
print('Final_array shape',dfsWeWant[0].shape[0], len(dfsWeWant), num_unique_columns)
X = np.empty((dfsWeWant[0].shape[0], len(dfsWeWant), num_unique_columns))
X[:] = nanfill #our masking value we will experiment with 
for timep in range(len(dfsWeWant)): 
   #final_array[:,timep,:] = dfsWeWant[timep][unique_columns].to_numpy()      
   for col in dfsWeWant[timep].columns:
      if col in ['subjectkey', 'eventname']:
         continue
      #need to get index of this in unique_columns
      bigindex = unique_columns.index(col)
      #THE BELOW LINE WILL COPY data where its time period should be. 
      #So 2 year variables will only appear in twoyear time period.
      X[:, timep, bigindex] = dfsWeWant[timep][col].to_numpy()
      runningColIndexes[bigindex] += 1
      coef_lookup[col].append(timelabels[timep])  

learn_test = X

#We need train for Shapley background so we have to construct all of that as well.
baseTrain = pd.read_csv(input_dir + target + '^baseline_tt_combined_DLFS.csv')
oneTrain = pd.read_csv(input_dir + target + '^1_year_tt_combined_DLFS.csv')
twoTrain = pd.read_csv(input_dir + target + '^2_year_tt_combined_DLFS.csv')
threeTrain = pd.read_csv(input_dir + target + '^3_year_tt_combined_DLFS.csv')

allDFsTrain = [baseTrain, oneTrain, twoTrain, threeTrain]
dfsWeWantTrain = [ allDFsTrain[i] for i in time_periods_list ] 

for i in range(len(dfsWeWantTrain)):
   if i == 0:
      subsInAllTrain = dfsWeWantTrain[0][['subjectkey']]
   else:
      subsInAllTrain = subsInAllTrain.merge(dfsWeWantTrain[i][['subjectkey']], how='inner',on='subjectkey')
for i in range(len(dfsWeWantTrain)):
   dfsWeWantTrain[i] = subsInAllTrain.merge(dfsWeWantTrain[i], how='left', on='subjectkey')
for i in range(1,len(dfsWeWantTrain)):
   otherDFTrain = dfsWeWantTrain[i]
   if dfsWeWantTrain[0][['subjectkey']].compare(otherDFTrain[['subjectkey']]).empty == False:
      print("subject mismatch. Exiting.")
      exit(1)
#count columns both by year and determine whether they are timeseries cols or not
columnsByYearTrain = {}
#want list of unique cols that appear in all dfs
all_cols_list = None
for i in range(len(dfsWeWantTrain)):
   columnsByYearTrain[timelabels[i]] = dfsWeWantTrain[i].columns.tolist()
   if i == 0:
      all_cols_list = columnsByYearTrain[timelabels[i]]
   else:
      all_cols_list = all_cols_list + columnsByYearTrain[timelabels[i]]
      
unique_columns_train = np.unique(all_cols_list) #NOTE this still has subjectkey which we will want to remove at the very end.
print("Uniques(one time only) + timeseries = num_unique_columns",  len(uniques), len(timeseries), len(unique_columns) )
print(dfsWeWantTrain[0].shape[0])
#subject list sanity check
for i in range(len(dfsWeWantTrain)):
   if dfsWeWantTrain[0]['subjectkey'].equals(dfsWeWantTrain[i]['subjectkey']) == False:
      print("subjectkey mismatch in final_array assembly. Exiting.")
      exit(1) 
      
unique_columns_train = unique_columns_train.tolist()
#Time for array conversion, even subjectkey has to go now
unique_columns_train.remove('subjectkey')
num_unique_columns_train = len(unique_columns_train)
print('Final_array shape',dfsWeWantTrain[0].shape[0], len(dfsWeWantTrain), num_unique_columns_train)
background = np.empty((dfsWeWantTrain[0].shape[0], len(dfsWeWantTrain), num_unique_columns_train))
background[:] = nanfill #our masking value we will experiment with 
for timep in range(len(dfsWeWantTrain)): 
   for col in dfsWeWantTrain[timep].columns:
      if col in ['subjectkey', 'eventname']:
         continue
      #need to get index of this in unique_columns
      bigindex = unique_columns_train.index(col)
      #Copy data where its time period should be, so 2 year variables will only appear in twoyear time period.
      background[:, timep, bigindex] = dfsWeWantTrain[timep][col].to_numpy()
#END train assembly

#Double check that feature indexes match up 
print(np.array_equal(unique_columns_train, unique_columns))

with tf.device('device:GPU:0'):
    #set number of models you wish to evaluate
    accuracy_threshold = 100
    
    # define eli5 score importances - permutation importance, we only use this for importance of each time period now.
    def score(X,y):
         y_predict_classes = np.argmax(model.predict(X), axis=-1)
         return accuracy_score(y,y_predict_classes)
    #Set runtime options
    optoption = 'AdamW'
    #Set output filepaths
    basepath = "IEL_ann_validation"
    if UseTPs == "True":
       basepath += "TPs"
    basepath += '_' + time_periods + "/" + str(SD) + "SD/" + target
    os.makedirs(basepath, exist_ok=True) #e.g. For 2 and 4 SD this has to be ok
    basename = target
    #Need to make Shapley directory here too for easy zipping later
    shapley_scores_dir = basepath + '/Shapley_scores'
    os.makedirs(shapley_scores_dir)

    #Get most important features from recursion (previous step in pipeline)
    input_basepath = "IEL_ann_recursive"
    if UseTPs == "True":
       input_basepath += "TPs"
    input_basepath += '_' + time_periods + "/" + target + os.path.sep + str(SD) + "SD" + os.path.sep + target + '_' + optoption
#filenames are like phenotypic_anx_pres_Adam_models.csv
    parameters_filename = input_basepath + '_models' + str(SD) + "SD"
    #models filename is the same
    parameters_filename += '.csv'
    parameters = pd.read_csv(parameters_filename)
    parameters = parameters.drop(parameters.columns[0], axis=1)
        
    features_filename = input_basepath + '_features_' + str(SD) + 'SD'
    if UseTPs == "True":
       features_filename += "_TP"    
    features_filename += '.csv'
    features = pd.read_csv(features_filename)
    features = features.drop(features.columns[0], axis=1)

    importances_filename = input_basepath + '_importances' + str(SD) + 'SD'
    if UseTPs == "True":
       importances_filename += '_TP'
    importances_filename += '.csv'
    importances = pd.read_csv(importances_filename)
    importances = importances.drop(importances.columns[0], axis=1)

    #determine hyperparameters for top accuracy models
    #IMPORTANT: argsort default is ascending order which is only for lower value = better like mse or fitness
    #Need to use np.flip with accuracy or r2 to reverse sort
    accuracy_idx = np.flip(np.argsort(parameters['accuracy']))
    top_accuracy_idx = accuracy_idx[:accuracy_threshold].tolist()
    learning_list = parameters['learning'].to_list()
    learning_select = [learning_list[i] for i in top_accuracy_idx]
    beta_1_list = parameters['beta_1'].to_list()
    beta_1_select = [beta_1_list[i] for i in top_accuracy_idx]
    beta_2_list = parameters['beta_2'].to_list()
    beta_2_select = [beta_2_list[i] for i in top_accuracy_idx]

    #If we're doing time periods and the importance of a feature/TP is 0, then it was nanfilled before and
    #we don't want to regard it as important now.
    if UseTPs == "True":
       bool_mask_importances = (importances > 0.0) | (importances < - 0.0)
       features = features[bool_mask_importances]

    #determine features for top accuracy models
    names_list = []
    for col in features:
        ser = features[col].dropna()
        names = ser.unique()
        names = names.tolist()
        for i in range(len(names)):
            names[i] = names[i].replace('_tt_','_')
        names_list.append(names)
    features_select = [names_list[i] for i in top_accuracy_idx]
          
    #create X,y
    y = y.to_numpy()
    y_true = y
    print("0s/1s count for",target,len(np.where(y == 0)[0]), len(np.where(y == 1)[0]))
   
    #create lists for performance measures
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    auc_list = []
    feature_importances_scores = []
    feature_importances_scores_withTPs = []
    #shapley_scores = []
    num_features_list = []
    fpr_list = []
    tpr_list = []
    tp_scores = [] 
    root_feature_list = []
    TP_feature_list = []
    count = 0   
    # perform validation
    #feature is actually a set of 1-15 features identified in recursion as being important.
    #learning/beta_selects are single values.
    for feature, learning, beta_1, beta_2 in zip(features_select, learning_select, beta_1_select, beta_2_select):
            print('FEATURE:',feature)   
            #The below section deals with naming incongruities. If feature X appears in multiple time periods, it
            #will be feature X in each individual timeperiod. But we need to name it separately e.g. X_TP0
            indexeses = [] 
            final_col_names = []        
            if UseTPs == "True":
               featuresTPs_dict = {}
               for fi in range(len(feature)):
                  tpIwant = int(feature[fi][-1])
                  root_feature_name = feature[fi][:-4]
                  if root_feature_name not in featuresTPs_dict.keys():
                     featuresTPs_dict[root_feature_name] = [tpIwant]
                  else:
                     featuresTPs_dict[root_feature_name].append(tpIwant)
               print(featuresTPs_dict)
               final_col_names = []
               for col in featuresTPs_dict.keys():
                  indexeses.append(unique_columns.index(col))               
                  final_col_names.append(col)
            else:
               for col in feature:
                  indexeses.append(unique_columns.index(col))
               final_col_names = feature.copy()
               #sanity check
               for c in range(len(final_col_names)):
                  if final_col_names[c] != feature[c]:
                     print('final col mismatch. exiting')
                     exit(1)

            features_TPs = []
            for f in final_col_names:
              for t in range(learn_test.shape[1]):
                 features_TPs.append(f + '_TP' + str(t))
            TP_feature_list.append(features_TPs)
            
            root_feature_list.append(final_col_names)
            #End of determining feature list. #Now we take X[features_we_want]
            X_sample = np.take(learn_test, indexeses, axis=2)
            num_features = len(final_col_names) #len(feature)
            num_features_list.append(num_features)

            if UseTPs == "True":
               for tp in range(X_sample.shape[1]):  
                  for f in range(X_sample.shape[2]): #THIS SHOULD BE same length as learn_train.shape[2]
                     if tp not in featuresTPs_dict[final_col_names[f]]:
                        X_sample[:,tp,f] = nanfill
                        print('Nanfilling time period',tp,'feature',final_col_names[f])
            
            y_true = y
            y_cat = tf.keras.utils.to_categorical(y, num_classes=2)
            #create model
            model = Sequential()
            model.add(Bidirectional(LSTM(300, activation='tanh', return_sequences=True))) #,input_shape=(final_array.shape[1], final_array.shape[2])))) #
            model.add(Bidirectional(LSTM(300, activation='tanh')))
            model.add(Dense(2, activation = 'softmax', name='output'))
            #opt = Adam(lr = learning, beta_1=beta_1, beta_2=beta_2)
            wd = .01 # pytorch default
            opt = tfa.optimizers.AdamW(learning_rate = learning, beta_1=beta_1, beta_2=beta_2, weight_decay=wd)
            model.compile(loss="categorical_crossentropy", optimizer = opt, metrics=['accuracy'])
            early_stopping_monitor = EarlyStopping(monitor='loss',patience=3)
            model.fit(X_sample,y_cat, epochs=200, callbacks=[early_stopping_monitor], verbose=0)
            print("GPU 0 running validation testing for", target,SD,"SD", type(X_sample), flush=True)
            y_predict_raw = model.predict(X_sample)
            y_predict_classes = np.argmax(y_predict_raw, axis=-1)

            base_score, score_decreases = get_score_importances(score, X_sample,y_true)
            print("after get_score_importances")
            tp_importances = np.mean(score_decreases, axis=0)
            tp_scores.append(str(tp_importances))
            #New Shapley feature importances method
            train_subset = np.take(background, indexeses, axis=2)   #background[feature] - this has to match the set of columns we used in test
            print("right after train_subset=")
            if UseTPs == "True": #NOTE THAT INDEXES SHOULD LINE UP BECAUSE train and test should have same columns
               for tp in range(train_subset.shape[1]):  
                  for f in range(train_subset.shape[2]): #THIS SHOULD BE same length as learn_train.shape[2]
                     if tp not in featuresTPs_dict[final_col_names[f]]:
                        train_subset[:,tp,f] = nanfill
                        print('Nanfilling time period',tp,'feature',final_col_names[f])
            
            print("X_sample.shape",X_sample.shape, "Train_subset.shape", train_subset.shape)
            ge = shap.GradientExplainer( model, train_subset)
            shap_values = ge.shap_values(X_sample, nsamples=X_sample.shape[0] * 2)
            if isinstance(shap_values, list):
               if len (shap_values) != 2:
                 print('shap_values list should be == numClasses', flush=True)
                 exit(1)
            else:
               print('SHAP_values is NOT a list. Exiting.',flush=True)
               exit(1)
            shap_values = shap_values[1] #0/1 targets, we take 1
            np.set_printoptions(threshold=np.inf)
            
            #next line averages the feature values across all rows
            feature_importances = np.mean(np.abs(shap_values), axis=0) 
            #Now feature_importances Shape is (numTimePeriods,numFeatures)  
            feature_importances_withTPs = copy.deepcopy(feature_importances)
                       
            #Now for the avgd over time periods way, now axis 0 = time period
            feature_importances = np.mean(feature_importances, axis=0)
            feature_importances_scores.append(feature_importances)
            #shap.summary_plot(shap_values, X_sample[feature], max_display=num_features, plot_size=(7,5), show=False, color_bar=None, color_bar_label=[], class_names=[],title=None)

            #BEGIN OF SHAP FEATURES ONLY
            feature_mean_shap = np.mean(shap_values, axis=1)
            #print(shap_values.shape)  # shape here is (numRows, numTimePeriods, numFeatures)
            #print(shap_values_reshape.shape) #shape here is (numRows, numTimePeriods * numFeatures)          
            print(feature_mean_shap.shape) #shape here is (numRows, numFeatures) 
            test_mean = np.mean(X_sample, axis=1) #average over time periods so shap has a baseline
            print(test_mean.shape)
            data_frame = pd.DataFrame( test_mean, columns=final_col_names)  
            shap.summary_plot(feature_mean_shap, data_frame, max_display=len(feature_importances), plot_size=(7,5), show=False, color_bar=None, color_bar_label=[], class_names=[],title=None)
            plt.title("") #("Shapley value-impact on model output", x=-10)
            plt.xlabel("")
            plt.ylabel("")
            plt.gca().yaxis.set_ticks_position('none')
            plt.tight_layout()
            plt.savefig(shapley_scores_dir + '/Model_' + str(count) + '_shapley_plot.png')
            plt.close()
            #We want to write out these indvidual files
            shapley_frame = pd.DataFrame(feature_mean_shap, columns=final_col_names)
            shapley_frame.to_csv(shapley_scores_dir + '/Model_' + str(count) + '_shapley_values.csv')
            #END OF SHAP FEATURES ONLY 
       
            #BEGIN SHAP_FEATURES_TPs   
            features_TPs_flatten = []
            countercheck = 0
            for f in range(len(final_col_names)):
               for t in range(X_sample.shape[1]):
                  if final_col_names[f] not in features_TPs[countercheck]:
                     print('feature name mismatch')
                     exit(1)
                  features_TPs_flatten.append(feature_importances_withTPs[t, f ])
                  countercheck += 1
            feature_importances_scores_withTPs.append(features_TPs_flatten) 
            #order=F is important here
            shap_values_reshape = np.reshape(shap_values, (shap_values.shape[0], shap_values.shape[1] * shap_values.shape[2]), order='F')        
            #NEED TO MAKE NEW DATAFRAME here with actual values of test
            X_sample_reshape = np.reshape(X_sample, (X_sample.shape[0], X_sample.shape[1] * X_sample.shape[2]), order='F') 
            print(len(features_TPs_flatten), len(features_TPs))
            data_frame = pd.DataFrame(X_sample_reshape, columns=features_TPs)
            data_frame.to_csv(shapley_scores_dir + '/Model_' + str(count) + '_CheckTPs.csv', index=False)
            shap.summary_plot(shap_values_reshape, data_frame, max_display=shap_values_reshape.shape[1], plot_size=(7,5), show=False, color_bar=None, color_bar_label=[], class_names=[],title=None)
            plt.title("") #("Shapley value-impact on model output", x=-10)
            plt.xlabel("")
            plt.ylabel("")
            plt.gca().yaxis.set_ticks_position('none')
            plt.tight_layout()
            plt.savefig(shapley_scores_dir + '/Model_' + str(count) + '_shapley_plotTPs.png')
            plt.close()
            shapley_frame = pd.DataFrame(shap_values_reshape, columns=features_TPs)
            shapley_frame.to_csv(shapley_scores_dir + '/Model_' + str(count) + '_shapley_valuesTPs.csv')            
            #END SHAP FEATURES TPS
            
            count += 1
            #Gather metrics
            accuracy = accuracy_score(y_true, y_predict_classes)
            accuracy_list.append(accuracy)
            precision = average_precision_score  (y_true, y_predict_classes)
            precision_list.append(precision)
            recall = recall_score(y_true,y_predict_classes)
            recall_list.append(recall)
            f1 = f1_score(y_true,y_predict_classes, average='binary')
            f1_list.append(f1)
            y_predict_raw_pos = []
            for tup in y_predict_raw:
               y_predict_raw_pos.append(tup[1])
            fpr, tpr, thresholds = roc_curve(y_true, y_predict_raw_pos, pos_label=1) #pos_label=1 because our y is in [0,1]
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            area_under_curve = auc(fpr, tpr)
            auc_list.append(area_under_curve)

    learning_array = np.array(learning_select)
    beta_1_array = np.array(beta_1_select)
    beta_2_array = np.array(beta_2_select)
    recall_array = np.array(recall_list)
    precision_array = np.array(precision_list)
    accuracy_array = np.array(accuracy_list)
    f1_array = np.array(f1_list)
    auc_array = np.array(auc_list)
    num_features_array = np.array(num_features_list)
    fpr_array = np.array(fpr_list)
    tpr_array = np.array(tpr_list)
    #ROC Curve plot saving - need to create ROC curve plot for most accurate model
    accuracy_idx = np.flip(np.argsort(accuracy_array))
    top_accuracy_idx = accuracy_idx[0]
    print("Top accuracy and it's corresponding f1 should be",accuracy_array[top_accuracy_idx], f1_array[top_accuracy_idx])
    plt.plot(fpr_array[top_accuracy_idx],tpr_array[top_accuracy_idx],label="auc="+str(auc_array[top_accuracy_idx]))
    plt.legend(loc=4)
    plt.title(target)
    plt.savefig(basepath + '/ga_ann_classification_test_unseen_elbow_' + str(SD) + '_roc_curve.png',bbox_inches='tight')
    plt.close()
      
    print(target, "accuracy max:", accuracy_array.max())
    print(target, "precision max:", precision_array.max())
    print(target, "recall max:", recall_array.max())
    #collect performance measures, features and importances
    best_ga_ann_features = pd.DataFrame(data=root_feature_list).T
    best_ga_ann_features_TPs = pd.DataFrame(data=TP_feature_list).T
    best_accuracy_models = np.stack((learning_select, beta_1_select, beta_2_select, accuracy_array, precision_array, recall_array, f1_array, auc_array, num_features_list, tp_scores), axis=1)
    best_ga_ann_models = pd.DataFrame(data=best_accuracy_models, columns = ['learning_rate', 'beta_1', 'beta_2', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'num_features','TP_Importances'])
    best_ga_ann_importances = pd.DataFrame(data=feature_importances_scores).T
    best_ga_ann_importances_TPs = pd.DataFrame(data=feature_importances_scores_withTPs).T
        
    #save features
    features_path = "ga_ann_classification_test_unseen_elbow_" + str(SD) + "SD_best_features_"
    features_path = os.path.join(basepath,features_path)
    features_filename = features_path + basename
    features_csv_filename = features_path + basename + '.csv'
    best_ga_ann_features.to_pickle(features_filename)
    best_ga_ann_features.to_csv(features_csv_filename)
    best_ga_ann_features_TPs.to_csv(features_csv_filename.replace('best_features','best_features_TP'))
     
    #save models
    models_path = "ga_ann_classification_test_unseen_elbow_" + str(SD) + "SD_best_models_"
    models_path = os.path.join(basepath,models_path)
    models_filename = models_path + basename
    models_csv_filename = models_path + basename + '.csv'
    best_ga_ann_models.to_pickle(models_filename)
    best_ga_ann_models.to_csv(models_csv_filename)
        
    #save importances
    importances_path = "ga_ann_classification_test_unseen_elbow_" + str(SD) + "SD_best_importances_"
    importances_path = os.path.join(basepath,importances_path)
    importances_filename = importances_path + basename
    importances_csv_filename = importances_path + basename + '.csv'
    best_ga_ann_importances.to_pickle(importances_filename)
    best_ga_ann_importances.to_csv(importances_csv_filename)
    best_ga_ann_importances_TPs.to_csv(importances_csv_filename.replace('best_importances', 'best_importances_TP'))
    #save shapley
    #importances = Shapley here so we don't need them here
 
    #save zip of shapley raw scores and remove the raw scores dir
    shapley_scores_path = "ga_ann_classification_test_unseen_elbow_" + str(SD) + "SD_FullShapley_scores"
    shapley_scores_path  = os.path.join(basepath, shapley_scores_path)
    shapley_scores_filename = shapley_scores_path + basename + '.zip'
    shutil.make_archive(shapley_scores_filename, 'zip', shapley_scores_dir)
    shutil.rmtree(shapley_scores_dir)

