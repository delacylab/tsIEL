import os
os.environ["OMP_NUM_THREADS"] = f"1"
os.environ['TF_NUM_INTEROP_THREADS'] = f"1"
os.environ['TF_NUM_INTRAOP_THREADS'] = f"1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from collections import Counter
from eli5.permutation_importance import get_score_importances
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, average_precision_score, recall_score
from sklearn.metrics import f1_score, roc_curve, auc, roc_auc_score
from datetime import datetime
import multiprocessing
from multiprocessing import Pool,Process,Queue, current_process
#import threading
#import contextlib
import time
import sys
import argparse
import psutil
import gc
import pickle
import shap
import copy
parser = argparse.ArgumentParser()
parser.add_argument('target',help='prefix of DLFS file to use, not including .csv')
parser.add_argument('numFeatures',help='Number of features (identified through elbow on fitness/num_features graph')
parser.add_argument('SD',help='Number of standard deviations, e.g. 0,2,4')
parser.add_argument('IDkey',help='DB Subject Identifier header')
parser.add_argument('numOfGPUs',type=int,help='Number of GPUs to use')
parser.add_argument('coresPerGPU',type=int, help='Cores Per GPU')
parser.add_argument('threshold',help='lasso threshold to use')
#parser.add_argument('weighted',help='Use weighted avg, True or False')
parser.add_argument('tps',help='should look like 0123')
args = parser.parse_args()

layer_size = 300 
spp = 100 
numGens = 401 
optoption = 'AdamW'

def makeMSEOneOutputModel(iq, outq):
      #Imports are done here because of an odd CUDA limitation that prevents
      #forked multiprocesses from getting another CUDA context.
   for rand,learning,beta_1, beta_2, partNum, gpuNum, recursive_features in iter(iq.get,'STOP'):
      import tensorflow as tf
      from tensorflow.keras.optimizers import Adam, Adamax, Nadam
      from tensorflow.keras.callbacks import EarlyStopping
      from tensorflow.keras.layers import Dense, LSTM, Bidirectional
      from tensorflow.keras.models import Sequential
      import tensorflow_addons as tfa
      tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
      from absl import logging
      logging.set_verbosity(logging.ERROR)
      tf.config.threading.set_inter_op_parallelism_threads(1)
      tf.config.threading.set_intra_op_parallelism_threads(1) 
      if gpuNum != 99:
         physical_devices = tf.config.list_physical_devices('GPU')
         tf.config.set_visible_devices(physical_devices[gpuNum], 'GPU')
         tf.config.experimental.set_memory_growth(physical_devices[gpuNum], True) 

      features = []
      indexeses = []
      #fun sanity check
      if X.shape[2] != len(warm_start_features):
         print("X.shape[2] != len(warm_start_features)")
         exit(1)
      if partNum == 1:  #X_sample = X.sample(n=rand, axis=1)
         rng = np.random.default_rng()
         #IMPORTANT point in next line - we don't want the + 1 after X.shape[2]. This is because indexes is going to be
         #used in warm_start_features, a 0-based array.
         indexeses = rng.choice(range(0,X.shape[2]), size=rand, replace=False) 
         #print("PART 1 indexeses should range from 0 to", X.shape[2], indexeses)
         for indx in indexeses:
            features.append(warm_start_features[indx])
         #print('in func,features is ',features)
      else: #      PART 2 - X_sample = X[recursive_features].sample(n=rand, axis=1) 
         rng = np.random.default_rng()
         if partNum == 3 and rand != len(recursive_features):
            print("In part 3,rand should = len(recursive_features). Exiting.")
            exit(1)
         rec_indexes = rng.choice(range(0,len(recursive_features)), size=rand, replace=False)
         for indx in rec_indexes:
            indexeses.append(warm_start_features.index(recursive_features[indx]))            
            features.append(recursive_features[indx])                   
            print("In Part",partNum,"recursive_features = ",recursive_features, "indx here is",indx," and should range from 0 to len(recursive_features)",len(recursive_features),'going to append warm_start_features_index',warm_start_features.index(recursive_features[indx]),'and recursive_feature',recursive_features[indx],'which should equal',warm_start_features[warm_start_features.index(recursive_features[indx])])
         if partNum == 3:
            print('these should be equal:',features,recursive_features)

      #easiest to maintain compatibility with how things were always done before
      features = pd.Index(features)
      
      X_sample = np.take(X, indexeses, axis=2)
         
      #END ORIGINAL
      num_features = len(features)
      np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
      k = np.floor(len(X_sample)/len(features)).astype('int64')
      k = np.where(k>10, 10, k).astype('int64').item()
      k = np.where(k > (np.sum(y == 1)/k), np.floor(np.sum(y == 1)/k), k).astype('int64').item()
      k = np.where(k > (np.sum(y == 0)/k), np.floor(np.sum(y == 0)/k), k).astype('int64').item()
      k = np.where(k<2, 2, k).astype('int64').item()
      skf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
      fitness_scores = []
      feature_importances_scores = []        
      accuracy_scores = []
      precision_scores = []
      recall_scores = []
      f1_scores = []
      auc_scores = []
      tp_scores = []      
      for train,test in skf.split(X_sample, y): 
         X_train = np.take(X_sample, train, axis=0)
         X_test = np.take(X_sample, test, axis=0)         
         y_train = tf.keras.utils.to_categorical(y[train], num_classes=2)
         y_test = tf.keras.utils.to_categorical(y[test], num_classes=2)
         model = Sequential()
         model.add(Bidirectional(LSTM(layer_size, activation='tanh', return_sequences=True))) #,input_shape=(final_array.shape[1], final_array.shape[2])))) #
         model.add(Bidirectional(LSTM(layer_size, activation='tanh')))
         #model.add(Dense(1)) #for regression
         model.add(Dense(2, activation = 'softmax'))
         opt = None
         if optoption == 'Nadam':
           opt = Nadam(learning_rate = learning, beta_1=beta_1, beta_2=beta_2)
         elif optoption == 'Adamax':
           opt = Adamax(learning_rate = learning, beta_1=beta_1, beta_2=beta_2)
         elif optoption == 'AdamW':
           # AdamW requires a weight decay parameter. I took default below from website
           step = tf.Variable(0, trainable=False)
           schedule = tf.optimizers.schedules.PiecewiseConstantDecay([10000, 15000], [1e-0, 1e-1, 1e-2])
           # lr and wd can be a function or a tensor
           #lr = 1e-1 * schedule(step)
           wd = .01 #Default in pytorch
           opt = tfa.optimizers.AdamW(learning_rate = learning, beta_1=beta_1, beta_2=beta_2, weight_decay=wd,clipnorm=1.)
         elif optoption == 'RMSprop':
           opt = RMSprop(learning_rate = learning, rho=beta_1) #beta_1=beta_1, beta_2=beta_2)         
         else:
           opt = Adam(learning_rate = learning, beta_1=beta_1, beta_2=beta_2)
          
         def acc_score(X,y):
            y_predict_classes = np.argmax(model.predict(X), axis=-1)
            return accuracy_score(y, y_predict_classes)   
         model.compile(loss="categorical_crossentropy", optimizer = opt, metrics=['accuracy'])
         early_stopping_monitor = EarlyStopping(monitor='val_loss',patience=3)
         model.fit(X_train,y_train, epochs=200, validation_data = (X_test, y_test), callbacks=[early_stopping_monitor], verbose=0) #callback used to be early_stopping_monitor
         #Used to be keras had model.predict_classes which would give us nice looking predictions
         #like [1,0] for the 2 output neurons. model.predict will
         #look like [0.9432, 0.0231]. For our accuracy metrics we need it to look like [1,0]

         #further note: below line is for networks with a softmax final layer 
         #(aka >1 output neurons) like we're using here. Not really, just says which one is highest.
         y_predict_classes = np.argmax(model.predict(X_test), axis=-1) 
         resid = y[test] - y_predict_classes
         sse = sum(resid**2)
         sample_size = len(X_sample)
         num_params = num_features
         #Below handles the sse=0 case
         with np.errstate(divide='raise'):
            try:
               fitness = (sample_size * np.log(sse/sample_size)) + (num_params * np.log(sample_size))  
            except FloatingPointError:
               fitness = (sample_size * np.log( 0.3 / sample_size)) + (num_params * np.log(sample_size)) 
         fitness_scores.append(fitness)

         if partNum == 1 or partNum == 3:
            ge = shap.GradientExplainer( model, X_train)
            shap_values = ge.shap_values(X_test)
            if isinstance(shap_values, list):
               if len (shap_values) != 2: #0/1 are possible values of target
                 print('shap_values list should be == numClasses', flush=True)
                 exit(1)
            else:
               print('SHAP_values is NOT a list. Exiting.',flush=True)
               exit(1)
            shap_values = shap_values[1]
            shap_values = np.abs(shap_values) # shape here is (numRows, numTimePeriods, numFeatures)
            #next line averages the feature values across all rows 
            feature_importances = np.mean(shap_values, axis=0) 
            #Now feature_importances Shape is (numTimePeriods,numFeatures)            
            base_score, score_decreases = get_score_importances(acc_score, X_test,y[test])
            tp_importances = np.mean(score_decreases, axis=0)
            tp_scores.append(tp_importances)
            if weighted == "False": #straight average over time periods
               feature_importances = np.mean(feature_importances, axis=0)
            else:
               if np.sum(tp_importances) == 0: #Have to take straight average
                  print("should be rare, in weighted section but weights add to 0", tp_importances)
                  feature_importances = np.mean(feature_importances, axis=0)
               else: #use weighted avg
                  print("Using weighted average")
                  feature_importances = np.average(feature_importances, axis=0, weights = tp_importances)
            #END OF WEIGHTED AVG SECTION
            feature_importances_scores.append(feature_importances)

         accuracy = accuracy_score(y[test], y_predict_classes)
         accuracy_scores.append(accuracy)
         if partNum == 2 or partNum == 3:
            precision = average_precision_score(y[test], y_predict_classes)
            precision_scores.append(precision)
            recall = recall_score(y[test],y_predict_classes)
            recall_scores.append(recall) 
            f1 = f1_score(y[test],y_predict_classes, average='binary')
            f1_scores.append(f1)
            fpr, tpr, thresholds = roc_curve(y[test], y_predict_classes, pos_label=1) #pos_label=1 because our y is in [0,1]
            area_under_curve = auc(fpr, tpr)
            auc_scores.append(area_under_curve)

      mean_accuracy = np.mean(accuracy_scores)             
      mean_fitness = np.mean(fitness_scores)
      if partNum == 1 or partNum == 3:  #Recursion requires importances calculated in part 1 as well.
         mean_tps = np.mean(np.array(tp_scores), axis=0 )
         feature_importances_scores_folds=pd.DataFrame(data=feature_importances_scores) 
         feature_importances_scores_mean = []
         for column in feature_importances_scores_folds:
            col_mean = feature_importances_scores_folds[column].mean()
            feature_importances_scores_mean.append(col_mean)      
      else:
         feature_importances_scores_mean = [0 for i in range(num_features)]
         mean_tps = [0 for i in range(num_features)]

      if partNum == 2 or partNum == 3:
         mean_precision = np.mean(precision_scores)
         mean_recall = np.mean(recall_scores)
         mean_f1 = np.mean(f1_scores)
         mean_auc = np.mean(auc_scores)
      else:
         mean_precision = -99999.0
         mean_recall = -99999.9
         mean_f1 = -999999.0
         mean_auc = -999999.0
         mean_auc = -999999.0

      outq.put( (features, mean_fitness, feature_importances_scores_mean, num_features, mean_accuracy, mean_precision,mean_recall, mean_f1, mean_auc,learning,beta_1, beta_2, mean_tps) )

prog_start_time = time.time()
target = args.target #Condition such as anxiety or somatic 
SD = args.SD         #standard deviations - see recursion script
IDkey = args.IDkey   #row level subject identification column for dataset - for ABCD V4, it is subjectkey
threshold = args.threshold #Lasso coefficient threshold for columns to be included in dataset, we used 0.0
fname = args.target
numFeatures = int(args.numFeatures)
numOfGPUs = args.numOfGPUs
coresPerGPU = args.coresPerGPU
time_periods = str(args.tps) #looks like 0123 (baseline,1yr,2yr,3yr)
#weighted = args.weighted
weighted = "False"
if weighted != "True" and weighted != "False":
   print ("weighted parameter required and must be True or False")
   exit(1)
   
QUEUE_SIZE = coresPerGPU * numOfGPUs  #This 32 will need to be adjusted depending on workload size
if QUEUE_SIZE > len(os.sched_getaffinity(0)):
    print("More cores requested than are avaialble. Exiting.")
    exit(1)
#I want to set output paths here now even though they're not used until the end
#create filepaths

#BEGINNING OF recursion code to bring in previous results
#INPUT PATHS HERE - get results of prior trained models from resample
#stored in dirs like IEL_ann_resample/phenotypic/anx_pres/
input_basepath = "IEL_ann_resample_" + time_periods
if weighted == "True":
   input_basepath += '_weightedavg'
input_basepath += os.path.sep + fname + os.path.sep + fname + '_' + optoption

#filenames are like phenotypic_anx_pres_Adam_models.csv
parameters_filename = input_basepath + '_models.csv'
parameters = pd.read_csv(parameters_filename)
parameters = parameters.drop(parameters.columns[0], axis=1)
learning_min = parameters['learning'].min()
learning_max = parameters['learning'].max()
beta_1_min = parameters['beta_1'].min()
beta_1_max = parameters['beta_1'].max() 
beta_2_min = parameters['beta_2'].min() 
beta_2_max = parameters['beta_2'].max()   

#get results of prior trained features from resample
features_filename = input_basepath + "_features.csv"
features = pd.read_csv(features_filename)
features = features.drop(features.columns[0], axis=1)
    
#get results of prior trained feature importances from resample
importances_filename = input_basepath + "_importances.csv"
importances = pd.read_csv(importances_filename)
importances = importances.drop(importances.columns[0], axis=1)
#END OF INPUT GETTING FROM PREVIOUS STAGE
#Now to set output paths
basepath = "IEL_ann_recursive_" + time_periods + os.path.sep + fname + os.path.sep + str(SD) + "SD" + os.path.sep
os.makedirs(basepath) #exist_ok=True is a param I could use but I don't think I want to.
features_path = fname + '_' + optoption + "_features_" + str(SD) + "SD"
full_features_path = os.path.join(basepath,features_path)
features_filename = full_features_path + '.csv'
models_path = fname + '_' + optoption + "_models" + str(SD) + "SD"
full_models_path = os.path.join(basepath,models_path)
models_filename = full_models_path + '.csv'
importances_path = fname + '_' + optoption + "_importances" + str(SD) + "SD"
full_importances_path = os.path.join(basepath,importances_path)
importances_filename = full_importances_path + '.csv'
#END OF code to bring in previous results
print("Running " + targ + " with this many GPUs: " + str(numOfGPUs) + " and total # of cores is " + str(QUEUE_SIZE), flush=True)
#becuase we need the GPU (mem size of 32 * numOfGPUs) < GPU_Max_Mem_Size (e.g. 40GB)
print("Before process spawning, total memory is:", psutil.virtual_memory().total / 1024 / 1024 / 1024, "available memory is:", (psutil.virtual_memory().available / 1024 / 1024 / 1024),flush=True)
nanfill = 1
#time_periods = '0123'
time_periods_list = [int(t) for t in time_periods] #e.g. 02 will be [0,2]
allTimeLabels = ['baseline', '1Y','2Y','3Y']
timelabels = [ allTimeLabels[i] for i in time_periods_list ]
base_input_dir = '../DLFS_V2Combined'
suffix = ''
if '3yearonset' in targ:
   base_input_dir += '_3yo'
base_input_dir += '/'
print('Using basedir',base_input_dir)
input_dir = base_input_dir + str(threshold) + '/'

targetdf = pd.read_csv('../targets/cbcl_3year_targets_after_methods_tt.csv')
base = pd.read_csv(input_dir + targ + '^baseline_tt_combined_DLFS.csv')
sixM = pd.read_csv(input_dir + targ + '^6_month_tt_combined_DLFS.csv')
one = pd.read_csv(input_dir + targ + '^1_year_tt_combined_DLFS.csv')
eighteenM = pd.read_csv(input_dir + targ + '^18_month_tt_combined_DLFS.csv')
two = pd.read_csv(input_dir + targ + '^2_year_tt_combined_DLFS.csv')
thirtyM = pd.read_csv(input_dir + targ + '^30_month_tt_combined_DLFS.csv')
three = pd.read_csv(input_dir + targ + '^3_year_tt_combined_DLFS.csv')

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
y = y[targ]
print('target length should match earliest row count',len(y))

#When constructing timeseries dataset, it is important to know whether columns are unique
#(only appearing in one time period) or a true timeseries col (column has data for multiple time periods)
#count columns both by year and determine whether they are timeseries cols or not
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
      
unique_columns = np.unique(all_cols_list) #NOTE this still has subjectkey which we will want to remove at the verrrry end.
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
#NOTE empty not zeros below. We want all nans for this experiment
print('Final_array shape',dfsWeWant[0].shape[0], len(dfsWeWant), num_unique_columns)
X = np.empty((dfsWeWant[0].shape[0], len(dfsWeWant), num_unique_columns))
X[:] = nanfill #our masking value we will experiment with 
for timep in range(len(dfsWeWant)):   
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

# SET PARAMETERS for genetic algorithm
sol_per_pop = spp
num_generations = numGens
generations_array = list(range(1,6))
FIFO_len = 30
# num parents mating and num parents mut should be even numbers
num_parents_mating = 40
num_parents_mut = 20
num_parents_rand = int(sol_per_pop - (num_parents_mating/2 + num_parents_mut))
queue_len = num_parents_mating + num_parents_mut

threshold_warm_start = np.float64(0.0005)
current_features = 999999  #-1
warm_start_features = []
begin_desired_range = 0;end_desired_range = 0
while numFeatures != current_features:  #current_features >= numFeatures - 1
   threshold_warm_start += 0.000005
   bool_mask_importances = (importances > threshold_warm_start) | (importances < - threshold_warm_start)
   thresh_features = features[bool_mask_importances]
   names_list = []
   for col in thresh_features:
        ser = thresh_features[col]  #ser is a Series
        #I am thinking this extra handling is due to version changes
        if 'Index' in str(type(ser.values[0])):
           names = ser.values[0].unique()
        else:
           names = ser.unique()
        names = names.tolist()
        names_list = names_list + names
   warm_start_features = pd.Series(data=names_list).dropna()
   warm_start_features = warm_start_features.unique()
   warm_start_features = warm_start_features.tolist()
   #A range of threshold values will actually work, want to try taking
   #midpoint of range instead of the first one that works.
   current_features = len(warm_start_features)
   if current_features == numFeatures and begin_desired_range == 0:
       begin_desired_range = threshold_warm_start
   if current_features < numFeatures : #loop about to exit, take step back
       end_desired_range = threshold_warm_start - 0.0005
   print("Threshold_warm_start",threshold_warm_start,"Number of features is:",len(warm_start_features), flush=True)

print(targ, "warm start features:", len(warm_start_features))
print("features are", warm_start_features)
fi = []
for col in warm_start_features:
    fi.append(unique_columns.index(col))
    print("Building X, appending",unique_columns.index(col),"which should range from 0 to X.shape[2] which is ",X.shape[2])
learn_train_warm_start = np.take(X, fi, axis=2)
features_pop = warm_start_features # learn_train_warm_start.columns.to_list()

#Now to calculate threshold_recursive
import_list = []
for col in importances:
    values = importances[col]
    values = values.dropna()
    values = values.tolist()
    import_list = import_list + values

df_import = pd.DataFrame(data=import_list)
threshold_recursive = df_import.std() * SD #where SD is user supplied param of 0,2,4,etc.
threshold_recursive = threshold_recursive.tolist()[0]

print("threshold_recursive",type(threshold_recursive),threshold_recursive) 
X = learn_train_warm_start 
print("Right before fork, X is ",X,flush=True)

balance_pct = y[y == 1]
balance_pct = balance_pct.shape[0] / y.shape[0]
if balance_pct < 0.45 or balance_pct > 0.55:
   print("Target balance percentage:",balance_pct, "using SMOTENN")
   sm = SMOTEENN(sampling_strategy = 1, random_state=42)
   X,y = sm.fit_resample(X, y)
else:
   print("targets balanced, not using SMOTENN")

if __name__ == '__main__':
 inputQueue = Queue();outputQueue = Queue()
 lock = multiprocessing.Lock()
 print("Right before process starting")
 for i in range(QUEUE_SIZE): #Number of sub procs to run
   Process(target=makeMSEOneOutputModel,args=(inputQueue,outputQueue)).start()
 print("Right after process starting")
  
 #set arrays of #features, learning rate, beta_1, beta_2 for ann initialization
 rand_pop = np.random.randint(3, high=X.shape[2] + 1, size=sol_per_pop).tolist()  #high=len(X.columns), size=sol_per_pop).tolist() 
 learning_pop = np.random.uniform(low=learning_min, high=learning_max, size=sol_per_pop).tolist()
 beta_1_pop = np.random.uniform(low=beta_1_min, high=beta_1_max, size=sol_per_pop).tolist()
 beta_2_pop = np.random.uniform(low=beta_2_min, high=beta_2_max, size=sol_per_pop).tolist()
 #initialize ann
 feature_list = []
 fitness_list = []
 feature_importances_list = []
 num_features_list = []
 learning_rates = []
 accuracy_scores = []
 beta_1_list = []
 beta_2_list = [] 
 tps_list = [] 
 overallCounter = 0
 counter = 0
 gpuNum = 0
 recursive_features = [] #This is dummy variable for part 1
 for rand, learning, beta_1, beta_2 in zip(rand_pop, learning_pop, beta_1_pop, beta_2_pop):
    if counter < QUEUE_SIZE:
      #GPUNum handling - we want to send in a param for which GPU to start function
      #on but ONLY THE FIRST TIME. We are not allowed to change it once process has 
      #started and we want to keep everything in the same queue for efficiency. As 
      #long as we balance out the queue at the beginning, we really don't care which
      #GPU it runs on. We send in real GPUNum code
      #the first time the function runs and then 99 as a flag to never set it again.
      #gpuNum should range from 0 to numOfGPUs
      if overallCounter < QUEUE_SIZE:
         gpuNum = (int  (  (int(overallCounter) / int(QUEUE_SIZE / numOfGPUs)  ) ))
      else:
         gpuNum = 99
      inputQueue.put( (rand,learning,beta_1, beta_2, 1, gpuNum, recursive_features) )
      counter += 1
      overallCounter += 1
    else: #Collect results
      for i in range(QUEUE_SIZE):
        (features, mean_fitness, feature_importances_scores_mean, num_features,mean_accuracy,dummy3,dummy4,dummy5,dummy6,l,b1, b2, mean_tps) = outputQueue.get()
        lock.acquire()
        feature_list.append(features)
        fitness_list.append(mean_fitness)
        feature_importances_list.append(feature_importances_scores_mean)
        num_features_list.append(num_features)
        accuracy_scores.append(mean_accuracy)
        learning_rates.append(l)
        beta_1_list.append(b1)
        beta_2_list.append(b2)
        tps_list.append(mean_tps)
        lock.release()
      #We also need to start next job running after collecting all results
      inputQueue.put( (rand,learning,beta_1, beta_2, 1, 99, recursive_features) )  #Cheating on gpuNum here, all procs should have their gpuNum already 
      counter = 1
      #overallCounter's job is done here, don't need to do anything with it here 
 #have to collect stragglers for when workload not evenly divisible by QUEUE_SIZE
 print("Collecting stragglers: ", counter)
 for i in range(counter):
    (features, mean_fitness, feature_importances_scores_mean, num_features,mean_accuracy,dummy3,dummy4, dummy5,dummy6,l,b1, b2, mean_tps) = outputQueue.get()
    lock.acquire()
    feature_list.append(features)
    fitness_list.append(mean_fitness)
    feature_importances_list.append(feature_importances_scores_mean)
    num_features_list.append(num_features)
    accuracy_scores.append(mean_accuracy)
    learning_rates.append(l)
    beta_1_list.append(b1)
    beta_2_list.append(b2)
    tps_list.append(mean_tps)
    lock.release()
    #mse_scores.append(mean_mse)
 learning_pop = np.copy(learning_rates)
 beta_1_pop = np.copy(beta_1_list)
 beta_2_pop = np.copy(beta_2_list) 
 np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
 ann_features = np.asarray(feature_list, dtype=object)
 ann_fitness = np.asarray(fitness_list)
 ann_importances = feature_importances_list  #np.asarray(feature_importances_list, dtype=object)
 print("first classification initialized for:", targ, " in ",time.time() - prog_start_time," seconds")
 print("ann_importances:",ann_importances)
 print("ann_features:",ann_features)
 #collect best models based on fitness
 recursive_best_fitness_list = []
 recursive_best_feature_list = []
 recursive_best_num_feature_list = []
 recursive_best_learning_list = []
 recursive_best_beta_1_list = []
 recursive_best_beta_2_list = []
 recursive_best_accuracy_list = []
 recursive_best_precision_list = []
 recursive_best_recall_list = []
 recursive_best_f1_list = []
 recursive_best_auc_list = []
 recursive_best_importances_list = []
 recursive_best_generation_list = []
 recursive_best_tps_list = []
 recursive_best_new_fitness_list = []
 
 #initialize recursive loop
 g = 1
 convergence_test = False
 while g in generations_array and convergence_test == False:
    print("Recursive generation is ",g)
    df_importances = pd.DataFrame(data=ann_importances).T
    df_importances = df_importances.astype('float64').fillna(0)
    df_features=pd.DataFrame(data=ann_features).T
    #threshold coef
    #print("df_importances",df_importances)
    bool_mask_importances = (df_importances > threshold_recursive) | (df_importances < - threshold_recursive)
    thresh_features = df_features[bool_mask_importances]
    #get list of recursive features at threshold
    names_list = []
    for col in thresh_features:
        ser = thresh_features[col]  #ser is a Series
        #I am thinking this extra handling is due to version changes
        if 'Index' in str(type(ser.values[0])):
           names = ser.values[0].unique()
        else:
           names = ser.unique()
        names = names.tolist()
        names_list = names_list + names

    recursive_features = pd.Series(data=names_list).dropna()
    recursive_features = recursive_features.unique()
    recursive_features = recursive_features.tolist()
    print("n=", len(recursive_features), "recursive features for target:", targ, "in recursive generation:", g )
    
    if len(recursive_features) > 2:
        print("continuing recursion")
    else:
        convergence_test = True
        print("converged recursive")

    features_pop = recursive_features
    learning_max = max(learning_pop)
    learning_min = min(learning_pop)
    beta_1_max = max(beta_1_pop)
    beta_1_min = min(beta_1_pop)
    beta_2_max = max(beta_2_pop)
    beta_2_min = min(beta_2_pop)
    init_fitness_sorted = np.sort(ann_fitness)
    #reduces to length of queue
    init_top_fitness = init_fitness_sorted[:queue_len]
    queue = init_top_fitness.reshape(queue_len,1)
    #collect best models based on fitness
    best_fitness_list = []
    best_feature_list = []
    best_num_feature_list = []
    best_importances_list = []
    best_learning_list = []
    best_beta_1_list = []
    best_beta_2_list = []
    best_accuracy_list = []
    best_precision_list = []
    best_recall_list = []
    best_f1_list = []
    best_auc_list = []
    best_generation_list = []
    best_tps_list = []
    best_new_fitness_list = []    
    #initialize ga while loop and collection lists
    generation=1
    is_converged = False 
    while generation in range(num_generations) and is_converged == False and convergence_test == False:
       print("Beginning generation loop " + str(generation), flush=True)
       gen_start_time = time.time()
       #generate parameter arrays for mating parents
       fitness_idx = np.argsort(ann_fitness)
       fitness_mating_idx = fitness_idx[:num_parents_mating].tolist()
       fitness_mating = [ann_fitness[i] for i in fitness_mating_idx]
       learning_mating = [learning_pop[i] for i in fitness_mating_idx]
       beta_1_mating = [beta_1_pop[i] for i in fitness_mating_idx]
       beta_2_mating = [beta_2_pop[i] for i in fitness_mating_idx]
    
       #determine features for num_parents_mating
       #features children generated by selecting top half of parent models
       bisect = np.array(num_parents_mating/2, dtype=int)

       # generate children by crossover at pivot point
       # learning rate children
       learning_first = learning_mating[:bisect]
       learning_second = learning_mating[bisect:]
       learning_mate_child = np.add(learning_first, learning_second)/2
       learning_mate_child = np.where(learning_mate_child<learning_min, learning_min, learning_mate_child)
       learning_mate_child = np.where(learning_mate_child>learning_max, learning_max, learning_mate_child)
       learning_mate_child = learning_mate_child.tolist()
       #beta_1 children
       beta_1_first = beta_1_mating[:bisect]
       beta_1_second = beta_1_mating[bisect:]
       beta_1_mate_child = np.add(beta_1_first, beta_1_second)/2
       beta_1_mate_child = np.where(beta_1_mate_child<beta_1_min, beta_1_min, beta_1_mate_child)
       beta_1_mate_child = np.where(beta_1_mate_child>beta_1_max, beta_1_max, beta_1_mate_child)
       beta_1_mate_child = beta_1_mate_child.tolist()
       #beta_2 children
       beta_2_first = beta_2_mating[:bisect]
       beta_2_second = beta_2_mating[bisect:]
       beta_2_mate_child = np.add(beta_2_first, beta_2_second)/2
       beta_2_mate_child = np.where(beta_2_mate_child<beta_2_min, beta_2_min, beta_2_mate_child)
       beta_2_mate_child = np.where(beta_2_mate_child>beta_2_max, beta_2_max, beta_2_mate_child)
       beta_2_mate_child = beta_2_mate_child.tolist()
       #determine remaining population after num_parents_mating removed from each array
       #note that features remainder is larger than parameters remainder since only half the number of parents mating are removed
       fitness_remainder = np.delete(ann_fitness, fitness_mating_idx)
       learning_remainder = np.delete(learning_pop, fitness_mating_idx)
       beta_1_remainder = np.delete(beta_1_pop, fitness_mating_idx)
       beta_2_remainder = np.delete(beta_2_pop, fitness_mating_idx)      
   
       print("After remainders ", flush=True)
       #determine arrays for parameters and features for mutation from remainders
       # note that features mutating are drawn from higher in the remainder queue (which is longer)
       fitness_idx = np.argsort(fitness_remainder)
       fitness_mut_idx = fitness_idx[:num_parents_mut]
       fitness_mut = fitness_remainder[fitness_mut_idx]
       learning_mut = learning_remainder[fitness_mut_idx]
       beta_1_mut = beta_1_remainder[fitness_mut_idx]
       beta_2_mut = beta_2_remainder[fitness_mut_idx]
       #add mutations for learning rate to num_parents_mut by splitting and shifting half 0.01 to right (+) and half 0.01 to left (-)
       #set pivot point
       bisect = np.array(num_parents_mut/2, dtype=int)
       #add mutations to learning rate by splitting and shifting half 0.0001 to right (+) and half 0.0001 to left (-)
       learning_left = learning_mut[:bisect,]
       learning_left_child = learning_left + 0.0001
       learning_right = learning_mut[bisect:,]
       learning_right_child = learning_right - 0.0001
       learning_mut_child = np.append(learning_right_child, learning_left_child)
       learning_mut_child = np.where(learning_mut_child<learning_min, learning_min, learning_mut_child)
       learning_mut_child = np.where(learning_mut_child>learning_max, learning_max, learning_mut_child)
       learning_mut_child = learning_mut_child.tolist()
    
       #add mutations for beta_1 to num_parents_mut by splitting and shifting half 0.001 to right (+) and half 0.001 to left (-)
       beta_1_left = beta_1_mut[:bisect,]
       beta_1_left_child = beta_1_left + 0.001
       beta_1_right = beta_1_mut[bisect:,]
       beta_1_right_child = beta_1_right - 0.001
       beta_1_mut_child = np.append(beta_1_right_child, beta_1_left_child)
       #add floor/ceiling to learning_mut_child
       beta_1_mut_child = np.where(beta_1_mut_child<beta_1_min, beta_1_min, beta_1_mut_child)
       beta_1_mut_child = np.where(beta_1_mut_child>beta_1_max, beta_1_max, beta_1_mut_child)
       beta_1_mut_child = beta_1_mut_child.tolist()
       #add mutations for beta_2 to num_parents_mut by splitting and shifting half 0.001 to right (+) and half 0.001 to left (-)
       beta_2_left = beta_2_mut[:bisect,]
       beta_2_left_child = beta_2_left + 0.001
       beta_2_right = beta_2_mut[bisect:,]
       beta_2_right_child = beta_2_right - 0.001
       beta_2_mut_child = np.append(beta_2_right_child, beta_2_left_child)
       beta_2_mut_child = np.where(beta_2_mut_child<beta_2_min, beta_2_min, beta_2_mut_child)
       beta_2_mut_child = np.where(beta_2_mut_child>beta_2_max, beta_2_max, beta_2_mut_child)
       beta_2_mut_child = beta_2_mut_child.tolist()
       #collect mated and mutated children
       new_learning_child = learning_mate_child + learning_mut_child
       new_beta_1_child = beta_1_mate_child + beta_1_mut_child
       new_beta_2_child = beta_2_mate_child + beta_2_mut_child
       #add new random parents
       learning_rand_child = np.random.uniform(low=learning_min, high=learning_max, size = num_parents_rand).tolist()
       beta_1_rand_child = np.random.uniform(low=beta_1_min, high=beta_1_max, size=num_parents_rand).tolist()
       beta_2_rand_child = np.random.uniform(low=beta_2_min, high=beta_2_max, size=num_parents_rand).tolist()
       feature_cap = len(recursive_features)
       new_rand_array = np.random.randint(2,high=feature_cap + 1, size=num_parents_rand)
    
       #set up new populations
       learning_pop = new_learning_child + learning_rand_child
       beta_1_pop = new_beta_1_child + beta_1_rand_child
       beta_2_pop = new_beta_2_child + beta_2_rand_child
       rand_pop = np.random.randint(2,high=feature_cap + 1, size=sol_per_pop).tolist() 
       #set up lists to collect metrics and features from CV 
       fitness_list = []
       accuracy_list = []
       precision_list = []
       recall_list = []
       f1_list = []
       auc_list = []
       feature_list = []
       feature_importances_list = []
       num_features_list = []
       parallel_learning_pop = []
       parallel_b1_pop = []
       parallel_b2_pop = []
       tps_list = []
       counter = 0
       print("Before feature loop",flush=True)
       total_loop_count = 0
       # run ANN and generate n+1 fitness array of shape (sol_per_pop,) 
       for rand, learning, beta_1, beta_2 in zip(rand_pop, learning_pop, beta_1_pop, beta_2_pop):
           total_loop_count += 1
           if counter < QUEUE_SIZE:
               #See part 1 for why this is set to 99 here
               gpuNum = 99 #(int  (  (int(counter) / int(QUEUE_SIZE / numOfGPUs) ) ))
               inputQueue.put( (rand,learning,beta_1, beta_2, 2, gpuNum, recursive_features) )
               counter += 1
           else: #Collect results
               for i in range(QUEUE_SIZE):
                  (features, mean_fitness, feature_importances_scores_mean, num_features, mean_accuracy, mean_precision,mean_recall, mean_f1, mean_auc,l,b1, b2, mean_tps) = outputQueue.get()
                  lock.acquire()
                  num_features_list.append(num_features)
                  feature_list.append(features)
                  fitness_list.append(mean_fitness)
                  accuracy_list.append(mean_accuracy)
                  precision_list.append(mean_precision)
                  recall_list.append(mean_recall)
                  f1_list.append(mean_f1)
                  auc_list.append(mean_auc)
                  #feature_importances_list.append(feature_importances_scores_mean)
                  parallel_learning_pop.append(l)
                  parallel_b1_pop.append(b1)
                  parallel_b2_pop.append(b2)
                  #tps_list.append(mean_tps)
                  lock.release()
               #We also need to start next job running after collecting all results
               inputQueue.put( (rand,learning,beta_1, beta_2, 2, 99, recursive_features) )  #Cheating on gpuNum here, this is first in the queue so will always be 0
               counter = 1
   #NEED TO COLLECT STRAGGLERS IN CASE WORKLOAD WAS NOT EVENLY DIVISIBLE BY QUEUE_SIZE
       print("Collecting stragglers: ", counter)
       for i in range(counter):
         (features, mean_fitness, feature_importances_scores_mean, num_features, mean_accuracy, mean_precision,mean_recall, mean_f1, mean_auc,l,b1, b2, mean_tps) = outputQueue.get()
         lock.acquire()
         num_features_list.append(num_features)
         feature_list.append(features)
         fitness_list.append(mean_fitness)
         accuracy_list.append(mean_accuracy)
         precision_list.append(mean_precision)
         recall_list.append(mean_recall)
         f1_list.append(mean_f1)
         auc_list.append(mean_auc)
         #feature_importances_list.append(feature_importances_scores_mean)
         parallel_learning_pop.append(l)
         parallel_b1_pop.append(b1)
         parallel_b2_pop.append(b2)
         #tps_list.append(mean_tps)
         lock.release()
       print("After feature loop, which had a total of " + str(total_loop_count) + " iterations",flush=True)
       
       #Resync the arrays that carry over with our parallel ones that may
       #have arrived out of order
       learning_pop = np.copy(parallel_learning_pop)
       beta_1_pop = np.copy(parallel_b1_pop)
       beta_2_pop = np.copy(parallel_b2_pop)
       ann_fitness = np.asarray(fitness_list)
       ann_features = np.asarray(feature_list, dtype=object)
       # pull out top models based on fitness and its parameters and join to repository of best fitness models in every generation
       best_fitness_idx = np.argsort(ann_fitness)
       best_fitness_idx = best_fitness_idx[:3].tolist()
       best_fitness = [ann_fitness[i] for i in best_fitness_idx]
       best_fitness_list = best_fitness_list + best_fitness
       best_learning = [learning_pop[i] for i in best_fitness_idx]
       best_learning_list = best_learning_list + best_learning
       best_beta_1 = [beta_1_pop[i] for i in best_fitness_idx]
       best_beta_1_list= best_beta_1_list + best_beta_1
       best_beta_2 = [beta_2_pop[i] for i in best_fitness_idx]
       best_beta_2_list= best_beta_2_list + best_beta_2
       best_features = [ann_features[i] for i in best_fitness_idx]
       best_feature_list = best_feature_list + best_features
       best_num_features = [num_features_list[i] for i in best_fitness_idx]
       best_num_feature_list = best_num_feature_list + best_num_features
       #NOW IT's TIME to do feature importances
       if len(feature_importances_list) > 0:
         print("ERROR - feature_importances_list at this point should be 0")
         exit(1)
       print("STARTING NEW FEATURE IMPORTANCE GENERATION, best features are", best_features)    
       #This is done this way because we know we want 3 (the top 3). I want a list but I want the indexes to already exist.  
       accuracy_list = [0,1,2]
       precision_list = [0,1,2]
       recall_list = [0,1,2]
       f1_list = [0,1,2]
       auc_list = [0,1,2]       
       feature_importances_list = [ 0, 1, 2 ]
       new_fitness_list = [0,1,2]
       tps_list = [0,1,2]
       for i in range(3):
          #we want to re-create model here as it was found to be best. A little trickier than usual cases
          #because we want to train model with the features it was trained with before and so we have to set
          #un-random selection. So we send best_features and len(best_features) to ensure all are selected.
          inputQueue.put( (len(best_features[i]),best_learning[i],best_beta_1[i], best_beta_2[i], 3, 99, best_features[i]) ) 
       for i in range(3):
          (features, mean_fitness, feature_importances_scores_mean, num_features, mean_accuracy, mean_precision,mean_recall, mean_f1, mean_auc,l,b1, b2, mean_tps) = outputQueue.get()      
          #which result did we get?
          answer = best_learning.index(l) 
          #print("received answer",answer)
          if answer == best_beta_1.index(b1) and answer == best_beta_2.index(b2):
             #print(answer,'checked out')
             #Now getting everything from the new model
             new_fitness_list[answer] = mean_fitness
             accuracy_list[answer] = mean_accuracy
             precision_list[answer] = mean_precision
             recall_list[answer] = mean_recall
             f1_list[answer] = mean_f1
             auc_list[answer] = mean_auc             
             feature_importances_list[answer] = feature_importances_scores_mean
             tps_list[answer] = str(mean_tps)
       print("Importances retrieved")       
       #We do NOT use best_fitness_idx because we have already sorted and already now have the feature_importances in the order we want  
       best_new_fitness = [new_fitness_list[i] for i in range(3)]
       best_new_fitness_list = best_new_fitness_list + best_new_fitness
       best_accuracy = [accuracy_list[i] for i in range(3)]
       best_accuracy_list = best_accuracy_list + best_accuracy
       best_precision = [precision_list[i] for i in range(3)]
       best_precision_list = best_precision_list + best_precision
       best_recall = [recall_list[i] for i in range(3)]
       best_recall_list = best_recall_list + best_recall
       best_f1 = [f1_list[i] for i in range(3)]
       best_f1_list = best_f1_list + best_f1
       best_auc = [auc_list[i] for i in range(3)]
       best_auc_list = best_auc_list + best_auc        
       best_importances = [feature_importances_list[i] for i in range(3)] #best_fitness_idx]
       best_importances_list = best_importances_list + best_importances
       best_tps = [tps_list[i] for i in range(3)] #best_fitness_idx]
       best_tps_list = best_tps_list + best_tps        
       best_generation_list.append(generation);best_generation_list.append(generation);best_generation_list.append(generation)
       # FIFO queue
       # pull out top fitness models and join to queue
       top_fitness_idx = np.argsort(ann_fitness) 
       top_fitness_idx = top_fitness_idx[:queue_len]  #takes top 60 (0-59)
       fitness_round = np.round(ann_fitness)
       top_fitness = fitness_round[top_fitness_idx]
       top_fitness = top_fitness.reshape(queue_len,1)  #so this typically is 60,1
       queue = np.hstack((queue, top_fitness)) #hstack is basically array concat
   
       print(str(queue.shape) + " being checked against a shape of " + str(queue_len) + " " + str(FIFO_len), flush=True)
           
       if queue.shape < (queue_len, FIFO_len):
          generation += 1
       else:
          queue = np.delete(queue,0,1)
          generation += 1
          print("Generation " + str(generation - 1) + " completed in " + str(time.time() - gen_start_time) + " seconds", flush=True)
          print("Available memory is now: " + str(psutil.virtual_memory().available / 1024 / 1024 / 1024) + "GB", flush=True)
    recursive_best_fitness_list = recursive_best_fitness_list + best_fitness_list
    recursive_best_feature_list = recursive_best_feature_list + best_feature_list
    recursive_best_num_feature_list = recursive_best_num_feature_list + best_num_feature_list
    recursive_best_learning_list = recursive_best_learning_list + best_learning_list
    recursive_best_beta_1_list = recursive_best_beta_1_list + best_beta_1_list
    recursive_best_beta_2_list = recursive_best_beta_2_list + best_beta_2_list
    recursive_best_accuracy_list = recursive_best_accuracy_list + best_accuracy_list
    recursive_best_precision_list = recursive_best_precision_list + best_precision_list
    recursive_best_recall_list = recursive_best_recall_list + best_recall_list
    recursive_best_f1_list = recursive_best_f1_list + best_f1_list
    recursive_best_auc_list = recursive_best_auc_list + best_auc_list
    recursive_best_importances_list = recursive_best_importances_list + best_importances_list
    recursive_best_generation_list = recursive_best_generation_list + best_generation_list
    recursive_best_tps_list = recursive_best_tps_list + best_tps_list
    recursive_best_new_fitness_list = recursive_best_new_fitness_list + best_new_fitness_list        
    g += 1
   #END OF while generation in range(num_generations)
   #convergence_test == is_converged #??? This statement should have no effect.
 #collect best recursive models from each generation based on fitness
 best_ann_ga_features = pd.DataFrame(data=recursive_best_feature_list).T
 best_ann_ga_importances = pd.DataFrame(data=recursive_best_importances_list).T
 best_fitness_models = np.stack((recursive_best_fitness_list, recursive_best_learning_list, recursive_best_beta_1_list, recursive_best_beta_2_list, recursive_best_accuracy_list, recursive_best_precision_list, recursive_best_recall_list, recursive_best_f1_list, recursive_best_auc_list, recursive_best_num_feature_list, recursive_best_generation_list, recursive_best_tps_list, recursive_best_new_fitness_list), axis=1)
 best_ann_ga_models = None
 best_ann_ga_models = pd.DataFrame(data=best_fitness_models, columns = ['fitness', 'learning', 'beta_1', 'beta_2', 'accuracy', 'precision', 'recall', 'f1','auc', 'num_features', 'generation', 'TP_importances', 'New_fitness'])
  #save output
 best_ann_ga_features.to_csv(features_filename)
 best_ann_ga_models.to_csv(models_filename)
 best_ann_ga_importances.to_csv(importances_filename)
 print("time elapsed is:", np.round((time.time() - prog_start_time)/60), "minutes")
 print("Gathering processes...",flush=True)
 for i in range(QUEUE_SIZE): #Number of sub procs to run
       inputQueue.put('STOP')
       #processList[i].join()
 print("Processess gathered...",flush=True)
 
 exit(0)

