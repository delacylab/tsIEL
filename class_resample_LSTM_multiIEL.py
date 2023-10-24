import pandas as pd
import numpy as np
from eli5.permutation_importance import get_score_importances
import eli5
from sklearn.metrics import accuracy_score,average_precision_score,recall_score
from sklearn.metrics import f1_score, roc_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from datetime import datetime
import multiprocessing
from multiprocessing import Pool,Process,Queue, current_process
import shap
import shutil
import copy
import sys
import argparse
import os
from collections import Counter
import time
import psutil
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('target',help='prefix of DLFS file to use, not including .csv')
parser.add_argument('IDkey',help='DB Subject Identifier header')
parser.add_argument('numOfGPUs',type=int,help='Number of GPUs to use')
parser.add_argument('coresPerGPU',type=int, help='Cores Per GPU')
parser.add_argument('threshold',help='lasso threshold to use')
#parser.add_argument('weighted',help='Use weighted avg, True or False')
parser.add_argument('tps',help='should look like 0123')
args = parser.parse_args()
prog_start_time = time.time()
target = args.target #Condition such as anxiety or somatic 
IDkey = args.IDkey   #row level subject identification column for dataset - for ABCD V4, it is subjectkey
threshold = args.threshold #Lasso coefficient threshold for columns to be included in dataset, we used 0.0
fname = args.target
numOfGPUs = args.numOfGPUs
coresPerGPU = args.coresPerGPU
QUEUE_SIZE = coresPerGPU * numOfGPUs  #This 32 will need to be adjusted depending on workload size
time_periods = str(args.tps) #Don't want any confusion here
print(time_periods)
weighted = "False" #args.weighted
if weighted != "True" and weighted != "False":
   print ("weighted parameter required and must be True or False")
   exit(1)

if QUEUE_SIZE > len(os.sched_getaffinity(0)):
    print("More cores requested than are avaialble. Exiting.")
    exit(1)
   
layer_size = 300 
spp = 100 
numGens = 401 
optoption = 'AdamW'

#create filepaths for output
basepath = "IEL_ann_resample_" + time_periods
if weighted == "True":
   basepath += '_weightedavg'
basepath += os.path.sep + fname + os.path.sep

os.makedirs(basepath, exist_ok=True) #exist_ok=True is a param I could use but I don't want to.
features_path = fname + '_' + optoption + "_features"
full_features_path = os.path.join(basepath,features_path)
features_filename = full_features_path + '.csv'
models_path = fname + '_' + optoption + "_models"
full_models_path = os.path.join(basepath,models_path)
models_filename = full_models_path + '.csv'
importances_path = fname + '_' + optoption + "_importances"
full_importances_path = os.path.join(basepath,importances_path)
importances_filename = full_importances_path + '.csv'
#New files for TP and importances
features_path = fname + '_' + optoption + "_TPfeatures"
full_features_path = os.path.join(basepath,features_path)
features_TP_filename = full_features_path + '.csv'
models_path = fname + '_' + optoption + "_TPmodels"
full_models_path = os.path.join(basepath,models_path)
models_TP_filename = full_models_path + '.csv'
importances_path = fname + '_' + optoption + "_TPimportances"
full_importances_path = os.path.join(basepath,importances_path)
importances_TP_filename = full_importances_path + '.csv'

# SET PARAMETERS for genetic algorithm
sol_per_pop = spp
num_generations = numGens
FIFO_len = 30
# num parents mating and num parents mut should be even numbers
num_parents_mating = 40
num_parents_mut = 20
num_parents_rand = int(sol_per_pop - (num_parents_mating/2 + num_parents_mut))
queue_len = num_parents_mating + num_parents_mut

learning_min = 0.00001
learning_max = 0.01
beta_1_min = 0.9
beta_1_max = 0.999
beta_2_min = 0.9
beta_2_max = 0.999

def parallel_func(iq, outq):
      #Imports are done here because of an odd CUDA limitation that prevents
      #forked multiprocesses from getting another CUDA context. 
   for rand,learning,beta_1, beta_2, partNum, gpuNum in iter(iq.get,'STOP'):
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
      import tensorflow as tf
      from tensorflow.keras.optimizers import Adam, Adamax, Nadam, RMSprop
      from tensorflow.keras.callbacks import EarlyStopping
      from tensorflow.keras.layers import Dense, LSTM, Bidirectional
      from tensorflow.keras.models import Sequential
      import tensorflow_addons as tfa
      tf.get_logger().setLevel('ERROR')   
      import logging
      logging.disable(logging.WARNING)

      tf.config.threading.set_inter_op_parallelism_threads(1)
      tf.config.threading.set_intra_op_parallelism_threads(1) 

      if gpuNum != 99:
         physical_devices = tf.config.list_physical_devices('GPU')
         tf.config.set_visible_devices(physical_devices[gpuNum], 'GPU')
         tf.config.experimental.set_memory_growth(physical_devices[gpuNum], True) 

      #NOTE in part 2 case, rand is really feature
      features = []
      indexeses = []
      if partNum == 1:
         rng = np.random.default_rng()
         #IMPORTANT point in next line - we don't want the + 1 after X.shape[2]. This is because indexes is going to be
         #used in unique_columns, a 0-based array.
         indexeses = rng.choice(range(0,X.shape[2]), size=rand, replace=False)  
         for indx in indexeses:
            features.append(unique_columns[indx])
      else:
         for col in rand:
            indexeses.append(unique_columns.index(col))
         features = rand
         
      #New part making a new features list for TP/features.         
      features_TPs = []
      for f in features:
         for t in range(X.shape[1]):
            features_TPs.append(f + '_TP' + str(t))
         
      X_sample = np.take(X, indexeses, axis=2)
      if 'Series' in str(type(X_sample)):
         print("X_sample is series,not df. Partnum,rand",partNum,rand)   
         exit(1)
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
      feature_importances_scores_withTPs = []
      
      for train,test in skf.split(X_sample, y): 
         #X_train = X_sample[train]
         #X_test = X_sample[test]
         X_train = np.take(X_sample, train, axis=0)
         X_test = np.take(X_sample, test, axis=0)
         #print("X_train and test shapes",type(X_train), type(X_test),X_train.shape, X_test.shape)
         y_train = tf.keras.utils.to_categorical(y[train], num_classes=2)  
         y_test = tf.keras.utils.to_categorical(y[test], num_classes=2)       
         model = Sequential()
         #NOTE: tanh activation function runs way faster because CUDnn is optimized for it.
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
            #return mean_squared_error(y, y_predict) 
            return accuracy_score(y,y_predict_classes)
 
         model.compile(loss="categorical_crossentropy", optimizer = opt, metrics=['accuracy'])
         early_stopping_monitor = EarlyStopping(monitor='val_loss',patience=3)
         model.fit(X_train,y_train, epochs=200, validation_data = (X_test, y_test), callbacks=[early_stopping_monitor], verbose=0) 
         y_predict = np.argmax(model.predict(X_test), axis=-1)  #model.predict(X_test)
         #INLINING BIC_pop_fitness here
         resid = y[test] - y_predict
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
         
         #Feature importances for Part 3 only due to time it takes to compute them.
         if partNum == 3:
            ge = shap.GradientExplainer( model, X_train)
            shap_values = ge.shap_values(X_test)
            if isinstance(shap_values, list):
               if len (shap_values) != 2:
                 print('shap_values list should be == numClasses', flush=True)
                 exit(1)
            else:
               print('SHAP_values is NOT a list. Exiting.',flush=True)
               exit(1)
            shap_values = shap_values[1]
            shap_values = np.abs(shap_values)
            #next line averages the feature values across all rows
            feature_importances = np.mean(shap_values, axis=0) 
            #Now feature_importances Shape is (numTimePeriods,numFeatures)
            feature_importances_withTPs = copy.deepcopy(feature_importances)
            feature_importances_scores_withTPs.append(feature_importances_withTPs)
            
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
             
         accuracy = accuracy_score(y[test], y_predict)
         accuracy_scores.append(accuracy)
         if partNum == 2 or partNum == 3:
            precision = average_precision_score(y[test], y_predict)
            precision_scores.append(precision)
            recall = recall_score(y[test],y_predict)
            recall_scores.append(recall)
            f1 = f1_score(y[test],y_predict, average='binary')
            f1_scores.append(f1)
            fpr, tpr, thresholds = roc_curve(y[test], y_predict, pos_label=1) #pos_label=1 because our y is in [0,1]
            area_under_curve = auc(fpr, tpr)
            auc_scores.append(area_under_curve)
            
      mean_accuracy = np.mean(accuracy_scores) #see above about MSE formerly only in part 2
      mean_fitness = np.mean(fitness_scores)

      if partNum == 3:
         mean_tps = np.mean(np.array(tp_scores), axis=0 )
         feature_importances_scores_folds=pd.DataFrame(data=feature_importances_scores) 
         feature_importances_scores_mean = []
         #print("OLD HONK", len(feature_importances_scores),feature_importances_scores)
         for column in feature_importances_scores_folds:
            col_mean = feature_importances_scores_folds[column].mean()
            feature_importances_scores_mean.append(col_mean)      
         feature_importances_scores_withTPs_mean = np.mean(feature_importances_scores_withTPs, axis=0)
         features_TPs_flatten = []
         countercheck = 0
         for f in range(len(features)):
            for t in range(X.shape[1]):
               #print('Feature,value',features[f], feature_importances_scores_withTPs_mean[t, f ])
               if features[f] not in features_TPs[countercheck]:
                  print('feature name mismatch')
               features_TPs_flatten.append(feature_importances_scores_withTPs_mean[t, f ])
               countercheck += 1                   
      else:
         feature_importances_scores_mean = [0 for i in range(num_features)]
         mean_tps = [0 for i in range(num_features)]
         features_TPs_flatten = [0 for i in range(num_features * X.shape[1])]
                   
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
      
      outq.put( (features, mean_fitness, feature_importances_scores_mean, num_features, mean_accuracy, mean_precision,mean_recall, mean_f1, mean_auc,learning,beta_1, beta_2, mean_tps, features_TPs, features_TPs_flatten ) )

#Notes from previous 0.1 versions - nanfill will always be 1. Time periods will always be
#0123
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
#I'll keep the monthlys here although I don't intend to use them for a while if ever
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
print("0s/1s count for",targ,len(np.where(y == 0)[0]), len(np.where(y == 1)[0]))
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
   #final_array[:,timep,:] = dfsWeWant[timep][unique_columns].to_numpy()      
   for col in dfsWeWant[timep].columns:
      if col in ['subjectkey', 'eventname']:
         continue
      #need to get index of this in unique_columns
      bigindex = unique_columns.index(col)
      #The following commented line would create a matrix stacked to fill the early time periods.
      #where even a column only appearing at 2 year would be pushed into the baseline time period
      #so occurences rather than actual time period
      #final_array[:, runningColIndexes[bigindex] ,bigindex] = dfsWeWant[timep][col].to_numpy()
      #THE BELOW LINE WILL COPY data where its time period should be. 
      #So 2 year variables will only appear in twoyear time period.
      X[:, timep, bigindex] = dfsWeWant[timep][col].to_numpy()
      runningColIndexes[bigindex] += 1
      coef_lookup[col].append(timelabels[timep])  
         
#shuffle shuffles along first dimension only so it's ok to use for 2D or 3D matrices
#parallel_func uses kfold so no shuffling necessary
#final_array, y = shuffle(final_array, y, random_state=0)
if __name__ == '__main__':
 inputQueue = Queue();outputQueue = Queue()
 lock = multiprocessing.Lock()
 print("Right before process starting")
 for i in range(QUEUE_SIZE): #Number of sub procs to run
   Process(target=parallel_func,args=(inputQueue,outputQueue)).start()
 print("Right after process starting")
 
 numColsToUse = X.shape[2]
 if numColsToUse > 15:
     numColsToUse = 15
 print("Capping feature count at",numColsToUse)    
 #set arrays of #features, learning rate, beta_1, beta_2 for ann initialization
 rand_pop = np.random.randint(2,high=numColsToUse + 1, size=sol_per_pop).tolist()
 learning_pop = np.random.uniform(low=learning_min, high=learning_max, size=sol_per_pop).tolist()
 beta_1_pop = np.random.uniform(low=beta_1_min, high=beta_1_max, size=sol_per_pop).tolist()
 beta_2_pop = np.random.uniform(low=beta_2_min, high=beta_2_max, size=sol_per_pop).tolist()
 #initialize ann
 feature_list = []
 fitness_list = []
 feature_importances_list = []
 featureTP_list = []
 feature_importances_TP_list = []
 accuracy_scores = []
 num_features_list = []
 learning_rates = []
 beta_1_list = []
 beta_2_list = []
 tps_list = []
 overallCounter = 0
 counter = 0
 gpuNum = 0
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
      inputQueue.put( (rand,learning,beta_1, beta_2, 1, gpuNum) )
      counter += 1
      overallCounter += 1
    else: #Collect results, NOTE LAST 2 output params are blank and not used here
      for i in range(QUEUE_SIZE):
        (features, mean_fitness, feature_importances_scores_mean, num_features,mean_accuracy,dummy3,dummy4,dummy5,dummy6,l,b1, b2, mean_tps, features_TPs, feature_TPs_flatten) = outputQueue.get()
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
        featureTP_list.append(features_TPs)
        feature_importances_TP_list.append(feature_TPs_flatten)        
        lock.release()
      #We also need to start next job running after collecting all results
      inputQueue.put( (rand,learning,beta_1, beta_2, 1, 99) )  #99=all procs should have their gpuNum already 
      counter = 1
      #overallCounter's job is done here, don't need to do anything with it here 
 #have to collect stragglers for when workload not evenly divisible by QUEUE_SIZE
 print("Collecting stragglers: ", counter)
 for i in range(counter):
    (features, mean_fitness, feature_importances_scores_mean, num_features,mean_accuracy,dummy3,dummy4,dummy5,dummy6,l,b1, b2, mean_tps, features_TPs, feature_TPs_flatten) = outputQueue.get()
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
    featureTP_list.append(features_TPs)
    feature_importances_TP_list.append(feature_TPs_flatten)     
    lock.release()

 #This is important because results will arrive out of order and we want all indexes to be synchronized.        
 learning_pop = np.copy(learning_rates)
 beta_1_pop = np.copy(beta_1_list)
 beta_2_pop = np.copy(beta_2_list) 
 print("PART 1 COMPLETE - first classification initialized for:", targ, " in ",time.time() - prog_start_time," seconds")
 
 np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
 ann_features = np.asarray(feature_list, dtype=object)
 ann_fitness = np.asarray(fitness_list)
 
 #initialize queue
 init_fitness_sorted = np.sort(ann_fitness)
 #reduces to length of queue
 init_top_fitness = init_fitness_sorted[:queue_len]
 init_top_fitness_round = np.round(init_top_fitness)
 queue = init_top_fitness_round.reshape(queue_len,1)
 #initialize ga while loop and collection lists
 generation=1
 is_converged = False 
 #collect best models based on fitness
 best_fitness_list = []
 best_feature_list = []
 best_num_feature_list = []
 best_importances_list = []
 best_featureTP_list = []
 best_importances_TPs_list = []
 best_learning_list = []
 best_beta_1_list = []
 best_beta_2_list = []
 best_new_fitness_list = []
 best_accuracy_list = []
 best_precision_list = []
 best_recall_list = []
 best_f1_list = []
 best_auc_list = []
 best_tps_list = []  #They will all be best
 roll_std_list = []
 roll_min_list = []
 roll_abs_std_list = []
 roll_abs_min_list = []
 std_std_list = []
 std_min_list = []
 mean_std_list = []
 std_mean_list = []
 percentError_list = []
 best_generation_list = []
 while generation in range(num_generations) and is_converged == False:
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
 
   #print("Before learning/mate_childs " + str(bisect), flush=True)
   # generate children by crossover at pivot point
   # learning rate children
   learning_first = learning_mating[:bisect]
   learning_second = learning_mating[bisect:]
   learning_mate_child = np.add(learning_first, learning_second)/2
   learning_mate_child = np.where(learning_mate_child<learning_min, learning_min, learning_mate_child)
   learning_mate_child = np.where(learning_mate_child>learning_max, learning_max, learning_mate_child)
   learning_mate_child = learning_mate_child.tolist()
   #print("After learning/mate_childs " + str(learning_mate_child), flush=True)
   #beta_1 children
   beta_1_first = beta_1_mating[:bisect]
   beta_1_second = beta_1_mating[bisect:]
   beta_1_mate_child = np.add(beta_1_first, beta_1_second)/2
   beta_1_mate_child = np.where(beta_1_mate_child<beta_1_min, beta_1_min, beta_1_mate_child)
   beta_1_mate_child = np.where(beta_1_mate_child>beta_1_max, beta_1_max, beta_1_mate_child)
   beta_1_mate_child = beta_1_mate_child.tolist()
   #print("After beta_1_mate_childs " + str(beta_1_mate_child), flush=True)
   #beta_2 children
   beta_2_first = beta_2_mating[:bisect]
   beta_2_second = beta_2_mating[bisect:]
   beta_2_mate_child = np.add(beta_2_first, beta_2_second)/2
   beta_2_mate_child = np.where(beta_2_mate_child<beta_2_min, beta_2_min, beta_2_mate_child)
   beta_2_mate_child = np.where(beta_2_mate_child>beta_2_max, beta_2_max, beta_2_mate_child)
   beta_2_mate_child = beta_2_mate_child.tolist()
   #print("After beta_2_mate_childs " + str(beta_2_mate_child), flush=True)
    #determine mating features: top half of parent models to pivot point
   fitness_for_features_mating_idx = fitness_idx[:bisect]
   features_mate_child = [ann_features[i] for i in fitness_for_features_mating_idx]
   #determine remaining population after num_parents_mating removed from each array
   #note that features remainder is larger than parameters remainder since only half the number of parents mating are removed
   fitness_remainder = np.delete(ann_fitness, fitness_mating_idx)
   fitness_remainder_for_features = np.delete(ann_fitness,fitness_for_features_mating_idx)
   features_remainder = np.delete(ann_features, fitness_for_features_mating_idx)
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
        #determine features mut child
   features_idx = np.argsort(fitness_remainder_for_features)
   features_mut_idx = features_idx[:num_parents_mut].tolist()
   features_mut_child = [features_remainder[i] for i in features_mut_idx]    
        #collect mated and mutated children
   new_learning_child = learning_mate_child + learning_mut_child
   new_beta_1_child = beta_1_mate_child + beta_1_mut_child
   new_beta_2_child = beta_2_mate_child + beta_2_mut_child
   new_features_child = features_mate_child + features_mut_child
    #add new random parents
   learning_rand_child = np.random.uniform(low=learning_min, high=learning_max, size = num_parents_rand).tolist()
   beta_1_rand_child = np.random.uniform(low=beta_1_min, high=beta_1_max, size=num_parents_rand).tolist()
   beta_2_rand_child = np.random.uniform(low=beta_2_min, high=beta_2_max, size=num_parents_rand).tolist()
   
   colsize = X.shape[2]
   if colsize > 15:
     colsize = 15     
   features_for_X_array = []
   rng = np.random.default_rng()
   #below is ok because duplication is ok here
   new_rand_array = rng.integers(low=2,high=colsize + 1, size=num_parents_rand)
   for rand in new_rand_array:
      sublist = []
      #IMPORTANT point in next line - we don't want the + 1 after X.shape[2]. This is because indexes is going to be
      #used in unique_columns, a 0-based array.
      indexeses = rng.choice(range(0,X.shape[2]), size=rand, replace=False) 
      for indx in indexeses:
         sublist.append(unique_columns[indx])
      features_for_X_array.append(sublist)
   features_rand_child = features_for_X_array
    
   #set up new populations
   learning_pop = new_learning_child + learning_rand_child
   beta_1_pop = new_beta_1_child + beta_1_rand_child
   beta_2_pop = new_beta_2_child + beta_2_rand_child
   features_pop = new_features_child + features_rand_child
    #set up lists to collect metrics and features from CV 
   fitness_list = []
   accuracy_list = []
   precision_list = []
   recall_list = []
   f1_list = []
   auc_list = []
   feature_list = []
   feature_importances_list = []
   featureTP_list = []
   feature_importances_TP_list = []      
   num_features_list = []
   tps_list = []
   counter = 0
   total_loop_count = 0
   parallel_learning_pop = []
   parallel_b1_pop = []
   parallel_b2_pop = []
   # run ANN and generate n+1 fitness array of shape (sol_per_pop,) 
   for feature, learning, beta_1, beta_2 in zip(features_pop, learning_pop, beta_1_pop, beta_2_pop):
     #print('total_loop_count, feature', total_loop_count, feature)
     total_loop_count += 1
     if counter < QUEUE_SIZE:
        #See part 1 for why this is set to 99 here
        gpuNum = 99 #(int  (  (int(counter) / int(QUEUE_SIZE / numOfGPUs)  ) ))
        inputQueue.put( (feature,learning,beta_1, beta_2, 2, gpuNum) )
        counter += 1
     else: #Collect results
        for i in range(QUEUE_SIZE):
           (features, mean_fitness, feature_importances_scores_mean, num_features, mean_accuracy, mean_precision,mean_recall, mean_f1, mean_auc,l,b1, b2, mean_tps, features_TPs, feature_TPs_flatten) = outputQueue.get()
           lock.acquire()
           num_features_list.append(num_features)
           feature_list.append(features)
           fitness_list.append(mean_fitness)
           accuracy_list.append(mean_accuracy)
           precision_list.append(mean_precision)
           recall_list.append(mean_recall)
           f1_list.append(mean_f1)
           auc_list.append(mean_auc)
           #tps_list.append(mean_tps)
           #feature_importances_list.append(feature_importances_scores_mean)
           parallel_learning_pop.append(l)
           parallel_b1_pop.append(b1)
           parallel_b2_pop.append(b2)
           featureTP_list.append(features_TPs)
           feature_importances_TP_list.append(feature_TPs_flatten)            
           lock.release()
        #We also need to start next job running after collecting all results
        inputQueue.put( (feature,learning,beta_1, beta_2, 2, 99) )  #Cheating on gpuNum here, this is first in the queue so will always be 0
        counter = 1
   #NEED TO COLLECT STRAGGLERS IN CASE WORKLOAD WAS NOT EVENLY DIVISIBLE BY QUEUE_SIZE
   print("Collecting stragglers: ", counter)
   for i in range(counter):
      (features, mean_fitness, feature_importances_scores_mean, num_features, mean_accuracy, mean_precision,mean_recall, mean_f1, mean_auc,l,b1, b2, mean_tps, features_TPs, feature_TPs_flatten) = outputQueue.get()
      lock.acquire()
      num_features_list.append(num_features)
      feature_list.append(features)
      fitness_list.append(mean_fitness)
      accuracy_list.append(mean_accuracy)
      precision_list.append(mean_precision)
      recall_list.append(mean_recall)
      f1_list.append(mean_f1)
      auc_list.append(mean_auc)
      #tps_list.append(mean_tps)
      #feature_importances_list.append(feature_importances_scores_mean)
      parallel_learning_pop.append(l)
      parallel_b1_pop.append(b1)
      parallel_b2_pop.append(b2)
      featureTP_list.append(features_TPs)
      feature_importances_TP_list.append(feature_TPs_flatten)       
      lock.release()
   print("After feature loop, which had a total of " + str(total_loop_count) + " iterations",flush=True)
   #we have an interesting problem with learning_pop and parallel_learning_pop, etc. here. We want these to carry over from part 1 to part 2.
   #But after one iteration, we need to resync aka we now need to call learning_pop, beta_1_pop, beta_2_pop their parallel counterparts.
   #But first let's verify that they are element-wise (NOT ORDER-wise matches)
   #This is because we do not get results in the order they were sent
   if len(learning_pop) != len(parallel_learning_pop):
      print("Learning pop size mismatch")
      exit(1)
   if len(beta_1_pop) != len(parallel_b1_pop):
      print("b1 pop size mismatch")
      exit(1)
   if len(beta_2_pop) != len(parallel_b2_pop):
      print("b2 pop size mismatch")
      exit(1)      
   for n in learning_pop:
      if n not in parallel_learning_pop:
         print("We have a serious problem with learning")
         exit(1)
   for n in beta_1_pop:
      if n not in parallel_b1_pop:
         print("We have a serious problem with beta1")
         exit(1)
   for n in beta_2_pop:
      if n not in parallel_b2_pop:
         print("We have a serious problem with beta2")
         exit(1)            
   learning_pop = np.copy(parallel_learning_pop)
   beta_1_pop = np.copy(parallel_b1_pop)
   beta_2_pop = np.copy(parallel_b2_pop)
   #end of re-ordering of learning/b1/b2
      
   #ann_fitness = fitness_list
   #ann_features = feature_list
   ann_fitness = np.asarray(fitness_list)
   ann_features = np.asarray(feature_list, dtype=object)
   #New thing.
   ann_TP_features = np.asarray(featureTP_list, dtype=object)
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
   best_features_TPs = [ann_TP_features[i] for i in best_fitness_idx]
   best_featureTP_list = best_featureTP_list + best_features_TPs
   best_num_features = [num_features_list[i] for i in best_fitness_idx]
   best_num_feature_list = best_num_feature_list + best_num_features
   #NOW IT's TIME to do feature importances
   if len(feature_importances_list) > 0:
      print("ERROR - feature_importances_list at this point should be 0")
      exit(1)  
   #This is done this way because we know we want 3 (the top 3). I want a list but I want the indexes to already exist.  
   accuracy_list = [0,1,2]
   precision_list = [0,1,2]
   recall_list = [0,1,2]
   f1_list = [0,1,2]
   auc_list = [0,1,2]
   feature_importances_list = [ 0, 1, 2 ]
   feature_importances_TP_list = [ 0, 1, 2 ]
   new_fitness_list = [0,1,2]
   tps_list = [0,1,2]
   for i in range(3):
      inputQueue.put( (best_features[i],best_learning[i],best_beta_1[i], best_beta_2[i], 3, 99) ) 
   for i in range(3):
      (features, mean_fitness, feature_importances_scores_mean, num_features, mean_accuracy, mean_precision,mean_recall, mean_f1, mean_auc,l,b1, b2, mean_tps, features_TPs, feature_TPs_flatten) = outputQueue.get()      
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
         feature_importances_TP_list[answer] = feature_TPs_flatten
         
   #importances retrieved.
   print("Importances retrieved")
   #We do NOT use best_fitness_idx because we have already sorted and already now have the feature_importances in the order we want 
   best_new_fitness = [new_fitness_list[i] for i in range(3)]
   best_new_fitness_list = best_new_fitness_list + best_new_fitness
   best_acc = [accuracy_list[i] for i in range(3)]
   best_accuracy_list = best_accuracy_list + best_acc
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
   best_importances_TPs = [feature_importances_TP_list[i] for i in range(3)]
   best_importances_TPs_list = best_importances_TPs_list + best_importances_TPs
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
   #END OF while generation in range(num_generations)
   #convergence_test == is_converged #??? This statement should have no effect.
 #collect best recursive models from each generation based on fitness
 best_ann_ga_features = pd.DataFrame(data=best_feature_list).T
 best_ann_ga_importances = pd.DataFrame(data=best_importances_list).T
 #interesting to see what happens here
 best_ann_ga_featuresTPs = pd.DataFrame(data=best_featureTP_list).T
 best_ann_ga_importancesTPs = pd.DataFrame(data=best_importances_TPs_list).T 
 
 best_fitness_models = np.stack((best_fitness_list, best_learning_list, best_beta_1_list, best_beta_2_list, best_accuracy_list, best_precision_list, best_recall_list, best_f1_list, best_auc_list, best_num_feature_list, best_generation_list, best_tps_list, best_new_fitness_list), axis=1)
 best_ann_ga_models = None
 best_ann_ga_models = pd.DataFrame(data=best_fitness_models, columns = ['fitness', 'learning', 'beta_1', 'beta_2', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'num_features', 'generation', 'time_period_importances', 'New_fitness'])
 
  #save output
 best_ann_ga_features.to_csv(features_filename)
 best_ann_ga_models.to_csv(models_filename)
 best_ann_ga_importances.to_csv(importances_filename)
 #Now save new TP output as well - note no models file
 best_ann_ga_featuresTPs.to_csv(features_TP_filename)
 #best_ann_ga_modelsTPs.to_csv(models_TP_filename)
 best_ann_ga_importancesTPs.to_csv(importances_TP_filename) 
 #Sanity check on features, make sure no duplicate bug
 for col in best_ann_ga_features.columns:
   curcol = best_ann_ga_features[col].dropna()
   if curcol.size != curcol.unique().shape[0]:
      print("UNIQUE FEATURE ERROR - col,curcol",col,curcol, '. Exiting.')
      for i in range(QUEUE_SIZE): #Number of sub procs to run
        inputQueue.put('STOP')
      exit(1)
       
 print("Ending properly,time elapsed is:", np.round((time.time() - prog_start_time)/60), "minutes")
 print("Gathering processes...",flush=True)
 for i in range(QUEUE_SIZE): #Number of sub procs to run
       inputQueue.put('STOP')
       #processList[i].join()
 print("Processess gathered...",flush=True)
 
 exit(0)

      
