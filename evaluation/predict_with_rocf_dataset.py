'''
Script structure:

Part 1. Imports.
Part 2. Argument parser.
Part 3. Global constants and variables.
Part 4. Main body.
'''

'''
Part 1. Imports.
'''

import argparse
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf

from tensorflow import keras
print( '\n' )

# In order to use sibling modules.
sys.path.append( ".." )
import utils.machine_learning_utils as ml_utils

'''
Part 2. Argument parser.
'''

parser = argparse.ArgumentParser( description = 'predict_with_rocf_dataset' )
parser.add_argument( "--dataset_path", type = str, default = "/home/jguerrero/Desarrollo/DATA/proyecto_REY/datasets/rocfd528.pickle", help = "Pickle with the ROCFD528 dataset." )
parser.add_argument( "--fold_info_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf_automatic_scoring/data/dataset_information/rocfd528_fold_information/", help = "Directory with fold information of ROCFD528 dataset." )
parser.add_argument( "--pretrained_models_dir", type = str, default = "/home/jguerrero/Desarrollo/DATA/proyecto_REY/models/regression_TL_MN2_with_QD_2_0_to_RD_3_0_i0/", help = "Directory with the pre-trained models." )
parser.add_argument( "--stopping_epochs_path", type = str, default = "/home/jguerrero/Desarrollo/rocf_automatic_scoring/data/stopping_epochs/SE_regression_TL_MN2_with_QD_2_0_to_RD_3_0_i0_16folds.csv", help = "CSV with stopping epochs for the 16 models." )
parser.add_argument( "--output_predictions_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf_automatic_scoring/data/predictions/", help = "Directory where to save the predictions." )
parser.add_argument( "--model_type", type = str, default = "MN2", help = "Choosing the model. Options: SaN, MN2, IC3, ENB1." )
parser.add_argument( "--learning_paradigm", type = str, default = "TL_QD", help = "Choosing the learning paradigm. Options: IL, TL_IN, TL_QD." )
parser.add_argument( "--execution_iteration", type = int, default = 0, help = "Execution iteration." )
parser.add_argument( "--gpu_id", type = str, default = '2', help = "Choosing which GPU to use by its index." )

args = parser.parse_args( )

'''
Part 3. Global constants and variables.
'''

dataset_path = args.dataset_path 
fold_info_dir = args.fold_info_dir
pretrained_models_dir = args.pretrained_models_dir

print( '\n' )
print( 'Loading models from: ', pretrained_models_dir )

stopping_epochs_path = args.stopping_epochs_path
STOPPING_EPOCHS_DELIMITER = ','

print( '\n' )
print( 'Loading stopping epochs from: ', stopping_epochs_path )

output_predictions_dir = args.output_predictions_dir

CUSTOM_MODEL_TYPE = args.model_type
CUSTOM_LEARNING_PARADIGM = args.learning_paradigm
CUSTOM_EXECUTION_ITERATION = args.execution_iteration

gpu_id = args.gpu_id

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

PROBLEM_TYPE = None
if CUSTOM_LEARNING_PARADIGM == 'IL':
  PROBLEM_TYPE = 'regression'
elif CUSTOM_LEARNING_PARADIGM == 'TL_IN' or CUSTOM_LEARNING_PARADIGM == 'TL_QD':
  PROBLEM_TYPE = 'regression_TL'
  
MODEL_ALIAS = None
if CUSTOM_LEARNING_PARADIGM == 'IL':
  MODEL_ALIAS = CUSTOM_MODEL_TYPE + '_with_RD_3_0'
elif CUSTOM_LEARNING_PARADIGM == 'TL_IN':
  MODEL_ALIAS = CUSTOM_MODEL_TYPE + '_with_IN_1_0_to_RD_3_0'
elif CUSTOM_LEARNING_PARADIGM == 'TL_QD':
  MODEL_ALIAS = CUSTOM_MODEL_TYPE + '_with_QD_2_0_to_RD_3_0'

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384

NUMBER_OF_FOLDS = 16
FOLD_CREATION_SEED = 33
  
MIN_SCORE = 0
MAX_SCORE = 36
  
input_model_base_name = PROBLEM_TYPE + '_' + MODEL_ALIAS + '_i' + str( CUSTOM_EXECUTION_ITERATION )

IMAGE_OUTPUT_CHANNELS = None
if CUSTOM_LEARNING_PARADIGM == 'IL' or CUSTOM_LEARNING_PARADIGM == 'TL_QD':
  IMAGE_OUTPUT_CHANNELS = 1
elif CUSTOM_LEARNING_PARADIGM == 'TL_IN':
  IMAGE_OUTPUT_CHANNELS = 3
  
DATASET_NAME = 'rocfd528'
    
# False is for debug mode.
SAVE_RESULTS = True
  
'''
Part 4. Main body.
'''

if __name__ == '__main__':
    
  # Loading stopping epochs.

  input_stopping_epoch_list = np.genfromtxt( stopping_epochs_path, delimiter = STOPPING_EPOCHS_DELIMITER, dtype = int )

  # Unpickling dataset.
  
  X, y, paths = ml_utils.unpickleDataset( dataset_path, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_OUTPUT_CHANNELS )
  
  # Image data generators (for real-time data augmentation AKA RTDA).
  
  # Input image pixel values: (0,255)
  # Image pixel values required for each of the models.
  # SaN = (0, 1). MN2, IC3 = (-1, 1). ENB1 = (0, 255).
  
  val_image_data_generator = None
  
  if CUSTOM_MODEL_TYPE == 'SaN':
    
    SAN_RESCALE_FACTOR = 1.0 / 255.0

    val_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
      rescale = SAN_RESCALE_FACTOR
    )
    
  else:
    
    custom_preprocessing_function = None
    if CUSTOM_MODEL_TYPE == 'MN2':
      custom_preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input
    elif CUSTOM_MODEL_TYPE == 'IC3':
      custom_preprocessing_function = tf.keras.applications.inception_v3.preprocess_input
    elif CUSTOM_MODEL_TYPE == 'ENB1':
      custom_preprocessing_function = tf.keras.applications.efficientnet.preprocess_input

    val_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
      preprocessing_function = custom_preprocessing_function
    )
  
  # Predicting.
  
  print( 'Predicting.' )
  print( 'Chosen model:', CUSTOM_MODEL_TYPE )
  print( 'Chosen learning paradigm:', CUSTOM_LEARNING_PARADIGM )
  print( '\n' )
  
  for fold_index in range( 0, NUMBER_OF_FOLDS ):
    
    fold_counter = fold_index + 1
    
    current_fold_as_string = '{:02d}'.format( fold_counter )
    
    print( 'Fold ' + current_fold_as_string + '.' )
    print( '\n' )
        
    val_indices_as_csv = fold_info_dir + DATASET_NAME + '_val_fold' + current_fold_as_string + '.csv'
    val_indices = np.genfromtxt( val_indices_as_csv, delimiter = ';', dtype = int )
    
    print( 'Validation set indices:', val_indices )
    print( '\n' )
        
    current_stopping_epoch_as_string = '{:04d}'.format( input_stopping_epoch_list[fold_index] )
    
    # Model loading.
    
    pretrained_models_dir_fold = pretrained_models_dir + input_model_base_name + '_fold' + current_fold_as_string + '/'
    pretrained_model_path = pretrained_models_dir_fold + input_model_base_name + '_fold' + current_fold_as_string +  '-' + current_stopping_epoch_as_string + '.hdf5'
    
    model = keras.models.load_model( pretrained_model_path )
    print( 'Loaded model:', pretrained_model_path )
    print( '\n' )

    # Model summary.
    #model.summary()
  
    # Image data iterator.
        
    val_data_iterator = val_image_data_generator.flow(
      x = X[val_indices],
      y = y[val_indices],
      batch_size = 1,
      shuffle = False
    )
        
    step_size_val = int( np.ceil( val_data_iterator.n / val_data_iterator.batch_size ) )
  
    # Model predict.

    predictions_matrix_keras = model.predict(
      x = val_data_iterator,
      steps = step_size_val,
      verbose = 1,
    )
    print( '\n' )
    
    # Processing the predictions.
    
    predicted_normalized_scores = predictions_matrix_keras[:,0]

    print( 'Predictions (0,1):', predicted_normalized_scores )
    print( '\n' )

    predictions = ml_utils.convertNormalizedScoresIntoContinuousOsterriethScores( predicted_normalized_scores, MIN_SCORE, MAX_SCORE )

    print( 'Predictions (0,36):', predictions )
    print( '\n' )
      
    # Saving the predictions to a CSV file.
    
    if SAVE_RESULTS:
      
      predictions_path = output_predictions_dir + 'P_' + input_model_base_name + '_fold' + current_fold_as_string + '.csv'
      
      print( 'Saving predictions to', predictions_path )
      print( '\n' )
      
      predictions_dataframe = pd.DataFrame( { "Filename": paths[val_indices], "Predictions": predictions } )
      predictions_dataframe.to_csv( predictions_path, index = False, header = True )