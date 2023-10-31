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
import sys
import tensorflow as tf
import time

from tensorflow import keras
print( '\n' )

# In order to use sibling modules.
sys.path.append( ".." )
import utils.machine_learning_utils as ml_utils
import utils.model_creator as model_creator

'''
Part 2. Argument parser.
'''

parser = argparse.ArgumentParser( description = 'transfer_from_imagenet_to_rocf_dataset' )
parser.add_argument( "--dataset_path", type = str, default = "/home/jguerrero/Desarrollo/DATA/proyecto_REY/datasets/rocfd528.pickle", help = "Pickle with the ROCFD528 dataset." )
parser.add_argument( "--fold_info_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf_automatic_scoring/data/dataset_information/rocfd528_fold_information/", help = "Directory with fold information of ROCFD528 dataset." )
parser.add_argument( "--output_model_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf_automatic_scoring/data/models/", help = "Directory where to save the partial models." )
parser.add_argument( "--execution_times_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf_automatic_scoring/data/execution_times/", help = "Directory where to save the execution times." )
parser.add_argument( "--histories_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf_automatic_scoring/data/histories/", help = "Directory where to save the histories." )
parser.add_argument( "--model_type", type = str, default = "ENB1", help = "Choosing the model to be used for training. Options: MN2, IC3, ENB1." )
parser.add_argument( "--execution_iteration", type = int, default = 0, help = "Execution iteration." )
parser.add_argument( "--batch_size", type = int, default = 32, help = "Batch size." )
parser.add_argument( "--learning_rate", type = float, default = 5e-5, help = "Learning rate." )
parser.add_argument( "--training_epochs", type = int, default = 1, help = "Training epochs." )
parser.add_argument( "--gpu_id", type = str, default = '1', help = "Choosing which GPU to use by its index." )

args = parser.parse_args( )

'''
Part 3. Global constants and variables.
'''

dataset_path = args.dataset_path 
fold_info_dir = args.fold_info_dir

output_model_dir = args.output_model_dir
execution_times_dir = args.execution_times_dir
histories_dir = args.histories_dir

CUSTOM_MODEL_TYPE = args.model_type
CUSTOM_EXECUTION_ITERATION = args.execution_iteration

CUSTOM_BATCH_SIZE = args.batch_size
CUSTOM_LEARNING_RATE = args.learning_rate
CUSTOM_TRAINING_EPOCHS = args.training_epochs

gpu_id = args.gpu_id

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

PROBLEM_TYPE = 'regression_TL'
MODEL_ALIAS = CUSTOM_MODEL_TYPE + '_with_IN_1_0_to_RD_3_0'

# RD 3.0 image size: 384 x 384.
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384

NUMBER_OF_FOLDS = 16
FOLD_CREATION_SEED = 33

MIN_SCORE = 0
MAX_SCORE = 36
  
histories_root = histories_dir + PROBLEM_TYPE + '_' + MODEL_ALIAS + '/'

output_model_base_name = PROBLEM_TYPE + '_' + MODEL_ALIAS + '_i' + str( CUSTOM_EXECUTION_ITERATION )
  
IMAGE_OUTPUT_CHANNELS = 3
  
DATASET_NAME = 'rocfd528'

CUSTOM_NUMBER_OF_CLASSES = 1
CUSTOM_OUTPUT_ACTIVATION_FUNCTION = 'linear'

CUSTOM_RANDOM_SEED = 333
  
'''
Part 4. Main body.
'''

if __name__ == '__main__':

  # Creating folder for histories.
  
  if not( os.path.isdir( histories_root ) ):
    os.mkdir( histories_root )
  
  # Unpickling dataset.
  
  X, y, paths = ml_utils.unpickleDataset( dataset_path, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_OUTPUT_CHANNELS )

  # Image data generators (for real-time data augmentation AKA RTDA).
  
  # Input image pixel values: (0,255)
  # Image pixel values required for each of the models.
  # MN2, IC3 = (-1, 1). ENB1 = (0, 255).
  
  custom_preprocessing_function = None
  if CUSTOM_MODEL_TYPE == 'MN2':
    custom_preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input
  elif CUSTOM_MODEL_TYPE == 'IC3':
    custom_preprocessing_function = tf.keras.applications.inception_v3.preprocess_input
  elif CUSTOM_MODEL_TYPE == 'ENB1':
    custom_preprocessing_function = tf.keras.applications.efficientnet.preprocess_input

  train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 10.0,
    zoom_range = [ 0.9, 1.1 ],
    fill_mode = 'constant',
    cval = 0.0,
    preprocessing_function = custom_preprocessing_function
  )

  val_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function = custom_preprocessing_function
  )
  
  # Training.
  
  execution_time_list = np.empty( shape = ( NUMBER_OF_FOLDS ), dtype = object )

  print( 'Training.' )
  print( 'Chosen model:', CUSTOM_MODEL_TYPE )
  print( '\n' )
  
  for fold_index in range( 0, NUMBER_OF_FOLDS ):
    
    fold_counter = fold_index + 1
    
    current_fold_as_string = '{:02d}'.format( fold_counter )
    
    print( 'Fold ' + current_fold_as_string + '.' )
    print( '\n' )
    
    train_indices_as_csv = fold_info_dir + DATASET_NAME + '_train_fold' + current_fold_as_string + '.csv'
    train_indices = np.genfromtxt( train_indices_as_csv, delimiter = ';', dtype = int )
    
    val_indices_as_csv = fold_info_dir + DATASET_NAME + '_val_fold' + current_fold_as_string + '.csv'
    val_indices = np.genfromtxt( val_indices_as_csv, delimiter = ';', dtype = int )
    
    print( 'Validation set indices:', val_indices )
    print( '\n' )
    
    # Model surgery.
    
    base_model = None
    if CUSTOM_MODEL_TYPE == 'MN2':
      base_model = model_creator.mobileNetV2WithImageNetWeights( IMAGE_HEIGHT, IMAGE_WIDTH )
    elif CUSTOM_MODEL_TYPE == 'IC3':
      base_model = model_creator.inceptionV3WithImageNetWeights( IMAGE_HEIGHT, IMAGE_WIDTH )
    elif CUSTOM_MODEL_TYPE == 'ENB1':
      base_model = model_creator.efficientNetB1WithImageNetWeights( IMAGE_HEIGHT, IMAGE_WIDTH )
    
    # Model summary.
    #base_model.summary()

    # Freezing all layers.

    base_model.trainable = False
    
    model = None
    
    if CUSTOM_MODEL_TYPE == 'MN2' or CUSTOM_MODEL_TYPE == 'IC3':
    
      # Adding some layers on top of the base model.

      inputs = keras.Input( shape = ( IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_OUTPUT_CHANNELS ) )

      # Putting base model on inference mode (so BatchNormalization layers remain unchanged).

      x = base_model( inputs, training = False )

      # It does the same as Flatten but different input shapes are allowed.

      x = keras.layers.GlobalAveragePooling2D( )( x )

      # New layers.

      outputs = keras.layers.Dense( CUSTOM_NUMBER_OF_CLASSES, activation = CUSTOM_OUTPUT_ACTIVATION_FUNCTION )( x ) 

      model = keras.Model( inputs, outputs )
      
    elif CUSTOM_MODEL_TYPE == 'ENB1':
      
      # Adding some layers on top of the base model.

      inputs = keras.Input( shape = ( IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_OUTPUT_CHANNELS ) )

      # Putting base model on inference mode (so BatchNormalization layers remain unchanged).

      x = base_model( inputs, training = False )

      # It does the same as Flatten but different input shapes are allowed.

      x = keras.layers.GlobalAveragePooling2D( )( x )

      EFFICIENT_NET_B1_DROPOUT = 0.2  
      x = keras.layers.Dropout( EFFICIENT_NET_B1_DROPOUT )( x )

      # New layers.

      outputs = keras.layers.Dense( CUSTOM_NUMBER_OF_CLASSES, activation = CUSTOM_OUTPUT_ACTIVATION_FUNCTION )( x ) 

      model = keras.Model( inputs, outputs )
    
    # Model summary.
    #model.summary()
    
    # Model compile.

    '''
    Adam optimizer.
    beta_1: related to gradient.
    beta_2: related to squared gradient.
    '''
    custom_adam_optimizer = keras.optimizers.Adam( 
      learning_rate = CUSTOM_LEARNING_RATE, 
      beta_1 = 0.9, beta_2 = 0.999, amsgrad = False )

    model.compile(
      optimizer = custom_adam_optimizer,
      loss = 'mean_squared_error',
      metrics = ['mean_absolute_error', 'mean_squared_error']
    )
    
    # Image data iterator.
    
    train_data_iterator = train_image_data_generator.flow(
      x = X[train_indices],
      y = y[train_indices],
      batch_size = CUSTOM_BATCH_SIZE,
      shuffle = True,
      seed = CUSTOM_RANDOM_SEED
    )
    
    val_data_iterator = val_image_data_generator.flow(
      x = X[val_indices],
      y = y[val_indices],
      batch_size = 1,
      shuffle = False
    )
    
    step_size_train = int( np.ceil( train_data_iterator.n / train_data_iterator.batch_size ) )
    step_size_val = int( np.ceil( val_data_iterator.n / val_data_iterator.batch_size ) )

    # Model fit.
    
    output_model_folder = output_model_dir + output_model_base_name + '_fold' + current_fold_as_string + '/'
    if not( os.path.isdir( output_model_folder ) ):
      os.mkdir( output_model_folder )

    output_model_checkpoints = output_model_folder + output_model_base_name + '_fold' + current_fold_as_string + '-{epoch:04d}.hdf5'
    
    custom_model_checkpoint = keras.callbacks.ModelCheckpoint(
      filepath = output_model_checkpoints,
      monitor = 'val_loss',
      verbose = 0,
      save_best_only = True,
      save_weights_only = False,
      mode = 'min',
      save_freq = 'epoch'
    )

    # Timing BEGIN.
    start_time = time.time()

    history = model.fit(
      x = train_data_iterator,
      epochs = CUSTOM_TRAINING_EPOCHS,
      callbacks = [ custom_model_checkpoint ],
      steps_per_epoch = step_size_train,
      validation_data = val_data_iterator,
      validation_steps = step_size_val
    )

    # Timing END.
    elapsed_time = time.time() - start_time
    
    # Save history.

    histories_path = histories_root + 'H_' + output_model_base_name + '_fold' + current_fold_as_string + '.npy'
    np.save( histories_path, history.history )
    
    # Format execution time.

    execution_time_as_string = ml_utils.convertToDDHHMMSSFormatAndPrint( elapsed_time )
    execution_time_list[ fold_index ] = execution_time_as_string

  # Saving execution times.

  execution_times_as_csv = execution_times_dir + 'ET_' + output_model_base_name + '_' + str( NUMBER_OF_FOLDS ) + 'folds.csv'
  np.savetxt( execution_times_as_csv, execution_time_list, fmt = '%s', delimiter = ',' )
  print( 'Execution times saved to:', execution_times_as_csv )