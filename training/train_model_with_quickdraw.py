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

parser = argparse.ArgumentParser( description = 'train_model_with_quickdraw' )
parser.add_argument( "--dataset_dir", type = str, default = "/home/jguerrero/Desarrollo/DATA/proyecto_REY/datasets/quickdraw_dataset_2_0_414k_binarize_tvt_split_345/", help = "Directory with the QD dataset." )
parser.add_argument( "--output_weight_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf_automatic_scoring/data/weights/", help = "Directory where to save the partial model weights." )
parser.add_argument( "--histories_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf_automatic_scoring/data/histories/", help = "Directory where to save the histories." )
parser.add_argument( "--model_type", type = str, default = "SaN", help = "Choosing the model to be used for training. Options: SaN, MN2, IC3, ENB1." )
parser.add_argument( "--execution_iteration", type = int, default = 0, help = "Execution iteration." )
parser.add_argument( "--batch_size", type = int, default = 64, help = "Batch size." )
parser.add_argument( "--learning_rate", type = float, default = 1e-4, help = "Learning rate." )
parser.add_argument( "--training_epochs", type = int, default = 5, help = "Training epochs." )
parser.add_argument( "--gpu_id", type = str, default = '1', help = "Choosing which GPU to use by its index." )

args = parser.parse_args( )

'''
Part 3. Global constants and variables.
'''

dataset_dir = args.dataset_dir

output_weight_dir = args.output_weight_dir
histories_dir = args.histories_dir

CUSTOM_MODEL_TYPE = args.model_type
CUSTOM_EXECUTION_ITERATION = args.execution_iteration

CUSTOM_BATCH_SIZE = args.batch_size
CUSTOM_LEARNING_RATE = args.learning_rate
CUSTOM_TRAINING_EPOCHS = args.training_epochs

gpu_id = args.gpu_id

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

PROBLEM_TYPE = 'classification'
MODEL_ALIAS = CUSTOM_MODEL_TYPE + '_with_QD_2_0'

# RD 2.0 image size: 256 x 256.
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

histories_root = histories_dir + PROBLEM_TYPE + '_' + MODEL_ALIAS + '/'

# Model checkpoint.
output_model_base_name = PROBLEM_TYPE + '_' + MODEL_ALIAS + '_i' + str( CUSTOM_EXECUTION_ITERATION )

train_dir = dataset_dir + 'train'
val_dir = dataset_dir + 'val'

# class_mode: categorical (2D one-hot encoded labels), sparse (1D integer labels).
LABEL_TYPE = 'sparse'

CUSTOM_NUMBER_OF_CLASSES = 345
CUSTOM_OUTPUT_ACTIVATION_FUNCTION = 'softmax'

CUSTOM_RANDOM_SEED = 333

'''
Part 4. Main body.
'''

if __name__ == '__main__':

  # Creating folder for histories.
  
  if not( os.path.isdir( histories_root ) ):
    os.mkdir( histories_root )

  # Image data generators (for real-time data augmentation AKA RTDA).
  
  # Input image pixel values: (0,255)
  # Image pixel values required for each of the models.
  # SaN = (0, 1). MN2, IC3 = (-1, 1). ENB1 = (0, 255).   

  train_image_data_generator = None
  val_image_data_generator = None
  
  if CUSTOM_MODEL_TYPE == 'SaN':
    
    SAN_RESCALE_FACTOR = 1.0 / 255.0
    
    train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
      rotation_range = 10,
      width_shift_range = 0.1,
      height_shift_range = 0.1,
      shear_range = 10.0,
      zoom_range = [ 0.9, 1.1 ],
      fill_mode = 'constant',
      cval = 0.0, 
      rescale = SAN_RESCALE_FACTOR
    )

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

  # Image data iterators.
  # WARNING: .ipynb_checkpoints directories can contain copies of images. Remove them!
  
  print( '\n' )
  
  train_data_iterator = train_image_data_generator.flow_from_directory(
    directory = train_dir,
    target_size = ( IMAGE_HEIGHT, IMAGE_WIDTH ),
    color_mode = 'grayscale',
    class_mode = LABEL_TYPE,
    batch_size = CUSTOM_BATCH_SIZE,
    shuffle = True,
    seed = CUSTOM_RANDOM_SEED
  )

  val_data_iterator = val_image_data_generator.flow_from_directory(
    directory = val_dir,
    target_size = ( IMAGE_HEIGHT, IMAGE_WIDTH ),
    color_mode = 'grayscale',
    class_mode = LABEL_TYPE,
    batch_size = CUSTOM_BATCH_SIZE,
    shuffle = True,
    seed = CUSTOM_RANDOM_SEED
  )

  STEP_SIZE_TRAIN = int( np.ceil( train_data_iterator.n / train_data_iterator.batch_size ) )
  STEP_SIZE_VAL = int( np.ceil( val_data_iterator.n / val_data_iterator.batch_size ) )
  
  print( '\n' )

  # Model creation.

  model = None
  if CUSTOM_MODEL_TYPE == 'SaN':
    model = model_creator.sketchANet( IMAGE_HEIGHT, IMAGE_WIDTH, CUSTOM_NUMBER_OF_CLASSES, CUSTOM_OUTPUT_ACTIVATION_FUNCTION, layer_6_kernel_size = 8 )
  elif CUSTOM_MODEL_TYPE == 'MN2':
    model = model_creator.mobileNetV2( IMAGE_HEIGHT, IMAGE_WIDTH, CUSTOM_NUMBER_OF_CLASSES, CUSTOM_OUTPUT_ACTIVATION_FUNCTION )
  elif CUSTOM_MODEL_TYPE == 'IC3':
    model = model_creator.inceptionV3( IMAGE_HEIGHT, IMAGE_WIDTH, CUSTOM_NUMBER_OF_CLASSES, CUSTOM_OUTPUT_ACTIVATION_FUNCTION )
  elif CUSTOM_MODEL_TYPE == 'ENB1':
    model = model_creator.efficientNetB1( IMAGE_HEIGHT, IMAGE_WIDTH, CUSTOM_NUMBER_OF_CLASSES, CUSTOM_OUTPUT_ACTIVATION_FUNCTION )

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
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
  )
  
  # Model fit.
  
  print( 'Training.' )
  print( 'Chosen model:', CUSTOM_MODEL_TYPE )
  print( '\n' )

  output_model_folder = output_weight_dir + output_model_base_name + '/'
  if not( os.path.isdir( output_model_folder ) ):
    os.mkdir( output_model_folder )
  
  output_model_checkpoints = output_model_folder + output_model_base_name + '-{epoch:04d}.hdf5'
    
  custom_model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath = output_model_checkpoints,
    monitor = 'val_accuracy',
    verbose = 0,
    save_best_only = True,
    save_weights_only = True,
    mode = 'max',
    save_freq = 'epoch'
  )
  
  # Timing BEGIN.
  start_time = time.time()
  
  history = model.fit(
    x = train_data_iterator,
    epochs = CUSTOM_TRAINING_EPOCHS,
    callbacks = [ custom_model_checkpoint ],
    steps_per_epoch = STEP_SIZE_TRAIN,
    validation_data = val_data_iterator,
    validation_steps = STEP_SIZE_VAL
  )
  
  # Timing END.
  elapsed_time = time.time() - start_time
  
  # Printing elapsed time.
  ml_utils.convertToDDHHMMSSFormatAndPrint( elapsed_time )
  
  # Save history.
  histories_path = histories_root + 'H_' + output_model_base_name + '.npy'
  np.save( histories_path, history.history )