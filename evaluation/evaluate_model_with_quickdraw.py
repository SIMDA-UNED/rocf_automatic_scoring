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

from tensorflow import keras
print( '\n' )

# In order to use sibling modules.
sys.path.append( ".." )
import utils.model_creator as model_creator

'''
Part 2. Argument parser.
'''

parser = argparse.ArgumentParser( description = 'evaluate_model_with_quickdraw' )
parser.add_argument( "--dataset_dir", type = str, default = "/home/jguerrero/Desarrollo/DATA/proyecto_REY/datasets/quickdraw_dataset_2_0_414k_binarize_tvt_split_345/", help = "Directory with the QD dataset." )
parser.add_argument( "--pretrained_weights_path", type = str, default = "/home/jguerrero/Desarrollo/DATA/proyecto_REY/weights/classification_SaN_with_QD_2_0_i0/classification_SaN_with_QD_2_0_i0-0089.hdf5", help = "Weights of the pre-trained model." )
parser.add_argument( "--model_type", type = str, default = "SaN", help = "Choosing the model to be used for training. Options: SaN, MN2, IC3, ENB1." )
parser.add_argument( "--execution_iteration", type = int, default = 0, help = "Execution iteration." )
parser.add_argument( "--batch_size", type = int, default = 64, help = "Batch size." )
parser.add_argument( "--learning_rate", type = float, default = 1e-4, help = "Learning rate." )
parser.add_argument( "--gpu_id", type = str, default = '1', help = "Choosing which GPU to use by its index." )

args = parser.parse_args( )

'''
Part 3. Global constants and variables.
'''

dataset_dir = args.dataset_dir
pretrained_weights_path = args.pretrained_weights_path

print( '\n' )
print( 'Loading weights from: ', pretrained_weights_path )
  
CUSTOM_MODEL_TYPE = args.model_type
CUSTOM_EXECUTION_ITERATION = args.execution_iteration

CUSTOM_BATCH_SIZE = args.batch_size
CUSTOM_LEARNING_RATE = args.learning_rate
  
gpu_id = args.gpu_id

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

PROBLEM_TYPE = 'classification'
MODEL_ALIAS = CUSTOM_MODEL_TYPE + '_with_QD_2_0'

# RD 2.0 image size: 256 x 256.
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

val_dir = dataset_dir + 'val'
test_dir = dataset_dir + 'test' 

# class_mode: categorical (2D one-hot encoded labels), sparse (1D integer labels).
LABEL_TYPE = 'sparse'

CUSTOM_NUMBER_OF_CLASSES = 345
CUSTOM_OUTPUT_ACTIVATION_FUNCTION = 'softmax'

CUSTOM_RANDOM_SEED = 333

'''
Part 4. Main body.
'''

if __name__ == '__main__':
  
  # Image data generators (for real-time data augmentation AKA RTDA).
  
  # Input image pixel values: (0,255)
  # Image pixel values required for each of the models.
  # SaN = (0, 1). MN2, IC3 = (-1, 1). ENB1 = (0, 255).  

  val_and_test_image_data_generator = None
  
  if CUSTOM_MODEL_TYPE == 'SaN':
    
    SAN_RESCALE_FACTOR = 1.0 / 255.0
    
    val_and_test_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
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
    
    val_and_test_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
      preprocessing_function = custom_preprocessing_function
    )
  
  # Image data iterators.
  # WARNING: .ipynb_checkpoints directories can contain copies of images. Remove them!
  
  print( '\n' )
  
  val_data_iterator = val_and_test_image_data_generator.flow_from_directory(
    directory = val_dir,
    target_size = ( IMAGE_HEIGHT, IMAGE_WIDTH ),
    color_mode = 'grayscale',
    class_mode = LABEL_TYPE,
    batch_size = CUSTOM_BATCH_SIZE,
    shuffle = True,
    seed = CUSTOM_RANDOM_SEED
  )

  test_data_iterator = val_and_test_image_data_generator.flow_from_directory(
    directory = test_dir,
    target_size = ( IMAGE_HEIGHT, IMAGE_WIDTH ),
    color_mode = 'grayscale',
    class_mode = LABEL_TYPE,
    batch_size = 1,
    shuffle = False
  )
  
  STEP_SIZE_VAL = int( np.ceil( val_data_iterator.n / val_data_iterator.batch_size ) )
  STEP_SIZE_TEST = int( np.ceil( test_data_iterator.n / test_data_iterator.batch_size ) )
  
  print( '\n' )

  # Model loading.
  
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
  
  # Loading pre-trained weights.
  model.load_weights( pretrained_weights_path )
  
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
  
  # Model evaluate.
  
  print( 'Evaluating.' )
  print( 'Chosen model:', CUSTOM_MODEL_TYPE )
  print( '\n' )
  
  ## Validation set accuracy.
  
  print( 'Evaluating with validation set.' )
  print( '\n' )
  
  val_metrics = model.evaluate(
    x = val_data_iterator,
    steps = STEP_SIZE_VAL
  )

  val_accuracy = val_metrics[1] * 100

  print( 'Validation accuracy.' )
  print( '%s: %.2f%%' % ( model.metrics_names[1], val_accuracy ) )
  print( '\n' )

  ## Test set accuracy.

  print( 'Evaluating with test set.' )
  print( '\n' )
  
  test_metrics = model.evaluate(
    x = test_data_iterator,
    steps = STEP_SIZE_TEST
  )

  test_accuracy = test_metrics[1] * 100

  print( 'Test accuracy.' )
  print( '%s: %.2f%%' % ( model.metrics_names[1], test_accuracy ) )
  print( '\n' )