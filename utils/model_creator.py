'''
Script structure:

Part 1. Imports.
Part 2. Global constants and variables.
Part 3. Functions.
'''

'''
Part 1. Imports.
'''

import tensorflow as tf
from tensorflow import keras

'''
Part 2. Global constants and variables.
'''

# Initializers (Keras default).
custom_kernel_initializer = keras.initializers.glorot_uniform()
custom_bias_initializer = keras.initializers.Zeros()

# Activations.
custom_activation_function = 'relu'

'''
Part 3. Functions.
'''

# Sketch-a-Net

def sketchANet( image_height, image_width, num_of_classes, activation_function, layer_6_kernel_size = 13 ):
    
  model = keras.Sequential()
  
  # Convolution and pooling layers.

  valid_padding = 'valid'
  same_padding = 'same'

  custom_pool_size = ( 3, 3 )
  # Original: ( 3, 3 )
  custom_pool_strides = ( 2, 2 )
  # Original: ( 2, 2 )

  # Layer 1.

  layer1_filters = 64
  # Original: 64
  layer1_kernel_size = ( 15, 15 )
  # Original: ( 15, 15 )
  layer1_strides = ( 3, 3 )
  # Original: ( 3, 3 )

  model.add( keras.layers.Conv2D( layer1_filters, 
                                  kernel_size = layer1_kernel_size,
                                  activation = custom_activation_function,
                                  kernel_initializer = custom_kernel_initializer,
                                  bias_initializer = custom_bias_initializer,
                                  strides = layer1_strides,
                                  padding = valid_padding,
                                  input_shape = ( image_height, image_width, 1 ),
                                  name = 'conv2d' ) )

  model.add( keras.layers.MaxPooling2D( pool_size = custom_pool_size,
                                        strides = custom_pool_strides,
                                        name = 'max_pooling2d' ) )

  # Layer 2.

  layer2_filters = 128
  # Original: 128
  layer2_kernel_size = ( 5, 5 )
  # Original: ( 5, 5 )
  layer2_strides = ( 1, 1 )
  # Original: ( 1, 1 )

  model.add( keras.layers.Conv2D( layer2_filters, 
                                  kernel_size = layer2_kernel_size,
                                  activation = custom_activation_function,
                                  kernel_initializer = custom_kernel_initializer,
                                  bias_initializer = custom_bias_initializer,
                                  strides = layer2_strides,
                                  padding = valid_padding,
                                  name = 'conv2d_1' ) )

  model.add( keras.layers.MaxPooling2D( pool_size = custom_pool_size,
                                        strides = custom_pool_strides,
                                        name = 'max_pooling2d_1' ) )

  # Layers 3, 4 and 5.

  layer345_filters = 256
  # Original: 256
  layer345_kernel_size = ( 3, 3 )
  # Original: ( 3, 3 )
  layer345_strides = ( 1, 1 )
  # Original: ( 1, 1 )

  model.add( keras.layers.Conv2D( layer345_filters, 
                                  kernel_size = layer345_kernel_size,
                                  activation = custom_activation_function,
                                  kernel_initializer = custom_kernel_initializer,
                                  bias_initializer = custom_bias_initializer,
                                  strides = layer345_strides,
                                  padding = same_padding,
                                  name = 'conv2d_2' ) )

  model.add( keras.layers.Conv2D( layer345_filters, 
                                  kernel_size = layer345_kernel_size,
                                  activation = custom_activation_function,
                                  kernel_initializer = custom_kernel_initializer,
                                  bias_initializer = custom_bias_initializer,
                                  strides = layer345_strides,
                                  padding = same_padding,
                                  name = 'conv2d_3' ) )

  model.add( keras.layers.Conv2D( layer345_filters, 
                                  kernel_size = layer345_kernel_size,
                                  activation = custom_activation_function,
                                  kernel_initializer = custom_kernel_initializer,
                                  bias_initializer = custom_bias_initializer,
                                  strides = layer345_strides,
                                  padding = same_padding,
                                  name = 'conv2d_4' ) )

  model.add( keras.layers.MaxPooling2D( pool_size = custom_pool_size,
                                        strides = custom_pool_strides,
                                        name = 'max_pooling2d_2' ) )
  
  # Layer 6.

  layer6_filters = 512
  # Original: 512
  
  # Input size: 384x384. Quick, Draw! images. layer_6_kernel_size = 13.
  # Input size: 256x256. ROCF RD and SD images. layer_6_kernel_size = 8.
  layer6_kernel_size = ( layer_6_kernel_size, layer_6_kernel_size )
  # Original: ( N, N ) => Depends on input image dimensions.
  
  layer6_strides = ( 1, 1 )
  # Original: ( 1, 1 )
  layer6_dropout = 0.5
  # Original: 0.5

  model.add( keras.layers.Conv2D( layer6_filters, 
                                  kernel_size = layer6_kernel_size,
                                  activation = custom_activation_function,
                                  kernel_initializer = custom_kernel_initializer,
                                  bias_initializer = custom_bias_initializer,
                                  strides = layer6_strides,
                                  padding = valid_padding,
                                  name = 'conv2d_5' ) )

  model.add( keras.layers.Dropout( layer6_dropout,
                                   name = 'dropout' ) ) 

  # Layer 7.
  
  layer7_filters = 512
  # Original: 512
  layer7_kernel_size = ( 1, 1 )
  # Original: ( 1, 1 )
  layer7_strides = ( 1, 1 )
  # Original: ( 1, 1 )
  layer7_dropout = 0.5
  # Original: 0.5

  model.add( keras.layers.Conv2D( layer7_filters, 
                                  kernel_size = layer7_kernel_size,
                                  activation = custom_activation_function,
                                  kernel_initializer = custom_kernel_initializer,
                                  bias_initializer = custom_bias_initializer,
                                  strides = layer7_strides,
                                  padding = valid_padding,
                                  name = 'conv2d_6' ) )

  model.add( keras.layers.Dropout( layer7_dropout,
                                   name = 'dropout_1' ) )
  
  # Output layer.
  
  model.add( keras.layers.Flatten( name = 'flatten' ) )
  
  model.add( keras.layers.Dense( num_of_classes, 
                                 activation = activation_function,
                                 name = 'dense' ) )
  
  return model

def sketchANetAddTop( base_model, new_image_height, new_image_width ):
  
  inputs = keras.Input( shape = ( new_image_height, new_image_width, 1 ) )

  # Putting base model on inference mode (so BatchNormalization layers remain unchanged).

  x = base_model( inputs, training = False )

  # Layer 6.

  layer6_filters = 512
  layer6_kernel_size = ( 13, 13 )
  layer6_strides = ( 1, 1 )
  layer6_dropout = 0.5

  x = keras.layers.Conv2D( 
    layer6_filters, 
    kernel_size = layer6_kernel_size,
    activation = custom_activation_function,
    kernel_initializer = custom_kernel_initializer,
    bias_initializer = custom_bias_initializer,
    strides = layer6_strides,
    padding = 'valid',
    name = 'conv2d_5' )( x )

  x = keras.layers.Dropout( 0.5, name = 'dropout' )( x )

  # Layer 7.

  layer7_filters = 512
  layer7_kernel_size = ( 1, 1 )
  layer7_strides = ( 1, 1 )
  layer7_dropout = 0.5

  x = keras.layers.Conv2D( 
    layer7_filters, 
    kernel_size = layer7_kernel_size,
    activation = custom_activation_function,
    kernel_initializer = custom_kernel_initializer,
    bias_initializer = custom_bias_initializer,
    strides = layer7_strides,
    padding = 'valid',
    name = 'conv2d_6' )( x )

  x = keras.layers.Dropout( 0.5, name = 'dropout_1' )( x )

  # Output layer.

  x = keras.layers.Flatten( name = 'flatten' )( x )

  outputs = keras.layers.Dense( 1, activation = 'linear', name = 'dense' )( x ) 

  model = keras.Model( inputs, outputs )

  return model

def sketchANetWithNewInput( pretrained_weights_path, new_image_height, new_image_width ):
    
  # Pre-trained model.
  QD_IMAGE_HEIGHT = 256
  QD_IMAGE_WIDTH = 256
  QD_CLASSES = 345
  QD_LAYER_6_KERNEL_SIZE = 8
  pretrained_model = sketchANet( QD_IMAGE_HEIGHT, QD_IMAGE_WIDTH, QD_CLASSES, 'softmax', QD_LAYER_6_KERNEL_SIZE )
  
  pretrained_model.load_weights( pretrained_weights_path )
  
  # Feedback.
  #pretrained_model.summary()
  
  # IMPORTANT: this resets layer numbers.
  # This is not necessary now as we are hard-coding the layer names.
  #tf.keras.backend.clear_session( )
  
  # Create a new model with different inputs and outputs.
  RD_LAYER_6_KERNEL_SIZE = 13
  new_model = sketchANet( new_image_height, new_image_width, 1, 'linear', RD_LAYER_6_KERNEL_SIZE )
  
  # Feedback.
  #new_model.summary()
  
  # Load pre-trained weights.
  
  for layer in new_model.layers:
    
    try:
      layer.set_weights( pretrained_model.get_layer( layer.name ).get_weights( ) )
      print( 'Weights loaded for layer ' + layer.name + '.' )
    except:
      print( 'Weights CANNOT be loaded for layer ' + layer.name + '.' )
  
  print( '\n' )
  
  # Removing some layers at the top of the model.
  
  # Option A: freeze fc layers.
  #last_layer_name = 'dropout_1'
  # Option B: unfreeze fc layers.
  last_layer_name = 'max_pooling2d_2'
  
  base_model = keras.Model( inputs = new_model.input, outputs = new_model.get_layer( last_layer_name ).output )
  
  return base_model

# MobileNetV2.

def mobileNetV2( image_height, image_width, num_of_classes, activation_function ):
    
  model = keras.applications.MobileNetV2(
  	input_shape = ( image_height, image_width, 1 ),
  	alpha = 1.0,
  	include_top = True,
  	weights = None,
  	input_tensor = None,
  	pooling = None,
  	classes = num_of_classes,
  	classifier_activation = activation_function )
  
  return model

def mobileNetV2WithNewInput( pretrained_weights_path, new_image_height, new_image_width ):
  
  # Feedback.
  sample_model = mobileNetV2( new_image_height, new_image_width, 1, 'linear' )
  #sample_model.summary()
  #print( "Sample model. Number of layers:", len( sample_model.layers ) )
  
  # input_tensor: changing input shape and all layer output shapes.
  # weights: loading previously trained weights (QuickDraw).
  # include_top (False): we are going to use new top layers.
  base_model = keras.applications.MobileNetV2(
    input_tensor = keras.layers.Input( shape = ( new_image_height, new_image_width, 1 ) ),
    alpha = 1.0,
    include_top = False,
    weights = None,
    pooling = None )
  
  # by_name (True): load weights for layers only if their names have not been changed.
  base_model.load_weights( pretrained_weights_path, by_name = True )
  #base_model.load_weights( pretrained_weights_path )
  
  return base_model

def mobileNetV2WithImageNetWeights( new_image_height, new_image_width ):
  
  # ImageNet requires number of channels to be always 3.
  
  base_model = keras.applications.MobileNetV2(
    input_shape = ( new_image_height, new_image_width, 3 ),
    alpha = 1.0,
    include_top = False,
    weights = 'imagenet' )
    
  # alpha: regulates the width of the network.
  # Non necesssary attributes: input_tensor, pooling, classes, classifier_activation.
  
  # include_top = True and input_shape = ( 224, 224, 3 ) to see default architecture.
  
  return base_model

# InceptionV3.

def inceptionV3( image_height, image_width, num_of_classes, activation_function ):
  
  model = keras.applications.InceptionV3(
    include_top = True,
    weights = None,
    input_tensor = None,
    input_shape = ( image_height, image_width, 1 ),
    pooling = None,
    classes = num_of_classes,
    classifier_activation = activation_function )
  
  return model

def inceptionV3WithNewInput( pretrained_weights_path, new_image_height, new_image_width ):
  
  # Feedback.
  sample_model = inceptionV3( new_image_height, new_image_width, 1, 'linear' )
  #sample_model.summary()
  #print( "Sample model. Number of layers:", len( sample_model.layers ) )
  
  # input_tensor: changing input shape and all layer output shapes.
  # weights: loading previously trained weights (QuickDraw).
  # include_top (False): we are going to use new top layers.
  base_model = keras.applications.InceptionV3(
    input_tensor = keras.layers.Input( shape = ( new_image_height, new_image_width, 1 ) ),
    include_top = False,
    weights = None,
    pooling = None )
  
  # by_name (True): load weights for layers only if their names have not been changed.
  base_model.load_weights( pretrained_weights_path, by_name = True )
  #base_model.load_weights( pretrained_weights_path )
  
  return base_model

def inceptionV3WithImageNetWeights( new_image_height, new_image_width ):
  
  base_model = keras.applications.InceptionV3(
    include_top = False,
    weights = 'imagenet',
    input_shape = ( new_image_height, new_image_width, 3 ) )
  
  # Non necesssary attributes: input_tensor, pooling, classes, classifier_activation.
  
  # include_top = True and input_shape = ( 299, 299, 3 ) to see default architecture.
  
  return base_model

# EfficientNet (B=1).

def efficientNetB1( image_height, image_width, num_of_classes, activation_function ):
  
  model = keras.applications.EfficientNetB1(
    include_top = True,
    weights = None,
    input_tensor = None,
    input_shape = ( image_height, image_width, 1 ),
    pooling = None,
    classes = num_of_classes,
    classifier_activation = activation_function )
  
  return model

def efficientNetB1WithNewInput( pretrained_weights_path, new_image_height, new_image_width ):
  
  # Feedback.
  sample_model = efficientNetB1( new_image_height, new_image_width, 1, 'linear' )
  #sample_model.summary()
  #print( "Sample model. Number of layers:", len( sample_model.layers ) )
  
  # input_tensor: changing input shape and all layer output shapes.
  # weights: loading previously trained weights.
  # include_top (False): we are going to use new top layers.
  base_model = keras.applications.EfficientNetB1(
    input_tensor = keras.layers.Input( shape = ( new_image_height, new_image_width, 1 ) ),
    include_top = False,
    weights = None,
    pooling = None )
  
  #print( "Base model. Number of layers:", len( base_model.layers ) )
  
  # by_name (True): load weights for layers only if their names have not been changed.
  base_model.load_weights( pretrained_weights_path, by_name = True )
  #base_model.load_weights( pretrained_weights_path )
  
  return base_model

def efficientNetB1WithImageNetWeights( new_image_height, new_image_width ):
  
  base_model = keras.applications.EfficientNetB1(
    include_top = False,
    weights = 'imagenet',
    input_shape = ( new_image_height, new_image_width, 3 ) )
  
  # Non necesssary attributes: input_tensor, pooling, classes, classifier_activation.
  
  # include_top = True and input_shape = ( 240, 240, 3 ) to see default architecture.
  
  return base_model