'''
Script structure:

Part 1. Imports.
Part 2. Functions.
'''

'''
Part 1. Imports.
'''

import numpy as np
import pickle

'''
Part 2. Functions.
'''

def unpickleDataset( pickle_file, image_height, image_width, image_output_channels ):
  
  with open( pickle_file, 'rb' ) as pickle_file_descriptor:
  	pickle_file_elements = pickle.load( pickle_file_descriptor )
  	X = pickle_file_elements['X']
  	y = pickle_file_elements['y']
  	paths = pickle_file_elements['paths']
  
  # Converting arrays to tensor format (using numpy library).    
  
  X = np.array( X, dtype = np.float32 )
  X = np.reshape( X, ( X.shape[0], image_height, image_width, 1 ) )
  if image_output_channels > 1:
    X = np.repeat( X, image_output_channels, axis = 3 )
    
  # Feedback.
  '''
  image = X[60]
  print( image[145:160,145:160,0] )
  print( image[145:160,145:160,1] )
  print( image[145:160,145:160,2] )
  '''
    
  y = np.array( y, dtype = np.float32 )
  
  paths = np.array( paths )
  
  # Feedback.
  #'''
  print( '\n' )
  print( 'Dataset information.' )
  print( '\n' )
  print( 'X shape:', X.shape )
  print( 'y shape:', y.shape )
  print( 'paths shape:', paths.shape )
  print( '\n' )
  #np.set_printoptions( threshold = sys.maxsize )
  print( 'First label (y):', y[0] )
  print( 'Label type. Regression: float. Classification: int.' )
  print( 'First label (y) type:', type( y[0] ) )
  print( 'First path:', paths[0] )
  print( '\n' )
  #print( y )
  #'''
  
  return X, y, paths

def convertToDDHHMMSSFormatAndPrint( seconds ):

  seconds_in_day = 86400
  seconds_in_hour = 3600
  seconds_in_minute = 60
  
  seconds = int( seconds )
  
  days = seconds // seconds_in_day
  seconds = seconds - (days * seconds_in_day)
  
  hours = seconds // seconds_in_hour
  seconds = seconds - (hours * seconds_in_hour)
  
  minutes = seconds // seconds_in_minute
  seconds = seconds - (minutes * seconds_in_minute)
  
  print("{0:.0f} days, {1:.0f} hours, {2:.0f} minutes, {3:.0f} seconds.".format(
      days, hours, minutes, seconds))
  print( '\n' )

  time_as_string = "{0:02d}:{1:02d}:{2:02d}:{3:02d}".format( int( days ), int( hours ), int( minutes ), int( seconds ) )

  return time_as_string

def convertNormalizedScoresIntoContinuousOsterriethScores( normalized_scores, min_score, max_score ):
  
  # Continuous Osterrieth scores: 0 to 36.
  
  continuous_osterrieth_scores = []
  
  for normalized_score in normalized_scores:
    
    unnormalized_score = normalized_score * ( max_score - min_score ) + min_score
    
    continuous_osterrieth_scores.append( unnormalized_score )
    
  continuous_osterrieth_scores_np = np.array( continuous_osterrieth_scores, dtype = np.float32 )
      
  return continuous_osterrieth_scores_np

def saveConfusionMatrixWithHeader( output_path, output_matrix, num_of_classes, matrix_value_separator ):
  
  confusion_matrix_header = ''
  for i in range( num_of_classes ):
    confusion_matrix_header += ',' + str( i )
  confusion_matrix_header += '\n'
    
  with open( output_path, 'w' ) as output_file:

    output_file.write( confusion_matrix_header )

    for i in range( num_of_classes ):
  
      cm_row = output_matrix[i]
      cm_row_as_list = cm_row.tolist()
      cm_row_as_string = matrix_value_separator.join( [str( element ) for element in cm_row_as_list] )
      output_line = str( i ) + matrix_value_separator + cm_row_as_string + '\n'
      output_file.write( output_line )