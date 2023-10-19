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
import pandas as pd
import sys

from scipy import stats as sps
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from statistics import mean

# In order to use sibling modules.
sys.path.append( ".." )
import utils.machine_learning_utils as ml_utils

'''
Part 2. Argument parser.
'''

parser = argparse.ArgumentParser( description = 'extract_metrics_and_confusion_matrices' )
parser.add_argument( "--dataset_information_path", type = str, default = "/home/jguerrero/Desarrollo/rocf_automatic_scoring/data/dataset_information/rocfd528_information.csv", help = "CSV with ROCFD528 dataset information." )
parser.add_argument( "--input_predictions_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf_automatic_scoring/data/predictions/", help = "Directory from which predictions are loaded." )
parser.add_argument( "--output_confusion_matrices_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf_automatic_scoring/data/confusion_matrices/", help = "Directory where to save the confusion matrices." )
parser.add_argument( "--model_type", type = str, default = "MN2", help = "Choosing the model. Options: SaN, MN2, IC3, ENB1." )
parser.add_argument( "--learning_paradigm", type = str, default = "TL_QD", help = "Choosing the learning paradigm. Options: IL, TL_IN, TL_QD." )
parser.add_argument( "--execution_iteration", type = int, default = 0, help = "Execution iteration." )

args = parser.parse_args( )

'''
Part 3. Global constants and variables.
'''

dataset_information_path = args.dataset_information_path

input_predictions_dir = args.input_predictions_dir

output_confusion_matrices_dir = args.output_confusion_matrices_dir

CUSTOM_MODEL_TYPE = args.model_type
CUSTOM_LEARNING_PARADIGM = args.learning_paradigm
CUSTOM_EXECUTION_ITERATION = args.execution_iteration

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

input_model_base_name = PROBLEM_TYPE + '_' + MODEL_ALIAS + '_i' + str( CUSTOM_EXECUTION_ITERATION )

dataset_information_dataframe = pd.read_csv( dataset_information_path, sep = ';' )

NUMBER_OF_FOLDS = 16
NUM_OF_CLASSES = 37
MATRIX_VALUE_SEPARATOR = ','

# Lists for the metric values in each fold.
# Metrics: pcc, r2, mae, rmse (mse), medae.

pcc_fold_list = np.zeros( shape = ( NUMBER_OF_FOLDS ), dtype = float )

r2_fold_list = np.zeros( shape = ( NUMBER_OF_FOLDS ), dtype = float )

mae_fold_list = np.zeros( shape = ( NUMBER_OF_FOLDS ), dtype = float )

mse_fold_list = np.zeros( shape = ( NUMBER_OF_FOLDS ), dtype = float )
rmse_fold_list = np.zeros( shape = ( NUMBER_OF_FOLDS ), dtype = float )

medae_fold_list = np.zeros( shape = ( NUMBER_OF_FOLDS ), dtype = float )

# Lists for the metric values.

confusion_matrix_cumulative = np.zeros( shape = ( NUM_OF_CLASSES, NUM_OF_CLASSES ), dtype = int )
regression_outliers = 0

# Score lists.

predicted_score_list = []
actual_score_list = []

'''
Part 4. Main body.
'''

if __name__ == '__main__':

  for fold_index in range( 0, NUMBER_OF_FOLDS ):

    fold_counter = fold_index + 1    
    current_fold_as_string = '{:02d}'.format( fold_counter )
    predictions_path = input_predictions_dir + 'P_' + input_model_base_name + '_fold' + current_fold_as_string + '.csv'
    
    print( 'Fold', fold_counter )
    print( 'Source:', predictions_path )
    print('\n')
    
    predictions_dataframe = pd.read_csv( predictions_path, sep = ',' )

    number_of_elements_in_fold = len( predictions_dataframe.index )
    
    predicted_score_list_fold = []
    actual_score_list_fold = []
    
    for index, row in predictions_dataframe.iterrows( ):
      
      current_figure_id_and_name = row['Filename']
      # current_figure_id_and_name is 528_figure_name. The first 4 characters are "120_".
      current_figure_name = current_figure_id_and_name[4:]

      predicted_score = float( row['Predictions'] )

      dataset_information_row = dataset_information_dataframe.loc[ dataset_information_dataframe['figure_name'] == current_figure_name ]
      actual_score = dataset_information_row['osterrieth_score_by_experts'].values[0]

      # Regression outliers.
      if predicted_score > 36:
        print( 'Predicted score > 36.' )
        print( current_figure_name )
        print( predicted_score )
        print( '\n' )
        predicted_score = 36
        regression_outliers += 1
      elif predicted_score < 0:
        print( 'Predicted score < 0.' )
        print( current_figure_name )
        print( predicted_score )
        print( '\n' )
        predicted_score = 0
        regression_outliers += 1

      # Filling lists.
      predicted_score_list_fold.append( predicted_score )
      actual_score_list_fold.append( actual_score )

      predicted_score_list.append( predicted_score )
      actual_score_list.append( actual_score )

      # Confusion matrix.
      rounded_actual_score = int( np.floor( actual_score ) )
      rounded_predicted_score = int( np.floor( predicted_score ) )
      confusion_matrix_cumulative[ rounded_actual_score, rounded_predicted_score ] += 1  

    # Pearson's correlation.
    pcc_fold, p_value_fold = sps.pearsonr( actual_score_list_fold, predicted_score_list_fold )
    pcc_fold_list[ fold_index ] = pcc_fold

    # R^2 AKA coefficient of determination.
    r2_fold = r2_score( actual_score_list_fold, predicted_score_list_fold )
    r2_fold_list[ fold_index ] = r2_fold

    # Mean absolute error.
    mae_fold = mean_absolute_error( actual_score_list_fold, predicted_score_list_fold )
    mae_fold_list[ fold_index ] = mae_fold

    # Mean squared error.
    # squared = True. MSE.
    # squared = False. RMSE.
    mse_fold = mean_squared_error( actual_score_list_fold, predicted_score_list_fold, squared = True )
    rmse_fold = mean_squared_error( actual_score_list_fold, predicted_score_list_fold, squared = False )
    mse_fold_list[ fold_index ] = mse_fold
    rmse_fold_list[ fold_index ] = rmse_fold

    # Median absolute error.
    medae_fold = median_absolute_error( actual_score_list_fold, predicted_score_list_fold )
    medae_fold_list[ fold_index ] = medae_fold

  print( 'Evaluation metrics.' )  
  print( '\n' )

  # Pearson's correlation.
  
  pcc, p_value = sps.pearsonr( actual_score_list, predicted_score_list )
  
  print( "Pearson's correlation coefficient:", pcc )
  print( "Pearson's correlation p-value:", p_value )

  print( "Pearson's correlation coefficient (per fold):", pcc_fold_list )
  print( "Pearson's correlation coefficient (per fold) (averaged):", mean( pcc_fold_list ) )

  print( '\n' )

  # R^2 AKA coefficient of determination.
  
  r2 = r2_score( actual_score_list, predicted_score_list )
  
  print( "R^2 AKA coefficient of determination:", r2 )

  print( "R^2 AKA coefficient of determination (per fold):", r2_fold_list )
  print( "R^2 AKA coefficient of determination (per fold) (averaged):", mean( r2_fold_list ) )

  print( "R^2 formula." )
  print( "output = 1 - ( numerator / denominator )" )
  print( "numerator = sum( ( y_true - y_pred ) ^ 2 )" )
  print( "denominator = sum( ( y_true - mean( y_true ) ) ^ 2 )" )

  print( '\n' )

  # Mean absolute error.
  
  mae = mean_absolute_error( actual_score_list, predicted_score_list )
  
  print( "Mean absolute error:", mae )
  print( "Mean absolute error (per fold):", mae_fold_list )
  print( "Mean absolute error (per fold) (averaged):", mean( mae_fold_list ) )
  print( '\n' )

  # Mean squared error.
  
  # squared = True. MSE.
  # squared = False. RMSE.
  
  mse = mean_squared_error( actual_score_list, predicted_score_list, squared = True )
  rmse = mean_squared_error( actual_score_list, predicted_score_list, squared = False )
  
  print( "Mean squared error:", mse )
  print( "Mean squared error (per fold):", mse_fold_list )
  print( "Mean squared error (per fold) (averaged):", mean( mse_fold_list ) )
  print( '\n' )

  print( "Root mean squared error:", rmse )
  print( "Root mean squared error (per fold):", rmse_fold_list )
  print( "Root mean squared error (per fold) (averaged):", mean( rmse_fold_list ) )
  print( '\n' )

  # Median absolute error.
  
  medae = median_absolute_error( actual_score_list, predicted_score_list )
  
  print( "Median absolute error:", medae )
  print( "Median absolute error (per fold):", medae_fold_list )
  print( "Median absolute error (per fold) (averaged):", mean( medae_fold_list ) )
  print( '\n' )

  # Regression outliers.
  
  print( 'Number of outliers due to regression:', regression_outliers )
  print('\n')

  # Saving confusion matrix.
  
  confusion_matrix_cumulative_as_csv = output_confusion_matrices_dir + 'CM_' + input_model_base_name + '_' + str( NUMBER_OF_FOLDS ) + 'folds.csv'
  ml_utils.saveConfusionMatrixWithHeader( confusion_matrix_cumulative_as_csv, confusion_matrix_cumulative, NUM_OF_CLASSES, MATRIX_VALUE_SEPARATOR )
  
  print( 'Cumulative confusion matrix saved to:', confusion_matrix_cumulative_as_csv )
  print( '\n' )