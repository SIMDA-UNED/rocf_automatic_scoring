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
import cv2
import natsort
import numpy as np
import os
import pandas as pd
import pickle

'''
Part 2. Argument parser.
'''

parser = argparse.ArgumentParser( description = 'dataset_to_pickle' )

parser.add_argument( "--dataset_dir", type = str, default = "/home/jguerrero/Desarrollo/DATA/proyecto_REY/datasets/rocf_RD_3_0_528_binarize_label_split/all_classes/", help = "Directory with the ROCFD528 dataset." )
parser.add_argument( "--dataset_information_path", type = str, default = "/home/jguerrero/Desarrollo/rocf_automatic_scoring/data/dataset_information/rocfd528_information.csv", help = "CSV with ROCFD528 dataset information." )
parser.add_argument( "--output_pickle_path", type = str, default = "/home/jguerrero/Desarrollo/DATA/proyecto_REY/datasets/rocf_RD_3_0_528.pickle", help = "Pickle with ROCFD528 dataset images and information." )

args = parser.parse_args( )

'''
Part 3. Global constants and variables.
'''

dataset_dir = args.dataset_dir
dataset_information_path = args.dataset_information_path
output_pickle_path = args.output_pickle_path

CSV_DELIMITER = ';'

MIN_SCORE = 0
MAX_SCORE = 36

'''
Part 4. Main body.
'''

if __name__ == '__main__':

  info_dataframe = pd.read_csv( dataset_information_path, sep = CSV_DELIMITER )

  X = []
  y = []
  paths = []

  image_list = os.listdir( dataset_dir )
  ordered_image_list = natsort.natsorted( image_list, reverse = False )

  for file in ordered_image_list:

    if file.endswith( ".png" ):

      file_name = os.path.splitext( file )[0]

      ## Processing paths.

      paths.append( file_name )

      ## Processing X.

      image_path = os.path.join( dataset_dir, file )
      image = cv2.imread( image_path, cv2.IMREAD_GRAYSCALE )
      # Normalizing. Pixels are kept in the format (0, 255).
      image = np.array( image, dtype = np.float32 )
      X.append( image )

      ## Processing y.

      # file_name is 528_figure_name. 4 characters "120_".
      figure_name = file_name[4:]
      row_of_interest = info_dataframe.loc[ info_dataframe['figure_name'] == figure_name ]
      ceiled_score = int( np.ceil( float( row_of_interest.iloc[0]['osterrieth_score_by_experts'] ) ) )
      normalized_score = ( ceiled_score - MIN_SCORE ) / ( MAX_SCORE - MIN_SCORE )
      y.append( normalized_score )

  # Creating the pickle.
  rocf_dataset = {}
  rocf_dataset['X'] = X
  rocf_dataset['y'] = y
  rocf_dataset['paths'] = paths

  # Saving the pickle.
  print( 'Generating pickle...' )
  with open( output_pickle_path, 'wb' ) as file:
    pickle.dump( rocf_dataset, file )
  print( 'Done.' )

  print( "Output path:", output_pickle_path )