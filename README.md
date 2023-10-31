# A Benchmark for Rey-Osterrieth Complex Figure Test Automatic Scoring

![](https://img.shields.io/badge/language-Python-{green}.svg)
![](https://img.shields.io/badge/license-GNU-{yellowgreen}.svg)

This code repository is the official source code of the paper ["A Benchmark for Rey-Osterrieth Complex Figure Test Automatic Scoring"](http://www.simda.uned.es/rocf_automatic_scoring/index.html) by [Juan Guerrero Martín et al.](http://www.simda.uned.es/)

## Requirements

Operating system: GNU/Linux Debian 11 (bullseye=stable 2021-08-14)

Hardware environment: 40 Intel(R) Xeon(R) Silver 4210 CPU @ 2.20Ghz, 100 GB RAM, 2 Tesla V100 GPUs, 2 Tesla V100S GPUs. All the following codes can run on a single Tesla V100S GPU.

Programming language: Python 3.9.2

Programming libraries: TensorFlow + Keras 2.7.0

Please, download the ROCFD528 and QDSD414k datasets from the following website: http://www.simda.uned.es/rocf_automatic_scoring/index.html. Make sure to convert the ROCFD528 dataset into a pickle using the script utils/dataset_to_pickle.py.

How to clone and use our environment will be detailed in the following.

## Usage

```
# 1. Choose your workspace and download our repository.
cd ${CUSTOMIZED_WORKSPACE}
git clone https://github.com/SIMDA-UNED/rocf_automatic_scoring.git

# 2. Enter the directory.
cd rocf_automatic_scoring

# 3. Make sure that the datasets are downloaded.

Default directory with the ROCFD528 dataset:
/home/jguerrero/Desarrollo/DATA/proyecto_REY/datasets/rocfd528/

Default directory with the QDSD414k dataset:
/home/jguerrero/Desarrollo/DATA/proyecto_REY/datasets/qdsd414k/

# 4. Execute any of our scripts.

Example:

cd training
python train_model_with_rocf_dataset.py
```

## Script Description

A. **training/train_model_with_rocf_dataset.py** : It allows you to train one of the four models (SaN, MN2, IC3, ENB1) with the ROCFD528 dataset.

B. **training/train_model_with_quickdraw.py** : It allows you to train one of the four models with Quick, Draw! dataset.

C. **training/transfer_from_imagenet_to_rocf_dataset.py** : It allows you to re-train the association layers of one of the three models (MN2, IC3, ENB1) using ROCFD528 dataset. The models have been previously trained with ImageNet dataset.

D. **training/transfer_from_quickdraw_to_rocf_dataset.py** : It allows you to re-train the association layers of one of the four models using ROCFD528 dataset. The models have been previously trained with Quick, Draw! dataset (refer to B).

E. **evaluation/predict_with_rocf_dataset.py** : Given one of the eleven configurations, it predicts the labels of all the images in ROCFD528 dataset. You need to pass to the script a CSV with the partial model (its corresponding training epoch) that you are going to use to make the predictions.

F. **evaluation/evaluate_model_with_quickdraw.py** : Given one of the four models, it calculates validation and test accuracy for Quick, Draw! dataset.

G. **evaluation/extract_metrics_and_confusion_matrices.py** : It compares the scores given by the experts and predicted by the eleven configurations and returns the values of 5 metrics (pcc, r2, mae, rmse, medae) and the confusion matrix.

H. **utils/machine_learning_utils.py** : Here you can find some useful functions for the other scripts.

I. **utils/model_creator.py** : This script allows you to manipulate in different ways the four machine learning models discussed in the article.

J. **utils/dataset_to_pickle.py** : It converts the ROCFD528 dataset into a pickle.

## Experimental Results

In this section we show the values of the evaluation metrics for each of the eleven configurations.

Configuration | PCC | $R^2$ | MAE | RMSE | MedAE
:---: | :---: | :---: | :---: | :---: | :---:
SaN - DL | 0.859 | 0.727 | 3.448 | 4.426 | 2.825
MN2 - DL | 0.614 | 0.351 | 5.791 | 6.973 | 5.293
I3 - DL | 0.753 | 0.541 | 4.714 | 5.879 | 4.031
ENB1 - DL | 0.820 | 0.665 | 3.889 | 4.948 | 3.227
MN2 - TL - IN | 0.778 | 0.563 | 4.546 | 5.619 | 4.032
I3 - TL - IN | 0.780 | 0.600 | 4.318 | 5.464 | 3.627
ENB1 - TL - IN | 0.786 | 0.544 | 4.432 | 5.815 | 3.328
SaN - TL - QD | 0.735 | 0.400 | 5.602 | 6.731 | 5.273
MN2 - TL - QD | 0.795 | 0.623 | 4.257 | 5.255 | 3.571
I3 - TL - QD | 0.729 | 0.526 | 4.722 | 5.925 | 4.023
ENB1 - TL - QD | 0.804 | 0.639 | 4.068 | 5.124 | 3.494

## Citations

If you find this code useful to your research, please cite our paper as the following bibtex:

```
TBA
```

## License

This project is licensed under the GNU General Public License v3.0.

## Acknowledgement

The authors gratefully acknowledge to research project CPP2021-009109 and a FPI-UNED-2021 scholarship.

## Contact

If you would have any discussion on this code repository, please feel free to send an email to Juan Guerrero Martín.  
Email: **jguerrero@dia.uned.es**