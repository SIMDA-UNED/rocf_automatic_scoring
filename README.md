# A Benchmark for Rey-Osterrieth Complex Figure Test Automatic Scoring

![](https://img.shields.io/badge/language-Python-{green}.svg)
![](https://img.shields.io/badge/license-GNU-{yellowgreen}.svg)

This code repository is the official source code of the paper ["A Benchmark for Rey-Osterrieth Complex Figure Test Automatic Scoring"]() ([JOURNAL Link](), [arXiv Link]()), by [Juan Guerrero Martín et al.](https://github.com/xxx/)

## Requirements

GNU/Linux Debian 11 (bullseye=stable 2021-08-14)

Python 3.9.2

TensorFlow + Keras 2.7.0

How to install (clone) our environment will be detailed in the following.

Our hardware environment: 40 Intel(R) Xeon(R) Silver 4210 CPU @ 2.20Ghz, 100 GB RAM, 2 Tesla V100 GPUs, 2 Tesla V100S GPUs.

All the following codes can run on a single Tesla V100S GPU.

## Usage

```
# 1. Choose your workspace and download our repository.
cd ${CUSTOMIZED_WORKSPACE}
git clone https://github.com/xxx/rocf_automatic_scoring

# 2. Enter the directory.
cd rocf_automatic_scoring

# 3. Download the datasets (e-cienciaDatos).

Directory with the ROCFD528 dataset:
/home/jguerrero/Desarrollo/DATA/proyecto_REY/datasets/rocf_RD_3_0_528_binarize_label_split/all_classes/

Directory with the QD dataset:
/home/jguerrero/Desarrollo/DATA/proyecto_REY/datasets/quickdraw_dataset_2_0_414k_binarize_tvt_split_345/

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
@article{guerrero2023rocf,
  title={A Benchmark for Rey-Osterrieth Complex Figure Automatic Scoring},
  author={Guerrero, Juan and Rincón, Mariano},
  journal={XXX},
  year={2023},
  publisher={XXX}
}
```

## License

This project is licensed under the GNU General Public License v3.0.

## Acknowledgement

The authors gratefully acknowledge to research project CPP2021-009109 and an FPI-UNED-2021 scholarship.

## Contact

If you would have any discussion on this code repository, please feel free to send an email to Juan Guerrero Martín.  
Email: **jguerrero@dia.uned.es**