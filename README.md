# Municipal Solid Waste Prediction

_Municipal Solid Waste (MSW) management enact a significant role in protecting public
health and the environment. The main objective of this paper is to explore the utility of using state-of-the-art machine learning and deep learning-based models for predicting future variations in MSW generation for a given geographical region, considering its past waste generation pattern. We consider nine different machine learning and deep-learning models to examine and evaluate their capability in forecasting the daily generated waste amount. In order to have a comprehensive evaluation, we explore the utility of two training and prediction paradigms, a single model approach and a multi-model ensemble approach. Three Sri Lankan datasets from; Boralesgamuwa, Dehiwala, and Moratuwa, and open-source daily waste datasets from the city of Austin and Ballarat, are considered in this study. In addition, we provide an in depth discussion on important considerations to make when choosing a model for predicting MSW generation._

## About

This repository contains a collection of experiments that attempts in predicting the municipal solid waste generation of five areas around the world.

If you use any of the code or datasets in here for your research for publication, please consider citing our [paper](https://ieeexplore.ieee.org/document/9950270).

```
@article{mudannayake2022exploring,
  title={Exploring Machine Learning and Deep Learning Approaches for Multi-Step Forecasting in Municipal Solid Waste Generation},
  author={Mudannayake, Oshan and Rathnayake, Disni and Herath, Jerome Dinal and Fernando, Dinuni K and Fernando, Mgnas},
  journal={IEEE Access},
  year={2022},
  publisher={IEEE}
}
```

## Installation guide

### Setup the project
Execute the following commands to setup the project.
1. `git clone https://github.com/ivantha/solid-waste-prediction.git`
2. `cd solid-waste-prediction`
3. `conda env create -f environment.yml`

### Preprocess data
1. Start the Jupyter Notebook by executing the `jupyeter notebook` command within the project conda environment.
2. Execute the notebooks in the **process_data** directory to process the relavant datasets.

### Run experiments
1. Go to **experiments** directory by `cd experiements`.
2. Run the desired experiment. e.g. - `python arima.py`

### Generate reports
1. Go to **reports** directory by `cd reports`.
2. Run the desired report. e.g. - `python 10_plot_image.py`
