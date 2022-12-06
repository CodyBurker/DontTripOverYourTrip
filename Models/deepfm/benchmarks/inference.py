import sys
sys.path.append('../FuxiCTR-main/')
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append('../')
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap, FeatureEncoder
from fuxictr.pytorch import models
from fuxictr.pytorch.torch_utils import seed_everything
import gc
import argparse
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np


# Read model and data parameters for feature mapping
experiment_id = "DeepFM_yelp_all_feature_019_eabe7106" # change to desired model name
config_dir = "./tuner_config/DeepFM_all_feature"
params = load_config(config_dir, experiment_id)
params['gpu'] = 0
params['version'] = "pytorch"
params['test_data'] = os.path.join(params['data_root'], 'inference.csv') # define inference data location
seed_everything(seed=params['seed'])

feature_encoder = FeatureEncoder(**params)
feature_map = feature_encoder.feature_map

# Preprocess data - form a data with one user and all business and trasform it to hdf5
def transform_data(input_user):
    # Select a user for testing
    user = pd.read_csv("../data/all_user.csv").loc[1]
    # Join the user data with all business data and save to a csv file
    df_inference = pd.read_csv("../data/all_business.csv")
    for col_name, col_val in user.iteritems():
        df_inference[col_name] = col_val
    df_inference["target"] = 0 # a redundant value in order to use the framework
    df_inference.to_csv("../data/inference.csv", index=False)

    # transform csv to hdf5 and save on disk
    datasets.build_dataset(feature_encoder, **params)
    print(params["test_data"])
    # Load hdf5 file and make it into a generator 
    dataset = params['dataset_id'].split('_')[0].lower()
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    params["test_data"] = os.path.join(data_dir, 'test*.h5')
    print(params["test_data"])
    return datasets.h5_generator(feature_map, stage='test', **params)

# Inference
def get_prediction(inference_gen):
    # Load model
    # initialize model
    model_class = getattr(models, params['model'])
    model = model_class(feature_map, **params)
    # print number of parameters used in model
    model.count_parameters()
    # load the best model checkpoint
    model.load_weights(model.checkpoint)
    # making inference
    print("Inferencing")
    inference_result = model.predict_generator(inference_gen)
    print(inference_result.size)
    return inference_result 

def render_prediction(inference_result):
    df_inference = pd.read_csv("../data/inference.csv")
    df_inference["score"] = inference_result
    df_inference = df_inference.sort_values("score", ascending=False)[:5]
    return df_inference[["business_name", "categories", "score"]]

if __name__ == '__main__':
#      app.run()
    inference_gen = transform_data("a")
    inference_result = get_prediction(inference_gen)
    recommendations = render_prediction(inference_result)
    print(recommendations)
