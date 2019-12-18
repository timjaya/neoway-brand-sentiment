import fire
# import config
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict
import spacy
from .runPrediction import Predictor
from .spacy_train import run_training
from .spacy_validate import evaluate_spacy
import benepar
import matplotlib.pyplot as plt
import ast
from scipy.stats import spearmanr

def preprocess(reviews, brandlist, sample_size=2000, validation_size=0.1, 
               test_size=0.25, verbose=0, **kwargs):
    """Function that generates the dataset for Spacy training. 
    
    Input: Yelp dataset
    Output: train/test CSV for ER model training
    Parameters:
    - reviews: pandas dataframe of reviews
    - brandlist: pandas dataframe containing list of products/brands
    - sample_size: total number of reviews to subset
    - validation_size: proportion of total sample_size to validate on
    - test_size: proportion of total sample_size that will serve as the test set
    
    NOTE 
    ----
    config.data_path: workspace/data
    You should use workspace/data to put data to working on.  Let's say
    you have workspace/data/iris.csv, which you downloaded from:
    https://archive.ics.uci.edu/ml/datasets/iris. You will generate
    the following:
    + workspace/data/test.csv
    + workspace/data/train.csv
    + workspace/data/validation.csv
    + other files
    With these files you can train your model!
    """
    if verbose == 1:
      print("==> GENERATING DATASETS FOR TRAINING YOUR MODEL")

    # Convert brands in brand list to lowercase
    brandlist.word = brandlist.word.str.lower()

    # Extract a sample of reviews to generate training/validation/test data from
    sample = reviews.sample(n=sample_size)

    # Convert reviews to format relevant for spacy training
    if verbose == 1:
      print("   ===> CONVERTING DATA FOR SPACY")
    train_data = []
    print("LENGTH OF DATASET: ", len(sample))
    for index, row in tqdm(sample.iterrows()):
        # print(index)
        brands_tmp = []
        for brand in brandlist.word:
            text = row.text.lower()
            start_index = 0
            while start_index < len(text):
                start_index = text.find(brand, start_index)
                end_index = start_index + len(brand)
                if start_index == -1:
                    break
                if not text[start_index-1].isalpha() and (end_index == len(text) or not text[end_index].isalpha()):
                    if brand not in ['place', 'restaurant', 'cafe', 'establishment', 'diner']:
                        brands_tmp.append((start_index, end_index, "PRODUCT"))
                    else:
                        brands_tmp.append((start_index, end_index, "PRODUCT"))

                start_index += len(brand)
        train_data.append((row.review_id, row.text, brands_tmp))

    result = pd.DataFrame(train_data, columns=['review_id', 'text', 'entities'])

    # Split processed data into train/validation/test sets
    if verbose == 1:
      print("   ===> SPLITTING INTO TRAIN/VALIDATION/TEST SETS")
    train_validation, test = train_test_split(result, test_size=test_size)
    train, validation = train_test_split(train_validation, test_size=validation_size / (1-test_size))

    # Output to CSV in data folder
    train.to_csv('./workspace/data/train.csv')
    validation.to_csv('./workspace/data/validation.csv')
    test.to_csv('./workspace/data/test.csv')
    
    if verbose == 1:
      print("==> DATASETS GENERATED")
    
    return train, validation, test

def train(**kwargs):
    """Function that will run your model, be it a NN, Composite indicator
    or a Decision tree, you name it.

    NOTE
    ----
    config.models_path: workspace/models
    config.data_path: workspace/data

    As convention you should use workspace/data to read your dataset,
    which was build from generate() step. You should save your model
    binary into workspace/models directory.
    """
    print("==> TRAINING YOUR SPACY MODEL!")

    # TODO: Load data from workspace/data
    # TODO: Save trained model to workspace/models
    run_training()

    print("==> SPACY TRAINING COMPLETE!")

def metadata_spacy(**kwargs):
    """Generate metadata for model governance using testing!
    TODO: since sentiment analysis validation takes a while, should take this as an option

    NOTE
    ----

    workspace_path: config.workspace_path

    In this section you should save your performance model,
    like metrics, maybe confusion matrix, source of the data,
    the date of training and other useful stuff.

    You can save like as workspace/performance.json:

    {
       'name': 'My Super Nifty Model',
       'metrics': {
           'accuracy': 0.99,
           'f1': 0.99,
           'recall': 0.99,
        },
       'source': 'https://archive.ics.uci.edu/ml/datasets/iris'
    }
    """

    # Metadata for ER 

    # Metadata for Sentiment Analysis
    # Use predict from Sentiment Analysis

    print("==> TESTING SPACY MODEL PERFORMANCE AND GENERATING METADATA")

    evaluate_spacy()

def predict(input_data):
    """Predict: load the trained model and score input_data
    
    
    Generates sentiment scores given a list of reviews using default settings.    

    Args:
        input_data: List of strings containing each review

    Returns:
        A list of list of tuples containing an entity and sentiment score pair
        For exmaple:
        
        [[('mango', 0.6369), ('service', 0.8016)], 
        [('buffet', -0.6639), ('dessert', -0.6639), ('Palace', 0.9072), 
        ('place', 0.9661), ('table', -0.4417), ('food', 0.0), 
        ('service', -0.0772), ('salad', 0.3182)]]
        
        
    NOTE
    ----
    As convention you should use predict/ directory
    to do experiments, like predict/input.csv.

    """

    print("==> PREDICT DATASET {}".format(input_data))
    predictor = Predictor()
    result = predictor.defaultPredict(input_data)
    return result

# Run all pipeline sequentially for training, create pickled models and subset data
def run(**kwargs):
    """Run the complete pipeline of the model.
    """
    print("Args: {}".format(kwargs))
    print("Running <@model> by <@author>")

    reviews = pd.read_csv('./workspace/data/restaurant_reviews_10k.csv')
    brandlist = pd.read_csv('./workspace/data/wordnet_food_beverages_list.csv', header=None, names=['word'])
    
    preprocess(reviews, brandlist, sample_size=2000, verbose=1)
    train(**kwargs)     # training model and save to filesystem
    metadata_spacy(**kwargs)  # performance report of ER Model with Sentiment Analysis

    example_input = "Best Thai food ever! Love the mango curry especially and everything. They have great bubble tea too. Very nice service very polite."
    print("EXAMPLE INPUT: ", example_input)
    print("RESULT: ", predict(example_input))

def cli():
    """Caller of the fire cli"""
    return fire.Fire()

if __name__ == '__main__':
    cli()
