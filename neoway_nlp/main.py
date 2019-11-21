import fire
from neoway_nlp import config
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess(reviews, brandlist, sample_size=20000, validation_size=0.1, 
               test_size=0.25, **kwargs):
    """Function that will generate the dataset for your model. It can
    be the target population, training or validation dataset. You can
    do in this step as well do the task of Feature Engineering.
    Input: Yelp dataset
    Output: train/test CSV for ER model training
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
    print("==> GENERATING DATASETS FOR TRAINING YOUR MODEL")

    # Convert brands in brand list to lowercase
    brandlist.word = brandlist.word.str.lower()

    # Extract a sample of reviews to generate training/validation/test data from
    sample = reviews.sample(n=sample_size)

    # Convert reviews to format relevant for spacy training
    print("   ===> CONVERTING DATA FOR SPACY")
    train_data = []
    for index, row in sample.iterrows():
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
                        brands_tmp.append((start_index, end_index, "ESTABLISHMENT"))

                start_index += len(brand)
        train_data.append((row.review_id, row.text, brands_tmp))

    result = pd.DataFrame(train_data, columns=['review_id', 'text', 'entities'])

    # Split processed data into train/validation/test sets
    print("   ===> SPLITTING INTO TRAIN/VALIDATION/TEST SETS")
    train_validation, test = train_test_split(result, test_size=test_size)
    train, validation = train_test_split(train_validation, test_size=validation_size / (1-test_size))

    # Output to CSV in data folder
    train.to_csv('../data/train.csv')
    validation.to_csv('../data/validation.csv')
    test.to_csv('../data/test.csv')

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
    print("==> TRAINING YOUR MODEL!")

    # TODO: Load data from workspace/data
    # TODO: Save trained model to workspace/models


def metadata(**kwargs):
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

    print("==> TESTING MODEL PERFORMANCE AND GENERATING METADATA")


def predict(input_data):
    """Predict: load the trained model and score input_data

    NOTE
    ----
    As convention you should use predict/ directory
    to do experiments, like predict/input.csv.
    """

    print("==> PREDICT DATASET {}".format(input_data))


    # TODO: Predict Entities Here using ER Model
    # input: str of comment text
    # output: list of entities

    # 1. Load saved ER Model using spacy.load
    # 2. Predict entities for each input data

    # TODO: Predict Sentiments of those entities using Sentiment Analysis
    # input: str of comment text, list of entities
    # output: list of tuples with str and score e.g. [('pasta', 0.3612)]
    
    # 1. Start NLP Server
    # 2. Predict results for each input data
    # nlp = SentimentAnalyzer()
    # nlp.predict(x)
    # nlp.stop_server()

    # TODO: return result

# Run all pipeline sequentially for training, create pickled models and subset data
def run(**kwargs):
    """Run the complete pipeline of the model.
    """
    print("Args: {}".format(kwargs))
    print("Running <@model> by <@author>")

    # TODO: Train ER Model
    preprocess(**kwargs)  # generate dataset for training
    train(**kwargs)     # training model and save to filesystem
    metadata(**kwargs)  # performance report of ER Model with Sentiment Analysis

def cli():
    """Caller of the fire cli"""
    return fire.Fire()

if __name__ == '__main__':
    cli()
