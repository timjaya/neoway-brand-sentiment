import fire
from neoway_nlp import config

def preprocess(comments, brandlist, **kwargs):
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
    prepare_ER_training_data(**kwargs)  # generate dataset for training
    train_ER(**kwargs)     # training model and save to filesystem
    metadata(**kwargs)  # performance report of ER Model with Sentiment Analysis

def cli():
    """Caller of the fire cli"""
    return fire.Fire()

if __name__ == '__main__':
    cli()
