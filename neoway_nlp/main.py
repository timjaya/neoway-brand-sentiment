import fire
# import config
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict
import spacy
from .runPrediction import prediction
from .spacy_train import run_training
from .spacy_validate import evaluate_spacy
import benepar
import matplotlib.pyplot as plt
import ast
from scipy.stats import spearmanr



def preprocess(reviews, brandlist, sample_size=20000, validation_size=0.1, 
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

def validate_end_to_end():

    benepar.download("benepar_en2")
    parser = benepar.Parser("benepar_en2")

    example = "I just EASILY had the BEST lunch I've ever eaten!  It was THAT good!\n\nThe chicken tortilla soup was out of this world...light and delicate...fresh and HOT!!!\nI had two fish tacos with no tortilla.  One was a regular fish taco and the other was a beer battered fish taco.l\n\nThe fire roasted salsa was EASILY the best salsa I have ever had, too!\n\nThis place is a serious gem!  I could go there every single day!\n\nThanks guys!!"

    print("   ==> READING SPACY MODEL")
    model_dir = "./workspace/models/er_model"
    nlp = spacy.load(model_dir)

    def get_entities(nlp_model, text):
        """
        Input nlp_model and text, retrieve a list of unique entities from the text.
        """
        doc = nlp_model(text)
        entities = set()
        for ent in doc.ents:
            if ent.label_ == "PRODUCT":
                entities.add(ent.text)
        return list(entities)

    print("entity example: ")
    print("TEXT: ", example)
    print("ENTITIES: \n") 
    example_entity_list = get_entities(nlp, example)
    print(example_entity_list)

    bus = pd.read_csv("./workspace/end_to_end_test_set.csv")

    business_ids_similar_stars = bus.business_id.unique()

    rule = 'rule_2'

    correlation_scores = []

    print("RULE: ", rule)
    for bus_id in tqdm(business_ids_similar_stars):
        print("Running on restaurant ", bus_id, "...")
        subset = bus[bus.business_id == bus_id]
        
        # only get reviews with enough amount of text
        reviews_subset = [review for review in subset.text if len(review) < 400]

        print("Number of Reviews left after subset length: ", len(reviews_subset))
        
        # get set of entities for this particular restaurant,
        # and count how many reviews each entity have
        entities_with_count = defaultdict(int) 
        review_entities = [] # extract entities for each review
        print("Extracting entities from each review...")
        for review in tqdm(reviews_subset):
            entities = get_entities(nlp, review)

            # add this review as a count to an entity
            for ent in entities:
                entities_with_count[ent.lower()] += 1

            review_entities.append(entities)
            
        # only grab entities that have enough reviews
        print("Filtering entities to have enough reviews...")
        entities_with_enough_reviews = []
        threshold = 30
        for key, value in entities_with_count.items():
            if value >= threshold:
                entities_with_enough_reviews.append(key)
                
        # TRUE RANKINGS CALCULATION
        # for each entity, average ratings
        true_rankings = defaultdict(list)

        print("Calculating Yelp Star Rankings... ")
        for entity in entities_with_enough_reviews:
            true_rankings['entity'] += [entity]
            entity_reviews = subset[subset.text.str.contains(entity, case=False)]
            true_rankings['average_stars'] += [np.mean(entity_reviews.stars)]

        true_rankings = pd.DataFrame(true_rankings)
        
        # PREDICTION RANKING CALCULATION
        print("Calculating Prediction Rankings...")
        # Filter entities of each review to be from the entities_with_enough_review set
        entity_filter = set(entities_with_enough_reviews)

        filtered_entities = []

        for entities in review_entities:
            filtered = []
            for ent in entities:
                ent = ent.lower()
                if ent in entity_filter:
                    filtered.append(ent)
            filtered_entities.append(filtered)

        # perform sentiment analysis for each review with filtered entities above
        predicted_scores = defaultdict(list)

        from .main import predict

        print("Performing sentiment analysis for each review... ")
        for i, review in enumerate(tqdm(reviews_subset)):
            entities = filtered_entities[i]

        #     print(review)

            scores = predict(review)
            print(scores)
            #, entities, parser = parser, sentiment_package='vader', rule=rule)

            # save results 
            for entity, score in scores:
                predicted_scores[entity] += [score]

        # create rankings from scores
        predicted_rankings = defaultdict(list)
        for entity, scores in predicted_scores.items():
            predicted_rankings['entity'] += [entity]
            predicted_rankings['predicted_score'] += [np.mean(scores)]

        predicted_rankings = pd.DataFrame(predicted_rankings)

        #### may not be necessary to do these castings
        predicted_rankings['entity'] = predicted_rankings['entity'].astype(str)
        true_rankings['entity'] = true_rankings['entity'].astype(str)
        ####
        
        full_rankings = true_rankings.merge(predicted_rankings, how='left').fillna(0)

        # spearman correlation metric
        print("Rankings result: ")
        print(full_rankings)
        
        corr, pvalue = spearmanr(full_rankings.average_stars, full_rankings.predicted_score)
        print("Spearman Correlation Score: ", corr)
        correlation_scores.append(corr)
    print("Final correlation score: ", np.mean(correlation_scores))


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
    
    runPrediction = prediction(input_data)
    result = runPrediction.defaultPredict()
    return result


def validate(result):
    """
    The input is the list of entities with an sentiment score outputted by
    the prediction function above.

    NOTE
    -----------
    Prints out intermediary results

    Returns
    -----------
    
    - list of correlation scores

    """
    pass



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
