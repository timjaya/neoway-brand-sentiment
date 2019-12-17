import pandas as pd
from neoway_nlp.main import preprocess, train

# READ AND PREPROCESS DATA
reviews = pd.read_json('./workspace/data/restaurant_reviews_500.json', lines=True)
brandlist = pd.read_csv('./workspace/data/wordnet_food_beverages_list.csv', header=None, names=['word'])
preprocess(reviews, brandlist, sample_size=100, verbose=1)

# TRAIN DATA
# train()

# VALIDATE SPACY

# VALIDATE END-TO-END

# PREDICT 