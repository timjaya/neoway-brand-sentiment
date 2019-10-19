import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words_set = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
import re
from contractions import contractions_dict
nltk.download('punkt')


def main_clean(df):
    # Read data
    reviews = pd.read_json('../../data/reviews_1500k.json', lines=True)
    reviews = split_reviews(reviews)
    reviews = remove_stop_words(reviews)
    reviews = lemmatize_stem(reviews)
    reviews = remove_contractions(reviews)
    reviews = remove_punctuations(reviews)
    return reviews


# Pre-processing steps
def split_reviews(reviews):
    # Note that this takes a while to run
    reviews['text'] = reviews['text'].str.lower().str.split()
    return reviews


def remove_stop_words(reviews):
    # Removing stop words
    reviews['text'] = reviews['text'].apply(lambda x: [item for item in x if item not in stop_words_set])
    # Convert back to string, takes a while to run
    reviews['text'] = reviews['text'].apply(lambda x: [' '.join(x)][0])
    return reviews


def lemmatize_stem(reviews):
    ps = PorterStemmer()
    reviews['lemmatized_stemmed_text'] = reviews['text'].apply(lambda x: [ps.stem(x)][0])
    return reviews


def remove_contractions(reviews):
    def expand_contractions(text, contractions_dict):
        contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contractions_dict.get(match) \
                if contractions_dict.get(match) \
                else contractions_dict.get(match.lower())
            expanded_contraction = expanded_contraction
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text

    reviews['lemmatized_stemmed_text'] = reviews['lemmatized_stemmed_text']. \
        apply(lambda x: [expand_contractions(x, contractions_dict)][0])
    return reviews


def remove_punctuations(reviews):
    reviews['text'] = reviews['text'].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))
    return reviews
