import pandas as pd
from sklearn.model_selection import train_test_split
import unittest
from main import preprocess

class TestPreprocess(unittest.TestCase):

    reviews = pd.read_json('../data/restaurant_reviews_500.json', lines=True)
    brandlist = pd.read_csv('../analysis/wordnet_db/wordnet_food_beverages_list.csv', header=None, names=['word'])

    def test_lengths(self):
        train, validation, test = preprocess(self.reviews, self.brandlist, sample_size=100, verbose=0)
        self.assertEqual(len(train), 65, "Should be 65")
        self.assertEqual(len(validation), 10, "Should be 10")
        self.assertEqual(len(test), 25, "Should be 25")


if __name__ == '__main__':
    unittest.main()