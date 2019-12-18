# example script to use the neoway_nlp

import pandas as pd
from neoway_nlp.runPrediction import Predictor
from neoway_nlp.main import preprocess, train, metadata_spacy, predict

def test_preprocessing():
    # READ AND PREPROCESS DATA

    # IF READ IN FULL DATA
    # reviews = pd.read_json('./workspace/data/restaurant_reviews_1900k.json', lines=True) # raw data

    # CODE TO SUBSET FULL DATA
    # subset = reviews.sample(10000).reset_index().drop(columns = ['index'])
    # subset.to_json('./restaurant_reviews_10k.json')

    # READ IN SUBSET DATA (10k)

    reviews = pd.read_csv('./workspace/data/restaurant_reviews_10k.csv')
    brandlist = pd.read_csv('./workspace/data/wordnet_food_beverages_list.csv', header=None, names=['word'])
    preprocess(reviews, brandlist, sample_size=2000, verbose=1)

def test_training():
    train()

def test_metadata_spacy():
    metadata_spacy()

def test_predict_single():
    input_data = "Best Thai food ever! Love the mango curry especially and everything. They have great bubble tea too. Very nice service very polite."
    results = predict(input_data)
    print("INPUT: ", input_data)
    print("RESULTS: ", results)

def test_predict():
    input_data = ["Best Thai food ever! Love the mango curry especially and everything. They have great bubble tea too. Very nice service very polite.",
                  "The service was pretty bad. There were these two grouchy ladies dealing with us at the entrance and we have to wait ridiculously long for a table even though the restaurant is almost empty. Then we have to sit as far away from the food as possible. When we ask to sit closer they say no. We ask why and they say no one is covering that area. Seems like they could easily fix that or make one exception, I mean its a buffet. we won't be needing much service if any. Luckily that lady who took our drink orders was pretty nice and came over often to clear our plates and refill our mimosas. The food at the buffet was one of the better buffets I've had in Vegas, definitely better than Monte Carlo's or Harrah's, but nothing compared to Paris or Mirage. I was told Caesar's Palace is the best, but I've never had it. From the looks of it, I'm sure its a million times better than this place. If you want a relatively less expensive buffet where most of the food is tasty and there's a fair amount of variety you might want to go here especially for the mimosa's. We got a great deal from tix4tonight. Definitely check out tix4tonight or a service like it before you go anywhere in Vegas. One really cool feature here is the salad bar, where they make you a salad fresh for you. I really enjoyed the Chinese Chicken Salad. There's also a dessert area that has more choices than a lot of the other buffets, though the execution of them aren't so great. Definitely worth going to for the food, but I really thought the service was the worst out of all the restaurants I went to in Vegas. For those reasons I probably won't go back, but I wouldn't tell people not to go."]
    results = predict(input_data)
    print("INPUT: ", input_data)
    print("RESULTS: ", results)

def test_all_rules():
    input_data = ["Best Thai food ever! Love the mango curry especially and everything. They have great bubble tea too. Very nice service very polite.",
                  "The service was pretty bad. There were these two grouchy ladies dealing with us at the entrance and we have to wait ridiculously long for a table even though the restaurant is almost empty. Then we have to sit as far away from the food as possible. When we ask to sit closer they say no. We ask why and they say no one is covering that area. Seems like they could easily fix that or make one exception, I mean its a buffet. we won't be needing much service if any. Luckily that lady who took our drink orders was pretty nice and came over often to clear our plates and refill our mimosas. The food at the buffet was one of the better buffets I've had in Vegas, definitely better than Monte Carlo's or Harrah's, but nothing compared to Paris or Mirage. I was told Caesar's Palace is the best, but I've never had it. From the looks of it, I'm sure its a million times better than this place. If you want a relatively less expensive buffet where most of the food is tasty and there's a fair amount of variety you might want to go here especially for the mimosa's. We got a great deal from tix4tonight. Definitely check out tix4tonight or a service like it before you go anywhere in Vegas. One really cool feature here is the salad bar, where they make you a salad fresh for you. I really enjoyed the Chinese Chicken Salad. There's also a dessert area that has more choices than a lot of the other buffets, though the execution of them aren't so great. Definitely worth going to for the food, but I really thought the service was the worst out of all the restaurants I went to in Vegas. For those reasons I probably won't go back, but I wouldn't tell people not to go."]
    predictor = Predictor()
    results = predictor.defaultPredict(input_data)
    print("INPUT: ", input_data)
    print("RESULTS: ", results)
    
    for rule_num in range(1, 10):
        print("RULE NUMBER: ", rule_num)
        predict_result = predictor.customPredict(input_data, rule_num)            
        print("INPUT: ", input_data)
        print("RESULTS: ", results)

# VALIDATE END-TO-END
# work in progress
# validate_end_to_end()

if __name__ == "__main__":
    test_preprocessing()
    test_training()
    test_metadata_spacy()
    test_predict_single()
    test_predict()
    test_all_rules()


