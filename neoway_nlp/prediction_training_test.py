import fire
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from collections import defaultdict
import spacy
from runPrediction import prediction
from spacy_train import run_training
from main import train, predict


def run_training_test():
    run_training()
    

def test_main_training():
    train()


def run_predict_test():
    input_data = ["Best Thai food ever! Love the mango curry especially and everything. They have great bubble tea too. Very nice service very polite.",
                  "The service was pretty bad. There were these two grouchy ladies dealing with us at the entrance and we have to wait ridiculously long for a table even though the restaurant is almost empty. Then we have to sit as far away from the food as possible. When we ask to sit closer they say no. We ask why and they say no one is covering that area. Seems like they could easily fix that or make one exception, I mean its a buffet. we won't be needing much service if any. Luckily that lady who took our drink orders was pretty nice and came over often to clear our plates and refill our mimosas. The food at the buffet was one of the better buffets I've had in Vegas, definitely better than Monte Carlo's or Harrah's, but nothing compared to Paris or Mirage. I was told Caesar's Palace is the best, but I've never had it. From the looks of it, I'm sure its a million times better than this place. If you want a relatively less expensive buffet where most of the food is tasty and there's a fair amount of variety you might want to go here especially for the mimosa's. We got a great deal from tix4tonight. Definitely check out tix4tonight or a service like it before you go anywhere in Vegas. One really cool feature here is the salad bar, where they make you a salad fresh for you. I really enjoyed the Chinese Chicken Salad. There's also a dessert area that has more choices than a lot of the other buffets, though the execution of them aren't so great. Definitely worth going to for the food, but I really thought the service was the worst out of all the restaurants I went to in Vegas. For those reasons I probably won't go back, but I wouldn't tell people not to go."]
    predict(input_data)


def test_all_rules():
    input_data = ["Best Thai food ever! Love the mango curry especially and everything. They have great bubble tea too. Very nice service very polite.",
                  "The service was pretty bad. There were these two grouchy ladies dealing with us at the entrance and we have to wait ridiculously long for a table even though the restaurant is almost empty. Then we have to sit as far away from the food as possible. When we ask to sit closer they say no. We ask why and they say no one is covering that area. Seems like they could easily fix that or make one exception, I mean its a buffet. we won't be needing much service if any. Luckily that lady who took our drink orders was pretty nice and came over often to clear our plates and refill our mimosas. The food at the buffet was one of the better buffets I've had in Vegas, definitely better than Monte Carlo's or Harrah's, but nothing compared to Paris or Mirage. I was told Caesar's Palace is the best, but I've never had it. From the looks of it, I'm sure its a million times better than this place. If you want a relatively less expensive buffet where most of the food is tasty and there's a fair amount of variety you might want to go here especially for the mimosa's. We got a great deal from tix4tonight. Definitely check out tix4tonight or a service like it before you go anywhere in Vegas. One really cool feature here is the salad bar, where they make you a salad fresh for you. I really enjoyed the Chinese Chicken Salad. There's also a dessert area that has more choices than a lot of the other buffets, though the execution of them aren't so great. Definitely worth going to for the food, but I really thought the service was the worst out of all the restaurants I went to in Vegas. For those reasons I probably won't go back, but I wouldn't tell people not to go."]
    test_predict = prediction(input_data)
    print(test_predict.defaultPredict())
    
    for rule_num in range(1, 10):
        print(rule_num)
        predict_result = test_predict.customPredict(rule_num)            
        print(predict_result)
    


if __name__ == "__main__":
    run_training_test()
    test_main_training()
    test_all_rules()
    run_predict_test()
    