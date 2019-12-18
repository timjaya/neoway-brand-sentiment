import time
import pandas as pd

sample_reviews = pd.read_json("D:/git_repos/neoway-brand-sentiment/sample_data/reviews_1000.json", lines = True)
speed_dict = dict()

# Calculate speed for Stanford NLP
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')
start_time = time.time()
count = 0
for reviews in sample_reviews["text"]:
    res = nlp.annotate(reviews,
                       properties={
                           'annotators': 'sentiment',
                           'outputFormat': 'json'
                       })
    count += 1
end_time = time.time()
speed_dict["stanford_nlp"] = end_time - start_time

from joblib import Parallel, delayed
import multiprocessing
     
# what are your inputs, and what operation do you want to 
# perform on each input. For example...
inputs = sample_reviews["text"]

def processInput(i):
    res = nlp.annotate(reviews,
                       properties={
                           'annotators': 'sentiment',
                           'outputFormat': 'json'
                       })
    return res
 
num_cores = multiprocessing.cpu_count()

start_time = time.time()
results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
end_time = time.time()

speed_dict["stanford_nlp_multi"] = end_time - start_time

print(speed_dict)
print(len(results))