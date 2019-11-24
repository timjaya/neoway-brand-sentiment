import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
from stanfordcorenlp import StanfordCoreNLP
from joblib import Parallel, delayed
import multiprocessing
import subprocess
import json
import numpy as np
import math
import time

class StanfordNLP:
    def __init__(self, 
                 host='http://localhost', 
                 port=9000,
                 folder_name = 'stanford-corenlp-full-2018-10-05'):
        
        # java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -sentiment.threads 8 -port 9000 -timeout 30000
        
        
        if sys.platform == 'win32':
            git_repo = os.path.join(os.path.dirname(os.path.dirname(__file__)))
            command = git_repo[0:2] + " && chdir " + git_repo + " && chdir " + folder_name + ' && java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -sentiment.threads 8 -port ' + str(port) + ' -timeout 30000'
            subprocess.Popen(command, 
                            shell=True,
                            stdin=None, 
                            stdout=None, 
                            stderr=None)
        elif sys.platform == 'darwin':
            git_repo = os.path.join(os.path.dirname(os.path.abspath(__file__)))
            command = "cd " + git_repo + "; cd .. " + "; cd " + folder_name + '; java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -sentiment.threads 8 -port ' + str(port) + ' -timeout 30000'
            subprocess.Popen(command, 
                            shell=True,
                            stdin=None, 
                            stdout=None, 
                            stderr=None)
            
        
        self.nlp = StanfordCoreNLP(host,
                                   port = port)
        
        self.props = {
            # 'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
            'annotators': 'sentiment',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }
        
        self.num_cores = multiprocessing.cpu_count()

    def word_tokenize(self, sentences):
        results = Parallel(n_jobs=self.num_cores)(delayed(self.nlp.word_tokenize)(i) for i in sentences)
        return results

    def pos(self, sentences):
        results = Parallel(n_jobs=self.num_cores)(delayed(self.nlp.pos_tag)(i) for i in sentences)
        return results

    def ner(self, sentences):
        results = Parallel(n_jobs=self.num_cores)(delayed(self.nlp.ner)(i) for i in sentences)
        return results
    
    def _create_batches(self, sentences):
        # Returns a list of list
        n = math.ceil(len(sentences) / 200)
        sentences_list = []
        for i in range(n):
            fix = 0
            if i == n-1:
                fix = len(sentences) % 200
            sentences_list.append(sentences[200 * i:(200 * (i+1)-fix)])
        return sentences_list
    
    def _internal_parse(self, sentence):
        return self.nlp.parse(sentence)
    
    def parse(self, sentences):
        # If number of sentences exceeds 200, we break it down into chunks of
        # 500 sentences
        if len(sentences) > 200:
            sentences_list = self._create_batches(sentences)
            results = []
            for sentence_group in sentences_list:
                print(len(sentence_group))
                result_temp = Parallel(n_jobs=self.num_cores,
                                       verbose=10)(delayed(self._internal_parse)(i) for i in sentence_group)
                results += result_temp
                time.sleep(1)
        else:
            results = Parallel(n_jobs=self.num_cores,
                               verbose=10)(delayed(self._internal_parse)(i) for i in sentences)
        return results

    def dependency_parse(self, sentences):
        results = Parallel(n_jobs=self.num_cores)(delayed(self.nlp.dependency_parse)(i) for i in sentences)
        return results
    
    def _internal_annotate(self, sentence):
        return self.nlp.annotate(sentence, properties=self.props)
    
    def annotate(self, sentences):
        results = Parallel(n_jobs=self.num_cores)(delayed(self._internal_annotate)(i) for i in sentences)
        return results
    
    def kill_host(self):
        if sys.platform == 'win32':
            process_str = subprocess.check_output('netstat -ano | findstr :9000', shell=True).decode()
            process_num = []
            for item in process_str.split():
                try:
                    pid = int(item)
                except ValueError:
                    continue
                if pid > 0:
                    process_num.append(pid)
            
            process_num = set(process_num)
            for pid in process_num:
                command = 'taskkill/pid ' + str(pid) + ' /F' 
                subprocess.Popen(command, 
                                 shell=True)
            print("Host terminated")
            
        elif sys.platform == 'darwin':
            process_str = subprocess.check_output("ps -ax | grep StanfordCoreNLPServer | awk '{print $1}'", shell=True).decode()
            process_num = process_str.split()
            for pid in process_num:
                command = 'kill ' + str(pid)
                subprocess.Popen(command, shell=True)
            print("Host terminated")

    
    def stanford_sentiment(self, entity_with_clause):
        entity_with_sentiment = []
        for entity, clause in entity_with_clause:
            result = self.annotate(clause)
            sentiment = np.dot(result['sentences'][0]['sentimentDistribution'], [-2, -1, 0, 1, 2])
            entity_with_sentiment.append((entity, sentiment))
        return entity_with_sentiment
    
    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens

# Example
"""
if __name__ == '__main__':
    sNLP = StanfordNLP()
    git_repo = os.path.join(os.path.dirname(os.path.dirname(__file__)))
    sample_reviews = pd.read_json(git_repo + "/sample_data/reviews_1000.json", lines = True)
    reviews = [i for i in sample_reviews['text']]
    # res = sNLP.annotate(reviews)
    start_time = time.time()
    res2 = sNLP.parse(reviews)
    end_time = time.time()
    print(end_time - start_time)
    # print(len(res))
    print(len(res2))
    sNLP.kill_host()
"""

