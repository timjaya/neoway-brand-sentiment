import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
from stanfordcorenlp import StanfordCoreNLP
from joblib import Parallel, delayed
import multiprocessing
import subprocess
import json
import numpy as np

git_repo = os.path.join(os.path.dirname(os.path.dirname(__file__)))

class StanfordNLP:
    def __init__(self, 
                 host='http://localhost', 
                 port=9000,
                 folder_name = 'stanford-corenlp-full-2018-10-05'):
        
        # java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -sentiment.threads 8 -port 9000 -timeout 30000
        
        
        if sys.platform == 'win32':
            command = git_repo[0:2] + " && chdir " + git_repo + " && chdir " + folder_name + ' && java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -sentiment.threads 8 -port 9000 -timeout 30000'
            subprocess.Popen(command, 
                            shell=True,
                            stdin=None, 
                            stdout=None, 
                            stderr=None)
        elif sys.platform == 'linux':
            command = git_repo[0:2] + "; cd " + git_repo + "; cd " + folder_name + '; java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -sentiment.threads 8 -port 9000 -timeout 30000'
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
    
    def _internal_parse(self, sentence):
        return self.nlp.parse(sentence)
    
    def parse(self, sentences):
        results = Parallel(n_jobs=self.num_cores)(delayed(self._internal_parse)(i) for i in sentences)
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
            
        elif sys.platform == 'linux':
            pass
    
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
if __name__ == '__main__':
    sNLP = StanfordNLP()
    sample_reviews = pd.read_json(git_repo + "/sample_data/reviews_1000.json", lines = True)
    reviews = [i for i in sample_reviews['text']][0:800]
    # res = sNLP.annotate(reviews)    
    res2 = sNLP.parse(reviews)
    # print(len(res))
    print(len(res2))
    sNLP.kill_host()


