import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import collections
from tqdm.notebook import tqdm
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import benepar
import re
import spacy
from joblib import Parallel, delayed
import multiprocessing

class prediction:
    def __init__(self,
                 input_data,
                 sentiment_package = "vader",
                 parse_package = "benepar",
                 model_dir = "ermodel"):
        
        # self.spacy_train = spacy_train
        # self.review = spacy_train['text']
        # self.entities = spacy_train['entities']
        
        self.sentiment_package = sentiment_package
        self.nlp = spacy.load(model_dir)
        self.input_data = input_data
        self.num_cores = multiprocessing.cpu_count()
        
        if parse_package == 'benepar':
            try:
                self.parser = benepar.Parser("benepar_en2")
            except LookupError:
                benepar.download('benepar_en2')
        elif parse_package == 'stanford':   
            pass
        else:
            raise Exception('incorrect parse package')
        
    # Helper function
    def _remove_nestings(self, lst): 
        output = []
        
        def _remove_nestings_recursive(l):
            for i in l: 
                if type(i) == list: 
                    _remove_nestings_recursive(i) 
                else: 
                    output.append(i)
        
        _remove_nestings_recursive(lst)
        
        return output
    
    def _continue_splitting(self, review, list_of_dividers):    
        temp = list_of_dividers.copy()
        l = [review]
        while len(temp) > 0:
            divider = temp.pop(0)
            l_new = []
            for i in l:
                l_new += i.split(divider)
            l = l_new
        return l
    
    
    def join_clause(self, review, list_of_split_clauses, list_of_dividers):
        output = []
        loc_of_split_clauses = []
        for clause in list_of_split_clauses:
            loc_of_split_clauses.append(review.find(clause))
        for divider in list_of_dividers:
            print(divider)
            loc_div = review.find(divider)
            print(loc_div)
            for i in range(len(loc_of_split_clauses)):
                if loc_div > loc_of_split_clauses[i]:
                    print(loc_div,loc_of_split_clauses[i])
    
    
    def join_partitions(self, long_review,entity_with_review):
        loclist = []
        for (_, clause) in entity_with_review:
            loclist.append((long_review.find(clause),long_review.find(clause)+len(clause)))
        starts = {i for (i,j) in loclist}
        ends = {j for (i,j) in loclist}
        starts.add(len(long_review))
        newends = {}
        for i in ends:
            newends[i] = min([x for x in starts if x >= i])
        for i in newends:
            pass
        new_entity_with_review = []
        for i in range(len(loclist)):
            tup = loclist[i]
            entity = entity_with_review[i][0]
            st = tup[0]
            en = newends[tup[1]]
            new_entity_with_review.append((entity,long_review[st:en]))
        return new_entity_with_review
    
    
    def split_long_string(self, review):
        num = len(review)
        split_list = []
        start = 0
        end = 0
        while num != end:
            #if one step away from end of review
            if num - end < 1000:
                end = num
                split_list.append(review[start:end])
            
            #otherwise, find the last full stop
            else:
                end = review[start:(start+1000)].rfind('.') + start
                if end == -1:
                    end = review[start:(start+1000)].rfind(' ') + start #if no '.', space will do
                    if end == -1:
                        end = min(start + 1000,num) + start #if there still isn't, then we simply split
                split_list.append(review[start:end])
                start = end
        return(split_list)
    
    
    def split_very_long_string(self, review):
        num = len(review)
        split_list = []
        start = 0
        end = 0
        while num != end:
            #if one step away from end of review
            if num - end < 1000:
                end = num
                split_list.append(review[start:end])
            
            #otherwise, find the last full stop
            else:
                end = review[start:(start+400)].rfind('.') + start
                if end == -1:
                    end = review[start:(start+400)].rfind(' ') + start #if no '.', space will do
                    if end == -1:
                        end = min(start + 400,num) + start #if there still isn't, then we simply split
                split_list.append(review[start:end])
                start = end
        return(split_list)
    
    
    def split_review_naive(self, review,entities):
        clauses = re.split('[.?!]',review)
        lenlist = [len(x) for x in clauses]
        clauses = [x for _, x in sorted(zip(lenlist,clauses),reverse=False)]
        entity_with_clause = []
        for entity in entities:
            for clause in clauses:
                if entity in clause:
                    entity_with_clause.append((entity,clause))
                    break
        return(self.join_partitions(review,entity_with_clause))
    
    
    def min_tree(self, review, entities, output = 'minimum'):
        
        #review is string, entities is list of strings, parser is parser object
        #possible outputs: no_parse, minimum, partition, all
        
        if output == 'no_parse':
            return(self.split_review_naive(review,entities))
            
        treelist = []
        lenlist = []
        temp = review.split('\n')
        
        if len(review) > 1000:
            split_reviews = self.split_long_string(review)
        else:
            split_reviews = [i for i in temp if len(i) > 1 and len(i) <= 1000 ]
        
        #if output is partition, we need to keep track of the full review
        if output == 'partition':
            full_review = ''
        
        #constituency parsers
        
        for rev in split_reviews:
            if rev and rev.strip():
                u = self.parser.parse(rev) # tree 
    
                if type(u) == str:
                    u = nltk.Tree.fromstring(u)
    
                for s in u.subtrees(): # subtrees 
                    if s.label() == 'S': # if sentence
                        treelist += [s]
                        lenlist += [len(s.leaves())] # how long clause
                            
                if output == 'partition':
                    full_review += ' '.join(u.leaves()) + ' '
    
        treelist = [x for _, x in sorted(zip(lenlist,treelist),reverse=False)] # sort by lenlisit
        clauses = [' '.join(tree.leaves()) for tree in treelist]
        
        #If there is no sentences detected, then the full review is the only clause.
        if not clauses:
            if output == 'partition':
                clauses.append(full_review)
            else:
                clauses.append(review)
        entity_with_clause = []
        
        if output == 'all':
            for entity in entities:
                clauselist = []
                for clause in clauses:
                    if entity in clause:
                        clauselist.append(clause)
                entity_with_clause.append((entity,clauselist))
        
        #TODO: create rules and test them
        elif output == 'minimum':
            for entity in entities:
                for clause in clauses:
                    if entity in clause:
                        entity_with_clause.append((entity,clause))
                        break
                        
        elif output == 'partition':
            #first find minimal clause
            for entity in entities:
                for clause in clauses:
                    if entity in clause:
                        entity_with_clause.append((entity,clause))
                        break
            #get location of minimal clause in review
            
            entity_with_clause = self.join_partitions(full_review,entity_with_clause)
        
        return entity_with_clause
    
    
    def dependency_tree(self, review, entities, output = 'split_min'):
        #possible output = split_min, split_all, tree_min, tree_all -> split only uses sentence splitter, while tree takes into account tree structure
        doc = self.parser(review)
        
        if output == 'split_min' or output == 'split_all' or output == 'split_part':
            clauses = list(doc.sents)
        #length of every clause
        
        lenlist = [len(str(x)) for x in clauses]
            
        #sort
        clauses = [str(x) for _, x in sorted(zip(lenlist,clauses),reverse=False)]
        
        
        entity_with_clause = []
        
        if output == 'split_min':
            for entity in entities:
                for clause in clauses:
                    if entity in clause:
                        entity_with_clause.append((entity,clause))
                        break
                        
        if output == 'split_all':
            for entity in entities:
                clauselist = []
                for clause in clauses:
                    if entity in clause:
                        clauselist.append(clause)
                entity_with_clause.append((entity,clauselist))
        
        if output == 'split_part':
            for entity in entities:
                for clause in clauses:
                    if entity in clause:
                        entity_with_clause.append((entity,clause))
                        break
            #get location of minimal clause in review
            
            entity_with_clause = self.join_partitions(review,entity_with_clause)
                
        
        return(entity_with_clause)


    def vader_sentiment(self, entity_with_clause):
        analyzer = SentimentIntensityAnalyzer()
        entity_with_sentiment = []
        for entity, clause in entity_with_clause:
            sentiment = analyzer.polarity_scores(clause)['compound']
            entity_with_sentiment.append((entity,sentiment))
        return(entity_with_sentiment)   


    def sentiment_analysis(self, entity_with_review, 
                           sentiment_package = 'stanford'):
        #takes in list of tuples
        if sentiment_package == 'stanford':
            return stanford_sentiment(entity_with_review)
        elif sentiment_package == 'vader':
            return self.vader_sentiment(entity_with_review)
        else:
            raise Exception('incorrect sentiment package')


    def sentiment_analysis_indiv(self, clause,sentiment_package = 'stanford'):
        #takes in a single review
        if sentiment_package == 'stanford':
            stanford_sentiment_start()
            result = nlp.annotate(clause,
                       properties={
                           'annotators': 'sentiment',
                           'outputFormat': 'json'
                       })
            return np.dot(result['sentences'][0]['sentimentDistribution'], [-2, -1, 0, 1, 2])
        elif sentiment_package == 'vader':
            analyzer = SentimentIntensityAnalyzer()
            return analyzer.polarity_scores(clause)['compound']
        else:
            raise Exception('incorrect sentiment package')

    
    def rule_1(self, review, entities):
        entity_with_review = self.min_tree(review, entities, output = 'minimum')
        entity_with_sentiment = self.sentiment_analysis(entity_with_review, 
                                                        self.sentiment_package)
        return entity_with_sentiment
    
    
    def rule_2(self, review, entities):
        entity_with_review = self.min_tree(review, entities, output = 'all')
        entity_with_sentiment = []
        sentiment = 0
        for ent, revlist in entity_with_review:
            for clause in revlist:
                sentiment = self.sentiment_analysis_indiv(clause,self.sentiment_package)
                if self.sentiment_package == 'vader' and sentiment != 0:
                    break
                elif self.sentiment_package == 'stanford' and abs(sentiment) > 0.5:
                    break
                    #if sentiment is not neutral, stop. If sentiment is neutral, keep going up tree.                    
            entity_with_sentiment.append((ent,sentiment))
        return entity_with_sentiment
    
    
    def rule_3(self, review, entities):
        entity_with_review = self.min_tree(review, entities, output = 'all')
        
        entity_with_sentiment = []
        for ent, revlist in entity_with_review:
            sentiment_list = []
            for clause in revlist:
                sentiment = self.sentiment_analysis_indiv(clause,self.sentiment_package)
                sentiment_list.append(sentiment)
            if not sentiment_list:
                sentiment_list.append(0)
            entity_with_sentiment.append((ent,np.mean(sentiment_list)))
        
        return entity_with_sentiment
    
    
    def rule_4(self, review, entities):
        entity_with_review = self.min_tree(review, entities, output = 'partition')
        entity_with_sentiment = self.sentiment_analysis(entity_with_review, self.sentiment_package)
        return entity_with_sentiment
    
    
    def rule_5(self, review, entities):
        entity_with_review = self.min_tree(review, entities, output = 'minimum')
        entity_with_review_p = self.min_tree(review, entities, output = 'partition')
        
        entity_with_sentiment = self.sentiment_analysis(entity_with_review, self.sentiment_package)
        for i in range(len(entity_with_sentiment)):
            sent = entity_with_sentiment[i][1]
            if self.sentiment_package == 'vader' and sent != 0:
                entity_with_sentiment[i] = (entity_with_sentiment[i][0],
                                            self.sentiment_analysis_indiv(entity_with_review_p[i][1],
                                                                          self.sentiment_package))
            elif self.sentiment_package == 'stanford' and abs(sent) > 0.5:
                entity_with_sentiment[i] = (entity_with_sentiment[i][0],
                                            self.sentiment_analysis_indiv(entity_with_review_p[i][1],
                                                                          self.sentiment_package))
    
        return entity_with_sentiment
    
    
    def rule_6(self, review, entities):
        entity_with_review = self.min_tree(review, entities, output = 'no_parse')
        return entity_with_review
    
    def rule_7(self, review, entities):
        self.parser = spacy.load("en_core_web_sm")
        entity_with_review = self.dependency_tree(review, entities, output = 'split_min')
        return entity_with_review
    
    def rule_8(self, review, entities):
        self.parser = spacy.load("en_core_web_sm")
        
        entity_with_review = self.dependency_tree(review, entities, output = 'split_all')
        new_entity_with_review = []
        entity_with_sentiment = []
        sentiment = 0
        for ent, revlist in entity_with_review:
            for clause in revlist:
                sentiment = self.sentiment_analysis_indiv(clause,self.sentiment_package)
                if self.sentiment_package == 'vader' and sentiment != 0:
                    new_entity_with_review.append((ent,clause))
                    break
                elif self.sentiment_package == 'stanford' and abs(sentiment) > 0.5:
                    new_entity_with_review.append((ent,clause))
                    break
                    #if sentiment is not neutral, stop. If sentiment is neutral, keep going up tree.                    
            entity_with_sentiment.append((ent,sentiment))
            
        entity_with_review = new_entity_with_review
        return entity_with_review 
        
    def rule_9(self, review, entities):
        self.parser = spacy.load("en_core_web_sm")
        
        entity_with_review = self.dependency_tree(review, entities, output = 'split_min')
        entity_with_review_p = self.dependency_tree(review, entities, output = 'split_part')
        
        entity_with_sentiment = self.sentiment_analysis(entity_with_review, self.sentiment_package)
        for i in range(len(entity_with_sentiment)):
            sent = entity_with_sentiment[i][1]
            if self.sentiment_package == 'vader' and sent != 0:
                entity_with_review[i] = entity_with_review_p[i]
            elif self.sentiment_package == 'stanford' and abs(sent) > 0.5:
                entity_with_review[i] = entity_with_review_p[i]
    
        return entity_with_review
    
    
    
    def kill_host(self):
        if self.sentiment_package == "stanford":
            self.parser.kill_host()
        else:
            print("Stanford server not initialized")
            
            
    def get_entities(self, text):
        """
        Input nlp_model and text, retrieve a list of unique entities from the text.
        """
        doc = self.nlp(text)
        entities = set()
        for ent in doc.ents:
            if ent.label_ == "PRODUCT":
                entities.add(ent.text)
        return list(entities)
    
    
    def _parallelize_default(self, review):
        entities = self.get_entities(review)    
        result = self.rule_2(review, entities)
        return result
    
    
    def parallelize_predict(self):
        entities_with_sentiment = Parallel(n_jobs=self.num_cores)(delayed(self._parallelize_default)(i) for i in self.input_data)
        return entities_with_sentiment
    
    
    def defaultPredict(self):
        entities_with_sentiment = []
        for review in self.input_data:
            entities = self.get_entities(review)    
            result = self.rule_2(review, entities)
            entities_with_sentiment.append(result)
        return entities_with_sentiment
        
    
    def customPredict(self, rule_number=2):
        entities_with_sentiment = []
        
        for review in tqdm(self.input_data):
            entities = self.get_entities(review)
            if rule_number == 1:
                result = self.rule_1(review, entities)
            elif rule_number == 2:
                result = self.rule_2(review, entities)
            elif rule_number == 3:
                result = self.rule_3(review, entities)
            elif rule_number == 4:
                result = self.rule_4(review, entities)
            elif rule_number == 5:
                result = self.rule_5(review, entities)
            elif rule_number == 6:
                result = self.rule_6(review, entities)
            elif rule_number == 7:
                result = self.rule_7(review, entities)
            elif rule_number == 8:
                result = self.rule_8(review, entities)
            elif rule_number == 9:
                result = self.rule_9(review, entities)
            else:
                raise Exception('Rule number invalid, please choose something between 1 and 9')
                
            entities_with_sentiment.append(result)
            
        return entities_with_sentiment
    
    
    def defaultValidate(predicted_scores):
        # TODO: Finish writing validation function
        
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
        corr, pvalue = spearmanr(full_rankings.average_stars, full_rankings.predicted_score)
        return corr




        
            