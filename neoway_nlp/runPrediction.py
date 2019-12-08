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


class prediction:
    def __init__(self,
                 input_data,
                 sentiment_package = "vader",
                 parse_package = "benepar"):
        
        # self.spacy_train = spacy_train
        # self.review = spacy_train['text']
        # self.entities = spacy_train['entities']
        
        self.sentiment_package = sentiment_package
        
        if parse_package == 'benepar':
            try:
                self.parser = benepar.Parser("benepar_en2")
            except LookupError:
                benepar.download('benepar_en2')
        elif parse_package == 'stanford':   
            pass
        else:
            raise Exception('incorrect parse package')
        
        model_dir = "D:/git_repos/neoway-brand-sentiment/models"
        self.nlp = spacy.load(model_dir)
        
        self.input_data = input_data
        
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
    
    
    def join_partitions(self, long_review, entity_with_review):
        loclist = []
        for (_, clause) in entity_with_review:
            loclist.append((long_review.find(clause),long_review.find(clause)+len(clause)))
        starts = {i for (i,j) in loclist}
        ends = {j for (i,j) in loclist}
        starts.add(len(long_review))
        newends = {}
        for i in ends:
            newends[i] = min([x for x in starts if x >= i])
        new_entity_with_review = []
        for i in range(len(loclist)):
            tup = loclist[i]
            entity = entity_with_review[i][0]
            st = tup[0]
            en = newends[tup[1]]
            new_entity_with_review.append((entity,long_review[st:en]))
        return new_entity_with_review
    
    
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
        return(join_partitions(review,entity_with_clause))
    
    
    def min_tree(self, output = 'minimum'):
    
        #review is string, entities is list of strings, parser is parser object
        # TODO: How well are each review punctuatd and so forth EDA
        
        if output == 'partition':
            full_review = ''
        
        treelist = []
        lenlist = []
        temp = self.review.split('\n')
        long_reviews = [i.split('.') for i in temp if len(i) > 1000] #TODO: add ! and ? #TODO find better split rule
        short_reviews = [i for i in temp if len(i) > 1 and len(i) <= 1000 ]
        long_reviews = self._remove_nestings(long_reviews)
        split_reviews = long_reviews + short_reviews
        
        for rev in split_reviews:
            u = self.parser.parse(rev) # tree 
            
            if type(u) == str:
                u = nltk.Tree.fromstring(u)
    
            for s in u.subtrees(): # subtrees 
                    if s.label() == 'S': # if sentence
                        treelist += [s]
                        lenlist += [len(s.leaves())] # how long clause
            
            if output == 'partition':
                full_review += ' '.join(u.leaves())

                
        treelist = [x for _, x in sorted(zip(lenlist,treelist),reverse=False)] # sort by lenlisit
        clauses = [' '.join(tree.leaves()) for tree in treelist]
        
        entity_with_clause = []
        
        if output == 'all':
            for entity in self.entities:
                clauselist = []
                for clause in clauses:
                    if entity in clause:
                        clauselist.append(clause)
                entity_with_clause.append((entity,clauselist))
        
        #TODO: create rules and test them
        elif output == 'minimum':
            for entity in self.entities:
                for clause in clauses:
                    if entity in clause:
                        entity_with_clause.append((entity,clause))
                        break
                        
        elif output == 'partition':
            #first find minimal clause
            for entity in self.entities:
                for clause in clauses:
                    if entity in clause:
                        entity_with_clause.append((entity,clause))
                        break        
            entity_with_clause = self.join_partitions(full_review,entity_with_clause)
        return entity_with_clause


    def vader_sentiment(self, entity_with_clause):
        analyzer = SentimentIntensityAnalyzer()
        entity_with_sentiment = []
        for clause in entity_with_clause:
            sentiment = analyzer.polarity_scores(clause)['compound']
            entity_with_sentiment.append(sentiment)
        return(entity_with_sentiment)      


    def vader_sentiment_with_entity(self, entity_with_clause):
        analyzer = SentimentIntensityAnalyzer()
        entity_with_sentiment = []
        for entity, clause in entity_with_clause:
            sentiment = analyzer.polarity_scores(clause)['compound']
            entity_with_sentiment.append((entity,sentiment))
        return(entity_with_sentiment)


    def sentiment_analysis(self, entity_with_review, 
                           sentiment_package = 'stanford',
                           bln_entity = False):
        #takes in list of tuples
        if sentiment_package == 'stanford':
            return self.parser.stanford_sentiment(entity_with_review)
        elif sentiment_package == 'vader':
            if bln_entity:
                return self.vader_sentiment_with_entity(entity_with_review)
            else:
                return self.vader_sentiment(entity_with_review)
        else:
            raise Exception('incorrect sentiment package')
    
    def rule_1(self):
        entity_with_review = self.min_tree(output = 'minimum')
        entity_with_sentiment = self.sentiment_analysis(entity_with_review, self.sentiment_package, 
                                                        bln_entity = True)
        return entity_with_sentiment
    
    
    def rule_2(self):
        entity_with_review = self.min_tree(output = 'all')
        
        entity_with_sentiment = []
        for ent, revlist in entity_with_review:
            for clause in revlist:
                sentiment = self.sentiment_analysis(clause,self.sentiment_package)
                if sentiment != 0:
                    entity_with_sentiment.append((ent,np.mean(sentiment)))
                    break
                    #if sentiment is not neutral, stop. If sentiment is neutral, keep going up tree.
        return entity_with_sentiment
    
    
    def rule_3(self):
        entity_with_review = self.min_tree(output = 'all')
        entity_with_sentiment = []
        
        for ent, revlist in entity_with_review:
            sentiment_list = []
            for clause in revlist:
                sentiment = self.sentiment_analysis(clause,self.sentiment_package)
                sentiment_list.append(sentiment)
            if len(sentiment_list) > 0:
                entity_with_sentiment.append((ent,np.mean(sentiment_list[0])))
        return entity_with_sentiment
    
    
    def rule_4(self):
        entity_with_review = self.min_tree(output = 'partition')
        entity_with_sentiment = self.sentiment_analysis(entity_with_review, self.sentiment_package,
                                                        bln_entity = True)
        return entity_with_sentiment
    
    def kill_host(self):
        if sentiment_package == "stanford":
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
    
    
    def defaultPredict(self):
        parser = benepar.Parser("benepar_en2")
        entities_with_count = defaultdict(int) 
        review_entities = [] # extract entities for each review
        for review in tqdm(self.input_data):
            entities = self.get_entities(review)
    
            # add this review as a count to an entity
            for ent in entities:
                entities_with_count[ent.lower()] += 1
                
            review_entities.append(entities)
            
        # subset entities that have enough reviews
        entities_with_enough_reviews = []
        
        # for now, the threshold is 20
        threshold = 20
        for key, value in entities_with_count.items():
            if value >= threshold:
                entities_with_enough_reviews.append(key)
    
        # PREDICTION RANKING CALCULATION

        # Filter entities attached to each review to be 
        # from the one with enough review
        entity_filter = set(entities_with_enough_reviews)
        
        filtered_entities = []
        
        for entities in tqdm(review_entities):
            filtered = []
            for ent in entities:
                ent = ent.lower()
                if ent in entity_filter:
                    filtered.append(ent)
            filtered_entities.append(filtered)
            
        
        predicted_scores = defaultdict(list)

        for i, review in enumerate(tqdm(review_entities)):
            entities = filtered_entities[i]
            
            scores = self.vader_sentiment(entities)
            
            # save results 
            for entity, score in scores:
                predicted_scores[entity] += [score]
        
        # create rankings from scores
        predicted_rankings = defaultdict(list)
        for entity, scores in predicted_scores.items():
            predicted_rankings['entity'] += [entity]
            predicted_rankings['predicted_score'] += [np.mean(scores)]
        
        return predicted_rankings


# DO NOT RUN
"""
if __name__ == "__main__":
    input_data = ["The fire roasted salsa was EASILY the best salsa I have ever had, too!"]
    test_predict = prediction(input_data)
    test_predict.defaultPredict()
"""
            
            
            