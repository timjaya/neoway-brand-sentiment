from __future__ import unicode_literals, print_function
import ast 
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import time
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def create_train_data(df):
  train_data = []
  newnlp = spacy.load("en_core_web_sm")

  for i in range(len(df)):
    doc = newnlp(df['text'].iloc[i])
    entity_list = df['entities_clean'].iloc[i]
    for ent in doc.ents:
      entity_list.append((ent.start_char, ent.end_char, ent.label_))
    entity_dict = {"entities": entity_list}
    train_data.append((df['text'].iloc[i], entity_dict))
  return train_data

def create_test_data(df):
  test_data = []
  newnlp = spacy.load("en_core_web_sm")

  for i in range(len(df)):
    doc = newnlp(df['text'].iloc[i])
    entity_list = df['entities_clean'].iloc[i]
    for ent in doc.ents:
      entity_list.append((ent.start_char, ent.end_char, ent.label_))
    entity_dict = {"entities": entity_list}
    test_data.append((df['text'].iloc[i], entity_dict))
  return test_data


# new entity label
def train(train_data, test_data, LABEL, model='en_core_web_sm', new_model_name="product", output_dir='./ermodel', n_iter=1):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(LABEL)  # add new entity label to entity recognizer
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        # batch the examples using spaCy's minibatch
        start = time.time()
        for itn in range(n_iter):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)            
            #print("Training Recall:",nlp.evaluate(random.sample(TRAIN_DATA,200)).ents_r)
            #print("Test Recall:",nlp.evaluate(TEST_DATA).ents_p) #COMMENT: isn't this precision?
            #COMMENT: so test data here is evaluating test_data which has the format 
            # of e.g. ("Uber blew through $1 million a week", {"entities": [(0, 4, "ORG")]}) right
            #print("Training Losses", losses)
        end = time.time()
    print("Total training time:",end-start)

    # test the trained model (small sample test)
    for i in range(10):
      test_text = test_data[i][0]
      doc = nlp(test_text)
      print("Entities in '%s'" % test_text)
      for ent in doc.ents:
          print(ent.label_, ent.text)

    # TODO: Abstract to another function
    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # COMMENT: Abstract to another function 
        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)
    return nlp


def run_training(file_name = "../data/spacy_train_clean_10k.csv", 
                 output_dir = './ermodel'):
    df = pd.read_csv(file_name)
    df['entities_clean']=[ast.literal_eval(i) for i in df['entities']]
    #train_df, test_df = train_test_split(df, test_size = .2)
    all_train, therest = train_test_split(df, train_size=1000)
    train_df, test_df = train_test_split(all_train, test_size=.2)
    
    # new entity label
    LABEL = "PRODUCT"
    
    TRAIN_DATA = create_train_data(train_df)
    TEST_DATA = create_test_data(test_df)

    
    model = train(TRAIN_DATA, TEST_DATA, LABEL=LABEL, output_dir=output_dir)


