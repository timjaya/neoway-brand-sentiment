# Model report - [`neoway_nlp`]
This report should contain all relevant information regarding your model. Someone reading this document should be able to easily understand and reproducible your findings.

## Checklist
Mark which tasks have been performed

- [ ] **Summary**: you have included a description, usage, output,  accuracy and metadata of your model.
- [ ] **Pre-processing**: you have applied pre-processing to your data and this function is reproducible to new datasets.
- [ ] **Feature selection**: you have performed feature selection while modeling.
- [ ] **Modeling dataset creation**: you have well-defined and reproducible code to generate a modeling dataset that reproduces the behavior of the target dataset. This pipeline is also applicable to generate the deploy dataset.
- [ ] **Model selection**: you have chosen a suitable model according to the project specification.
- [ ] **Model validation**: you have validated your model according to the project specification.
- [ ] **Model optimization**: you have defined functions to optimize hyper-parameters and they are reproducible.
- [ ] **Peer-review**: your code and results have been verified by your colleagues and pre-approved by them.
- [ ] **Acceptance**: this model report has been accepted by the Data Science Manager. State name and date.

## Summary

The model takes in a Yelp review, detects entities mentioned in the review, and performs sentiment analysis towards each of these entities.

It completes the task above in three steps:
1. Entity Recognition (Spacy)
2. Parsing to determine relevant context per entity (Benepar, Spacy, Stanford NLP)
3. Sentiment Analysis, taking in relevant context from step 2 (VADER, Stanford NLP)

### Usage ________TIM__________
> Provide a reproducible pipeline to run your model

1. clone the repository
2. download test dataset from cloud storage (ex.: AWS S3)
3. run the following code on command line

```python
pip install -e .
python -m neoway_nlp on_my_data.csv save_to_my_file.csv
```

### Output

The model inputs a review (string). The output can be either the sentiment towards individual entities entities or the relevant portions of the review that discusses the entity in question. This output is in the form of a list of tuples - the first element in the tuple is the entity and the second is either the sentiment/context.
```python
input = 'I loved the mac and cheese in the restaurant, but I hated that they only offered cheap beer such as Miller Lite'
output_example1 = [('mac and cheese', 0.5994), ('beer', -0.6369)]
output_example2 = [('mac and cheese', 'I loved the mac and cheese in the restaurant'), ('beer', 'I hated that they only offered cheap beer such as Miller Lite')]
```

#### Metadata ________TIM__________
> Your model's metada should be provided in a machine-readable
> format (e.g. a json file) and include the following items:

* a brief description: this model predicts the type of a restaurant
  cuisine
* a measure of accuracy applicable for use of the model in other
  setups (if applicable): standard deviation, accuracy, error matrix.
* model version
* author
* date created
* link to training data

Make sure that the final consumer of your model can make use of your metadata.

#### Coverage ________TIM/FERNANDO__________

> Describe the population covered by your model in comparison to the
> target population.

## Pre-processing ________FERNANDO__________

(Talk about filtering out Reviews, etc)

We don't require a preprocessing of the reviews themselves. Our model is able to take in the original Yelp review and perform targeted sentiment analysis in a robust manner.

## Modeling

A trained `Spacy` model is used to perform entity recognition. We use a list of food & beverage nouns from WordNet as our training data. Once trained, the `Spacy` model is able to generalize to new entities not in the training dataset.

We have several choices for parsing. We compared several parsing methods, namely `Benepar`'s constituency parser
and `Spacy`'s dependency parser. In our test cases, constituency parsers perform better than dependency parsers.

We also have several choices for Sentiment Analyzers. We tried both `VADER` and `Stanford NLP`. `VADER` performed best due to its sensible labelling of neutral sentences, and works especially well with Rule 2, as described below.

### Performance Metrics ________CHARLENE__________

#Entity Recognition

| metric    | `''`   | `'pizza'`   | `'mexicano'`   | `'bar'`   | `'churrascaria'` |
| --------- | ------ | ----------- | -------------- | --------- | ---------------- |
| precision | .98    | .8          | .9             | .95       | .99              |
| recall    | .98    | .8          | .9             | .95       | .99              |

We can evaluate the performance of Sentiment Analysis models, Parsing Rules, etc using the validation method, described in our report. ______TIM_____


#Comparison of Sentiment Analysis Models

| Sentiment Analysis | `Rank correlation score`   | 
| ------------------ | -------------------------- | 
| VADER              | .24                        |
| Stanford NLP       | -.17                       |

#Comparison of Parsing Rules ______TIM_____

Detailed description of each rule can be found in the report. The best performing rule is Rule 2, which involves traversing up a constituency tree until we reach a context with non-neutral sentiment.

| Sentiment Analysis | `Rank correlation score`   | 
| ------------------ | -------------------------- | 
| Rule 1             | .25                        |
| Rule 2             | .58                        |
| Rule 3             | .51                        |
| Rule 4             | .40                        |
| Rule 5             | .23                        |
| Rule 6             | .24                        |


### Model selection & validation

We selected our models using a validation technique outlined below, taking advantage of the fact that sentiment of reviews are quasi-labelled by the star ratings. Please see our report for a justification of our validation technique.

1. A subset of restaurants with sufficient reviews is filtered.(Contains > 10 entities each with > 30 reviews)
2. For each of the entities above, calculate the average star rating. Rank the sentiment of each entity using this average score. This will be the 'true' ranking.
3. High vs low variance ___ TIM_____
4. Calculate the average sentiment for each entity using our model (with a particular Sentiment Analyzer, a particular parsing rule). Rank the sentiment of each entity using this average score. This will be our model-generated ranking.
5. Calculate the rank correlation of the rankings in step 2 and 4. Repeat using a different model in step 4 to create comparisons between models.

Note that we are not using the Yelp dataset to train our models, only to validate. Hence, no train-test splits are needed and we use the entire dataset for validation.

## Additional resources ___TIM____
> Provide links to additional documentation and presentations regarding your model.
