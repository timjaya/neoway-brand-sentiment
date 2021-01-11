# Neoway: Brand Sentiment Analyzer

This project aims to create a model that is able to do two tasks simultaneously: detect entities from Yelp reviews and assign sentiment scores towards those entities (e.g. food brands). We evaluate various open-source tools: VADER, Stanford NLP, and Benepar. Results can be found on the notebooks. 

## Stakeholders

| Role                 | Responsibility         | Full name                | e-mail       |
| -----                | ----------------       | -----------              | ---------    |
| Project Owner       | Stakeholder                 | Felipe Penha            | felipe.penha@alumni.usp.br |
| Collaborator        | Co-author / Project Lead              | Tim Kartawijaya | tak2151@columbia.edu   |
| Collaborator        | Co-author              | Charlene Luo | cl3788@columbia.edu   |
| Collaborator        | Co-author              | Fernando Troeman | ft2515@columbia.edu   |
| Collaborator        | Co-author              | Nico Winata | nw2408@columbia.edu   |
| Collaborator        | Co-author              | Jefferson Zhou | jyz2111@columbia.edu   |


## Usage

To reproduce results in the paper:
* For final validation results (spearman correlation ranking results) run [sentiment_and_parsing_rules_end_to_end_validation](./analysis/final_validation/sentiment_and_parsing_rules_end_to_end_validation.ipynb) end to end. The dataset used in the notebook (restaurant_reviews_1900k.json) is restricted due to Yelp policy, so please contact Tim Kartawijaya. 

* For qualitative results, run [qualitative_testings_VADER_Stanford_NLP_Benepar](./analysis/final_validation/qualitative_testings_VADER_Stanford_NLP_Benepar.ipynb).

To use the package for your own dataset / brand list, follow the steps done in [usage_example](./usage_example.ipynb). Documentation on how neoway_nlp works can be found in the [main](./neoway_nlp/main.py) file. (Further documentation needed here for better access). Data used in run() (restaurant_reviews_10k.csv and brand_list.csv) is restricted due to Yelp policy, so please contact Tim Kartawijaya.

#### Folder structure

* [docs](./docs): contains documentation of the project (NOT COMPLETED).
* [analysis](./analysis/): contains notebooks for modeling experimentation.
    * [final_validation](./analysis/): contains notebooks that produce the final qualitative/quantitative results.
    * [end_to_end_rules](./analysis/): contains notebooks that test the different parsing rules we developed.
    * [entity_recognition](./analysis/): contains notebooks that produce the Spacy ER model.
    * [preprocess](./analysis/): contains code to preprocess data from the Raw Yelp Reviews Dataset to digestable data.
* [tests](./tests/): contains files used for unit tests. (NOT COMPLETED).
* [neoway_nlp](./neoway_nlp/): main Python package with source of the model.

## Data Source
Complete Yelp Reviews Dataset - https://www.yelp.com/dataset
