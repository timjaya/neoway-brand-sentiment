# Neoway Brand Sentiment Software

(software name still in the works...)

This model is an NLP model that detects sentiments in an entity-level to solve brand sentiment detection for Neoway.

## Stakeholders
> Describe the people involved in this project

| Role                 | Responsibility         | Full name                | e-mail       |
| -----                | ----------------       | -----------              | ---------    |
| Project Owner       | Author                 | Felipe Penha            | felipe.penha@alumni.usp.br |
| Collaborator        | Co-author              | Tim Kartawijaya | tak2151@columbia.edu   |
| Collaborator        | Co-author              | Charlene Luo | cl3788@columbia.edu   |
| Collaborator        | Co-author              | Fernando Troeman | ft2515@columbia.edu   |
| Collaborator        | Co-author              | Nico Winata | nw2408@columbia.edu   |
| Collaborator        | Co-author              | Jefferson Zhou | jyz2111@columbia.edu   |


## Usage
> Describe how to reproduce your model

Usage is standardized across models. There are two main things you need to know, the development workflow and the Makefile commands.

Both are made super simple to work with Git and Docker while versioning experiments and workspace.

All you'll need to have setup is Docker and Git, which you probably already have. If you don't, feel free to ask for help.

Makefile commands can be accessed using `make help`.


Make sure that **docker** is installed.

Clone the project from the analytics Models repo.
```
git clone https://github.com/timjaya/neoway-brand-sentiment.git
cd neoway-brand-sentiment
```


## Final Report (to be filled once the project is done)

### Model Frequency

> Describe the interval frequency and estimated total time to run

### Model updating

> Describe how your model may be updated in the future

### Maintenance

> Describe how your model may be maintained in the future

### Minimum viable product

> Describe a minimum configuration that would be able to create a minimum viable product.

### Early adopters

> Describe any potential paying users for this product if it was available today. Also state a point of contact for each of them.

## Documentation

* [project_specification.md](./docs/project_specification.md): gives a data-science oriented description of the project.

* [model_report.md](./docs/model_report.md): describes the modeling performed.


#### Folder structure
>Explain you folder strucure

* [docs](./docs): contains documentation of the project
* [analysis](./analysis/): contains notebooks of data and modeling experimentation.
* [tests](./tests/): contains files used for unit tests.
* [<@model>](./<@model>/): main Python package with source of the model.

## Data Source
Complete Yelp Reviews Dataset - https://www.yelp.com/dataset
