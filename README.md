# Data Analytics Challenge - AG Team

### University of Dayton Data Analytics Club 2026

This repository contains our team's approach to the broad statement presented to us: propose a research question and find a creative analysis of bank data from 2000-2026 that develops your skills in the fields of statistics and computation.

We sought to answer the following question *Can quarterly financial ratios predict failures within four quarters, and do local economic conditions add predictive power beyond the bank’s own financials?* The answer we discovered was that Machine Learning models are extremely adept at recognizing banking trends and making predictions based on only a couple years of training data, as well as a clear class distinction.

## Data

[The data we used is linked here](data/)

We used a downsampled dataset taken from the original pool given as part of this project to balance a much larger class of succeeding banks, compared to a very select set of failing banks.

Initial graphs were created to help visualize which features would be significant for our ML models to evaluate when predicting bank failures. These are provided in `graphs/`. The following [graph](graphs/04_densities_top_features.png) is particularly beneficial in noting the most important feature from the original data pool.

## Models

[The models we used are linked here](models/)

We first used a Naive Bayes model, before working with an XGBoost model. This decision was made to test if there would be a significant improvement in prediction precision recall with a more advanced model.

**Note all results are stored within this repository, there is no need to run locally to see the outcome of each model*

## Results

[The results from each model are linked here](results/)

The results show no direct need for a more advanced model, as a well formulated dataset is capable of resulting similar predictions. However, the XGBoost model does improve upon the base Naive Bayes model, as seen [here](results/model_comparison_table.png)

**Read through our official report for a more in-depth analysis of our results, description of each model, and explanation of the significant variables used in prediction**

[Read the report](docs/Report.pdf)