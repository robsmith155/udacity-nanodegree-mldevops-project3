# Model Card
Random Forest classifier trained on the [US census dataset](https://archive.ics.uci.edu/ml/datasets/census+income). The goal is to predict whether a person earns more than $50k based on certain census information (such as age, gender, education and race).

## Model Details
Random Forest classifier from the [Imbalanced-learn](https://imbalanced-learn.org/stable/) package. This is based on the Scikit-learn API but is designed for imbalaned datasets as found here. The best model hyperparameters were found using a random search of the hyperparameter space. For now this was run in a [Jupyter Notebook](https://github.com/robsmith155/udacity-nanodegree-mldevops-project3/blob/main/notebooks/random-forest_hyparam-search.ipynb) found under the notebooks folder in the project repo. The model was trained by Rob Smith.

The best hyperparameters used were:
- num_estimators: 140
- min_samples_split: 25
- min_samples_leaf: 1
- max_features: sqrt
- max_depth: 42
- bootstrap: False

## Intended Use
The model is designed to be use dto predict whether an individual earns more or less than $50k per year based on information such as age, gender, education and race.

## Training Data
### Raw data
The raw data for the project can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income). The dataset contains the following features:

- age: continuous.
- workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- fnlwgt: continuous.
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- education-num: continuous.
- marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- sex: Female, Male.
- capital-gain: continuous.
- capital-loss: continuous.
- hours-per-week: continuous.
- native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

### Data cleaning and processing
First the data was cleaned to remove rows with missing data and output as `census_cleaned.csv`. 

Next the data processed ready for training. This included one-hot encoding of the categorical features and binarizing of the target label.

Data was split randomly into training and test datasets, with 20% of the data reserved for blind testing.

## Evaluation Data
For the hyperparameter search, stratified K-Fold cross-validation was used on the training dataset. The full dataset was used to train the final model once the best hyperparameters had been determined.

Twenty percent of the data was held back fro blind testing.

## Metrics
The `precision`, `recall`, and `F1` metrics were used to evaluate the model.

### Training data scores (all)
- precision: 0.626
- recall: 0.904
- F1: 0.740

### Test data scores (all)
- precision: 0.605
- recall: 0.862
- F1: 0.711

### Data slicing
Metrics were also run for all unique categories in the categorical features. These results can be found in the [outputs](https://github.com/robsmith155/udacity-nanodegree-mldevops-project3/tree/main/outputs) file in the repo. An example of the results for the race category are shown below.

#### F1 scores for race categories
- White: 0.713
- Asian-Pac-Islander: 0.680
- Black: 0.699
- Other: 0.5
- Amer-Indian-Eskimo: 0.545

## Ethical Considerations
Care should be taken when using the model since it has not been fully assessed for bias. The data slicing metrics that were output can give us some guidance. For instance, in the results shown above, the model does significantly worse for people that fall under the `other` and `Amer-Indian-Eskimo` categories compared to the other three. The model should be carefully analysed for the impact of things such as age, race and gender on the results.

## Caveats and Recommendations
The Random Forest model could likely be improved by analysing the feature importance and removing features that don't contribute to the results. The dependency between the input features could also be investigated and redundant features removed.

Here I only attempted to use the Random Forest model, but this may not be the best option. Additional models should be investigated, such as AdaBoost, XGBoost and neural networks.

The model has not been assessed for bias, so should not be put into production until this has been conducted.
