# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This XGBoost model is trained on 100+ features and uses >30,000 records. 

## Intended Use
This model should be used as an evaluation for current status of the salary output, but not to justify the gaps/discrepancies in the existing salary across population.

## Training Data
Dataset is split via 20% for test dataset so 80% of the data is used for training.

## Evaluation Data
Evaluation was done for precision, recall, and fbeta for the whole dataset as well as slices (see "Metrics" for details),

## Metrics
_Please include the metrics used and your model's performance on those metrics._
Under data, we calculated the slice based model performance in the file named "slice_based_output_metrics.csv." Before using this model, it is essential to evaluate how the model compares across these categorical groups that the data was sliced on:

"workclass",
"education",
"marital-status",
"occupation",
"relationship",
"race",
"sex",
"native-country"

## Ethical Considerations
This dataset contains potential fields that can negatively influence hiring, should this be in production at a real company. As a result, it is recommended to serve as a reflective, more so than predictive.

## Caveats and Recommendations
Please see Ethical Consideration and Intended Use.