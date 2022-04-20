from sklearn.metrics import fbeta_score, precision_score, recall_score
from xgboost import XGBClassifier
import numpy as np

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = XGBClassifier()
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : XGB model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def slice_output_generator(slice_columns,df,y_test,y_pred):
    """
    Calculates slice-based output calculations

    Inputs
    ------
    slice_columns: columns to slice metrics on
    df: dataframe to refer for slicing
    y_test: list of actual y values related to df
    y_pred: list of predicted y values related to df

    Returns
    ------
    slice_results : list
        Slice containing column name_category value and fbeta, precision, recall
    """

    # Model evaluation on the test dataset
    slice_results = [('columnName_sliceName'), ("precisionVal, recallVal, fbetaVal")]

    slice_results.append( (("Total_NoSlice"), compute_model_metrics(y_test, y_pred) ))

    # define categories
    for col in slice_columns:
        col_categ = np.unique(df[col].values)
        for categ in col_categ:
            categ_ind = df[df[col] == categ].index
            y_test_sub = y_test[categ_ind]
            y_pred_sub = y_pred[categ_ind]
            slice_results.append( ((col + "_" + categ), compute_model_metrics(y_test_sub, y_pred_sub)) )

    return slice_results
