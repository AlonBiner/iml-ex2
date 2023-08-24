from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    train_x = X.dropna().drop_duplicates()

    train_x = train_x.drop(["id", "date", "lat", "long", "sqft_living15", "sqft_lot15"], axis=1)

    for feature in ["bedrooms", "bathrooms"]:
        train_x = train_x[train_x[feature] >= 1]

    for feature in ["floors", "waterfront", "view", "condition", "sqft_above", "sqft_basement",
                    "yr_built", "yr_renovated"]:
        train_x = train_x[train_x[feature] >= 0]
    train_x["zipcode"] = train_x["zipcode"].astype(int)
    train_x = train_x[train_x["grade"].isin(range(1, 14))]
    train_x = pd.get_dummies(train_x, prefix="zipcode_", columns=["zipcode"])


    if y is None:
        return train_x, train_x["Price"]
    else:
        y = y.loc[train_x.index].dropna()
        train_x = train_x.loc[y.index]
    return train_x, y

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    X = X.loc[:, ~(X.columns.str.contains('^zipcode_', case=False))]

    for feature_name in X.columns:
        feature_values = X[feature_name]

        pearson_correlation = np.cov(feature_values, y)[0, 1] / (np.std(feature_values) * np.std(y))
        fig = px.scatter(pd.DataFrame({'x': feature_values, 'y': y}), x="x", y="y", trendline="ols",
                   color_discrete_sequence=["black"],
                   title=f"Pearson Correlation between {feature_name} and response."
                         f"value={pearson_correlation}",
                   labels={"x": f"{feature_name} Values", "y": "Response Values"},
                    width=900)
        fig.write_image(output_path + "/pearson_correlation_for_" + feature_name + ".png")

if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets

    prices = df["price"]
    rest_of_data = df.loc[:, df.columns != "price"]
    train_x, train_y, test_x, test_y = split_train_test(rest_of_data, prices)

    # Question 2 - Preprocessing of housing prices dataset
    preprocessed_x, preprocessed_y = preprocess_data(train_x, train_y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(preprocessed_x, preprocessed_y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    losses = []
    percentages = list(range(10, 101))
    test_x = test_x.drop(["id", "date", "lat", "long", "sqft_living15", "sqft_lot15"], axis=1).dropna()
    test_y = test_y.loc[test_x.index]
    test_x["zipcode"] = test_x["zipcode"].astype(int)
    test_x = pd.get_dummies(test_x, prefix="zipcode_", columns=["zipcode"])
    # test_x = test_x.drop(["zipcode__0"], axis=1)

    for p in percentages:
        loss_per_feature = []
        for i in range(10):
             train_x_partial = preprocessed_x.sample(frac=(float(p) / 100.0))
             train_y_partial = preprocessed_y.loc[train_x_partial.index]
             linear_regression = LinearRegression()
             linear_regression.fit(train_x_partial.to_numpy(), train_y_partial.to_numpy())
             loss = linear_regression.loss(test_x.to_numpy(), test_y.to_numpy())
             loss_per_feature += [loss]
        losses += [loss_per_feature]

    mean_losses = np.array(losses).mean(axis=1)
    std_losses = np.array(losses).std(axis=1)
    conf_int_top, conf_int_bottom = mean_losses + 2 * std_losses, mean_losses - 2 * std_losses

    fig = go.Figure([go.Scatter(x=percentages, y=mean_losses, mode='lines', line=dict(color='black')),
               go.Scatter(x=percentages, y=conf_int_top, mode='lines', line=dict(color='blue')),
               go.Scatter(x=percentages, y=conf_int_bottom, mode='lines', line=dict(color='blue'))],
              layout=go.Layout(title='Mean loss as a function of %p',
                               xaxis=dict(title='p%'),
                               yaxis=dict(title='Mean Loss'),
                               height=300,
                               width=400,
                               showlegend=False))

    fig.write_image("mean_loss_as_function_of_p.png")