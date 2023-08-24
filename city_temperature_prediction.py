import pandas

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    df.drop(["Day"], axis=1)
    df = pd.get_dummies(df, prefix="City_", columns=["City"])
    df["DayOfYear"] = df["Date"].dt.dayofyear #pandas.to_datetime(df["Date"])
    df = df[df["Temp"] > 0]
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    df_isr = df[df["Country"] == "Israel"]
    fig2_1 = px.scatter(df_isr, x="DayOfYear", y="Temp", color="Year",
                        color_discrete_sequence=px.colors.qualitative.Set1)
    fig2_1.write_image("daily_temperatures_israel.png")

    monthly_temps = df_isr.groupby("Month")["Temp"].agg("std").reset_index()

    fig2_2 = px.bar(monthly_temps, x="Month", y="Temp", title="Variation of Daily Temperatures by Month")
    fig2_2.update_layout(xaxis_title="Month", yaxis_title="Standard Deviation of Daily Temperatures In Israel")
    fig2_2.write_image("daily_temperatures_israel_by_month.png")

    # Question 3 - Exploring differences between countries
    grouped_data = df.groupby(["Country", "Month"]).agg({"Temp": ["mean", "std"]})
    grouped_data.columns = ["_".join(col) for col in grouped_data.columns]
    grouped_data = grouped_data.reset_index()

    fig = px.line(grouped_data, x="Month", y="Temp_mean", color="Country", error_y="Temp_std",
                  title="Average Monthly Temperature by Country")
    fig.update_layout(xaxis_title="Month", yaxis_title="Temperature")
    fig.write_image("average_monthly_temperature_by_country.png")

    # Question 4 - Fitting model for different values of `k`
    df_isr_training_x, df_isr_training_y, df_isr_test_x, \
            df_isr_test_y = split_train_test(df_isr["DayOfYear"], df_isr["Temp"])

    ks = list(range(1, 11))
    losses = []
    for k in ks:
        poly_fit = PolynomialFitting(k)
        poly_fit.fit(df_isr_training_x.to_numpy(), df_isr_training_y.to_numpy())
        loss = round(poly_fit.loss(df_isr_test_x.to_numpy(), df_isr_test_y.to_numpy()), 2)
        losses += [loss]

    print(losses)
    fig = px.bar(x=ks, y=losses, title="Test errors for ks", text=losses)
    fig.update_layout(xaxis_title="ks", yaxis_title="losses")
    fig.write_image("test_errors_for_ks.png")

    # Question 5 - Evaluating fitted model on different countries
    chosen_k = 5
    poly_fit = PolynomialFitting(chosen_k)
    poly_fit.fit(df_isr["DayOfYear"].to_numpy(), df_isr["Temp"].to_numpy())

    df_the_netherlands = df[df["Country"] == "The Netherlands"]
    loss_netherlands = round(poly_fit.loss(df_the_netherlands["DayOfYear"], df_the_netherlands["Temp"]), 2)

    df_jordan = df[df["Country"] == "Jordan"]
    loss_jordan = round(poly_fit.loss(df_jordan["DayOfYear"], df_jordan["Temp"]), 2)

    df_south_africa = df[df["Country"] == "South Africa"]
    loss_south_africa = round(poly_fit.loss(df_south_africa["DayOfYear"], df_south_africa["Temp"]), 2)

    fig = px.bar([{"country": "The Netherlands", "loss": loss_netherlands},
                  {"country": "Jordan", "loss": loss_jordan},
                  {"country": "South Africa", "loss": loss_south_africa}],
                 x="country", y="loss", text="loss",
                 title="Loss over other countries")

    fig.update_layout(xaxis_title="Country", yaxis_title="Loss")
    fig.write_image("test_errors_for_countries.png")