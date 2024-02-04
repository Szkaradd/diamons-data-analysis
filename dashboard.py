import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

clean_data = pd.read_csv("data/clean_data.csv")
clean_data_no_missing_values = pd.read_csv("data/clean_data_no_missing_values.csv")
continuous_vars = ['carat', 'length', 'width', 'height', 'depth', 'table', 'price']
categorical_vars = ['cut', 'color', 'clarity']

def plot_histplot(data, col):
    plt.figure(figsize=(10, 4))
    sns.histplot(data [col], bins=20)
    plt.title(f'Distribution of {col}')
    st.pyplot(plt)

def plot_piechart(data, col):
    plt.figure(figsize=(10, 4))
    data[col].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title(f'Pie chart of {col}')
    plt.axis('equal')
    st.pyplot(plt)

def regression_plot(y, y_pred):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(y, y_pred, color='blue')
    ax.set_xlabel('Real prices')
    ax.set_ylabel('Expected prices')
    ax.set_title('Real vs Expected diamond prices')
    plt.plot([min(y), max(y)], [min(y_pred), max(y_pred)], color='red')
    st.pyplot(fig)
    
def plot_scatterplot(data, col):
    plt.figure(figsize=(10, 4))
    sns.scatterplot(x=col, y='price', data=data)
    plt.title(f'Scatterplot of price and {col}')
    st.pyplot(plt)

# Page Title
st.title("Diamond data analysis")

# Author:
st.header("Author: Mikołaj Szkaradek")


st.header("Data")

# Show data
st.write(clean_data)
clean_data = clean_data[(clean_data['price'] > clean_data['price'].quantile(0.05)) & (clean_data['price'] < clean_data['price'].quantile(0.95))]

# Show data with missing values replaced by the mean value of the column
st.write("Data with missing values replaced by the mean value of the column")
st.write(clean_data_no_missing_values)

st.header("Diamonds by category")
category = st.selectbox("Select a category", categorical_vars)

plot_piechart(clean_data, category)

choice = st.selectbox(f'Select a {category}', clean_data[category].unique())

st.write(clean_data[clean_data[category] == choice])

# Continuous variables distribution
st.header("Continuous variables distribution")

col = st.selectbox("Select a variable", continuous_vars)
plot_histplot(clean_data, col)

# Price dependence on continuous variables
st.header("Price dependence on continuous variables")
col = st.selectbox("Select a continuous variable", continuous_vars[:-1])

plot_scatterplot(clean_data, col)

# Model

X = pd.read_csv("data/model_train_data.csv")
y = pd.read_csv("data/model_expected_data.csv")
y = y['price']

model = joblib.load("model.joblib")

# Plot linear regression of encoded data

st.header("Linear regression of encoded data")

st.write("The model is working on data with missing values replaced by the mean value of the column")
st.write("The model is working on encoded data with categorical variables replaced by numerical values")

y_pred = model.predict(X)

show_score = st.checkbox("Show model score")
if show_score:
    st.write("Model score: ", model.score(X, y))

show_plot = st.checkbox("Show regression plot")
if show_plot:
    regression_plot(y, y_pred)