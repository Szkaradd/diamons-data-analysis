import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import datetime

st.title("Diamond data analysis")

df = pd.read_csv("clean_data.csv")

st.write(df)

st.header("Diamonds by cut")

cut = st.selectbox("Select a cut", df["cut"].unique())

st.write(df[df["cut"] == cut])

occupation = st.selectbox("Your Occupation", ("Programmer", "Data Analyst", "Teacher"))
st.write("You selected this occupation: ", occupation)

data = st.file_uploader("Upload your dataset", type=['csv'])
if data is not None:
    df = pd.read_csv(data)
    st.dataframe(df.head(10))