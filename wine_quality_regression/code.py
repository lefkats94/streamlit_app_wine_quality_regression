import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
np.set_printoptions(suppress=True)

data = pd.read_csv(os.getcwd() + "/winequality.csv",sep=";")

st.sidebar.header("Options")
options = ["Home", "Data Exploration", "Preprocessing", "Prediction"]
choice = st.sidebar.selectbox("",options)

def main():
    if choice == "Home":
        home()

    if choice == "Data Exploration":
        cho(data)

    if choice == "Preprocessing":
        pre(data)
    
    if choice == "Prediction":
        predict(data)
    return None

def home():
    st.markdown("<h1 style='text-align: center; color: black;'>White wine quality</h1>", unsafe_allow_html=True)
    img = Image.open("image.png")
    st.image(img, width=750)
    return None

def cho(data):
    st.markdown("The Dataset:")
    st.dataframe(data)
    st.latex("-------------------------------------")
    st.markdown("The distribution of quality:")
    df = data.quality.value_counts()
    df = pd.DataFrame(df)
    df.reset_index(inplace=True)
    fig, ax = plt.subplots()
    plt.bar(df["index"], df.quality)
    st.pyplot(fig)
    st.latex("-------------------------------------")
    st.markdown("correlation within columns:")
    cor_matrix = data.corr().style.background_gradient(cmap='coolwarm')
    st.dataframe(cor_matrix)
    return None

def pre(data):
    features = pd.DataFrame(data.drop("quality",axis=1))
    target = pd.DataFrame(data.quality)
    st.markdown("Features:")
    st.dataframe(features)
    st.markdown("Target:")
    st.dataframe(target)

    st.latex("-------------------------------------")

    st.markdown("The features normilized:")
    features = Normalizer().fit_transform(features)
    features = pd.DataFrame(features)
    st.dataframe(features)

    st.latex("-------------------------------------")

    model = PCA()
    model.fit(features)
    features = model.transform(features)
    values = model.explained_variance_ratio_
    rounded_values = [round(num, 2) for num in values]
    st.markdown("The first three components are selected as final features and they contain around 99% of variance:")
    components = [i + 1 for i in range(len(values))]
    fig, ax = plt.subplots()
    ax.plot(components, rounded_values)
    ax.scatter(components, rounded_values)
    for i, txt in enumerate(rounded_values):
        ax.annotate( txt, ( components[i] + .01, rounded_values[i] + .01 ) )
    plt.xticks(components)
    st.pyplot(fig)

    st.latex("-------------------------------------")

    model = PCA(n_components=3)
    model.fit(features)
    features = model.transform(features)
    st.markdown("The final features:")
    st.dataframe(features)
    return features, target

def predict(data):

    features = pd.DataFrame(data.drop("quality",axis=1))
    target = pd.DataFrame(data.quality)
    features = Normalizer().fit_transform(features)
    features = pd.DataFrame(features)
    model = PCA(n_components=3)
    model.fit(features)
    features = model.transform(features)
    feat_train, feat_test, tar_train, tar_test = train_test_split(features, target, test_size=0.2)

    fixed_acidity = st.slider("fixed acidity", min_value = 0.0 , max_value = 20.0, step = 0.01)
    volatile_acidity = st.slider("volatile acidity", min_value = 0.0 , max_value = 2.0, step = 0.01)
    citric_acid = st.slider("citric acid", min_value = 0.0 , max_value = 3.0, step = 0.01)
    residual_sugar = st.slider("residual sugar", min_value = 0.0 , max_value = 100.0, step = 0.1)
    chlorides = st.slider("chlorides", min_value = 0.0 , max_value = 0.5, step = 0.001)
    free_sulfur_dioxide = st.slider("free sulfur dioxide", min_value = 0.0 , max_value = 300.0, step = 1.0)
    total_sulfur_dioxide = st.slider("total sulfur dioxide", min_value = 0.0 , max_value = 500.0, step = 1.0)
    density = st.slider("density", min_value = 0.90 , max_value = 1.1, step = 0.01)
    pH = st.slider("pH", min_value = 2.0 , max_value = 4.0, step = 0.01)
    sulphates = st.slider("sulphates", min_value = 0.0 , max_value = 2.0, step = 0.01)
    alcohol = st.slider("alcohol", min_value = 5.0 , max_value = 20.0, step = 0.1)

    algorithm = st.selectbox("",["select algorithm", "Linear Regression", "Random Forest", "Gradient Boosting Trees", "Support Vector Machines"])
    
    inputs = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol]])

    inputs = Normalizer().fit_transform(inputs)
    inputs = pd.DataFrame(inputs)
    inputs = model.transform(inputs)

    if algorithm == "Linear Regression":
        regressor = LinearRegression().fit(feat_train, tar_train)
        predictions = regressor.predict(feat_test)
        error = mean_squared_error(tar_test,predictions)
        result = regressor.predict(inputs)
    elif algorithm == "Random Forest":
        regressor = RandomForestRegressor().fit(feat_train, tar_train)
        predictions = regressor.predict(feat_test)
        error = mean_squared_error(tar_test,predictions)
        result = regressor.predict(inputs)
    elif algorithm == "Gradient Boosting Trees":
        regressor = GradientBoostingRegressor().fit(feat_train, tar_train)
        predictions = regressor.predict(feat_test)
        error = mean_squared_error(tar_test,predictions)
        result = regressor.predict(inputs)
    elif algorithm == "Support Vector Machines":
        regressor = SVR().fit(feat_train, tar_train)
        predictions = regressor.predict(feat_test)
        error = mean_squared_error(tar_test,predictions)
        result = regressor.predict(inputs)
    
    push = st.button("get the quality")

    if push:
        if algorithm != "select algorithm":
            st.header("The quality of your wine is " + str(int(result)))
            st.markdown("The mean squared error of this algorithm is " + str(round(error,2)))
        else:
            st.markdown("Please select algorithm")
    return None

if __name__ == "__main__":
    main()