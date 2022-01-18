import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from kneed import KneeLocator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

#-----------------------------------#
# Page layout
st.set_page_config(page_title='The Segmentation App',
    layout='wide')

#-----------------------------------#
# Preprocess Data
# def preprocess_data(X):
#       df = X.copy()
#       num_features = X.select_dtypes(include='number').columns
#       cat_features = X.select_dtypes(exclude='number').columns
#
#       # Impute Missing Values
#       # fill cat features with mode
#       cat_mode = X[cat_features].mode().squeeze()
#       X[cat_features] = X[cat_features].fillna(0)
#
#       # fill num features with median
#       num_median = X[num_features].medien()
#       X[num_features] = X[num_features].fillna(0)
#
#       # OneHotEncoder
#       X = pd.get_dummies(X, columns=cat_features)
#
#       # Scaler
#         scaler = scalaer
#
#       X[num_features] = scaler.fit_transform(X[num_features])
#
#       return X

#-----------------------------------#
# Find Cluster






#-----------------------------------#
# Model building
def auto_seg(X):
    X = preprocess_data(X)

    st.markdown('**Dataset**')
    st.write('Sample Data')
    st.info(X.head)

    model = KMeans(n_clusters=parameter_n_clusters,
                    init=parameter_init,
                    n_init=parameter_n_init,
                    max_iter=parameter_max_iter,
                    tol=parameter_tol,
                    precompute_distances=parameter_precompute_distances,
                    verbose=parameter_verbose,
                    random_state=parameter_random_state,
                    copy_x=parameter_copy_x,
                    n_jobs=parameter_n_jobs,
                    algorithm=parameter_algorithm)
    model.fit(X)
    cluster = model.predict(X)

    st.subheader('Clustering Result')

    st.markdown('Cluster')
    st.write('Predict cluster index for each sample.')
    st.info(cluster)

    st.markdown('Model Parameters')
    st.write(model.get_params())

#-----------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    df = pd.read_csv(uploaded_file)
    drop_col = st.sidebar.multiselect('Drop columns', df.columns.tolist())
# Transform options
with st.sidebar.header('Transform Data'):
    st.sidebar.subheader('Fill missing values')
    fill_numerical = st.sidebar.selectbox('Fill numerical features', ['mean','most_frequent','median','constant'], index=0)
    num_value = None
    if fill_numerical == 'constant':
        num_value = st.sidebar.number_input('Constant value for filling in numerical features', value=0)
    fill_categorical = st.sidebar.selectbox('Fill categorical features', ['most_frequent','constant'], index=0)
    cat_value = None
    if fill_categorical == 'constant':
        cat_value = st.sidebar.text_input('Constant value for filling in categorical features', value='NoData')

    st.sidebar.subheader('Encode Categorical Data')
    num_transformer = Pipeline([('imputer', SimpleImputer(strategy=fill_numerical, fill_value=num_value)),
                                ('scaler', MinMaxScaler())])
    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy=fill_categorical, fill_value=cat_value)),
                                ('encode', OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnPreprocessor([('num', num_transformer, selector(dtype_include='number')),
                                            ('cat', cat_transformer, selector(dtype_exclude='number'))])

# Sidebar - Specify parameter settings
with st.sidebar.subheader('Set Parameters **KMeans**'):
    parameter_n_clusters = st.sidebar.slider('Number of clusters (n_clusters)', 0, 100, 2, 1)
    parameter_init = st.sidebar.selectbox('Method for initialization (init)', ['k-means++', 'random'])
    parameter_n_init = st.sidebar.slider('Initial cluster centroids (n_init)', 1, 10, 2, 1)
    parameter_max_iter = st.sidebar.slider('Maximum number of iterations (max_iter)', 1, 2000, 300, 1)
    parameter_verbose = st.sidebar.slider('Verbosity mode (verbose)', 1, 10, 0, 1)
    parameter_random_state = st.sidebar.slider('Random State (random_state)', 1, 100, 42, 0)
    parameter_algorithm = st.sidebar.selectbox('K-means algorithm to use (algorithm)', ['auto', 'full', 'elkan'])

#-----------------------------------#
# Main panel

# Displays the dataset
st.header('Dataset')
