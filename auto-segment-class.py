import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from kneed import KneeLocator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from data_encoding import *
from sklearn.compose import make_column_selector as selector
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_diabetes

#-----------------------------------#
# Page layout
st.set_page_config(page_title='The Segmentation App',
    layout='centered')

#-----------------------------------#
# Find Cluster
class AutoClustering():
  def __init__(self, method='elbow', max_cluster=11, **kwargs):
      self.n_clusters = n_clusters
      # self.max_cluster = max_cluster
      self.method = method
      self.kwargs = kwargs

  def find_cluster(self, X):
    X = X.copy()

    clusters = range(2, 50)
    wcss = {}
    cluster = []
    score = []

    if self.method=='auto':
      for k in clusters:
          kmeans = KMeans(n_clusters=k, init="k-means++",  random_state=42)
          kmeans.fit(X)
          wcss[k] = kmeans.inertia_
        #   print(k)

        #   if k >= 3:
        #       dif = (wcss[k] - wcss[k-1])/wcss[k-1]
        #       if dif < -0.02:
        #           break

      kn = KneeLocator(x=list(wcss.keys()),
                  y=list(wcss.values()),
                  curve='convex',
                  direction='decreasing')
      k = kn.knee

      self.wcss = wcss
      #
      # self.wcss_df = pd.DataFrame({'cluster': self.wcss.keys(),
      #                           'wcss': self.wcss.values()})
      #
      # self.wcss_df['pct'] = self.wcss_df['wcss'].pct_change()

    # elif self.method=='silhouette':
    #   for k in clusters:
    #       kmeans = KMeans(n_clusters=k, init="k-means++",  random_state=42)
    #       cluster_labels = kmeans.fit_predict(X)
    #
    #       silhouette_avg = silhouette_score(X, cluster_labels)
    #       cluster.append(k)
    #       score.append(silhouette_avg)
    #
    #   silho_df = pd.DataFrame(list(zip(cluster, score)),
    #                         columns =['n_clusters', 'silhouette_score'])
    #
    #   k = int(silho_df[silho_df['silhouette_score'] == max(silho_df['silhouette_score'])]['n_clusters'])
    #
    #   self.cluster = cluster
    #   self.score = score
    #   self.silho_df = silho_df

    elif self.method=='manual':
      k = self.n_clusters

    return k

  def fit(self, X):
    X = X.copy()
    self.k = self.find_cluster(X)

    self.model = KMeans(n_clusters=self.k, **self.kwargs, random_state=42)
    self.model.fit(X)

    return self

  def predict(self, X):
    cluster = self.model.predict(X)

    return cluster

  def get_params(self):
      return self.model.get_params()


  def plot_elbow(self):
    # data = self.wcss_df[(self.wcss_df['pct'] < -0.1) | self.wcss_df['pct'].isna()]
    fig = px.line(x=list(self.wcss.keys()),
              y=list(self.wcss.values()),
              markers=True)

    fig.update_layout(width = 620,
                        height = 400,
                        title = 'The Elbow Method')

    fig.update_xaxes(title_text='the number of clusters(k)')
    fig.update_yaxes(title_text='Intra sum of distances(WCSS)')
    return fig

  def plot_silho(self):
    data = pd.DataFrame(list(zip(self.cluster, self.score)),
                        columns =['n_clusters', 'silhouette_score'])

    fig = px.line(data, x='n_clusters', y='silhouette_score',
              markers=True)

    fig.update_layout(width = 620,
                        height = 400,
                        title = 'The Silhouette Method')

    fig.update_xaxes(title_text='the number of clusters(k)')
    fig.update_yaxes(title_text='Silhouette Score')
    return fig


#-----------------------------------#
# Visualization
class AutoVisualized():
    def __init__(self, data):
        self.data = data
        self.X = self.num_feature(X)
        self.summary = self.data.groupby('Cluster')[self.X].mean()
        self.norm_sum = self._norm_sum(self.summary)


    def _norm_sum(self, summary):
        norm_sum = (summary - summary.min()) / (summary.max() - summary.min())
        return norm_sum

    def num_feature(self, X):
        num_features = X.select_dtypes(include='number').columns
        return num_features

    def plot_radar_chart(self, n):
        radar_cluster = self.norm_sum.loc[n].to_frame().reset_index()
        radar_cluster.columns = ['theta','r']

        # radar plot
        fig = px.line_polar(radar_cluster, r='r', theta='theta', line_close=True)
        fig.update_layout(
            polar = dict(
                radialaxis = dict(range=[0, 1], visible=True)
            )
        )
        return fig

    def compare_radar_chart(self, list_n):
        fig = go.Figure()
        for n in list_n:
            radar_cluster = self.norm_sum.loc[n].to_frame().reset_index()
            radar_cluster.columns = ['theta','r']
            fig.add_trace(go.Scatterpolar(
                r=radar_cluster['r'],
                theta=radar_cluster['theta'],
                fill='toself',
                name=f'Cluster: {n}'
            ))
        fig.update_layout(
            polar = dict(radialaxis_tickfont_size = 15,
                angularaxis = dict(
                    tickfont_size = 15,
                    rotation = 90,
                    direction = "clockwise" )),
            width = 600,
            height = 600,
        )
        return fig

    def plot_cluster(self):
        count_clus = self.data['Cluster'].value_counts(normalize=True)
        fig = px.bar(count_clus)
        fig.update_layout(
            title="The total number of clusters",
            xaxis_title="Cluster",
            yaxis_title="Ratio",
            legend_title=None)

        return fig


#------------------------------------#

st.write("""
# The Auto Segmentation App

In this implementation, the *KMeans()* function is used this app for build a clustering model using the **KMeans** algorithm.

Try uploading the file and adjusting the hyperparameter!

""")

#-----------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/PraewaPnc/auto-segmentation/ba5138f38ec21a2c1f0225d039584d673efb49ca/small_titanic.csv)
""")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        drop_col = st.sidebar.multiselect('Drop columns', df.columns.tolist())
    else:
        st.info('Awaiting for CSV file to be uploaded. **!!!**')

# Transform options
with st.sidebar.header('2. Transform Data'):
    st.sidebar.subheader('Fill missing values')
    fill_numerical = st.sidebar.selectbox('Fill numerical features', ['mean','most_frequent','median','constant'], index=0)
    num_value = None
    if fill_numerical == 'constant':
        num_value = st.sidebar.number_input('Constant value for filling in numerical features', value=0)
    fill_categorical = st.sidebar.selectbox('Fill categorical features', ['most_frequent','constant'], index=0)
    cat_value = None
    if fill_categorical == 'constant':
        cat_value = st.sidebar.text_input('Constant value for filling in categorical features', value='NoData')

    num_transformer = Pipeline([('imputer', SimpleImputer(strategy=fill_numerical, fill_value=num_value)),
                                ('scaler', MinMaxScaler())])
    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy=fill_categorical, fill_value=cat_value)),
                                ('encode', OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnPreprocessor([('num', num_transformer, selector(dtype_include='number')),
                                            ('cat', cat_transformer, selector(dtype_exclude='number'))])

# Select method for choose n_clusters
with st.sidebar.subheader('3. Select number of cluster'):
    # max_cluster = st.sidebar.slider('Maximum number of clusters', 1, 100, 10, 1)
    method = st.sidebar.selectbox('Method', ['auto', 'manual'])
    if method == 'manual':
        n_clusters = st.sidebar.number_input('Number of clusters (n_clusters)', value=0)
    else:
        n_clusters = 20

# Sidebar - Specify parameter settings
# with st.sidebar.subheader('4. Set Parameters in **KMeans**'):
#     init = st.sidebar.selectbox('Method for initialization (init)', ['k-means++', 'random'])
#     n_init = st.sidebar.slider('Initial cluster centroids (n_init)', 1, 10, 10, 1)
#     max_iter = st.sidebar.slider('Maximum number of iterations (max_iter)', 1, 2000, 300, 1)
#     verbose = st.sidebar.slider('Verbosity mode (verbose)', 1, 10, 0, 1)
#     random_state = st.sidebar.slider('Random State (random_state)', 1, 100, 42, 1)
#     algorithm = st.sidebar.selectbox('K-means algorithm to use (algorithm)', ['auto', 'full', 'elkan'])

with st.sidebar.subheader('4. Dimension Reduction [Optional]'):
    pca = st.sidebar.checkbox('PCA')
    if pca:
        n_components = st.sidebar.number_input('Number of Dimension (n_components)', value=0)

#-----------------------------------#
# Main panel

if uploaded_file:
# Displays the dataset
    st.subheader('Dataset')

    # autoClustering = AutoClustering(method=method,
    #                                 max_cluster=max_cluster,
    #                                 init=init,
    #                                 n_init=n_init,
    #                                 max_iter=max_iter,
    #                                 verbose=verbose,
    #                                 random_state=random_state,
    #                                 algorithm=algorithm)

    autoClustering = AutoClustering(method=method)

    X = df.drop(columns=drop_col, axis=1)
    st.markdown('**Glimpse of dataset**')
    st.write(X.head(5))
    st.write('Shape of Dataset')
    st.info(X.shape)
    st.write('Features for clustering')
    st.info(X.columns.to_list())

    transformed_X = preprocessor.fit_transform(X)
    if pca:
        pca = PCA(n_components=n_components, random_state=42)
        pca_X = pca.fit_transform(transformed_X)
        autoClustering.fit(pca_X)
        clusters = autoClustering.predict(pca_X)
        number_clusters = autoClustering.find_cluster(pca_X)
    else:
        autoClustering.fit(transformed_X)
        clusters = autoClustering.predict(transformed_X)
        number_clusters = autoClustering.find_cluster(transformed_X)

    df['Cluster'] = clusters + 1

    st.subheader('Clustering Result')

    if method == 'auto':
        fig = autoClustering.plot_elbow()
        st.plotly_chart(fig)
    elif method == 'silhouette':
        fig = autoClustering.plot_silho()
        st.plotly_chart(fig)
    else:
        pass

    st.write('Optimal Number of Clusters')
    st.info(number_clusters)
    st.write('Predict cluster index for each sample.')
    st.info(clusters + 1)

    st.markdown('Model Parameters')
    st.write(autoClustering.get_params())

    st.subheader('Dataset with Cluster')
    st.write(df)

    #---------------------------#
    # Visualization

    st.subheader('Visualization')
    autoVisualized = AutoVisualized(data=df)
    # plot the total number of cluster
    pc = autoVisualized.plot_cluster()
    st.plotly_chart(pc)

    list_clust = st.multiselect('Select cluster', df.Cluster.unique().tolist())

    # Compare 2 clusters
    plot = autoVisualized.compare_radar_chart(list_n=list_clust)
    st.plotly_chart(plot)
