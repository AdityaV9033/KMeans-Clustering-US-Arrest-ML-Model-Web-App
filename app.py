import streamlit as st
import pandas as pd
import numpy as np
st.title('ClusterID of Countries for US Arrest Dataset')
df = pd.read_csv('USArrests.csv')
df.head()
X = df.iloc[:,1:]
X.head()
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3,random_state=2021)
model.fit(X)
Y = pd.read_csv('USArrests.csv',index_col=0)
Y.head()

st.sidebar.header("Select the ML model you want to use")
Drop_options = ["K Means Clustering"]
Model_choice = st.sidebar.selectbox("Drop_options", options=Drop_options)

Index = st.slider("Country Name", min_value=0, max_value=49, step=1)
st.text(f"Country Name: {Y.index[Index]}")
Murder = st.slider("Murder", min_value=0.8, max_value=17.4, step=0.1)
st.text(f"Muder value: {Murder}")
Assault = st.slider("Assault", min_value=45, max_value=337, step=1)
st.text(f"Assault value: {Assault}")
UrbanPop = st.slider("UrbanPop", min_value=32, max_value=91,step=1)
st.text(f"UrbanPop value: {UrbanPop}")
Rape = st.slider("Rape", min_value=7.3, max_value=46.0,step=0.1)
st.text(f"Rape value: {Rape}")

input={'index':Index,
       'murder':Murder,
       'assault':Assault,
       'urbanpop':UrbanPop,
       'rape':Rape
       }
input_X=pd.DataFrame(input, index=['value'])
st.text("Input value of features:")
input_X.T
Z=input_X.iloc[:,1:]
ypred=model.predict(Z)
st.text(f"ClusterID of the country is: {ypred}")
