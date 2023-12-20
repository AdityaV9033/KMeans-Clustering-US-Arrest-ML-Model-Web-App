import streamlit as st
import pandas as pd
import numpy as np
st.title('Clustering Countries for US Arrest Dataset')
df = pd.read_csv('USArrests.csv',index_col=0)
df.head()
X = df.iloc[:,:]
X.head()
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3,random_state=2021)
model.fit(X)


st.sidebar.header("Select the ML model you want to use")
Drop_options = ["K Means Clustering"]
Model_choice = st.sidebar.selectbox("Drop_options", options=Drop_options)

Murder = st.slider("Murder", min_value=0.8, max_value=17.4, step=0.1)
st.text(f"Muder value: {Murder}")
Assault = st.slider("Assault", min_value=45, max_value=337, step=1)
st.text(f"Assault value: {Assault}")
UrbanPop = st.slider("UrbanPop", min_value=32, max_value=91,step=1)
st.text(f"UrbanPop value: {UrbanPop}")
Rape = st.slider("Rape", min_value=7.3, max_value=46.0,step=0.1)
st.text(f"Rape value: {Rape}")

input={'murder':Murder,
       'assault':Assault,
       'urbanpop':UrbanPop,
       'rape':Rape,
       }
input_X=pd.DataFrame(input, index=['value'])
st.text("Input value of features:")
input_X.T
ypred=model.predict(X)
st.text(f"clusterID of the country is: {ypred}")