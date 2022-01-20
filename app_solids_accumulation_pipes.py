import pickle
import zipfile
import sklearn
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import streamlit as st


path_to_zip_file = 'regressor_pipes.zip'
directory_to_extract_to = ''
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)

with open('regressor_pipes.pkl', 'rb') as file:
    regressor = pickle.load(file)
    

with open('StandardScaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('pca.pkl', 'rb') as file:
    pca = pickle.load(file)
    
with open('train_location.pkl', 'rb') as file:
    train_location, point_round = pickle.load(file)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 

def power(x, POWER, bias):
    return x**POWER / (x**POWER+bias)


def if_train_location(x, pca, train_location, point_round, scaler): 
    return tuple(pca.transform(scaler.transform(x))[0].round(point_round)) in train_location


def prediction(x, threshold, regressor, pca, train_location, point_round):
    # the point is feasible?
    warning = ""
    in_train = if_train_location(x, pca, train_location, point_round, scaler)
    if in_train is False:
        warning = "A combination of the pipe features selected is not feasible"
    
    # calculte the probabilty:
    a, a0 = 1.5, 0.1
    b, b0 = 1.5, 0.5
    i = regressor.predict(x)[0]
    if i-threshold < 0:
        prob = power(abs(i-threshold),a,a0)
    else:
        prob = power(abs(i-threshold),b,b0)
    probably = f'(probably: {100*(prob):.0f}%)'
    
    # calculate the prediction:
    if regressor.predict(x)[0] < threshold:
        pred = f'accumulate {probably}, {regressor.predict(x)[0]:.2f}'
    else: 
        pred = f'doesn\'t accumulate {probably}, {regressor.predict(x)[0]:.2f}'
        
    # return
    return pred, warning, regressor.predict(x)[0], in_train

# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Tool for predicting solids accumulation in sewer pipes</h1> 
    </div> 
    """
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">(based on machine learning model)</h1> 
    </div> 
    """
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    # following lines create boxes in which user can enter data required to make prediction 

    x = np.array([[
    1.88200000e-01, 4.41795230e-02, 7.00000000e-01, 1.00000000e+00,
    1.00000000e+00, 1.06000000e+02, 8.33333333e-01, 2.62135673e+01,
    0.00000000e+00, 6.63697100e-02, 4.45434000e-04, 1.07529531e+03,
    7.29700590e-02, 1.39022858e+01, 1.00000000e+00
    ]])
    x[0][0]  = st.select_slider('Select a diameter of pipe [m]:', options=[0.1882, 0.2354, 0.2966, 0.3766, 0.4708, 0.5932])
    x[0][1] = st.slider('Select a slope of pipe [%]:', min_value=0.29, max_value=10.01, value=(4.42))/100
    x[0][3] = 1 #st.slider("What is Type reduction                ?", min_value=0.0000, max_value=1.0000, value=(1.0000))
    x[0][4] = st.slider("Select a stream order of pipe:", min_value=1, max_value=75, value=(1))
    x[0][5] = st.slider("Select a pipe residents of pipe", min_value=9, max_value=62146, value=(106))
    x[0][6] = st.slider("What is Aspect ratio of network? [-]", min_value=0.133, max_value=7.5, value=(0.83))
    x[0][7] = st.slider("What is Density of network? [person/km^2]", min_value=4505, max_value=32060, value=(26210))/1000
    x[0][8] = st.slider("Select a Betweenness centrality of pipe [-]:", min_value=0.0000, max_value=0.7041, value=(0.0000))
    x[0][9] = st.slider("Select a Closeness centrality of pipe [-]:", min_value=0.0209, max_value=0.1944, value=(0.0664))
    x[0][10] = st.slider("Select a Current flow closeness centrality of pipe [-]:", min_value=0.01, max_value=0.40, value=(0.04))/100
    x[0][11] = st.slider("Select a Second order centrality of pipe [-]:", min_value=129.0504, max_value=4789.9608, value=(1075.2953))
    x[0][12] = st.slider("Select a Katz centrality of pipe [-]:", min_value=0.0514, max_value=0.1742, value=(0.0730))
    x[0][13] = st.slider("Select a Harmonic centrality of pipe [-]:", min_value=5.7243, max_value=28.8047, value=(13.9023))
    x[0][14] = st.slider("Select a Node degree of pipe [-]:", min_value=1, max_value=5, value=(1))
    
    st.write('Parameters for prediction:')
    x[0][2] = st.slider("What is DWES Scenario:", min_value=0.1000, max_value=1.0000, value=(0.7000), step=0.1)
    threshold = st.number_input("Set a threshold value (diurnal max shear stress) below which the pipe is defined as solid-accumulated \n(2 in the literature)",min_value=0.0, max_value=100.0, value=(2.0)) 

    
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        
        result = prediction(x, threshold, regressor, pca, train_location, point_round)
        
        if result[1] != "":
            st.success('{}'.format(result[1]))
        st.success('Your pipe {}'.format(result[0]))
#         print(LoanAmount)
     
if __name__=='__main__': 
    main()
    