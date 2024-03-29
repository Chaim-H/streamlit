import pickle
import zipfile
import sklearn
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import streamlit as st
import joblib



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
        pred = f'accumulate solids' # {probably}' , {regressor.predict(x)[0]:.2f}'
    else: 
        pred = f'doesn\'t accumulate solids' # {probably}' #, {regressor.predict(x)[0]:.2f}'
        
    # return
    return pred, warning, in_train

# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:bisque;"> 
    <h1 style ="color:black;text-align:center;font-size:23px;">Tool for predicting solids and sulphides accumulation in sewer pipes</h1> 
    </div> 
    """
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    html_temp = """ 
    <div style ="background-color:bisque;"> 
    <h1 style ="color:black;text-align:center;font-size:20px;">Based on a machine learning model \n(Harpaz et al., 2022)</h1> 
    </div> 
    """
    # display the front end aspect
#     st.markdown(html_temp, unsafe_allow_html = True) 
    
    # following lines create boxes in which user can enter data required to make prediction 

    x = np.array([[
    1.88200000e-01, 4.41795230e-02, 7.00000000e-01, 1.00000000e+00,
    1.00000000e+00, 1.06000000e+02, 8.33333333e-01, 2.62135673e+01,
    0.00000000e+00, 6.63697100e-02, 4.45434000e-04, 1.07529531e+03,
    7.29700590e-02, 1.39022858e+01, 1.00000000e+00,
    ]])
    
#     [diameter, slope, q_flow, type_reduction, SO, 
#      link_residents, aspect_ratio, density, betweenneess, closennes, current_flow, second_order, kats, harmonic, node_degree, temp, cod]
     
#      'Closeness centrality', 'Betweenness centrality', 'Current flow closeness centrality',
# 'Second order centrality', 'Katz centrality', 'Harmonic centrality',
# 'Pipe diameter', 'Node degree', 'Pipe slope', 'Stream order', 'Flow proportion',
#  'Pipe residents', 'Aspect ratio', 'Density', 'COD', 'Temperature']
      

    st.subheader('Case:')
    case = st.radio("Classification according to the availability and collection effort of the features",('Difficult (all features)', 'Medium (a range of slope is needed and the number of residents is not required)'))
    st.subheader('Pipe and network parameters:')
    if case == 'Difficult (all features)':
        x[0][1] = st.slider('Slope of the pipe [%]:', min_value=0.29, max_value=10.01, value=(4.42))/100
        x[0][5] = st.slider("Number of residents discharging wastewater to the pipe [-]:", min_value=9, max_value=62146, value=(106))
        x[0][4] = st.slider("Stream order of the pipe [-]:", min_value=1, max_value=75, value=(1))
    if case == 'Medium (a range of slope is needed and the number of residents is not required)':
        slope_dict = {'0.03-1': 0.65, '1-2': 1.5, '2-3': 2.5, '3-5': 4, '5-10': 7.5}
        x[0][1] = slope_dict[st.select_slider('Range of slope of the pipe [%]:', options=['0.03-1', '1-2', '2-3', '3-5', '5-10'])]
        x[0][4] = st.slider("Stream order of the pipe [-]:", min_value=1, max_value=75, value=(1))
        x[0][5] = 10 + 550 * x[0][4]
    x[0][3] = 1 # st.slider("What is Type reduction                ?", min_value=0.0000, max_value=1.0000, value=(1.0000))
    x[0][0]  = 0.5#get_num(st.select_slider('Diameter of the pipe [m]:', options=[0.1882, 0.2354, 0.2966, 0.3766, 0.4708, 0.5932]))
    x[0][8] = st.slider("Betweenness centrality of the pipe [-]:", min_value=0.0000, max_value=0.7041, value=(0.0000))
    x[0][9] = st.slider("Closeness centrality of the pipe [-]:", min_value=0.0209, max_value=0.1944, value=(0.0664))
    x[0][10] = st.slider("Current flow closeness centrality of the pipe [-]:", min_value=0.01, max_value=0.40, value=(0.04))/100
    x[0][11] = st.slider("Second order centrality of the pipe [-]:", min_value=129.0504, max_value=4789.9608, value=(1075.2953))
    x[0][12] = st.slider("Katz centrality of the pipe [-]:", min_value=0.0514, max_value=0.1742, value=(0.0730))
    x[0][13] = st.slider("Harmonic centrality of the pipe [-]:", min_value=5.7243, max_value=28.8047, value=(13.9023))
    x[0][14] = st.slider("Node degree of the pipe [-]:", min_value=1, max_value=5, value=(1))
    x[0][6] = st.slider("Aspect ratio of the network [-]:", min_value=0.133, max_value=7.5, value=(0.83))
    x[0][7] = st.slider("Density of the network? [person/km\u00b2]", min_value=4505, max_value=32060, value=(26210))/1000
    st.subheader('Simulation parameters:')
    x[0][2] = st.slider("Proportion of reduction in wastewater flow (due to DWES scenario) [-]:", min_value=0.1000, max_value=1.0000, value=(0.7000), step=0.1)
    temperature = st.slider("Temperature [C"+u"\u00b0" +"]:", min_value=10, max_value=30, value=(20), step=1)     #Temperature = "Temperature [C"+u"\u00b0" +"]"
    cod = st.slider("Increase in COD (due to DWES scenario) [-]:", min_value=1.0, max_value=4.0, value=(1.0), step=0.1)
    st.subheader('Thresholds:')

    threshold = st.slider(
        "Threshold value of the maximum shear stress below the pipe will accumulate solids (2 Pa typical value) [Pa]:", 
        min_value=1.5, max_value=2.5, value=(2.0), step=0.1)
    threshold_h2s = st.slider(
        "Threshold value of sulphides that if it is present for more than an hour a day the pipe is classified as accumulating sulphides (typical values are 0.5 mgS/l for moderate and 2 mgS/l for higher probability of hazards) [mgS\l]:", 
        min_value=0.25, max_value=2.5, value=(2.0), step=0.25)
    st.subheader('ML predictions:')

    result =""
    # when 'Predict' is clicked, make the prediction and store it 
#     if st.button("Predict solids"):
    result = prediction(x, threshold, regressor, pca, train_location, point_round)
    if result[1] != "":
        st.success('{}'.format(result[1]))
    st.success('Your pipe {}'.format(result[0]))
    

    x = scaler.transform(x)
    x = x.reshape(-1,1).reshape(-1,1)[[8,9,10,11,12,13,0,14,1,4,2,5,6,7]]
    x = np.append(x,
                  [
                    [StandardScaler().fit([[10], [20], [30]]).transform([[temperature]])[0][0]],
                    [StandardScaler().fit([[1], [1.5], [2], [3],[ 4]]).transform([[cod]])[0][0]],
                  ])
    loaded_rf = joblib.load("rrf_h2s.joblib")
#     if st.button("Predict H2S"):
    loaded_rf.predict(x.reshape(1,-1))
    if loaded_rf.predict(x.reshape(1,-1)) > threshold_h2s:
        st.success('Your pipe accumulate sulphides')
    else:
        st.success('Your pipe doesn\'t accumulate sulphides')

#     html_temp = """ 
#     <div style ="background-color:azure;"> 
#     <h1 style ="color:black;text-align:left;font-size:18px;">Harpaz C., Russo S., Leitão J.P., Penn  R., 2022. Potential of supervised machine learning algorithms for estimating the impact of water efficient scenarios on solids accumulation in sewers. Water Research, 216</h1> 
#     </div> 
#     """
    # display the front end aspect
#     st.markdown(html_temp, unsafe_allow_html = True)      
if __name__=='__main__': 
    main()
