import pickle
import streamlit as st
import zipfile
import sklearn
from scipy.special import logit, expit
import numpy as np


path_to_zip_file = 'classifier_pipes.zip'
directory_to_extract_to = ''
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)
  

pickle_in = open('classifier_pipes.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 

def sigmoid_moved_to_2(x):
    y = np.exp(x-2)/(np.exp(x-2) + 1)
    return y

def power(x):
    if x<1:
        POWER = 1
        return x**POWER / (x**POWER+0.5)
    if x>1:
        POWER = 2
        return x**POWER / (x**POWER+0.5)

def logit_2(x):
    return logit(x)+2

def prediction(
    linkdiameter, linkslope, Q_flow, Q_type_uniform,
    streamorderSH ,Link_residents ,Aspect_ratio ,real_density,
    betweeness, closeness, current_flow_closeness, second_order, katz_cent, harmonic_centrality, degree,
    threshold,
            ):   
 
    pickle_in = open('classifier_pipes.pkl', 'rb') 
    classifier = pickle.load(pickle_in)
 
    prediction = classifier.predict_proba( 
        [[linkdiameter, linkslope, Q_flow, Q_type_uniform,
    streamorderSH ,Link_residents ,Aspect_ratio ,real_density,
    betweeness, closeness, current_flow_closeness, second_order, katz_cent, harmonic_centrality, degree]])[0][1]
#     for threshold in [0,0.5,1,1.5,2,2.5,3,3.5,4,10,100]:
    new_threshold = sigmoid_moved_to_2(threshold)

    if prediction >= new_threshold:
        pred = f'does\'nt accumulate ({logit_2(prediction)}, probably: {100*(power(abs(threshold-logit_2(prediction)))):.0f}%)'

    else:
        pred = f'accumulate ({logit_2(prediction)}, probably: {100*(power(abs(threshold-logit_2(prediction)))):.0f}%)'
#     print(prediction, threshold, new_threshold, pred)

    return pred

      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">ML for sewer pipes</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 

    linkdiameter  = st.select_slider(
     'Select a diameter of pipe',
     options=[0.1882, 0.2354, 0.2966, 0.3766, 0.4708, 0.5932])
    st.write('My diameter pipe is', linkdiameter)
    
    linkslope = st.slider("What is linkslope [%]? (defalut is median)", min_value=0.0, max_value=10.0, value=(4.0), step=0.0005)/100
    Q_flow = st.slider("What is Q_flow? (defalut is median)", min_value=0.10, max_value=1.00, value=(0.50), step=0.1)
    Q_type_uniform = 1 #st.slider("What is Q_type_uniform? (defalut is median)", min_value=0.00, max_value=1.00, value=(1.00), step=1)
    streamorderSH = st.slider("What is streamorderSH? (defalut is median)", min_value=1.00, max_value=75.00, value=(2.00), step=1.0)
    Link_residents = st.slider("What is Link_residents? (defalut is median)", min_value=9.00, max_value=62146.00, value=(1037.00))
    Aspect_ratio = st.slider("What is Aspect_ratio? (defalut is median)", min_value=0.13, max_value=7.50, value=(0.83))
    real_density = st.slider("What is real_density? (defalut is median)", min_value=4.51, max_value=32.07, value=(18.79))
    betweeness = st.slider("What is betweeness? (defalut is median)", min_value=0.00, max_value=0.70, value=(0.05))
    closeness = st.slider("What is closeness? (defalut is median)", min_value=0.02, max_value=0.19, value=(0.05))
    current_flow_closeness = st.slider("What is current_flow_closeness? (defalut is median)", min_value=0.0007, max_value=0.0039, value=(0.001))
    second_order = st.slider("What is second_order? (defalut is median)", min_value=127.86, max_value=4789.96, value=(1419.78))
    katz_cent = st.slider("What is katz_cent? (defalut is median)", min_value=0.05, max_value=0.17, value=(0.08))
    harmonic_centrality = st.slider("What is harmonic_centrality? (defalut is median)", min_value=5.51, max_value=28.80, value=(13.74))
    degree = st.slider("What is degree? (defalut is median)", min_value=1, max_value=5, value=(2), step=1)
    threshold = st.number_input("Set threshold for predictin",min_value=0.0, max_value=100.0, value=(2.0)) 

    
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        
        result = prediction(
                        linkdiameter, linkslope, Q_flow, Q_type_uniform,
                        streamorderSH ,Link_residents ,Aspect_ratio ,real_density,
                        betweeness, closeness, current_flow_closeness, second_order, katz_cent, harmonic_centrality, degree,
                        threshold
                        )
        st.success('Your pipe {}'.format(result))
#         print(LoanAmount)
     
if __name__=='__main__': 
    main()
    
