#================================ /   IMPORTING LIBRARIES   / ======================================#

#File handling libraries
import numpy as np
import pickle

#Dashboard Library
import streamlit as st

#warnings
import warnings
warnings.filterwarnings("ignore")

#================================ /   DASHBOARD SETUP   / ======================================#

#Page_configuration
st.set_page_config(
    page_title = 'Singapore Resale Flats',
    page_icon = '🏙️',
    layout = 'wide')

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background: linear-gradient(#42c9a1,#b4f0de);
ckground-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0); 
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

#================================ /   LOADING MODEL   / ======================================#

with open("RandomForestRegressor.pkl", 'rb') as file:
    model = pickle.load(file)

with open("scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

with open("onehot_encoder.pkl", 'rb') as f:
    ohe = pickle.load(f)

#================================ /   USER INPUT FORM    / ======================================#

st.header(":orange[SINGAPORE RESALE FLAT PRICE PREDICTOR🏙️]")
st.markdown("**Enter the following parameters to get a Flat's Resale Price in Singapore**")

town_options = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI',
       'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG', 'SERANGOON', 
       'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN','LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS', 'PUNGGOL']

flat_type_options = ['1 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']

storey_range_options = ['10 TO 12', '04 TO 06', '07 TO 09', '01 TO 03', '13 TO 15', '19 TO 21', '16 TO 18', '25 TO 27', '22 TO 24', 
        '28 TO 30', '31 TO 33', '40 TO 42', '37 TO 39', '34 TO 36', '06 TO 10', '01 TO 05', '11 TO 15', '16 TO 20', '21 TO 25', '26 TO 30',
       '36 TO 40', '31 TO 35', '46 TO 48', '43 TO 45', '49 TO 51']

flat_model_options = ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified', 'Model A-Maisonette', 'Apartment', 'Maisonette', 
        'Terrace', '2-Room', 'Improved-Maisonette', 'Multi Generation', 'Premium Apartment', 'Adjoined Flat', 'Premium Maisonette',
        'Model A2', 'Dbss', 'Type S1', 'Type S2', 'Premium Apartment Loft', '3Gen']

with st.form('Predict Resale Price Form '):

    col1,col2 = st.columns(2)
    with col1:
        st.write(' ')
        town = st.selectbox(":green[**Select Town**]", sorted(town_options), index=None, key='town')
        flat_type = st.selectbox(":green[**Select Flat Type**]", sorted(flat_type_options), index=None, key='flat_type')
        storey_range = st.selectbox(":green[**Select Storey Range**]", sorted(storey_range_options), index=None, key='storey_range')
        flat_model = st.selectbox(":green[**Select Flat Model**]", sorted(flat_model_options), index=None, key='flat_model')

    with col2:
        st.write(' ')
        year = st.text_input(":green[**Enter the Flat Sale year**]", placeholder='Eg:1990', key='year')
        lease_comm_date = st.text_input(":green[**Enter the Lease Commencement Year**]", placeholder='Eg:2020', key='lease_date')
        floor_area = st.text_input(":green[**Enter the Floor area in square meters**]", placeholder='Eg:20', key='floor_area')
        st.write(' ')
        st.write(' ')
        submit_button = st.form_submit_button(label="**PREDICT RESALE PRICE**")

    if submit_button:
        user_data= np.array([[floor_area, lease_comm_date, year, town, flat_type, storey_range, flat_model]])
        user_data_ohe = ohe.transform(user_data[:, [3,4,5,6]])
        user_data = np.concatenate((user_data[:, [0,1,2]], user_data_ohe), axis=1)
        user_data1 = scaler.transform(user_data)
        pred = model.predict(user_data1)
        st.subheader(f"**:orange[PREDICTED RESALE PRICE:] {pred[0]:.2f} SGD**")

st.divider()

st.subheader(":orange[INFO]")

st.caption("ABOUT")
st.write("**This web page is to help Buyers and Sellers of flats in Singapore to predict the market resale \
    price of the flat. The resale price predicted depends on the town the flat is located in, floor size, flat type, \
    flat model and lease commencement year. The value predicted is based on the historical data of resale flat prices \
    in Singapore obtained from Government of Singapore's website.**")

st.caption("CREATOR")
st.text("App created by Abinaya Ganesh as a project for GUVI Geeks Netwrok Pvt. Ltd., India")

st.divider()