import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import bioread
import neurokit2 as nk
from glob import glob
import scipy

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

st.set_page_config(layout="wide")

with open( "style.css" ) as css:
   st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
   
st.title('Application for Analyzing Psychophysiological Data')

option = st.selectbox(
    'Select option',
    ['list folder with signals', 'selection of a single file'])

if option == 'selection of a single file':
    biopac_file = st.file_uploader("upload biopac file", type={"acq", "csv"}, key=1)
    data = bioread.read_file(biopac_file)
    orig_sr = data.samples_per_second
    st.markdown(orig_sr)
    df = pd.DataFrame()
    df['EDA_signal'] = data.channels[0].data
        
    sr = st.slider("Select target sampling rate", 0, 1000, 25)
    eda_res = scipy.signal.resample(df['EDA_signal'], sr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=eda_res,
                        mode='lines',
                        name='EDA_signal_raw'))
    fig.update_layout(title=f'EDA timeline: sampling rate: {sr} Hz')
    st.plotly_chart(fig, use_container_width=True)
    
    
    df['EDA_signal_clean'] = nk.eda_clean(df['EDA_signal'], sampling_rate=sr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['EDA_signal_clean'],
                        mode='lines',
                        name='EDA_signal_clean',
                        marker_color='red'))
    fig.update_layout(title='Cleaned Timeline')
    

    
    
