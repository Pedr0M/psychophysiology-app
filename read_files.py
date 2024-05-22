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
   

st.header('tool for visualizing EDA data')


shift = 20
quantile_exc = .999
cons_signal = 20

option = st.selectbox(
    'Select option',
    ['list all folders', 'manually upload file'])

if option == 'manually upload file':
    biopac_file = st.file_uploader("upload biopac file", type={"acq", "csv"}, key=1)
    

    cols = {}
    cols[1], cols[2] = st.columns(2)
    data = bioread.read_file(biopac_file)
    sr = data.samples_per_second
    #st.markdown(f'sampling rate: {sr}')

    len_c = len(data.channels[0].data)
    len_t = len(data.channels[1].data)
    min_len = min(len_c, len_t)
    
    df = pd.DataFrame()
    df['EDA_C'] = data.channels[0].data[:min_len]
    df['EDA_T'] = data.channels[1].data[:min_len]
    df['EDA_C'] = df['EDA_C'].astype('float')
    df['EDA_T'] = df['EDA_T'].astype('float')
    
    sr = int(len(df['EDA_C'])/250)
    eda_c_res = scipy.signal.resample(df['EDA_C'], sr)
    eda_t_res = scipy.signal.resample(df['EDA_T'], sr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=eda_c_res,
                        mode='lines',
                        name='C_raw'))
    fig.update_layout(title='client timeline')
    cols[1].plotly_chart(fig, use_container_width=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=eda_t_res,
                        mode='lines',
                        name='T_raw'))
    fig.update_layout(title='therapist timeline')
    cols[2].plotly_chart(fig, use_container_width=True)

else:
    pair = glob('EDA/*')
    pair.sort()
    option_pair = st.selectbox(
        'Select pair',
        pair)

    sessions = glob(f'{option_pair}/*')
    sessions.sort()
    option_session = st.selectbox(
        'Select session',
        sessions)

    biopac_file = glob(f'{option_session}/*.acq')[0]
#biopac_file = st.file_uploader("upload biopac file", type={"acq", "csv"}, key=1)



#if biopac_file is not None:
    option = st.selectbox(
    'Select option',
    [.999, .995, .99])
    
    cols = {}
    cols[1], cols[2] = st.columns(2)
    data = bioread.read_file(biopac_file)
    sr = data.samples_per_second
    st.markdown(f'sampling rate: {sr}')

    len_c = len(data.channels[0].data)
    len_t = len(data.channels[1].data)
    min_len = min(len_c, len_t)
    


    ### cliente
    df = pd.DataFrame()
    df['EDA_C'] = data.channels[0].data[:min_len]
    df['EDA_T'] = data.channels[1].data[:min_len]
    df['EDA_C'] = df['EDA_C'].astype('float')
    df['EDA_T'] = df['EDA_T'].astype('float')
    
    

    
    df['EDA_C_shift_10'] = df['EDA_C'].shift(shift) ### encontrar o valor de x pontos antes (definido em shift)
    df['diff_C_10'] = abs(df['EDA_C'] - df['EDA_C_shift_10']) ### comparar valor atual com o valor x unidades antes
    df.loc[df['diff_C_10'] >= df['diff_C_10'].quantile(option), 'EDA_C'] = np.nan ### verificar que pontos estão acima de threshold (definido em quantile_exc)
    df.loc[:shift-1, 'EDA_C_shift_10'] = 0
    df.loc[:shift-1, 'diff_C_10'] = 0
    df_nonan = df.dropna().reset_index() ### todas as observações acima do threshold são substituídas por nan (valor omisso)
    df_nonan['index_diff'] = df_nonan['index'].diff() ### encontrar blocos de pontos de dados sucessivos
    df_nonan['block'] = df_nonan['index_diff'].ne(1).cumsum()
    block_count = df_nonan.groupby('block').count().reset_index() ### visualizar extensão do bloco - blocos com menos de yy pontos são removidos (definido em cons_signal)
    blocks_to_include = block_count[block_count['index'] >= cons_signal]['block'].unique()
    df_cons_blocks = df_nonan[df_nonan['block'].isin(blocks_to_include)]
    df_final = pd.merge(df.reset_index()['index'], df_cons_blocks[['index', 'EDA_C']], on = 'index', how = 'left') ### interpolação dos valores em falta
    df_final = df_final.interpolate()
    df_client = df_final.copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['EDA_C'],
                        mode='lines',
                        name='C_raw'))
    fig.add_trace(go.Scatter(y=df_final['EDA_C'],
                        mode='lines',
                        name='C_outliers_removed'))
    fig.update_layout(title='client timeline: before and after removal of outliers')
    cols[1].plotly_chart(fig, use_container_width=True)
    
    ### terapeuta
    df = pd.DataFrame()
    df['EDA_C'] = data.channels[0].data[:min_len]
    df['EDA_T'] = data.channels[1].data[:min_len]
    df['EDA_C'] = df['EDA_C'].astype('float')
    df['EDA_T'] = df['EDA_T'].astype('float')
    
    df['EDA_T_shift_10'] = df['EDA_T'].shift(shift) ### encontrar o valor de x pontos antes (definido em shift)
    df['diff_T_10'] = abs(df['EDA_T'] - df['EDA_T_shift_10']) ### comparar valor atual com o valor x unidades antes
    df.loc[df['diff_T_10'] >= df['diff_T_10'].quantile(quantile_exc), 'EDA_T'] = np.nan ### verificar que pontos estão acima de threshold (definido em quantile_exc)
    df.loc[:shift-1, 'EDA_T_shift_10'] = 0
    df.loc[:shift-1, 'diff_T_10'] = 0
    df_nonan = df.dropna().reset_index() ### todas as observações acima do threshold são substituídas por nan (valor omisso)
    df_nonan['index_diff'] = df_nonan['index'].diff() ### encontrar blocos de pontos de dados sucessivos
    df_nonan['block'] = df_nonan['index_diff'].ne(1).cumsum()
    block_count = df_nonan.groupby('block').count().reset_index() ### visualizar extensão do bloco - blocos com menos de yy pontos são removidos (definido em cons_signal)
    blocks_to_include = block_count[block_count['index'] >= cons_signal]['block'].unique()
    df_cons_blocks = df_nonan[df_nonan['block'].isin(blocks_to_include)]
    df_final = pd.merge(df.reset_index()['index'], df_cons_blocks[['index', 'EDA_T']], on = 'index', how = 'left') ### interpolação dos valores em falta
    df_final = df_final.interpolate()
    df_therapist = df_final.copy()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['EDA_T'],
                        mode='lines',
                        name='T_raw'))
    fig.add_trace(go.Scatter(y=df_final['EDA_T'],
                        mode='lines',
                        name='T_outliers_removed'))
    fig.update_layout(title='therapist timeline: before and after removal of outliers')
    cols[2].plotly_chart(fig, use_container_width=True)
    
    df_client['EDA_C_clean'] = nk.eda_clean(df_client['EDA_C'], sampling_rate=20)
    df_therapist['EDA_T_clean'] = nk.eda_clean(df_therapist['EDA_T'], sampling_rate=20)


    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df_client['EDA_C_clean'],
                        mode='lines',
                        name='C_clean',
                        marker_color='red'))
    fig.add_trace(go.Scatter(y=df_therapist['EDA_T_clean'],
                        mode='lines',
                        name='T_clean',
                        marker_color='purple'))
    fig.update_layout(title='cleaned timelines: client and therapist')
    st.plotly_chart(fig, use_container_width=True)