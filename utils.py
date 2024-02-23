import os 
import pickle
import streamlit as st
import pandas as pd

from pysentimiento import create_analyzer

@st.cache_data
def load_data_pickle(path, file):
    """Load data from pickle file"""
    df = pd.read_pickle(os.path.join(path,file))
    return df

@st.cache_data
def load_data_csv(path, file):
    "Load data from csv file"
    df = pd.read_csv(os.path.join(path,file))
    return df

@st.cache_data
def load_model_pickle(path, file):
    """Load model from pickle file"""
    path_file = os.path.join(path, file)
    model = pickle.load(open(path_file, 'rb'))
    return model

