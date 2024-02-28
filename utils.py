import os 
import io
import pickle
import base64
import streamlit as st
import pandas as pd

from google.oauth2 import service_account
#from googleapiclient.discovery import build
from htbuilder import HtmlElement, div, hr, a, p, img, styles
from pathlib import Path



######################## LOAD DATA FROM REPO ##########################

@st.cache_data(ttl=3600, show_spinner=False)
def load_data_pickle(path, file):
    """Load data from pickle file"""
    df = pd.read_pickle(os.path.join(path,file))
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_data_csv(path, file):
    "Load data from csv file"
    df = pd.read_csv(os.path.join(path,file))
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_model_pickle(path, file):
    """Load model from pickle file"""
    path_file = os.path.join(path, file)
    model = pickle.load(open(path_file, 'rb'))
    return model


#################### LOAD DATA FROM GOOGLE DRIVE ###################

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(file, sheet_name, **kwargs):
    df = pd.read_excel(file, sheet_name=sheet_name, **kwargs)
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_model(file):
    """Load model from pickle file"""
    model = pickle.load(file)
    return model


# @st.cache_data(show_spinner=False) #3600 seconds
# def authenticate_drive():
#     creds = service_account.Credentials.from_service_account_info(
#         st.secrets["connections_gcs"],
#         scopes=["https://www.googleapis.com/auth/drive.readonly"]
#     )
#     drive_service = build('drive', 'v3', credentials=creds)
    
#     return drive_service

@st.cache_data(ttl=3600, show_spinner=False)
def load_content_drive(file_id, _drive_service):
    """ Load content from google drive
    """
    request = _drive_service.files().get_media(fileId=file_id)
    file_content = io.BytesIO(request.execute())
    
    return file_content
    

@st.cache_data(ttl=3600, show_spinner=False)
def load_data_drive(file_content, sheet_name=None, **kwargs):
    """ Load data using file_content    
    """
    if sheet_name is None:
        df = pd.read_excel(file_content, **kwargs)
    else:
        df = pd.read_excel(file_content, sheet_name=sheet_name, **kwargs)
    
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_model_drive(file_content):
    """ Load model using file_content
    """
    model = pickle.load(file_content)
    return model


# def files_in_drive(folder_id, drive_service):
#     results = drive_service.files().list(q=f"'{folder_id}' in parents").execute()
#     files_dict= results.get('files', [])
    
#     return files_dict



#################### PASSEWORD #####################

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if "password" in st.session_state and st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True






###################### OTHER ######################

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)

@st.cache_data
def convert_df(df):
# IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')