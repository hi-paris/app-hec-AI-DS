import os
import streamlit as st
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
import cv2 as cv
import imutils

from PIL import Image
from tensorflow.keras.models import load_model

def predictTumor(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv.threshold(gray, 45, 255, cv.THRESH_BINARY)[1]
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    image = cv.resize(new_image, dsize=(240, 240), interpolation=cv.INTER_CUBIC)
    image = image / 255.

    image = image.reshape((1, 240, 240, 3))

    res = model.predict(image)

    return res


st.set_page_config(layout="wide")

st.markdown("# Image Classification")
st.markdown("### What is Image Classification ?")

model = load_model('pretrained_models/brain_tumor_detector.h5', compile=False)

run_model = st.button("**Run the model**", type="primary")

if run_model:
    image = Image.open("data/brain_tumor/Te-me_0015.jpg")
    prediction = predictTumor(image)
    st.markdown(prediction)