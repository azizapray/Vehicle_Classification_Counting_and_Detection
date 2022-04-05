import streamlit as st
import pandas as pd
import tensorflow
import pathlib
import numpy as np

# Inference
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps


def app():
    st.markdown("<h1 style='text-align: center; color: yellow;'>DEMO PREDICTION</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center'>Web-app ini merupakan visualisasi dari hasil project yang telah kami buat mengenai <b>Object Detection<b>.<br>Dataset yang digunakan dalam project ini dapat dilihat pada <a href='https://www.kaggle.com/datasets/rishabkoul1/vechicle-dataset'>Kaggle</a>.</p>", unsafe_allow_html=True)
    st.write('---')

    model_v3 = load_model('model_vehicle_v3.h5')
    model_v4 = load_model('model_vehicle_v4.h5')
    model_v5 = load_model('model_vehicle_v5.h5')
    
    data_dir = pathlib.Path("vehicles/train")
    vehicle = np.array(sorted([item.name for item in data_dir.glob('*')]))
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Upload Foto')
        foto = st.file_uploader('Upload foto/video:', type=['png','jpg'])
        # if foto is not None:
        #     st.image(foto)
        # else:
        #     st.write('Please upload the correct format file.')
    
    # with col2:
    #     st.subheader('Upload Video')
    #     video = st.file_uploader('Upload foto/video:', type=['mp4'])
        # if video is not None:
        #     st.video(video)
        # else:
        #     st.write('Please upload the correct format file.')
    
    def import_and_predict(image_data, model):
        size = (224,224)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        pred = model.predict(img_reshape)
        return pred
    
    if foto:
        image = Image.open(foto)
        st.image(image, use_column_width=True)
        # if st.button('predict!'):
        prediction = import_and_predict(image, model_v3)
        st.write(f"### This image classify as {vehicle[np.argmax(prediction)]}")