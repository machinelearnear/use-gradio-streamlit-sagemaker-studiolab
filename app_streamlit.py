# Use Streamlit in Sagemaker Studio Lab
# Author: https://github.com/machinelearnear

# import dependencies
import streamlit as st
import numpy as np
import requests
import io
import json
import base64
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from layers import BilinearUpSampling2D
from tensorflow.keras.models import load_model
from huggingface_hub import from_pretrained_keras
from utils import load_images, predict
from streamlit_image_comparison import image_comparison

# helper funcs
def infer(model, image):
    inputs = load_images([image])
    outputs = predict(model, inputs)
    plasma = plt.get_cmap('plasma')
    rescaled = outputs[0][:, :, 0]
    rescaled = rescaled - np.min(rescaled)
    rescaled = rescaled / np.max(rescaled)
    image_out = plasma(rescaled)[:, :, :3]
    return image_out

def show_images(input_img, output_img):
    f = plt.figure(figsize=(20,20))
    f.add_subplot(1,2,1)
    plt.imshow(input_img)
    f.add_subplot(1,2,2)
    plt.imshow(output_img)
    plt.show(block=True)
    st.pyplot(bbox_inches='tight')

@st.cache(allow_output_mutation=True)
def load_model(keras_model="keras-io/monocular-depth-estimation"):
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
    model = from_pretrained_keras(keras_model, custom_objects=custom_objects, compile=False)
    return model

# streamlit app
# -----------------------------------------------------------
def main():
    with st.spinner(text='Loading model into Keras..'):
        model = load_model() # first things, first.. load your model
    st.title('Monocular Depth Estimation')
    st.header('Introduction')
    st.markdown('''
    Depth estimation is a crucial step towards inferring scene geometry from 2D images. 
    The goal in monocular depth estimation is to predict the depth value of each pixel 
    or inferring depth information, given only a single RGB image as input. 
    This example will show an approach to build a depth estimation model with a 
    convnet and simple loss functions.
    
    This demo is a Keras implementation of U-Net architecture with DenseNet201 backbone for
    estimating monocular depth from RGB images.
    ''')

    st.image(
        'https://cs.nyu.edu/~silberman/images/nyu_depth_v2_web.jpg',
        caption='Source: Indoor Segmentation and Support Inference from RGBD Images ECCV 2012 (NYU Depth Dataset V2)',
        use_column_width=True)

    # read input image
    st.header('Read input image')
    options = st.radio('Please choose any of the following options',
        (
            'Choose example from library',
            'Download image from URL',
            'Upload your own image',
        )
    )

    input_image = None
    if options == 'Choose example from library':
        image_files = list(sorted([x for x in Path('examples').rglob('*.png')]))
        selected_file = st.selectbox(
            'Select an image file from the list', image_files
        )
        st.write(f'You have selected `{selected_file}`')
        input_image = Image.open(selected_file)
    elif options == 'Download image from URL':
        image_url = st.text_input('Image URL')
        try:
            r = requests.get(image_url)
            input_image = Image.open(io.BytesIO(r.content))
        except Exception:
            st.error('There was an error downloading the image. Please check the URL again.')
    elif options == 'Upload your own image':
        uploaded_file = st.file_uploader("Choose file to upload")
        if uploaded_file:
            input_image = Image.open(io.BytesIO(uploaded_file.decode()))
            st.success('Image was successfully uploaded')

    if input_image:
        st.image(input_image, use_column_width=True)
        st.info('Note: Larger images will take longer to process.')
    else:
        st.warning('There is no image loaded.')
                
    # model inference
    st.header('Run your model prediction')
    st.write('')
    if input_image and st.button('Submit'):
        try:
            with st.spinner():
                output = infer(model, np.asarray(input_image))
                output_image = Image.fromarray((output * 255).astype(np.uint8))
                image_comparison(
                    img1=input_image, img2=output_image,
                    label1='Original', label2='Depth Estimation',
                )
        except Exception as e:
            st.error(e)
            st.error('There was an error processing the input image')
    if not input_image: st.warning('There is no image loaded')
    
    # footer
    st.header('References')
    st.markdown('''
    - https://github.com/machinelearnear/use-gradio-streamlit-sagemaker-studiolab
    - https://github.com/nicolasmetallo/deploy-streamlit-on-fargate-with-aws-cdk
    - https://keras.io/examples/vision/depth_estimation/
    ''')

# run application
# -----------------------------------------------------------
if __name__ == '__main__':
    main()