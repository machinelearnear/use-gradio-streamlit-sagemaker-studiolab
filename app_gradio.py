# Use Gradio in Sagemaker Studio Lab
# https://huggingface.co/spaces/keras-io/Monocular-Depth-Estimation/blob/main/app.py

from layers import BilinearUpSampling2D
from tensorflow.keras.models import load_model
from utils import load_images, predict
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
from huggingface_hub import from_pretrained_keras

custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
print('Loading model...')
model = from_pretrained_keras("keras-io/monocular-depth-estimation", custom_objects=custom_objects, compile=False)
print('Successfully loaded model...')
examples = ['examples/00015_colors.png', 'examples/00084_colors.png', 'examples/00033_colors.png']

def infer(image):
    inputs = load_images([image])
    outputs = predict(model, inputs)
    plasma = plt.get_cmap('plasma')
    rescaled = outputs[0][:, :, 0]
    rescaled = rescaled - np.min(rescaled)
    rescaled = rescaled / np.max(rescaled)
    image_out = plasma(rescaled)[:, :, :3]
    return image_out

iface = gr.Interface(
    fn=infer,
    title="Monocular Depth Estimation",
    description = "Keras Implementation of Unet architecture with Densenet201 backbone for estimating the depth of image 📏",
    inputs=[gr.inputs.Image(label="image", type="numpy", shape=(640, 480))],
    outputs="image",
    article = "Author: <a href=\"https://huggingface.co/vumichien\">Vu Minh Chien</a>. Based on the Keras example from <a href=\"https://keras.io/examples/vision/depth_estimation/\">Victor Basu</a>. Repo: https://github.com/machinelearnear/use-gradio-streamlit-sagemaker-studiolab",
    examples=examples).launch(inline=False, server_port=6006, debug=True, cache_examples=True)