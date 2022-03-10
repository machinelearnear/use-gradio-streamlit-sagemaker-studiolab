# Use Gradio or Streamlit on SageMaker Studio Lab

This repository shows a quick demo for how to run `Gradio` or `Streamlit` applications on SageMaker Studio Lab. Following the same capability for [Tensorboard](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-tensorboard.html) on SageMaker Studio, you can now apply the same to work with your Streamlit/Gradio application, except the default port (8051) set by Streamlit is not open.

https://user-images.githubusercontent.com/78419164/157738783-93bbbabd-08a0-42e7-ab27-68d43c6559d5.mov

## Getting started
- https://streamlit.io/
- https://gradio.app/
- https://github.com/nicolasmetallo/deploy-streamlit-on-fargate-with-aws-cdk
- https://huggingface.co/spaces/keras-io/Monocular-Depth-Estimation
- https://docs.aws.amazon.com/sagemaker/latest/dg/studio-tensorboard.html

## Requirements
- [SageMaker Studio Lab](https://studiolab.sagemaker.aws/) account. [See video](https://www.youtube.com/watch?v=FUEIwAsrMP4&ab_channel=machinelearnear) for more information.
- Python 3.8+
- Keras
- Numpy
- Streamlit
- Gradio

## How to run your apps

- `0_demo_notebook.ipynb` Notebook that runs Monocular Depth Estimation in Keras.
- `1_launch_gradio_streamlit.ipynb` Notebook with quick start to launch your apps.
- `app_gradio.py` Gradio application
- `app_streamlit.py` Streamlit application

### Gradio

Use `inline=False` and `server_port=6006`.

```python
gr.Interface(
    fn=infer,
    title="Monocular Depth Estimation",
    description = "Keras Implementation of Unet architecture with Densenet201 backbone for estimating the depth of image 📏",
    inputs=[gr.inputs.Image(label="image", type="numpy", shape=(640, 480))],
    outputs="image",
    article = "Author: <a href=\"https://huggingface.co/vumichien\">Vu Minh Chien</a>. Based on the Keras example from <a href=\"https://keras.io/examples/vision/depth_estimation/\">Victor Basu</a>. Repo: https://github.com/machinelearnear/use-gradio-streamlit-sagemaker-studiolab",
    examples=examples).launch(inline=False, server_port=6006, debug=True, cache_examples=True)
```

Then run with `!python app_gradio.py` either from the Terminal or from the `1_launch_gradio_streamlit.ipynb` Notebook.


### Streamlit

Use `server.port 6006` and run from the Terminal or from the `1_launch_gradio_streamlit.ipynb` Notebook.

```sh
!streamlit run app_streamlit.py --server.port 6006 # or 80/8080
```

## References
See more implementations here https://paperswithcode.com/task/monocular-depth-estimation
