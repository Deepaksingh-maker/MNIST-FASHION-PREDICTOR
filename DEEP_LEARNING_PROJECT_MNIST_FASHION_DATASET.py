import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D, Flatten, Dense
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "fashion_mnist_model.keras"

st.set_page_config(
    page_title="Fashion MNIST Dataset",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
    <div class='hero'>
        <h1>Fashion MNIST Dataset</h1>
        <p>An end-to-end Machine Learning dashboard — from raw data to model inference</p>  
        """ , unsafe_allow_html=True)
@st.cache_data
def load_data():
    (x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1,28,28,1)
    x_test = x_test.reshape(-1,28,28,1)
    return x_train,y_train,x_test,y_test
@st.cache_resource
def train_model(x_train,y_train):
    model = Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        Conv2D(32,(3,3),activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64,activation='relu'),
        Dense(32,activation='relu'),
        Dense(16,activation='relu'),
        Dense(10,activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train,y_train,epochs=5,batch_size=32,validation_split=0.2)
    model.save(MODEL_PATH)
    return model

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def prepare_uploaded_image(file):
    image = tf.keras.preprocessing.image.load_img(
        file, target_size=(28, 28), color_mode="grayscale")
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array.astype("float32") / 255.0

    # Fashion-MNIST has a dark background and a light object. Most uploaded
    # photos use the opposite colors, so normalize that difference.
    border = np.concatenate([
        image_array[0, :, 0],
        image_array[-1, :, 0],
        image_array[:, 0, 0],
        image_array[:, -1, 0],
    ])
    if np.mean(border) > 0.5:
        image_array = 1.0 - image_array

    return image, np.expand_dims(image_array, axis=0)

def get_demo_image(x_test, y_test, label):
    demo_index = np.flatnonzero(y_test == label)[0]
    return x_test[demo_index], demo_index

x_train,y_train,x_test,y_test = load_data()
model = None
if MODEL_PATH.exists():
    model = load_model()
    st.sidebar.success("✅ Saved Model Loaded")
else:
    if st.sidebar.button("🚀 Train Model"):
        with st.spinner("Training model... Please wait."):
            model = train_model(x_train,y_train)
    else:
        st.sidebar.info("Click the button to train the model.")

if model is not None:
    loss, accuracy = model.evaluate(x_test,y_test, verbose=0)
    st.sidebar.info(f"Test Loss: {loss:.4f}")
    st.sidebar.info(f"Test Accuracy: {accuracy:.4f}")

    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    input_mode = st.radio(
        "Choose an image source",
        ["Upload your image", "Demo: T-shirt", "Demo: Sneaker"],
        horizontal=True,
    )

    image = None
    image_array = None
    if input_mode == "Upload your image":
        file = st.file_uploader(
            "Upload an image for prediction",
            type=["png", "jpg", "jpeg"])
        if file is not None:
            image, image_array = prepare_uploaded_image(file)
    else:
        demo_label = 0 if input_mode == "Demo: T-shirt" else 7
        demo_image, demo_index = get_demo_image(x_test, y_test, demo_label)
        image_array = np.expand_dims(demo_image, axis=0)
        image = (demo_image[:, :, 0] * 255).astype("uint8")
        st.caption(
            f"Demo image from the Fashion-MNIST test set "
            f"(actual class: {class_names[demo_label]}, sample #{demo_index}).")

    if image_array is not None:
        prediction = model.predict(image_array, verbose=0)
        predicted_class = np.argmax(prediction)
        st.subheader("Prediction Result")
        st.info(f"Predicted Class: {class_names[predicted_class]}")
        st.info(f"Prediction Confidence: {np.max(prediction) * 100:.2f}%")
        st.image(image, caption="Image used for prediction", width=200)
    else:
        st.info("Upload an image or choose one of the demo images above.")
        

 
