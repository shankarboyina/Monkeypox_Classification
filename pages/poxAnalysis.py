
import torch
import streamlit as st
from lib import commons
from PIL import Image

from torchvision import datasets, models, transforms

def app():
    st.title("Pox Detection")
    st.subheader("Test whether an area is affected by pox using ResNet18 or YOLOv8")

    # Model selection dropdown
    model_option = st.selectbox("Choose the model", ["ResNet18", "YOLOv8"])
    image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "jfif"])

    if image_file is not None:
        st.image(Image.open(image_file), width=250)
        model = commons.load_model()
        predictions = commons.predict(model, image_file)
        if model_option == "ResNet18":
                    i=1
                    if predictions[0]!="Monkeypox":
                         
                        st.text("Not a case of Monkey Pox")
                        st.subheader("Pox types arranged in order of probability (highest first):")
                        

                        print(predictions)
                        for pred in predictions:
                            st.text(str(i)+". "+pred)    
                            i+=1            
                    else:
                        st.text("It is a case of Monkey Pox")
            # model = commons.load_model()
            #predictions = commons.predict(model, image_file)
            # # st.write("Predicted Pox Types (ResNet18):", predictions)
        elif model_option == "YOLOv8":
            yolov8_model = commons.load_yolov8_model()
            detected_class, confidences = commons.predict_yolov8(yolov8_model, image_file)
            st.write(f"Detected Condition (YOLOv8): {detected_class}")
            st.write("Confidence Scores:", confidences)
