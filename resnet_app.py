#!/usr/bin/env python
# coding: utf-8

# In[6]:


from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
from PIL import Image

# RESNet modelini yükle
model = ResNet50(weights='imagenet')

# Resim işleme fonksiyonunu tanımla
def img_preprocessing(image_path_):
    img = image.load_img(image_path_, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Streamlit uygulamasını oluştur
def main():
    st.title("Resim Sınıflandırma Uygulaması")
    uploaded_file = st.file_uploader("Resim dosyasını yükleyin", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Yüklenen Resim")
        image_path = "uploaded_image.jpg"
        image.save(image_path)

        img = img_preprocessing(image_path)
        pred = model.predict(img)
        predictions = decode_predictions(pred, top=1)

        st.write("Tahmin Edilen Sınıf: {}".format(predictions[0][0][1]))
        st.write("Tahmin Edilen Olasılık: {:.2f}%".format(predictions[0][0][2] * 100))

if __name__ == "__main__":
    main()


# In[ ]:




