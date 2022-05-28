# Example origin:
# https://habr.com/ru/post/664076/

# Source code:
# https://github.com/sozykin/streamlit_demo_app

# to run use ".\" before filename: 
# streamlit run .\vehicle-detection_app.py


import io
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow
import cv2


def load_image():
    """Создание формы для загрузки изображения"""
    # Форма для загрузки изображения средствами Streamlit
    uploaded_file = st.file_uploader(
        label='Виберіть зображення для розпізнавання')
    if uploaded_file is not None:
        # Получение загруженного изображения
        image_data = uploaded_file.getvalue()
        # Показ загруженного изображения на Web-странице средствами Streamlit
        st.image(image_data)
        # Возврат изображения в формате PIL
        return Image.open(io.BytesIO(image_data))
    else:
        return None

# загружает нейронную сеть

def load_model():
    model = tensorflow.keras.models.load_model('CNN_mobilenet_save.pkl')
    return model

# выполняет предварительную обработку изображения для подготовки к распознаванию

def preprocess_image(img):
    img = np.array(img); # PIL to BGR
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # model was trained on RGB images so convert to RGB
    img=cv2.resize(img, (75,75))                    # model was trained on images of size 64  X 64 X 3 so resize the images
    img=img/255                                     # model was trained with pixel value scalled between -1 to +1 so convert the pixel range    
    img=np.expand_dims(img, axis=0)                 # model predict expects the input to have dimension (batch_size, width, height, bands)
    
    return img

# печатает названия и вероятность для ТОП 3 классов, выданных моделью
def print_predictions(prediction):
    if np.round(prediction) == 1:                       
        st.write("Це транспортний засіб з імовірністю ", prediction[0][0])
    else:
        st.write("Це не транспортний засіб з імовірністю ", 1-prediction[0][0])    
    


###########################################################################

# Загружаем предварительно обученную модель
model = load_model()

# Выводим заголовок страницы
st.title('Розпізнавання транспортних засобів')

# Выводим форму загрузки изображения и получаем изображение
img = load_image()

# Показывам кнопку для запуска распознавания изображения
result = st.button('Розпізнати зображення')

# Если кнопка нажата, то запускаем распознавание изображения
if result:
    # Предварительная обработка изображения
    x = preprocess_image(img)

    # Распознавание изображения
    preds = model.predict(x)

    # Выводим заголовок результатов распознавания жирным шрифтом
    # используя форматирование Markdown
    st.write('**Результати розпізнавання:**')

    # Выводим результаты распознавания
    print_predictions(preds)
