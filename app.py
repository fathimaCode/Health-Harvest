from PIL import Image
import io
import streamlit as st
from streamlit_navigation_bar import st_navbar
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
from deep_translator import GoogleTranslator
from gtts import gTTS
from playsound import playsound
import json
LANGUAGES = {
'English': 'en',
'Afrikaans': 'af',
'Albanian': 'sq',
'Amharic': 'am',
'Arabic': 'ar',
'Armenian': 'hy',
'Azerbaijani': 'az',
'Basque': 'eu',
'Belarusian': 'be',
'Bengali': 'bn',
'Bosnian': 'bs',
'Bulgarian': 'bg',
'Catalan': 'ca',
'Cebuano': 'ceb',
'Chichewa': 'ny',
'Chinese (Simplified)': 'zh-cn',
'Chinese (Traditional)': 'zh-tw',
'Corsican': 'co',
'Croatian': 'hr',
'Czech': 'cs',
'Danish': 'da',
'Dutch': 'nl',
'English': 'en',
'Esperanto': 'eo',
'Estonian': 'et',
'Filipino': 'tl',
'Finnish': 'fi',
'French': 'fr',
'Frisian': 'fy',
'Galician': 'gl',
'Georgian': 'ka',
'German': 'de',
'Greek': 'el',
'Gujarati': 'gu',
'Haitian Creole': 'ht',
'Hausa': 'ha',
'Hawaiian': 'haw',
'Hebrew': 'iw',
'Hindi': 'hi',
'Hmong': 'hmn',
'Hungarian': 'hu',
'Icelandic': 'is',
'Igbo': 'ig',
'Indonesian': 'id',
'Irish': 'ga',
'Italian': 'it',
'Japanese': 'ja',
'Javanese': 'jw',
'Kannada': 'kn',
'Kazakh': 'kk',
'Khmer': 'km',
'Korean': 'ko',
'Kurdish (Kurmanji)': 'ku',
'Kyrgyz': 'ky',
'Lao': 'lo',
'Latin': 'la',
'Latvian': 'lv',
'Lithuanian': 'lt',
'Luxembourgish': 'lb',
'Macedonian': 'mk',
'Malagasy': 'mg',
'Malay': 'ms',
'Malayalam': 'ml',
'Maltese': 'mt',
'Maori': 'mi',
'Marathi': 'mr',
'Mongolian': 'mn',
'Myanmar (Burmese)': 'my',
'Nepali': 'ne',
'Norwegian': 'no',
'Odia': 'or',
'Pashto': 'ps',
'Persian': 'fa',
'Polish': 'pl',
'Portuguese': 'pt',
'Punjabi': 'pa',
'Romanian': 'ro',
'Russian': 'ru',
'Samoan': 'sm',
'Scots Gaelic': 'gd',
'Serbian': 'sr',
'Sesotho': 'st',
'Shona': 'sn',
'Sindhi': 'sd',
'Sinhala': 'si',
'Slovak': 'sk',
'Slovenian': 'sl',
'Somali': 'so',
'Spanish': 'es',
'Sundanese': 'su',
'Swahili': 'sw',
'Swedish': 'sv',
'Tajik': 'tg',
'Tamil': 'ta',
'Telugu': 'te',
'Thai': 'th',
'Turkish': 'tr',
'Ukrainian': 'uk',
'Urdu': 'ur',
'Uyghur': 'ug',
'Uzbek': 'uz',
'Vietnamese': 'vi',
'Welsh': 'cy',
'Xhosa': 'xh',
'Yiddish': 'yi',
'Yoruba': 'yo',
'Zulu': 'zu'
}
def translate_lang(text_to_translate, lang):
    translated_text = GoogleTranslator(source='auto', target=lang).translate(text_to_translate)
    return translated_text

def word_audio(translation, language):
    print(f"Translation in {language}: {translation}")
    tts = gTTS(text=translation, lang=language)
    file_path = f"play/{language}.mp3"
    tts.save(file_path)

def convert_response(data,language_to_convert):
    final_data = []
    name =f"name: {data[0]['name']}" 
    scientific = f"scientific name: {data[0]['scientific_name']}"
    benefits = f"benefits: {', '.join(data[0]['benefits'])}"
    bad_things = f"Drawback: {', '.join(data[0]['bad_things'])}"
    how_to_eat = f"how to eat: {', '.join(data[0]['how_to_eat'])}"
    who_shouldnt_eat = f"who_shouldnt_eat: {', '.join(data[0]['who_shouldnt_eat'])}"
    vitamins_minerals = f"vitamins_minerals: {', '.join(data[0]['vitamins_minerals'])}"
    daily_effect = f"daily_effect: {', '.join(data[0]['daily_effect'])}"

    final_data.append(name)
    final_data.append(scientific)
    final_data.append(benefits)
    final_data.append(bad_things)
    final_data.append(how_to_eat)
    final_data.append(who_shouldnt_eat)
    final_data.append(vitamins_minerals)
    final_data.append(daily_effect)
    converted_text = []
    for data in final_data:
        translated_text = translate_lang(data,language_to_convert)
        converted_text.append(translated_text)


    print(f"final:{converted_text}")
    word_audio(', '.join(converted_text),language_to_convert)
    print(f"audio completed")
    return converted_text
    


if 'class_predicted' not in st.session_state:
    st.session_state['class_predicted'] = None
# Define the model architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load and preprocess the image
def load_and_preprocess_image(image):
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def load_model(model_path, num_classes):
    model = SimpleCNN(num_classes=num_classes)  
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  
    return model

model_path = "F:/2024/hackthon/fruit_Classification/best_model.pth"
num_classes = 36  
model = load_model(model_path, num_classes)
device = torch.device("cpu")
model.to(device)

path = "F:/2024/hackthon/fruit_Classification/"

def show_home():
    st.title("Fruit & Veggie Dietician")
    st.write("This project provides users with comprehensive classification, benefits, uses, and dietary guidelines.")
    st.write("Data source: https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition")
    st.write("Image classes: 'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon'")
    st.image(path + 'assets/1.jpg', caption='Fruit & Veggie Dietician',  width=500)
    st.image(path + 'assets/2.png', caption='Sample Input Images',  use_column_width=True)



def show_testing():
    global class_predicted
    st.title("Testing")
    st.write("Model loaded successfully.")     
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image',  width=500)
        st.write("File name:", uploaded_file.name)
        st.write("File type:", uploaded_file.type)
        st.write("File size:", uploaded_file.size, "bytes")

        # Preprocess the image
        image = load_and_preprocess_image(image)
        image = image.to(device)

        # Run inference
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)

        # Load class names if available
        class_names = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']
        predicted_class_index = predicted.item()
        predicted_class_label = class_names[predicted_class_index]
        st.session_state['class_predicted'] = predicted_class_label
        st.write("Predicted Class:", predicted_class_label)

def load_info(name):
    
    with open('data.json','r') as json_data:
        out = json.load(json_data)
    info = [item for item in out if item['name']==name]
    return info


def show_info():
    st.title("More Info")
    if st.session_state['class_predicted']:
        st.write(f"Classes: {st.session_state['class_predicted']}")
        info = load_info(st.session_state['class_predicted'])
        selected_audio = st.selectbox('Select an language', options=list(LANGUAGES.keys()))
        lang_key = LANGUAGES[selected_audio]
        print(lang_key)
        translate_word = convert_response(info,lang_key)
        if not translate_word:
            st.write("Waiting to get Information 1min")
        else:
            st.audio(f'play/{lang_key}.mp3', format='audio/mp3')
            for word in translate_word:
                st.write(word)
    else:
        st.write("No class predicted yet, please go to the testing page and upload an image")

    
        
        
        

     
pages = ["Home","Testing","More Info"]
page = st_navbar(pages)




if page == "Home":
    show_home()

elif page == "Testing":
    show_testing()
elif page == "More Info":
    show_info()
else:
    st.write("Page not found")
