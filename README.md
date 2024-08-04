# Health-Harvest
"Health Harvest Specialist" project comes from the growing need for accessible, comprehensive, and user-friendly nutritional information about fruits and vegetables.
## Inspiration
To be successful in life and happy with your family- health play a vital part in our life. To improve our health, we rely on the green vegetables, fruits, and fibers. People has to search for different website or sources to understand its benefits, drawback, who can it, who can't it. All the foods are not suitable for everyone. In order to overcome these hurdles, we motivated to create an app on **Health harvest specialist** using **streamlit tool**. **We aimed to create a tool that not only classifies produce but also educates users about the health benefits and dietary guidelines associated with each item**. The goal is to help people make informed choices about their diet and enhance their overall well-being.

## What it does
Initially it will classify the vegetables and fruits. After that it provide nutritional information about the items.  Users receive insights into its health benefits, potential drawbacks, recommended ways to eat it, and essential vitamins and minerals. The application also supports multiple languages, offering translations and audio guides for a diverse audience
## How we built it
**Image Classification**: The pytorch framework is used to create a simple cnn architecture for classification. The dataset is taken from [https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition]. The model is generated and tested in jupyter notebook. 
**Data Handling**: Create a new json file to store the nutritional information of the vegetables and fruits
**Translation and Audio**: Integrated Google Translator for translating information into different languages and gTTS (Google Text-to-Speech) to generate audio guides for the translated text.
**Streamlit**: Finally, interface into streamlit app.
## Challenges we ran into
1. The conversion of finalized information into different language and audio
## Accomplishments that we're proud of
we use pytorch to create simple cnn which produce 93% accuracy with in 10 epoch. The converting the text and audio into multilanguage.
## What we learned
We learn detail layers in the pytorch, usage of Google Text-to-Speech, audio conversion. The streamlit is new for us.
## What's next for Health Harvest Specialist
So there is 36 classes but i searched for some other fiber and healthy items but there is no enough dataset on it. so we are planning to add some extra nutrition items, food varieties in future.  Make all this as **One Stop app for nutrition**
## How to run
streamlit run app.py