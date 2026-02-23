from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import requests
from IPython.display import Audio
import os
import streamlit  as st

load_dotenv(find_dotenv())
HUGGING_FACE_HUB_API_TOKEN = os.getenv("HUGGING_FACE_HUB_API_TOKEN")
OpenAI_key = os.environ.get("OPENAI_API_KEY")


#  Writing a function to geenrate text from image
def convert_img_text(url):
    image_to_text = pipeline("image-to-text", model="salesforce/blip-image-captioning-base")

    '''list of tasks are available at - huggingface.co/tasks '''
    text =image_to_text(url)[0]['generated_text']
    return text

# """ Writing a function to geenrate story from image"""
def generate_story_from_img(scenario):
    template ="""
    A photographer standing near Mt Fuji, Japan to taking pictures for his blog.
    CONTEXT : {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    print(os.getenv("OPENAI_API_KEY"))
    story_llm =LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=1), prompt=prompt,verbose=True)

    story = story_llm.predict(scenario=scenario)
    return story


# """ Writing a function to geenrate story from image"""
def convert_text_speech(message_from_image):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_HUB_API_TOKEN}"}
    payload = {
         "inputs":message_from_image
    }
 
    response = requests.post(API_URL, headers=headers, json=payload)
    with open('gen_audio_from_photo.flac','wb') as file:
        file.write(response.content)


def main():
   
    # Commenting for now
    st.set_page_config(page_title="Story from uploaded image",
                   page_icon="🧬",
                   initial_sidebar_state="expanded")
    st.header("Story from uploaded image")
    file_upload = st.file_uploader("Select an image...", type="jpeg")

    if file_upload is not None:
        bytes_data=file_upload.getvalue()
        with open(file_upload.name, "wb") as file:
            file.write(bytes_data)
        st.image(file_upload, caption="Uploaded Image.",
                 use_column_width=True)
        scenario = convert_img_text(file_upload.name)
        story=generate_story_from_img(scenario)
        convert_text_speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio("gen_audio_from_photo.flac")    


if __name__ == '__main__':
    main()