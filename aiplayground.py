import transformers
import torch
import streamlit as st
import time

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
st.set_page_config(layout="wide")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: visible;}
            footer:before {
               content:'Copyright @ 2022 Siew Kai Ming';
               display:block;
               position:relative;
               color:tomato;
            }
            #welcome-to-our-aiplayground {font-size: 3rem} !important
            span.css-10trblm {font-size: 1.5rem}
            .css-8hc8vl {font-size: 1.1rem; font-family: "Source Code Pro", monospace;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

import base64
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-repeat: no-repeat;
    background-size: cover;
    background-position: center center;
    background-attachment: fixed;
    background-color:#3d3d3d;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('./images/bkgnd.png')


# caching using singleton for non-data object
# preparing pre-trained model for code generator
@st.experimental_singleton
def code_load_data():
   tokenizer_code = AutoTokenizer.from_pretrained("AhmedSSoliman/MarianCG_NL-to-Code")
   model_code = AutoModelForSeq2SeqLM.from_pretrained("AhmedSSoliman/MarianCG_NL-to-Code")
   return tokenizer_code, model_code


# caching using singleton for non-data object
# preparing pre-trained model for chat bot
@st.experimental_singleton
def chat_load_data():    
     tokenizer_chat = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
     model_chat = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
     return tokenizer_chat, model_chat

st.header('Welcome to our aiPlayground')
st.warning('Loading the models for the first time will take a bit of time. So please be patient.')

tokenizer_code, model_code = code_load_data()
tokenizer_chat, model_chat = chat_load_data()


# splitting into two columns
col1,col2 = st.columns(2)

# column one for code generator
col1.markdown('#### Code Generator')
col1.markdown('The generator can create code structure snippets based on natural language input.')
input_code = col1.text_area('Sentence to Code', value='convert a list to dictionary in python')
output_code = model_code.generate(**tokenizer_code(input_code, padding="max_length", truncation=True, max_length=512, return_tensors='pt'))
output_code = tokenizer_code.decode(output_code[0], skip_special_tokens=True)
col1.text_area('Codebot', value=output_code)

expander = col1.expander("See explanation")
expander.write("""
    This is a fine tuned transformer model that was based on CMU CoNaLa, 
    the Code/Natural Language Challenge, a joint project of the Carnegie Mellon University NeuLab and STRUDEL Lab!
    \nThe challenge was designed to test systems for generating program snippets 
    from natural language.
    \nFind out more from
    \nhttps://conala-corpus.github.io/
""")


# column two for chat bot
col2.markdown('#### Let\'s Get Chatty')
col2.markdown('Chatty bot can chit chat with you, but it may politely mumble if it does not have an answer.')
input = col2.text_area('You', value = 'What is haiku?')
# playing with sessions
if 'count' not in st.session_state or st.session_state.count == 6:
    st.session_state.count = 0 
    st.session_state.chat_history_ids = None
    st.session_state.old_response = ''
else:
     st.session_state.count += 1

new_user_input_ids = tokenizer_chat.encode(input + tokenizer_chat.eos_token, return_tensors='pt')
bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if st.session_state.count > 1 else new_user_input_ids
st.session_state.chat_history_ids = model_chat.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer_chat.eos_token_id)
response = tokenizer_chat.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

col2.text_area('Chatty bot', value=response)
st.session_state.old_response = response
expander = col2.expander("See explanation")
expander.write("""
    This chatbot uses the Microsoft DialoGPT model and can only handle simple conversation.
    \nIt is said the responses generated from DialoGPT is comparable to human response. But is rather limited.
    \nThe model was trained on 147M multi-turn dialogue from Reddit discussion thread.
""")