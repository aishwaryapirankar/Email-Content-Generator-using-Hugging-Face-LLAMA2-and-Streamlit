import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

def getLLMResponse(form_input,email_sender,email_recipient,email_style):
    llm = CTransformers(model='llama-2-7b-chat.ggmlv3.q8_0.bin',     #https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin
                    model_type='llama',
                    config={'max_new_tokens': 256,
                            'temperature': 0.01})
    
    template = """Write a email with {style} style and includes topic :{email_topic}.\n\nSender: {sender}\nRecipient: {recipient} \n\nEmail Text:"""

    prompt = PromptTemplate(input_variables=["style", "email_topic", "sender", "recipient"], template=template,)

    response=llm(prompt.format(email_topic = form_input, sender = email_sender, recipient = email_recipient, style = email_style))
    print(response)

    return response


st.set_page_config(page_title = "Email Content Generator",
                    page_icon = '📧',
                    layout='centered',
                    initial_sidebar_state='collapsed')
st.header("Email Content Generator 📧")

form_input = st.text_area('Enter the topic of the email', height=275)

col1, col2, col3 = st.columns([10, 10, 5])
with col1:
    email_sender = st.text_input('Sender Name')
with col2:
    email_recipient = st.text_input('Recipient Name')
with col3:
    email_style = st.selectbox('Tone of the email', ('Formal', 'Appreciating', 'Not Satisfied', 'Neutral'), index=0)

submit = st.button("Generate")

if submit:
    st.write(getLLMResponse(form_input, email_sender, email_recipient, email_style))
