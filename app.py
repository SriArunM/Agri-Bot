# Import Required Libraries
# import langdetect
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
import os
import streamlit as st
from streamlit_chat import message
from langchain_community.document_loaders import OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import CohereEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import Cohere

# from langchain_cohere import CohereEmbeddings

# from langdetect import DetectorFactory, detect_langs, lang_detect_exception

st.set_page_config(page_title="🌴 Eco-AgriXpert ", page_icon=":smile:")

if not os.path.exists("./Data"):
    os.makedirs("./Data")

pdf_file_path = "Data/pmt.pdf"

pdf_file_path2 = "Data/Chemicals.pdf"
pdf_file_path3 = "Data/agrigpt.pdf"

st.markdown(
    "<h1 style='text-align: center; background-color:#85B88F; padding: 20px; color: white; border-radius: 10px;'>ECO-AGRI BOT 🌴</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
        body {
            background-color: #2b2727;
        }
        .reportview-container>.main>.block-container {
            background-color: #2b2727;
            border-radius: 10px;
            padding: 20px;
            box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 12px;
            backdrop-filter: blur(10px);
        }
        .stTextInput>div>div>input {
            border: 1px solid #85B88F;
            border-radius: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
<style>
    .message-container {
        background-color: transparent;
        padding: 0;
        margin: 0;
    }
    .message-content {
        background-color: #F0F0F0;
        border-radius: 10px;
        padding: 10px;
    }
    .message-content.is-user {
        background-color: #85B88F;
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialzing Text Splitter
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=10)

# Intializing Cohere Embdedding
embeddings = CohereEmbeddings(
    model="large", user_agent="ui2", cohere_api_key=st.secrets["cohere_apikey"]
)


def PDF_loader(document):
    loader = OnlinePDFLoader(document)
    documents = loader.load()
    prompt_template = """ 
    You are an AI Chatbot developed to help users by suggesting eco-friendly farming methods, alternatives to chemical pesticides and fertilizers, and maximizing profits. Use the following pieces of context to answer the question at the end. Greet Users!!
    {context}

    {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    texts = text_splitter.split_documents(documents)
    global db
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    global qa
    qa = RetrievalQA.from_chain_type(
        llm=Cohere(
            model="command-xlarge-nightly",
            temperature=0.98,
            cohere_api_key=st.secrets["cohere_apikey"],
        ),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs=chain_type_kwargs,
    )
    return "Ready"


PDF_loader(pdf_file_path)

# Session State
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []

# User Information
user_name = st.sidebar.text_input("Your Name:", key="user_name")
agriculture_type = st.sidebar.selectbox(
    "Type of Agriculture:",
    ["", "Organic", "Conventional", "Biodynamic"],
    key="agriculture_type",
)

welcome_message = f"Hello, {user_name}! I am Your AgriGuardian, and I'm here to help you with eco-friendly farming methods and alternatives to chemical pesticides and fertilizers. I see that you are interested in or practicing {agriculture_type} agriculture. Let's discuss how I can help you further."


# Generating Response
def generate_response(query):
    result = qa({"query": query, "chat_history": st.session_state["chat_history"]})
    result["result"] = result["result"]

    return result["result"]


response_container = st.container()
container = st.container()

with container:
    st.markdown(
        "<style>.stTextInput>div>div>input {border: 1px solid #85B88F;}</style>",
        unsafe_allow_html=True,
    )
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_input("You:", key="input")
        submit_button = st.form_submit_button(label="Send")

        if user_input and submit_button:
            output = generate_response(user_input)
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)
            st.session_state["chat_history"] = [(user_input, output)]

if st.session_state["generated"]:
    with response_container:
        message(
            welcome_message,
            key="welcome_message",
            avatar_style="adventurer",
            seed=123,
        )
        for i in range(len(st.session_state["generated"])):
            message(
                st.session_state["past"][i],
                is_user=True,
                key=str(i) + "_user",
                avatar_style="adventurer",
                seed=123,
            )
            message(st.session_state["generated"][i], key=str(i))

clear_button = st.sidebar.button("Clear Conversation", key="clear")

if clear_button:
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["chat_history"] = []


# Add background from URL
def add_bg_from_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("https://agrierp.com/blog/wp-content/uploads/2024/01/why-is-agriculture-important.webp");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True,
    )


add_bg_from_url()
