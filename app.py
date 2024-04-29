# Import streamlit
import streamlit as st
from streamlit_chat import message
from pathlib import Path

from src.helper import read_yaml
from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
config_details=read_yaml(CONFIG_FILE_PATH)

####################################################################
#            WEB Application interface using streamlit  
####################################################################

st.set_page_config(page_title="Chat With Your Document")

st.title("🤖 RAG chatbot")

# API keys
st.session_state.openai_api_key = ""
st.session_state.google_api_key = ""
st.session_state.cohere_api_key = ""
st.session_state.hf_api_key = ""


def expander_model_sidebar(
    LLM_provider="OpenAI",
    text_input_API_key="OpenAI API Key - [Get an API key](https://platform.openai.com/account/api-keys)",
    list_llm_models=["gpt-3.5-turbo-0125", "gpt-3.5-turbo", "gpt-4-turbo-preview"],
):
    '''API KEY sidebar expanders for streamlit with models and respective parameters   '''
    st.session_state.LLM_provider = LLM_provider

    if LLM_provider == "OpenAI":
        st.session_state.openai_api_key = st.text_input(
            text_input_API_key,
            type="password",
            placeholder="insert your API key",
        )
        st.session_state.google_api_key = ""
        st.session_state.hf_api_key = ""

    if LLM_provider == "Google":
        st.session_state.google_api_key = st.text_input(
            text_input_API_key,
            type="password",
            placeholder="insert your API key",
        )
        st.session_state.openai_api_key = ""
        st.session_state.hf_api_key = ""

    if LLM_provider == "HuggingFace":
        st.session_state.hf_api_key = st.text_input(
            text_input_API_key,
            type="password",
            placeholder="insert your API key",
        )
        st.session_state.openai_api_key = ""
        st.session_state.google_api_key = ""

    with st.expander("**Models and parameters**"):
        st.session_state.selected_model = st.selectbox(
            f"Choose {LLM_provider} model", list_llm_models
        )

        # model parameters
        st.session_state.temperature = st.slider(
            "temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
        )
        st.session_state.top_p = st.slider(
            "top_p",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.05,
        )

def sidebar_and_documentSelector():
    """Set up a sidebar and a tabbed pane: the initial tab should feature a document selector for generating 
        a new vectorstore, while the subsequent tab should offer a vectorstore selector for accessing an existing one.."""

    with st.sidebar:
        st.caption(
            "🚀 A retrieval augmented generation chatbot powered by 🔗 Langchain, Cohere, OpenAI, Google Generative AI and 🤗"
        )
        st.write("")
        list_LLM_providers=config_details.LLM_providers  #getting from the config file
        llm_selection = st.radio(
            "Select provider",
            list_LLM_providers,
            captions=[
                "[OpenAI pricing page](https://openai.com/pricing)",
                "Rate limit: 60 requests per minute.",
                "**Free access.**",
            ],
        )

        st.divider()
        if llm_selection == list_LLM_providers[0]:
            expander_model_sidebar(
                LLM_provider="OpenAI",
                text_input_API_key="OpenAI API Key - [Get an API key](https://platform.openai.com/account/api-keys)",
                list_models=config_details.OpenAI_models
            )

        if llm_selection == list_LLM_providers[1]:
            expander_model_sidebar(
                LLM_provider="Google",
                text_input_API_key="Google API Key - [Get an API key](https://makersuite.google.com/app/apikey)",
                list_models=["gemini-pro"],
            )
        if llm_selection == list_LLM_providers[2]:
            expander_model_sidebar(
                LLM_provider="HuggingFace",
                text_input_API_key="HuggingFace API key - [Get an API key](https://huggingface.co/settings/tokens)",
                list_models=["mistralai/Mistral-7B-Instruct-v0.2"],
            )
        # Assistant language
        st.write("")
        st.session_state.assistant_language = st.selectbox(
            f"Assistant language", list(config_details.welcome_message.keys())
        )
        st.divider()
        st.subheader("Retrievers")
        retrievers = config_details.retriever_types
        if st.session_state.selected_model == "gpt-3.5-turbo":
            # for "gpt-3.5-turbo", we will not use the vectorstore backed retriever
            # there is a high risk of exceeding the max tokens limit (4096).
            retrievers = retrievers[:-1]

        st.session_state.retriever_type = st.selectbox(
            f"Select retriever type", retrievers
        )
        st.write("")
        if st.session_state.retriever_type == retrievers[0]:  # Cohere
            st.session_state.cohere_api_key = st.text_input(
                "Coher API Key - [Get an API key](https://dashboard.cohere.com/api-keys)",
                type="password",
                placeholder="insert your API key",
            )

        st.write("\n\n")
        st.write(
            f"ℹ _Your {st.session_state.LLM_provider} API key, '{st.session_state.selected_model}' parameters, \
            and {st.session_state.retriever_type} are only considered when loading or creating a vectorstore._"
        )
         # Tabbed Pane: Create a new Vectorstore | Open a saved Vectorstore

    tab_new_vectorstore, tab_open_vectorstore = st.tabs(
        ["Create a new Vectorstore", "Open a saved Vectorstore"]
    )
    with tab_new_vectorstore:
        # 1. Select documnets
        st.session_state.uploaded_file_list = st.file_uploader(
            label="**Select documents**",
            accept_multiple_files=True,
            type=(["pdf", "txt", "docx", "csv"]),
        )
        # 2. Process documents
        st.session_state.vector_store_name = st.text_input(
            label="**Documents will be loaded, embedded and ingested into a vectorstore (Chroma dB). Please provide a valid dB name.**",
            placeholder="Vectorstore name",
        )
        # 3. Add a button to process documnets and create a Chroma vectorstore
        st.button("Create Vectorstore", on_click=chain_RAG_blocks)
        try:
            if st.session_state.error_message != "":
                st.warning(st.session_state.error_message)
        except:
            pass

