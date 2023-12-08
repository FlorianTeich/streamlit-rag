import streamlit as st
import random
import time
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings.llamacpp import LlamaCppEmbeddings
from langchain import hub
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from transformers import pipeline


st.title("🔥 Private RAG")

with st.sidebar:
    temperature = st.number_input('Temperature',
                                    value=0.8,
                                    min_value=0.0,
                                    max_value=1.0)
    max_tokens = st.number_input('Temperature',
                                    value=1000,
                                    min_value=2,
                                    max_value=1024)
    relevant_docs = st.number_input('Relevant Documents',
                                    value=2,
                                    min_value=0,
                                    max_value=1000)
    st.session_state.n_gpu_layers = st.number_input('GPU Layers',
                                    value=0,
                                    min_value=0,
                                    max_value=128)
    context_length = st.number_input('Context Length',
                                    value=1024,
                                    min_value=0,)
    chunk_size = st.number_input('Chunk Size',
                                value=1000,
                                min_value=0)
    chunk_overlap = st.number_input('Chunk Size',
                                    value=200,
                                    min_value=0,
                                    max_value=context_length)
    embedding_option = st.selectbox(
        'Embedding Model',
        ('all-MiniLM-L6-v2', ))

    llm_option = st.text_input('LLM Model',
                               value='./zephyr-7b-beta.Q4_0.gguf')

    embeddings = HuggingFaceEmbeddings(model_name=embedding_option)

st.session_state.llm = LlamaCpp(model_path=llm_option,
                temperature=temperature,
                max_tokens=max_tokens,
                n_ctx=context_length,
                n_gpu_layers=st.session_state.n_gpu_layers,
                verbose=False,
)

vectorstore = Chroma(
    persist_directory="./chromadb",
    embedding_function=embeddings)
vectorstore.persist()

promptt = hub.pull("rlm/rag-prompt")

retriever = vectorstore.as_retriever(search_type="similarity",
                                     search_kwargs={"k": relevant_docs})

uploaded_files = st.file_uploader("Choose Text files",
                                  accept_multiple_files=True)
for uploaded_file in uploaded_files:
    lines = uploaded_file.readlines()
    lines = [x.decode('UTF-8') for x in lines]
    docs = [Document(page_content="".join(lines))]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    print("Splits:", len(splits))
    progress_text = "Calculate Embeddings. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for num, split in enumerate(splits):
        vectorstore.add_documents(splits)
        my_bar.progress((num + 1) / len(splits),
                        text=progress_text)
    vectorstore.persist()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | promptt
            | st.session_state.llm
            | StrOutputParser()
        )

        for chunk in rag_chain.stream(prompt):
            full_response = full_response + chunk
            message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})