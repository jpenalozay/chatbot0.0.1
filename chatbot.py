import streamlit as st
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    ServiceContext
)
from llama_index.llms.openai import OpenAI
import os
os.environ["OPENAI_API_KEY"] = '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'


st.header("ChatBot 💬 📚")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Hola, dime como te puedo ayudar."}
    ]


PERSIST_DIR = "./storage"

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Cargando los datos para poder ayudarte. Espere!"):        
        if not os.path.exists(PERSIST_DIR):
            docs = SimpleDirectoryReader("datos").load_data()        
            service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="Eres un experto en el PMBook, tienes el grado de PMP y ademas tienes que contestar las preguntas de manera especificas y con gran detalle."))
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
    return index

index = load_data()

chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)

if prompt := st.chat_input("Su Pregunta!"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
