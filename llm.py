import os

if not os.path.isdir("./memory"):
    raise Exception("Crea tu base de datos!")

from langchain.llms import LlamaCpp, GPT4All
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chains import create_qa_with_sources_chain, RetrievalQA, RetrievalQAWithSourcesChain, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager, CallbackManagerForRetrieverRun

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain.schema.retriever import BaseRetriever, Document

from langchain.vectorstores import Chroma

from langchain.vectorstores.base import VectorStoreRetriever

import time

import streamlit as st

from random_word import RandomWords

from falcon import create_ChromaDB, Falcon_LLM

st.write("# Alejandria")


# @st.cache_resource(hash_funcs={Falcon: lambda p: p.hash})
# def db_retriever(f):
#embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#db = Chroma(persist_directory="./memory", embedding_function=embedding_function)



n_batch = 512

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

#llm_src = GPT4All(
#    model="./ggml-model-gpt4all-falcon-q4_0.bin",
#     embedding=True,
#     backend="LLaMA",
#     n_batch=n_batch,
#     callback_manager=callback_manager,
#     temp=0.6,
#     verbose=False
# )


f = Falcon_LLM()
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# files = [*relative paths*]
# loader = PyPDFLoader
# persist_directory = "./memory"
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 512,
#     chunk_overlap  = 0,
#     length_function = len,
#     is_separator_regex = False,
# )

st.write("### Modelo cargado :)")

# crp = f.crp

question = st.text_input(label="Ingresa tu pregunta: ")

# rdocs = crp.get_relevant_documents(question)
response = f.pregunta(question)
st.write("the answer is:")
st.write(response["result"])

#st.write("--- %s seconds ---" % (time.time() - start_time))

st.write("sources:")

# for i in rdocs:
#     st.write("source: ", i.metadata['source'])
#     st.write("page: ", i.metadata['page'])
