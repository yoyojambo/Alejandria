import os

if not os.path.isdir("./memory"):
    raise Exception("Crea tu base de datos!")

from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

import time

from prompt_toolkit import PromptSession

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./memory", embedding_function=embedding_function)

callback_manager = CallbackManager([])

n_gpu_layers = 40
n_batch = 512

llm_src = GPT4All(
    model="./ggml-model-gpt4all-falcon-q4_0.bin",
    embedding=True,
    backend="LLaMA",
    n_batch=n_batch,
    callback_manager=callback_manager,
    temp=0.6,
    verbose=False
)

crp = db.as_retriever(
    search_type = "mmr",
    search_kwargs={
        "score_threshold": 0.9,
        "k": 5
    }
)

qa = RetrievalQA.from_chain_type(
    llm = llm_src,
    chain_type = "stuff",
    retriever = crp,
    verbose = False
)

os.system('clear')

question = ""

while question != "q":
    question = PromptSession(message='>>> Tu pregunta: ').prompt()

    if question == "q":
        break

    if question == "":
        continue

    rdocs = crp.get_relevant_documents(question)

    start_time = time.time()

    response = qa({
        "query": question
    })

    print("-> La respuesta es:")
    print(response["result"])

    print("-> %s segundos" % ( round(time.time() - start_time) ) )

    print("-> fuentes:")

    for i in rdocs:
        print( f"--> {i.metadata['source']} pagina {i.metadata['page']}"  )

    print("\n-------------------\n")
