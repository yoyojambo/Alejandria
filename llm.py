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

# from chromaviz import visualize_collection

from prompt_toolkit import PromptSession

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./memory", embedding_function=embedding_function)

# visualize_collection(db._collection)

# exit()

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 40
n_batch = 512

sf = "THIS IS VERY IMPORTANT, PLEASE DO NOT EXTEND FROM THE ANSWER. YOU JUST NEED TO ANSWER THE QUESTION AND THAT'S IT"

# llm_src = LlamaCpp(
#     model_path="./llama-2-7b.Q4_K_M.gguf",
#     n_gpu_layers=n_gpu_layers,
#     n_batch=n_batch,
#     callback_manager=callback_manager,
#     n_ctx=4096,
#     suffix=sf,
#     temperature=0.5,
#     verbose=False
# )

llm_src = GPT4All(
    model="./ggml-model-gpt4all-falcon-q4_0.bin",
    embedding=True,
    backend="LLaMA",
    # n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    # n_ctx=4096,
    # suffix=sf,
    temp=0.6,
    verbose=False
)

# huggingfacehub_api_token = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# llm_src = HuggingFaceHub(
#     huggingfacehub_api_token = huggingfacehub_api_token, 
#     repo_id = "databricks/dolly-v2-3b", 
#     model_kwargs = {
#         "temperature": 0.6,
#         "max_new_tokens": 250 
#     }
# )
#

# question = "¿Como se relaciona el caso de Southwest Airlines y los factores criticos de exito? ¿Cuales factores crees que aprovecharon?"

# prompt_template = """
# You are an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. Below is some information. 
# {context}
#
# Based on the above information only, answer the below question. 
#
# {question}
# """

# crp = db.similarity_search(
#         question,
#         k = 1,
#         search_type = "mmr",
#         score_threshold = 0.8
#     )



crp = db.as_retriever(
    search_type = "mmr",
    search_kwargs={
        "score_threshold": 0.9,
        "k": 5
    }
)

# class CustomRetriever(VectorStoreRetriever):
#     vectorstore: VectorStoreRetriever
#     search_type: str = "similarity"
#     search_kwargs: dict = Field(default_factory=dict)
#
#     def get_relevant_documents(self, query: str) -> List[Document]:
#         results = self.vectorstore.get_relevant_documents(query=query)
#         
#         print( len(results) )
#
#         return results

# ccrp = CustomRetriever(crp)

# print(rdocs)

# f = open("esther.txt", "w")
#

#     f.write( i.page_content )
#
#     f.write("\n\n------------------------------\n\n")
#
# f.close()

qa = RetrievalQA.from_chain_type(
    llm = llm_src,
    chain_type = "stuff",
    retriever = crp,
    verbose = False
)


question = ""

while question != "q":
    question = PromptSession(message='tu pregunta: ').prompt()

    rdocs = crp.get_relevant_documents(question)

    start_time = time.time()

    response = qa({
        "query": question
    })

    print("--> the answer is:")
    print(response["result"])

    print("--- %s seconds ---" % (time.time() - start_time))

    for i in rdocs:
        print("source: ", i.metadata['source'])
        print("page: ", i.metadata['page'])

exit()
####

qa_chain = create_qa_with_sources_chain(llm_src)

doc_prompt = PromptTemplate(
    template="Content: {page_content}\nSource: {source} \n Page:{page}", # look at the prompt does have page#
    input_variables=["page_content", "source","page"],
)

final_qa_chain = StuffDocumentsChain(
    llm_chain=qa_chain, 
    document_variable_name='context',
    document_prompt=doc_prompt,
)

retrieval_qa = RetrievalQA(
    retriever=db.as_retriever(),
    combine_documents_chain=final_qa_chain
)

query = "Which pizza is synonymous of street food culture?"
ans = retrieval_qa.run(query)

