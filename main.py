from langchain.chains import RetrievalQA
from langchain_cohere.llms import Cohere
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Load from local storage
vectorstore = FAISS.load_local("faiss_index_constitution", embeddings, allow_dangerous_deserialization=True)

# Use RetrievalQA chain for orchestration
qa = RetrievalQA.from_chain_type(llm=Cohere(), chain_type="stuff", retriever=vectorstore.as_retriever())
query="How is affected by the CSRD?"
query="Companies of which size need to obey the Corporate Sustainability Reporting Directive?"
query="Who is the author of the underlying document that you are using in this RAG system?"
query="What is the content of the underlying document in three sentences?"
result = qa.invoke(query)
result
