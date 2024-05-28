# QA LLM EU Directives

This repository hosts a minimum code example to show how complex directives/regulations can be made easier accesible via a LLM. The LLM serves as vehicle to answer user questions concerning the underlying directive/regulation. Technically Langchain is used to abstract the user interaction with the document and LLM, Sentence Transformers is used to convert the irective/regulation into embeddings, FAISS is used to to store the embeddings in a vector database and a CohereAI LLM is used for answering user questions.