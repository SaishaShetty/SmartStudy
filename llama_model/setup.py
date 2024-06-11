from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from data_loader import pdf_to_text


# pdf_paths = ["Chemistry.pdf", "Biology.pdf", "Physics.pdf"]
pdf_paths = ["Chemistry.pdf",  "Physics.pdf"]
documents = pdf_to_text(pdf_paths)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                   model_kwargs=model_kwargs)
store = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory='cb'
)
store.persist()
