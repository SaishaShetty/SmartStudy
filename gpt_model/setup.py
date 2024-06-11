from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from data_loader import pdf_to_text
import os


os.environ['OPENAI_API_KEY'] = '<your_api_token>'
pdf_paths = ["Chemistry.pdf", "Biology.pdf", "Physics.pdf"]
documents = pdf_to_text(pdf_paths)

embeddings = OpenAIEmbeddings()

store = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory='cb'
)
store.persist()
