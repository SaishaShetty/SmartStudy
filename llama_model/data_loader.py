import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re


def pdf_to_text(pdf_paths):
    documents = []
    for pdf_path in pdf_paths:
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                paragraphs = re.split(r'\n\n+', text)

                doc_list = [Document(page_content=paragraph)
                            for paragraph in paragraphs]
                documents.extend(doc_list)
        except KeyError as e:
            print(f"Error processing {pdf_path}: {e}")
        except Exception as e:
            print(f"An error occurred while processing {pdf_path}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False
    )
    split_documents = text_splitter.split_documents(documents)

    return split_documents


if __name__ == "__main__":
    pdf_paths = ["Chemistry.pdf", "Biology.pdf",
                 "Physics.pdf"]  # need the downloaded pdf
    documents = pdf_to_text(pdf_paths)
