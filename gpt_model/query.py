from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import os
import pprint
import nltk
from nltk.translate.bleu_score import sentence_bleu

os.environ['OPENAI_API_KEY'] = '<your_token>'
nltk.download('punkt')
embeddings = OpenAIEmbeddings()

store = Chroma(
    persist_directory='cb',
    embedding_function=embeddings
)

template = """
Use the following context to answer the question:
Context: {context}
Question: {question}
"""
Prompt = PromptTemplate(template=template, input_variables=['context', 'question'])


llm = ChatOpenAI(temperature=0, model='gpt-4')

qa_with_source = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=store.as_retriever(),
    chain_type_kwargs={"prompt": Prompt},
    return_source_documents=True
)

def ask_question_bleu(question, references):
    answer = qa_with_source(question)
    generated_text = answer['result']

    reference_texts = [nltk.word_tokenize(ref) for ref in references]
    generated_tokens = nltk.word_tokenize(generated_text)
    bleu_score = sentence_bleu(reference_texts, generated_tokens)

    print("Generated Answer:")
    pprint.pprint(generated_text)
    print("\nBLEU Score:", bleu_score)
    return answer

def ask_question(question):
    answer = qa_with_source(question)
    generated_text = answer['result']

    print("Generated Answer:")
    pprint.pprint(generated_text)
    return answer

# Example question
"""question = "Most fungi get organic compounds from what?"
ask_question_bleu(question)"""

question = "What protects a developing flower while it is still a bud?"
#references = ["it is 100 Celsius"]
ask_question(question)
