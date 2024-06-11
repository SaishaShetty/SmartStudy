from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import os
import pprint
import nltk
from nltk.translate.bleu_score import sentence_bleu
from data_loader import pdf_to_text
from datasets import load_dataset
import random
# load dataset (questions)
dataset = load_dataset("allenai/sciq", split="test")
dataset = dataset.select(range(1000))
all_questions = [doc["question"] for doc in dataset]
all_ground_truth_answers = [doc["correct_answer"] for doc in dataset]

pdf_paths = ["Chemistry.pdf", "Physics.pdf"]
documents = pdf_to_text(pdf_paths)
os.environ['OPENAI_API_KEY'] = 'sk-proj-AvYq8YNz9wT73y3Cw83aT3BlbkFJJ2QNh7nnpsBGOHq2Ry31'
nltk.download('punkt')
# initialize the bm25 retriever and faiss retriever
bm25_retriever = BM25Retriever.from_documents(
    documents
)
bm25_retriever.k = 2

embedding = OpenAIEmbeddings()
faiss_vectorstore = FAISS.from_documents(
    documents,
    embedding
)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})
# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)


template = """
Use the following context to answer the question:
Context: {context}
Question: {question}
"""
Prompt = PromptTemplate(template=template, input_variables=[
                        'context', 'question'])


llm = ChatOpenAI(temperature=0, model='gpt-4')

qa_with_source = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=ensemble_retriever,
    chain_type_kwargs={"prompt": Prompt},
    return_source_documents=True
)


def ask_question_bleu(question, references):
    answer = qa_with_source.invoke(question)
    generated_text = answer['result']

    reference_texts = [nltk.word_tokenize(ref) for ref in references]
    generated_tokens = nltk.word_tokenize(generated_text)
    bleu_score = sentence_bleu(reference_texts, generated_tokens)

    print("Generated Answer:")
    pprint.pprint(generated_text)
    print("\nBLEU Score:", bleu_score)
    return answer, bleu_score


def ask_question(question):
    answer = qa_with_source.invoke(question)
    generated_text = answer['result']

    print("Generated Answer:")
    pprint.pprint(generated_text)
    return answer


# Example question
# """question = "Most fungi get organic compounds from what?"
# ask_question_bleu(question)"""

# question = "What protects a developing flower while it is still a bud?"
# question = "What is the unit used to measure air pressure?"
# references = ["it is 100 Celsius"]
# ask_question(question)
# ask_question_bleu(question, ["millibar"])

# Evaluation (BLEU Score)
questions, ground_truth_answers = zip(
    *random.sample(list(zip(all_questions, all_ground_truth_answers)), 100))
# Ask questions and calculate BLEU scores
bleu_scores = []
for question, answer in zip(questions, ground_truth_answers):
    ans, bleu = ask_question_bleu(question, [answer])
    bleu_scores.append(bleu)
average_bleu = sum(bleu_scores) / len(bleu_scores)
print("Average BLEU Score:", average_bleu)
