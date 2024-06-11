import random
from langchain_community.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import pprint
import nltk
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_dataset


# load  dataset(questions)
dataset = load_dataset("allenai/sciq", split="test")
dataset = dataset.select(range(1000))
all_questions = [doc["question"] for doc in dataset]
all_ground_truth_answers = [doc["correct_answer"] for doc in dataset]

# print(all_questions, all_ground_truth_answers)
nltk.download('punkt')

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                   model_kwargs=model_kwargs)


store = Chroma(
    persist_directory='cb',
    embedding_function=embeddings
)


template = """
You are a helpful assistant that answers questions based on the given document context. You generate short answer in simple words. Use the following context to answer the question:
Context: {context}
Question: {question}
"""
# template = """
# <<SYS>>\n You are a helpful assistant that answers questions based on the given document context. \n<</SYS>>\n\n [INST] Using the following context, please answer the question: {question}.\n\n{context}[/INST]
# """

Prompt = PromptTemplate(template=template, input_variables=[
                        'context', 'question'])

# llama
# download model llama-3-8B gguf https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF
model_path = "/Users/yvonne/Downloads/Meta-Llama-3-8B.Q2_K.gguf"
# model_path = "/Users/yvonne/Downloads/llama-2-7b-chat.Q2_K.gguf"  # llama2 https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=100,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
    temperature=0.8,
    top_p=0.95,
    top_k=10,
    repeat_penalty=1.2,
)

# llm("What protects a developing flower while it is still a bud?")

qa_with_source = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=store.as_retriever(),
    chain_type_kwargs={"prompt": Prompt},
    return_source_documents=True
)


def ask_question_bleu(question, references):
    answer = qa_with_source.invoke(question)
    generated_text = answer['result']
    print("Source Document:", answer["source_documents"])
    reference_texts = [nltk.word_tokenize(ref) for ref in references]
    generated_tokens = nltk.word_tokenize(generated_text)
    bleu_score = sentence_bleu(reference_texts, generated_tokens)

    print("Generated Answer:")
    pprint.pprint(generated_text)
    print("\nBLEU Score:", bleu_score)
    return answer, bleu_score


def ask_question(question):
    answer = qa_with_source(question)
    generated_text = answer['result']
    print("Generated Answer:")
    pprint.pprint(generated_text)
    return answer


# Example question
# """question = "Most fungi get organic compounds from what?"
# ask_question_bleu(question)"""

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
