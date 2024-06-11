from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import pandas as pd
import os
import pprint
import nltk
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_dataset
import random
# load  dataset(questions)
dataset = load_dataset("allenai/sciq", split="test")
dataset = dataset.select(range(1000))
all_questions = [doc["question"] for doc in dataset]
all_ground_truth_answers = [doc["correct_answer"] for doc in dataset]

os.environ['OPENAI_API_KEY'] = '<token>'
nltk.download('punkt')


template = """
Question: {question}
"""
Prompt = PromptTemplate(template=template, input_variables=[
    'question'])

llm = ChatOpenAI(temperature=0, model='gpt-4')
chain = LLMChain(llm=llm, prompt=Prompt)


def ask_question_bleu(question, references):
    answer = chain.invoke(question)
    generated_text = answer['text']

    reference_texts = [nltk.word_tokenize(ref) for ref in references]
    generated_tokens = nltk.word_tokenize(generated_text)
    bleu_score = sentence_bleu(reference_texts, generated_tokens)
    print("Generated Answer:")
    pprint.pprint(generated_text)
    print("\nBLEU Score:", bleu_score)
    return answer, bleu_score


def ask_question(question):
    answer = chain.invoke(question)
    generated_text = answer['text']

    print("Generated Answer:")
    pprint.pprint(generated_text)
    return answer


questions, ground_truth_answers = zip(
    *random.sample(list(zip(all_questions, all_ground_truth_answers)), 100))
bleu_scores = []
for question, answer in zip(questions, ground_truth_answers):
    ans, bleu = ask_question_bleu(question, [answer])
    bleu_scores.append(bleu)
average_bleu = sum(bleu_scores) / len(bleu_scores)
print("Average BLEU Score:", average_bleu)
