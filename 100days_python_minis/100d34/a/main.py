from question_model import Question
# from data import question_data
from quiz_brain import QuizBrain
import tkinter as tk
import requests as rqst
import json
import html

from ui import QuizInterface

parameters = {
    "amount": 10,
    "type": "boolean"
}

response = rqst.get("https://opentdb.com/api.php", params=parameters)
response.raise_for_status()
question_data = response.json()["results"]


print(question_data)
question_bank = []
for question in question_data:
    question_text = question["question"]
    q_text = html.unescape(question_text)
    question_answer = question["correct_answer"]
    new_question = Question(q_text, question_answer)
    question_bank.append(new_question)


quiz = QuizBrain(question_bank)
print(question_bank)
quiz_ui = QuizInterface(quiz)

# while quiz.still_has_questions():
#     quiz.next_question()

print("You've completed the quiz")
print(f"Your final score was: {quiz.score}/{quiz.question_number}")
