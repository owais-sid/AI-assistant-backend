import os
import io
import re
import time
import json
import base64
import uuid
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import Response, JSONResponse,StreamingResponse
# from polars import datetime
# from pydub import AudioSegment
from io import BytesIO
import openai
import csv
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict,Any,List, Optional
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
# from langdetect import detect

# from lingua import Language, LanguageDetectorBuilder

# from sentence_transformers import SentenceTransformer
# from pyparsing import Optional
# from google.genai import types

# --- Configuration & Initialization ---

# lingua_languages = [Language.ENGLISH, Language.URDU, Language.HINDI]
# detector = LanguageDetectorBuilder.from_languages(*lingua_languages).build()


# client = OpenAI(api_key="openapi key")
MODEL = "gpt-4o-mini"

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment")

client = OpenAI(api_key=OPENAI_API_KEY)
# File Paths
EMPLOYEE_FILE = r"C:\Users\Lenovo\Desktop\voice_agent\backend\employee.csv"
# QUESTIONS_FILE = r"C:\Users\Lenovo\Desktop\voice_agent\backend\questions.csv"
RESPONSES_FILE = r"C:\Users\Lenovo\Desktop\voice_agent\backend\responses.csv"

employee_df = pd.read_csv(EMPLOYEE_FILE)

# Combine columns and convert to a list
full_name = (employee_df["firstname"] + " " + employee_df["lastname"]).tolist()
print("Employee Names: ", full_name)


def load_questions_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert dataframe to questions format"""
    try:
        # Validate required columns
        required_columns = {"id", "question", "options"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {', '.join(required_columns)}")
        
        # Normalize options: split by '|' and clean
        df["options_list"] = df["options"].apply(
            lambda x: [opt.strip().lower() for opt in str(x).split("|")] if pd.notna(x) else []
        )
        
        return df
    except Exception as e:
        raise ValueError(f"Error processing CSV: {str(e)}")



def build_message(text: str, qid: str, options=None):
    options = options or []
    
    # Create a combined text for TTS
    tts_text = text
    if options:
        tts_text += " Options are: " + ", ".join(options)
    print("TTS Text: ", tts_text)
    return {
        "role": "assistant",
        "text": text,
        "audio": text_to_speech(tts_text),
        "options": options or [],
        "qid": qid
    }


def text_to_speech(text: str) -> str:
    tts = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
        response_format="opus"
    )
    return base64.b64encode(tts.content).decode("utf-8")

def extract_number(user_answer: str) -> str:
    """Extracts a number from text using Gemini."""
    prompt = f"Extract the number from the following text(find in words also like two, three). Return only the number. If no number is present, return 'NO_NUMBER'.\n\nText: {user_answer}"
    response = client.chat.completions.create(
        model=MODEL,    
        messages=[{"role": "system", "content": prompt + "\n\nText: " + user_answer}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def map_answer_llm(question: str, options: List[str], user_answer: str) -> str:
    """Maps user answer to one of the provided options."""
    if not options:
        return user_answer # Open-ended
   
    prompt = f"""
You are validating a survey response.


Your task:
- Understand the user's meaning.
- Match it to EXACTLY ONE of the given options.
- Return ONLY the exact matching option text.
- Do NOT return multiple options.
- Do NOT translate options.
- Do NOT explain.
- If nothing matches, return NO_MATCH.

Options:
{options}

User Answer:
{user_answer}
"""

    response = client.chat.completions.create(
        model=MODEL,
        # messages=[{"role": "system", "content": prompt}],
        messages=[
            {"role": "system", "content": "You are a strict survey answer mapper."},
            {"role": "user", "content": prompt}
            ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

def linguitic_answer_llm(question: str, options: List[str], user_answer: str) -> str:
    """Maps user answer to one of the provided options."""
    if not options:
        return user_answer # Open-ended
   
    system_prompt = f"""
You are a survey response validator and classifier.

Survey Question:
"{question}"

Your task:

1. Determine whether the user's answer is relevant to the survey question.
2. If relevant, classify the answer into EXACTLY ONE of the following categories:
{options}

Return JSON strictly in this format:

{{
  "relevant": true or false,
  "category": "<one of the options>" or null
}}

Rules:
- If answer is not related to the question, set relevant=false and category=null.
- If relevant=true, category MUST be exactly one of the provided options.
- Do not explain anything.
- Return only valid JSON.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_answer}
        ]
    )

    output_text = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(output_text)
        
        if not parsed.get("relevant"):
            return "NOT_RELEVANT"
        
        return parsed.get("category", "NOT_RELEVANT")
    
    except json.JSONDecodeError:
        return "NOT_RELEVANT"


# emb_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # small & fast

def normalize_options(options):
    if isinstance(options, str):
        return [opt.strip() for opt in options.split(",")]

    # If it's a list but contains one comma-separated string
    if isinstance(options, list):
        if len(options) == 1 and "," in options[0]:
            return [opt.strip() for opt in options[0].split(",")]
        return options

    return []

# def local_embed(text):
#     vector = emb_model.encode(text)
#     return vector.tolist()
def local_embed(text):
    return emb_model.encode(text).tolist()

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def map_with_embeddings(user_answer, options):
    user_emb = local_embed(user_answer)
    best_option = None
    best_score = -1
    if isinstance(options, str):
        options = [opt.strip() for opt in options.split(",")]
    elif isinstance(options, list):
        if len(options) == 1 and "," in options[0]:
            options = [opt.strip() for opt in options[0].split(",")]

    print("Normalized options:", options)
       
    for option in options:
        opt_emb = local_embed(option)
        score = cosine_similarity(user_emb, opt_emb)
        if score > best_score:
            best_score = score
            best_option = option
            print("score: ", best_score, " for option: ", option)
    print("Best score =", best_score)
    return best_option if best_score > 0.60 else "NO_MATCH"

def detect_intent_llm(user_input: str) -> Dict[str, Any]:
    """Detects user intent using Openai with JSON output."""
    system_prompt = """
You are an intent classifier for a survey system.
User said: "{user_input}"

You MUST return ONLY valid JSON.
Do NOT add explanations.
Do NOT add markdown.

If change_answer, extract question number if mentioned.
Schema:
{
  "intent": "answer | change_answer | list_options | summary | submit | repeat_question",
  "question_id": number or null,
  "new_answer": null or string
}
"""
    user_prompt = f'User input: "{user_input}"'

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # HARD FAIL SAFE
        return {
            "intent": "answer",
            "question_id": None,
            "new_answer": None
        }



def quick_intent_check(text: str) -> Optional[Dict[str, Any]]:
    """Fast keyword-based intent check."""
    t = text.lower().strip()
    if "option" in t:
        return {"intent": "list_options", "question_id": None, "new_answer": None}
    if t == "submit":
        return {"intent": "submit", "question_id": None, "new_answer": None}
    if "summary" in t:
        return {"intent": "summary", "question_id": None, "new_answer": None}
    if "repeat" in t:
        return {"intent": "repeat_question", "question_id": None, "new_answer": None}
    if "change" in t:
        return {"intent": "change_answer", "question_id": None, "new_answer": None}
    return None


async def generate_survey_intro(questions_df, language="en", full_name=None, orchestrator=None, total_questions=None):
    """
    Creates a short 2–3 line summary describing:
    - What the survey is about
    - Its general purpose
    """
    # Combine first few questions to understand theme
    sample_questions = questions_df["question"].dropna().tolist()

    combined_text = " ".join(sample_questions)

    name_instruction = f"Greet {full_name} naturally." if full_name else "Greet the user naturally."

    prompt1 = f"""
Generate a voice-friendly survey opening in {language}.
Keep the tone low as much as you can, and make it sound like a natural conversation. Because the user is not literate enough. Do not be too formal or robotic.

STRICT RULES:
- Do NOT use headings.
- Do NOT use markdown.
- Do NOT use bullet points.
- Do NOT number anything.
- Return ONLY plain text.
- You MUST separate the Greeting and Introduction using line breaks.

Output format must exactly be:
Greeting
Introduction

1. {name_instruction}
2. Based on these questions: "{combined_text}", write a 2-3 line introduction explaining the survey that what this survey is about, tell the total count of questions {total_questions} but don't list the questions.
"""

    # Assuming 'client' is defined globally in your environment
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt1,
        temperature=0
    )
    
    greeting_text = response.output[0].content[0].text
    print("LLM Intro Response: ", greeting_text)
    
    first_q = orchestrator.state.current_question()
    question_text = first_q['question']
    options_val = first_q.get("options")
    
    # question_text = f"First question: {first_q['question']}" if first_q is not None else intro_text
    # 1. Safely get the value
    if pd.notna(options_val) and str(options_val).lower() != "nan":
        if language == 'en':
            options_text = f"Choices for this questions are: {options_val}"
        if language == 'ur':
            options_text = f"اس سوال کے لیے آپ کے پاس یہ آپشنز ہیں: {options_val}"
    else:
        if language == 'en':
            options_text = "This is an open-ended question."
        if language == 'ur':
            options_text = "اس سوال کا جواب آپ اپنی مرضی سے دے سکتے ہیں۔"

#     prompt2 = f"""
# You are a voice assistant conducting a survey in {language}.
# Please read the following question to the user.

# STRICT RULES:
# - Do NOT use headings, markdown, bullet points, or numbers.
# - Return ONLY plain text.
# - The question and options MUST be in their exact original wordings. Do NOT paraphrase them.
# - Translate only the conversational filler words (like "The first question is:" or "The options are:") into {language}.

# Question to ask: "{question_text}"
# {options_instruction}
# """

#     response = client.responses.create(
#         model="gpt-4o-mini",
#         input=prompt2,
#         temperature=0
#     )
    
    # question_text = response.output[0].content[0].text
    print("LLM First Question Response: ", question_text + options_text)
    
    return greeting_text, question_text + "\n" + options_text
# --- Survey Logic ---

# survey_tools =[
#     {
#         "type": "function",
#         "function": {
#             "name": "execute_survey_action",
#             "description": "Determines the user's intent and generates the exact spoken reply in the correct language.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "intent": {
#                         "type": "string",
#                         "enum":[
#                             "answer", "change_answer", "repeat_question", 
#                             "list_options", "summary", "submit", "wrong_language"
#                         ],
#                         "description": "The action the user wants to take."
#                     },
#                     "mapped_answer": {
#                         "type": "string",
#                         "description": "If intent is 'answer', the exact option they chose. If their answer doesn't match the options, output 'NO_MATCH'."
#                     },
#                     "target_question_id": {
#                         "type": "integer",
#                         "description": "If intent is 'change_answer', the ID of the question they want to go back to."
#                     },
#                     "reply_to_speak": {
#                         "type": "string",
#                         "description": "The conversational reply to speak to the user IN THE EXPECTED LANGUAGE. If they answered correctly, acknowledge it and ask the Next Question in this string."
#                     }
#                 },
#                 "required": ["intent", "reply_to_speak"]
#             }
#         }
#     }
# ]


survey_tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_survey_action",
            "description": "Determines user intent and generates the conversational reply.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["answer", "change_answer", "repeat_question", "list_options", "summary", "submit", "wrong_language", "out_of_context"],
                        "description": "The detected action."
                    },
                    "mapped_answer": {"type": "string", "description": "The mapped option or NO_MATCH."},
                    "target_question_id": {"type": "integer"},
                    "reply_to_speak": {"type": "string", "description": "The reply in the EXPECTED LANGUAGE."}
                },
                "required": ["intent", "reply_to_speak"]
            }
        }
    }
]


class SurveyState:
    def __init__(self, questions_df: pd.DataFrame, language: str = "en"):
        self.questions = questions_df
        self.responses: Dict[int, str] = {}
        self.language = language
        self.current_index = 0
        self.max_index = 0
        self.completed = False
        self.halfway = False
        self.q_completed = False

    def current_question(self):
        if self.current_index >= len(self.questions):
            return None
        return self.questions.iloc[self.current_index]

    def is_done(self):
        return len(self.responses) >= len(self.questions)

class SurveyOrchestrator:
    def __init__(self, state: SurveyState):
        self.state = state


    # def handle_input(self, user_input: str, session_id: str, detected_language: str = "en") -> str:
        
    #     if not user_input:
    #         if detected_language == 'en':
    #             return "I couldn't hear anything. Could you please repeat that?"
    #         if detected_language == 'ur':
    #             return "معذرت، میں کچھ سن نہیں سکا۔ کیا آپ اسے دوبارہ دہرا سکتے ہیں؟"

        
    #     # intent = quick_intent_check(user_input)
    #     # if intent is None:
    #     intent = detect_intent_llm(user_input)
        
    #     # SUMMARY
    #     if intent["intent"] == "summary":
    #         if detected_language == 'en':
    #             return f"Here is your summary so far:\n{self.get_summary(detected_language)}."
    #         if detected_language == 'ur':
    #             return f"اب تک کا خلاصہ یہ ہے:\n{self.get_summary(detected_language)}"


    #     # SUBMIT
    #     if intent["intent"] == "submit":
    #         if not self.state.is_done():
    #             if detected_language == 'en':
    #                 return f"Please answer all questions before submitting.\n{self.get_summary(detected_language)}"
    #             if detected_language == 'ur':
    #                 return f"براہِ کرم جمع کروانے سے پہلے تمام سوالات کے جوابات دیں۔\nخلاصہ:\n{self.get_summary(detected_language)}"

            
    #         self.save_responses(session_id)
    #         self.state.completed = True
    #         self.state.current_index += 1
    #         if detected_language == 'en':
    #             return "Survey submitted successfully. Thank you for your time!"
    #         if detected_language == 'ur':
    #             return "سروے کامیابی سے جمع ہو گیا ہے۔ آپ کے وقت کا شکریہ!"


    #     # CHANGE ANSWER
    #     if intent["intent"] == "change_answer":
    #         qid_str = extract_number(user_input)
    #         try:
    #             qid = int(qid_str)
    #         except:
    #             if detected_language == 'en': 
    #                 return "I couldn't identify the question number. Please say 'change answer for question 1'."
    #             if detected_language == 'ur':
    #                 return "میں سوال کا نمبر نہیں پہچان سکا۔ براہِ کرم کہیں 'سوال 1 کا جواب تبدیل کریں'۔"

    #         if qid not in self.state.responses:
    #             if detected_language == 'en':
    #                 return f"Question {qid} has not been answered yet."
    #             if detected_language == 'ur':
    #                 return f"سوال {qid} کا جواب اب تک نہیں دیا گیا۔"

            
    #         self.state.current_index = qid - 1
    #         self.state.max_index -= 1
    #         print(f"Changing answer for question {qid}. Question: {self.state.questions.iloc[qid-1]['question']}. Current answer: {self.state.responses[qid]}")
    #         if detected_language == 'en':
    #             return f"What is your new answer for question {qid}?"
    #         if detected_language == 'ur':
    #             return f"سوال نمبر {qid} کے لیے آپ کا نیا جواب کیا ہے؟"

    #     question = self.state.current_question()
    #     if question is None:
    #         if detected_language == 'en':
    #             return f"The survey is complete. Here are your answers:\n{self.get_summary(detected_language)}"
    #         if detected_language == 'ur':
    #             return f"سروے مکمل ہو گیا ہے۔ آپ کے جوابات یہ ہیں:\n{self.get_summary(detected_language)}"


    #     qid = int(question["id"])

    #     # LIST OPTIONS
    #     if intent["intent"] == "list_options":
    #         opts = question.get('options')
    #         if pd.notna(opts) and str(opts).lower() != "nan":
    #             if detected_language == 'en':
    #                 options_str = f"The options are: {opts}."
    #             if detected_language == 'ur':
    #                 options_str = f"آپ کے پاس یہ آپشنز ہیں: {opts}"
                    
    #         else:
    #             if detected_language == 'en':
    #                 options_str = "No specific options. This is an open-ended question."
    #             if detected_language == 'ur':
    #                 options_str = "کوئی مخصوص آپشنز نہیں ہیں۔ یہ ایک آزادانہ سوال ہے۔"
            
    #         return f"{options_str}"
        
    #     # REPEAT
    #     if intent["intent"] == "repeat_question":
    #         if detected_language == 'en':
    #             return f"Sure, I'll repeat:\n{question['question']}"
    #         if detected_language == 'ur':
    #             return f"جی بالکل، میں دوبارہ دہرا دیتا ہوں:\n{question['question']}"


    #     # ANSWER
    #     if intent["intent"] == "answer":
    #         mapped = map_answer_llm(
    #             question["question"],
    #             question["options_list"],
    #             user_input,
    #         )
    #         print("LLM mapped answer: ", mapped)
    #         if (isinstance(mapped, str) and "," in mapped):
    #             mapped = "NO_MATCH"
    #         if mapped == "NO_MATCH":
    #             # if detected_language == self.state.language:
    #             # mapped = "NO_MATCH"
    #                 mapped = map_with_embeddings(
    #                     user_input,
    #                     question["options_list"]
    #                 )
    #                 if mapped == "NO_MATCH":
    #                     mapped = linguitic_answer_llm(
    #                         question["question"],
    #                         question["options_list"],
    #                         user_input
    #                     )
    #                     if mapped == "NOT_RELEVANT":
    #                         opts = question.get('options', '')
    #                         if detected_language == 'en': 
    #                             return f"Sorry! I didn't Understand. Please choose one of: {opts}"
    #                         if detected_language == 'ur':
    #                             return f"معذرت! میں سمجھ نہیں سکا۔ براہِ کرم ان میں سے کسی ایک کا انتخاب کریں: {opts}"

    #             # else:
    #             #     return f"Sorry! I didn't Understand. Please answer in selected language. The options are: <options>{question['options']}</options>"

    #         self.state.responses[qid] = mapped
    #         print(f"Recorded answer for question {qid}: {mapped}")     
            

    #         # if self.state.current_index == self.state.max_index:
    #         self.state.max_index += 1
    #         self.state.current_index = self.state.max_index

    #         next_q = self.state.current_question()
    #         if next_q is None:
    #             if detected_language == 'en':
    #                 return f"Your Response is {mapped}\n\nGreat! You've answered all questions.\n{self.get_summary(detected_language)}\n\nSay 'submit' to finalize."
    #             if detected_language == 'ur':
    #                 return f"آپ کا جواب ہے {mapped}\n\nزبردست! آپ نے تمام سوالات کے جوابات دے دیے ہیں۔\n{self.get_summary(detected_language)}\nفائنل کرنے کے لیے 'جمع کریں' یا 'submit' کہیں۔"

            
    #         options_val = next_q.get("options")
    #         if pd.notna(options_val):
    #             if detected_language == 'en':    
    #                 options_text = f"Choices for this questions are: {options_val}"
    #             if detected_language == 'ur':
    #                 options_text = f"اس سوال کے لیے آپ کے پاس یہ آپشنز ہیں: {options_val}"

    #         else:
                
    #             if detected_language == 'en':
    #                 options_text = "This is an open-ended question."
    #             if detected_language == 'ur':
    #                 options_text = "اس سوال کا جواب آپ اپنی مرضی سے دے سکتے ہیں۔"

                
    #         if(self.state.current_index >= int(len(self.state.questions)) /2 and self.state.halfway == False):
    #             self.state.halfway = True
    #         # if self.state.halfway:
    #             if detected_language == 'en':
    #                 return f"Your Response is {mapped} \n\nYou're halfway through! Keep going. \nNext question: Question: \n{self.state.current_index+1} {next_q['question']}\n{options_text}"
    #             if detected_language == 'ur':
    #                 return f"آپ کا جواب ہے {mapped} \n\nآدھا راستہ طے ہو گیا! بس ہمت جاری رکھیں۔ \nاگلا سوال: سوال {self.state.current_index+1}: {next_q['question']}\n{options_text}"

    #         if detected_language == 'en':
    #             return f"Your Response is {mapped} \n\nGot it. Next question: Question:  {self.state.current_index+1} {next_q['question']}\n{options_text}"
    #         if detected_language == 'ur':
    #             return f"آپ کا جواب ہے {mapped} \n\nٹھیک ہے، اگلا سوال یہ رہا: سوال {self.state.current_index+1}: {next_q['question']}\n{options_text}"


    #     if detected_language == 'en':    
    #         return "I'm sorry, I didn't understand. Could you please answer the question or ask for options?"
    #     if detected_language == 'ur':
    #         return "معذرت، میں سمجھ نہیں سکا۔ براہِ کرم سوال کا جواب دیں یا آپشنز کے بارے میں پوچھیں۔"

    
    
    def handle_input(self, user_text: str, session_id: str) -> str:
        # 1. Handle silent/empty audio instantly
        if not user_text.strip():
            # Quick fallback if Whisper heard nothing
            return "Please repeat that." if self.state.language == "en" else "براہ کرم دوبارہ کہیں۔"

        # 2. Gather State Information for the LLM
        current_q = self.state.current_question()
        expected_lang = self.state.language
        summary_so_far = self.get_summary(expected_lang)
        total_q = len(self.state.questions)
        
        if current_q is not None:
            # We are DURING the survey
            q_context = f"""
            CURRENT QUESTION:
            - ID: {current_q['id']}
            - Text: {current_q['question']}
            - Options: {current_q.get('options_list', 'Open-ended')}
            """

            if self.state.current_index < self.state.max_index:
                if self.state.max_index >= total_q:
                    # User finished everything and is just editing.
                    next_context = "NEXT STEP: You have already finished all questions. After acknowledging this change, show the SUMMARY and ask to SUBMIT."
                else:
                    # User is mid-survey and went back.
                    resumed_q = self.state.questions.iloc[self.state.max_index]
                    next_context = f"NEXT STEP: After this change, you will resume from Question {self.state.max_index + 1}: {resumed_q['question']}"
            else:
                # Normal flow (no editing)
                next_idx = self.state.current_index + 1
                if next_idx < total_q:
                    next_q = self.state.questions.iloc[next_idx]
                    next_context = f"NEXT QUESTION: Question {next_idx + 1}: {next_q['question']} Options: {next_q.get('options_list', '')}"
                else:
                    next_context = "NEXT STEP: This was the last question. Provide the SUMMARY and ask to SUBMIT."
        else:
            # Survey is already completely finished
            q_context = "SURVEY STATUS: All questions answered. Reviewing summary."
            next_context = "The user can now either 'submit' or 'change' a specific question ID."

        # Lookahead: Get the next question so the LLM can ask it automatically
        # next_q = None
        # if self.state.current_index + 1 < len(self.state.questions):
        #     next_q = self.state.questions.iloc[self.state.current_index + 1]

        # 3. Create the Ultimate System Prompt
        system_prompt = f"""
        You are a smart, conversational voice survey assistant.
        The user's EXPECTED LANGUAGE is: '{expected_lang}'. 
        
        {q_context}
        Next Question:
        {next_context}

        SURVEY SUMMARY SO FAR:
        {summary_so_far}
        
        
        STRICT RULES:
        1. IF {expected_lang} is 'ur' (Urdu) and user speaks in Hindi/Devanagari script, treat it as valid Urdu and reply in Urdu script.
        2. LANGUAGE REVOKE: If the user speaks a language other than {expected_lang}(if not hindi urdu case), set intent to 'wrong_language'. In 'reply_to_speak', tell them IN {expected_lang} that you only understand {expected_lang} and to please repeat their answer.
        3. OUT OF CONTEXT: If the user talks about unrelated things (pizza, weather, etc.), set intent to 'out_of_context'. In 'reply_to_speak', politely ask them to focus on the survey.
        4. NATIVE SCRIPT: Always use the proper script for the language (e.g., Nastaliq for Urdu, Arabic script for Arabic).
        5. NATURAL REPLIES: Always repeat the user's answer naturally (e.g., "Got it, so your choice is X").
        6. ANSWERING: If they answer, set intent to 'answer', map the value, and write 'reply_to_speak' to acknowledge their answer and it MUST follow this structure:
           [Acknowledgement of current answer] + [The Next Question with Options or Step from the Next Question section above]..
        7. SUBMITTING: If they ask to submit but haven't finished, politely tell them they must finish the questions first. and if finished then say goodbye greetings.
        8. RESPOND IN SELECTED LANGUAGE: Your 'reply_to_speak' MUST ALWAYS be in {expected_lang}.
        """
        

        # 4. Call the OpenAI Agent (Forcing it to use our Omni-Tool)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            tools=survey_tools,
            tool_choice={"type": "function", "function": {"name": "execute_survey_action"}}, # Forces structured JSON output
            temperature=0.2 # Keep it focused
        )

        # 5. Extract the AI's Decisions
        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        
        intent = args.get("intent")
        mapped_answer = args.get("mapped_answer")
        target_qid = args.get("target_question_id")
        reply_to_speak = args.get("reply_to_speak")

        print(f"Agent Detected Intent: {intent} | Mapped Answer: {mapped_answer}")

        if intent in ["wrong_language", "out_of_context"]:
            print(f"Guardrail Triggered: {intent}")
            return reply_to_speak
    
        # 6. Update Your Database/State based on the Agent's intent
        if intent == "answer" and mapped_answer != "NO_MATCH":
            qid = int(current_q["id"])
            self.state.responses[qid] = mapped_answer
            print(f"question: {qid} c: {self.state.current_index} m: {self.state.max_index}")
            # Advance the state
            # if self.state.current_index == self.state.max_index:
            #     self.state.max_index += 1
            # self.state.current_index += 1
            
            
            # If answering the LATEST question (normal flow)
            if self.state.current_index == self.state.max_index:
                self.state.current_index += 1
                self.state.max_index += 1
            else:
                # If answering a CHANGED question (jump back to bookmark)
                self.state.current_index = self.state.max_index
            print(f"question: {qid} c: {self.state.current_index} m: {self.state.max_index}")
            
            # Completion check
            if self.state.max_index >= total_q:
                print(f"q complete triggered")
                self.state.q_completed = True
            
            
            
            # if self.state.q_completed:
            #     self.state.current_index = total_q
            
            # self.state.max_index +=1
            # self.state.current_index = self.state.max_index
            # if self.state.max_index == len(self.state.questions):
            #     self.state.q_completed = True

        elif intent == "change_answer" and target_qid:
            print(f"change triggered for and at: {self.state.current_index}  {self.state.max_index}" )
            # Prevent going to a question they haven't answered yet
            if 1 <= target_qid <= self.state.max_index + 1:
                if target_qid <= self.state.max_index + 1:
                    self.state.current_index = target_qid - 1
                # self.state.max_index -=1
            else:
                return "You haven't reached that question yet." if expected_lang == "en" else "آپ ابھی اس سوال تک نہیں پہنچے۔" 

        elif intent == "submit" and self.state.is_done():
            self.save_responses(session_id)
            self.state.completed = True
            self.state.current_index += 1

        elif intent == "summary":
            fresh_summary = self.get_summary(expected_lang)
        
            prompt = f"""
            You are voice survey assistant
            The user asked for a summary. Read the following summary and ask them if they want to change anything or submit: {fresh_summary}
            Always respond in {expected_lang}
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_text}],
                temperature=0.3
            )
    
            return response.choices[0].message.content

        # 7. Return the dynamically translated text back to your TTS!
        return reply_to_speak
    
    
    def get_summary(self, detected_language: str) -> str:
        if detected_language == 'en':
            lines = ["Here is a summary of your responses:\n"]
            for _, row in self.state.questions.iterrows():
                qid = int(row["id"])
                ans = self.state.responses.get(qid, "Not answered")
                lines.append(f"Question {qid}: {ans}")
            return " \n".join(lines)
        if detected_language == 'ur':
            lines = ["آپ کے جوابات کا خلاصہ یہ ہے:\n"]
            for _, row in self.state.questions.iterrows():
                qid = int(row["id"])
                ans = self.state.responses.get(qid, "جواب نہیں دیا گیا")
                lines.append(f"سوال نمبر {qid}: {ans}")
            return " \n".join(lines)



    def save_responses_csv(self, session_id: str):
        data = []
        for _, row in self.state.questions.iterrows():
            qid = int(row["id"])
            data.append({
                "session_id": session_id,
                "question": row["question"],
                "answer": self.state.responses.get(qid, "Not answered"),
                "timestamp": time.time()
            })
        
        df = pd.DataFrame(data)
        
        # Check if file exists to decide on writing the header
        file_exists = os.path.isfile(RESPONSES_FILE)
        
        df.to_csv(
            RESPONSES_FILE, 
            mode='a',           # 'a' stands for Append
            index=False, 
            header=not file_exists  # Only write header if the file is new
        )
    def save_responses(self, session_id: str):
        data = []

        for _, row in self.state.questions.iterrows():
            qid = int(row["id"])
            data.append({
                "question_id": qid,
                "question_text": row["question"],
                "user_answer": self.state.responses.get(qid, "Not answered"),
                "timestamp": time.time()
            })

        SESSION_STORE[session_id] = data

# --- FastAPI App ---

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global session storage (In-memory for this example)
SESSION_STORE = {}
SESSIONS: Dict[str, SurveyOrchestrator] = {}
SURVEYS: dict[str, pd.DataFrame] = {}
TTS_STORE = {}

emb_model = None  # global reference

# @app.on_event("startup")
# async def load_embedding_model():
#     global emb_model
#     print("Loading embedding model...")
#     emb_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
#     print("Embedding model loaded.")

@app.get("/")
def read_root():
    return {"message": "Voice Survey API is running"}

@app.post("/upload_survey_csv")
async def upload_survey_csv(csv_file: UploadFile = File(...)):
    """
    Handle CSV file upload for custom surveys
    Saves to global dataframe
    
    Required columns: id, question, options
    Options should be separated by '|' (e.g., "Option1|Option2|Option3")
    """
    # global uploaded_questions_df
    
    try:
        # Validate file extension
        if not csv_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        # Read the uploaded file
        contents = await csv_file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate and process the dataframe
        validated_df = load_questions_from_dataframe(df)
        
        survey_id = str(uuid.uuid4())
        SURVEYS[survey_id] = validated_df
        # Save to global variable
        # questions_df = validated_df
        print("Uploaded Questions DataFrame:")
        print(validated_df[["options", "options_list"]].head())
        return {
            "survey_id": survey_id
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# from openai import OpenAI

def translate_text(text: str, target_language: str) -> str:
    """
    Translate given text into target_language.
    Returns translated string only.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",   # fast + cheap + multilingual
        temperature=0,        # deterministic translation
        messages=[
            {
                "role": "system",
                "content": f"""
You are a translator.

Translate the text into {target_language}.

IMPORTANT:
- Use simple, everyday spoken language.
- Do NOT use formal, literary, or classical vocabulary.
- Keep sentences short and natural.
- Make it sound like normal conversation.
- Return ONLY the translated text.
"""
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )
    

    return response.choices[0].message.content.strip()

@app.post("/start_session")
async def start_session(survey_id: str = None, language: str = "en"):
    if survey_id not in SURVEYS:
        raise HTTPException(status_code=400, detail="Invalid survey ID")
    questions_df = SURVEYS[survey_id]
    
    session_id = str(uuid.uuid4())
    state = SurveyState(questions_df, language=language)
    orchestrator = SurveyOrchestrator(state)
    SESSIONS[session_id] = orchestrator
    start_g = time.perf_counter()
    greeting_text, question_text = await generate_survey_intro(questions_df, language,full_name[0].upper(),orchestrator , int(len(state.questions)))
    end_g = time.perf_counter()
    print(f"⏳ [TIMER] Survey Intro Generation took: {end_g - start_g:.2f} seconds")
    # question_tts = client.audio.speech.create(
    #     model="tts-1",
    #     voice="alloy",
    #     input=question_text,
    # )
    # question_audio = base64.b64encode(
    #     question_tts.content
    # ).decode("utf-8")
    start_gtts = time.perf_counter()
    # question_audio = text_to_speech(question_text)
    # greeting_audio = text_to_speech(greeting_text)
    end_gtts = time.perf_counter()
    print(f"⏳ [TIMER] greeetings Text-to-Speech Generation took: {end_gtts - start_gtts:.2f} seconds")
    # greeting_tts = client.audio.speech.create(
    #     model="tts-1",
    #     voice="alloy",
    #     input=greeting_text,
    # )
    # greeting_audio = base64.b64encode(
    #     greeting_tts.content
    # ).decode("utf-8")
    tts_id_1 = str(uuid.uuid4())
    # tts_id_2 = str(uuid.uuid4())
    TTS_STORE[tts_id_1] = greeting_text + question_text
    # TTS_STORE[tts_id_2] = question_text
    return {
        "session_id": session_id,
        "total_questions": int(len(state.questions)),
        "messages": [
            {
                "role": "assistant",
                "type": "greeting",
                "text": greeting_text,
                "audio_url": f"/tts/{tts_id_1}"
            },
            {
                "role": "assistant",
                "type": "question",
                "question_id": 1,
                "text": question_text,
                # "options": options_list,
                # "audio_url": f"/tts/{tts_id_2}"
            }
        ]
    }


# @app.post("/process_input")
# async def process_input(
#     session_id: str = Form(...),
#     audio: UploadFile = File(...)
# ):
#     if session_id not in SESSIONS:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     orchestrator = SESSIONS[session_id]
#     messages = []
#     qid = orchestrator.state.current_index
#     qid = int(qid)
    
#     try:
#         start_stt = time.perf_counter()
#         print("Transcribing audio: ", audio)
#         # user_text = transcribe_audio(audio)
#         # audio.file.seek(0)
#         audio_bytes = await audio.read()
#         transcription = client.audio.transcriptions.create(
#             model="whisper-1",
#             file=(audio.filename or "audio.webm", audio_bytes, audio.content_type or "audio/webm"),
#             # file=(audio.filename or "audio.wav", audio_bytes, audio.content_type or "audio/wav"),
#             # language=orchestrator.state.language
#         )
#         end_stt = time.perf_counter()
#         print(f"⏳ [TIMER] Whisper Transcription took: {end_stt - start_stt:.2f} seconds")
        
#         user_text = transcription.text.strip()
#         detected_lang_enum = detector.detect_language_of(user_text)
#         print("User text: ", user_text)
#         print(f"Detected Language: {detected_lang_enum}")
        
#         if detected_lang_enum == Language.ENGLISH:
#             detected_lang = "en"
#         elif detected_lang_enum in [Language.URDU, Language.HINDI]:
#             # We treat Hindi script as Urdu because spoken they are identical, 
#             # and your GPT-4o agent can read Hindi script perfectly anyway.
#             detected_lang = "ur"
#         else:
#             detected_lang = "unknown"
        
#         expected_lang = orchestrator.state.language
#         if detected_lang != expected_lang:
#             print(f"Language Mismatch! Expected {expected_lang}, got {detected_lang}")
        
#             if expected_lang == "ur":
#                 reject_msg = "براہ کرم اپنا جواب اردو میں دیں۔" # "Please answer in Urdu"
#             else:
#                 reject_msg = "Please provide your answer in English."
#             tts_id = str(uuid.uuid4())
#             TTS_STORE[tts_id] = reject_msg
#             messages.append({
#                 "role": "assistant",
#                 "type": "question",
#                 "question_id": qid,
#                 "text": reject_msg,
#                 "audio_url": f"/tts/{tts_id}"  # NEW: Pass a URL!
#             })
#             # messages.append(build_message(reject_msg, qid))
#             return {"messages": messages, "user_text": "-"}
        
#     except Exception as e:
#         print("Error transcribing audio: ", e)
#         raise HTTPException(status_code=400, detail=f"Audio transcription failed: {str(e)}")
    
#     start_hi = time.perf_counter()
#     agent_text = orchestrator.handle_input(user_text, session_id, detected_language=detected_lang)
#     end_hi = time.perf_counter()
#     print(f"⏳ [TIMER] handle input took: {end_hi - start_hi:.2f} seconds")
#     # 4. Now you have the final, correctly translated text!
#     # agent_text = response.choices[0].message.content
#     print(f"Agent: {agent_text}")
#     pattern = r"</?(question|answer|options|summary)>"
#     cleaned_text = re.sub(pattern, "", agent_text)
#     print(f"Cleaned Agent Text: {cleaned_text}")
    
#     tts_id = str(uuid.uuid4())
#     TTS_STORE[tts_id] = cleaned_text
#     # 3. TTS
#     #audio_b64 = text_to_speech(agent_text)
#     if(orchestrator.state.completed):
#         return {
#             "messages": [
#             {
#                 "role": "assistant",
#                 "text": cleaned_text,
#                 "audio_url": f"/tts/{tts_id}",  # This is a URL that the frontend can call to get the audio stream
#                 "iscomplete": True
#             }],
#             "user_text": user_text
#         }
#     qid = orchestrator.state.current_index
#     qid = int(qid)
    
#     # messages.append(build_message(translate_text(f"OK, your response is '{user_text}'", target_language=orchestrator.state.language), qid))
#     start_tts = time.perf_counter()
#     # messages.append(build_message(cleaned_text, qid))
#     messages.append({
#         "role": "assistant",
#         "type": "question",
#         "question_id": qid,
#         "text": cleaned_text,
#         "audio_url": f"/tts/{tts_id}"  # NEW: Pass a URL!
#     })
#     end_tts = time.perf_counter()
#     # print(f"⏳ [TIMER] TTS generation took: {end_tts - start_tts:.2f} seconds")
    
#     if detected_lang == "ur" and detected_lang_enum == Language.HINDI:
#         convert_prompt = f"""
#             Convert the following text into Urdu script.
#             Do NOT translate. Keep meaning identical.

#             Text: {user_text}
#             """
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": convert_prompt}],
#             temperature=0
#         )

#         user_text = response.choices[0].message.content.strip()
    
#     return {"messages": messages, "user_text": user_text}
    
    

@app.post("/process_input")
async def process_input(
    session_id: str = Form(...),
    audio: UploadFile = File(...)
):
    # 1. Validate Session
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    
    orchestrator = SESSIONS[session_id]
    messages =[]
    
    try:
        # 2. Transcribe Audio using Whisper
        start_stt = time.perf_counter()
        print(f"Transcribing audio for session: {session_id}")
        
        audio_bytes = await audio.read()
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=(audio.filename or "audio.webm", audio_bytes, audio.content_type or "audio/webm")
        )
        end_stt = time.perf_counter()
        print(f"⏳ [TIMER] Whisper Transcription took: {end_stt - start_stt:.2f} seconds")
        
        user_text = transcription.text.strip()
        print(f"User text: {user_text}")
        
    except Exception as e:
        print("Error transcribing audio: ", e)
        raise HTTPException(status_code=400, detail=f"Audio transcription failed: {str(e)}")

    # 3. Pass text to the Agent (Omni-Tool Handles Everything!)
    start_hi = time.perf_counter()
    
    # Notice we no longer pass 'detected_lang' because the System Prompt handles it natively
    agent_text = orchestrator.handle_input(user_text, session_id) 
    
    end_hi = time.perf_counter()
    print(f"⏳ [TIMER] Agent handle_input took: {end_hi - start_hi:.2f} seconds")
    print(f"Agent Reply: {agent_text}")

    # 4. Store Text for TTS Streaming
    tts_id = str(uuid.uuid4())
    TTS_STORE[tts_id] = agent_text
    
    # 5. Build Response for the Frontend
    is_completed = orchestrator.state.completed
    qid = int(orchestrator.state.current_index)
    
    messages.append({
        "role": "assistant",
        "type": "question" if not is_completed else "completion",
        "question_id": qid,
        "text": agent_text,           # The perfectly formatted Urdu or English text
        "audio_url": f"/tts/{tts_id}", # Frontend will call this to play audio
        "iscomplete": is_completed
    })

    return {
        "messages": messages, 
        "user_text": user_text
    }


@app.get("/tts/{tts_id}")
async def stream_tts(tts_id: str):
    """
    The frontend will call this URL via an <audio> tag or JS Audio object.
    This streams the audio bytes directly from OpenAI to the user's speakers!
    """
    text_to_speak = TTS_STORE.get(tts_id)
    
    if not text_to_speak:
        raise HTTPException(status_code=404, detail="TTS audio not found")

    # text_to_speak = cleaned_text
    # Generate audio
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text_to_speak,
        response_format="opus"
    )
    del TTS_STORE[tts_id]
    # Stream the bytes directly to the browser. 
    # The browser will start playing the audio while it is still downloading!
    return StreamingResponse(
        response.iter_bytes(), 
        media_type="audio/ogg" 
    )
    
@app.get("/summary/{session_id}")
async def get_summary(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    
    orchestrator = SESSIONS[session_id]
    summary_text = orchestrator.get_summary()
    
    return {
        "text": summary_text,
        "audio": text_to_speech(summary_text)
    }

from fastapi import HTTPException

@app.get("/session_summary/{session_id}")
async def get_session_summary(session_id: str):
    try:
        if session_id not in SESSION_STORE:
            raise HTTPException(status_code=404, detail="Session not found")

        questions_with_answers = SESSION_STORE[session_id]

        return {
            "session_id": session_id,
            "total_questions": len(questions_with_answers),
            "questions_with_answers": questions_with_answers
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/submit_responses/{session_id}")
async def submit_responses(session_id: str):
    try:
        # Get survey instance for this session
        orchestrator = SESSIONS.get(session_id)

        if not orchestrator:
            raise HTTPException(status_code=404, detail="Session not found")

        if not orchestrator.state.responses:
            raise HTTPException(status_code=400, detail="No responses to save")

        orchestrator.save_responses_csv(session_id)

        return {"status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.get("/filled_surveys")
async def list_filled_surveys():
    
    df = pd.read_csv(RESPONSES_FILE)
    
    grouped = []

    for session_id, group in df.groupby("session_id"):
        responses = group[["question", "answer", "timestamp"]] \
            .to_dict(orient="records")

        grouped.append({
            "session_id": str(session_id),
            "responses": responses
        })

    return grouped


@app.get("/filled_surveys/{session_id}")
async def get_filled_survey(session_id: str):
    df = pd.read_csv(RESPONSES_FILE)
    session_df = df[df["session_id"].astype(str) == session_id]

    if session_df.empty:
        raise HTTPException(status_code=404, detail="Session not found")

    responses = session_df[["question", "answer", "timestamp"]] \
        .to_dict(orient="records")

    return {
        "session_id": session_id,
        "responses": responses
    }
    
    
    
@app.post("/check_input_language")
async def check_input_language(
    selected_language: str = "ur",
    audio: UploadFile = File(...)
):

    audio_bytes = await audio.read()
    
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=(audio.filename, audio_bytes, audio.content_type),
        # response_format="verbose_json",
        # prompt=whisper_prompt
    )

    user_text = transcription.text.strip()
    print(f"Raw Whisper Text: {user_text}")

        # 2. LINGUA LANGUAGE DETECTION
    detected_lang_enum = detector.detect_language_of(user_text)
    print(f"Lingua Detected: {detected_lang_enum}")
    
    # Map Lingua's Enum to your expected string format
    if detected_lang_enum == Language.ENGLISH:
        detected_lang = "en"
    elif detected_lang_enum in [Language.URDU, Language.HINDI]:
        # We treat Hindi script as Urdu because spoken they are identical, 
        # and your GPT-4o agent can read Hindi script perfectly anyway.
        detected_lang = "ur"
    else:
        detected_lang = "unknown"

    # 3. STRICT LANGUAGE RESTRICTION
    expected_lang = selected_language # 'en' or 'ur'

    if detected_lang != expected_lang:
        print(f"Language Mismatch! Expected {expected_lang}, got {detected_lang}")
        
        if expected_lang == "ur":
            reject_msg = "براہ کرم اپنا جواب اردو میں دیں۔" # "Please answer in Urdu"
        else:
            reject_msg = "Please provide your answer in English."
            
        return {
            "messages":[{
                "role": "assistant",
                "text": reject_msg,
                # "audio": text_to_speech(reject_msg), # Optional
                "iscomplete": False
            }],
            "user_text": user_text
        }
    
    return {
        "detected_language": detected_lang,
        "is_matching": detected_lang in ["urdu", "hindi"] if selected_language == "ur" else detected_lang == selected_language,
        "user_text": user_text,
        # "converted_text": converted_text
    }
    
    