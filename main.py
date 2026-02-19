import os
import io
import time
import json
import base64
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, JSONResponse
from pydub import AudioSegment
from io import BytesIO
import openai
import csv
from fastapi.middleware.cors import CORSMiddleware
from typing import Any


openai.api_key = "open ai key"

QUESTIONS_FILE = r"C:\Users\Lenovo\Desktop\voice_agent\questions.csv"
RESPONSES_FILE = r"C:\Users\Lenovo\Desktop\voice_agent\backend\responses.csv"

# Load questions
questions_df = pd.read_csv(QUESTIONS_FILE)
questions_list = questions_df["question"].tolist()
total_questions = len(questions_list)

def save_response(session_id, question, transcription, mapped_option):
    file_exists = os.path.isfile(RESPONSES_FILE)

    with open(RESPONSES_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Write header only if file doesn't exist
        if not file_exists:
            #writer.writerow(["session_id", "question", "answer", "timestamp"])


            writer.writerow([
                "session_id",
                "question",
                "transcription",
                "mapped_option",
                "timestamp"
            ])
        # writer.writerow([
        #     session_id,
        #     question,
        #     answer,
        #     time.time()
        # ])
        
        
        writer.writerow([
            session_id,
            question,
            transcription,
            mapped_option,
            time.time()
        ])


#change
def validate_with_llm(question, options, answer):
    """
    Returns mapped option OR None
    """

    # Open-ended question → auto accept
    if not options or str(options).strip() == "":
        return {"mappedOption": None, "valid": True}

    prompt = f"""
Question:
{question}

Options:
{options}

User Answer:
"{answer}"

Task:
- If the answer clearly matches ONE option, return that option exactly.
- If it does NOT clearly match, return null.

Return ONLY valid JSON:
{{
  "mappedOption": string or null
}}
"""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"} 
    )
    print("message: ",response.choices[0].message)
    result = json.loads(response.choices[0].message.content)

    return {
        "mappedOption": result.get("mappedOption"),
        "valid": result.get("mappedOption") is not None
    }


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # must be explicit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Voice Survey Backend!"}

@app.get("/questions/")
def get_all_questions():
    return {"questions": questions_list}

@app.get("/get_question/{q_index}")
async def get_question(q_index: int):
    if q_index >= len(questions_list):
        return JSONResponse({"status": "completed"})

    question_text = questions_list[q_index]

    tts = openai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=question_text,
        response_format="wav"
    )

    audio_bytes = tts.read()

    return Response(content=audio_bytes, media_type="audio/wav")


# @app.post("/submit_answer/")
# async def submit_answer(
#     session_id: str,
#     q_index: int,
#     file: UploadFile = File(...)
# ):
#     audio_bytes = await file.read()

#     transcription = openai.audio.transcriptions.create(
#         model="whisper-1",
#         file=(file.filename, audio_bytes, file.content_type)
#     )

#     answer_text = transcription.text.strip()
#     question_text = questions_list[q_index]

#     save_response(session_id, question_text, answer_text)
#     return {
#         "transcription": answer_text,
#         "mappedOption": None
#     }
#     #return {"answer": answer_text}

@app.post("/submit_answer/")
async def submit_answer(
    session_id: str,
    question_index: int,
    audio: UploadFile | Any = File(None)
):
    audio_bytes = await audio.read()

    # Speech to text
    transcription = openai.audio.transcriptions.create(
        model="whisper-1",
        file=(audio.filename, audio_bytes, audio.content_type)
    ).text.strip()

    row = questions_df.loc[questions_df["id"] == question_index].iloc[0]

    question_text = row["question"]
    options = row.get("options", "")

    validation = validate_with_llm(
        question_text,
        options,
        transcription
    )

    # ❌ Not validated → frontend will re-record
    if not validation["valid"]:
        # --- INVALID ANSWER ---
        clarification_text = f"Please choose one of the listed options.{options}"
        
        # Generate Audio (Use standard model name 'tts-1')
        tts_response = openai.audio.speech.create(
            model="tts-1", 
            voice="alloy",
            input=clarification_text
        )
        
        # Encode binary audio to Base64 string
        # JSON cannot send raw bytes like tts.read()
        audio_b64 = base64.b64encode(tts_response.content).decode('utf-8')

        return {
            "transcription": transcription,
            "mappedOption": None,
            "clarificationText": clarification_text,
            "clarificationAudio": audio_b64 # Send string, not bytes
        }
    
    else:
        return {
            "transcription": transcription,
            "mappedOption": validation["mappedOption"],
            "clarificationText": None, # Set to None
            "clarificationAudio": None # Set to None
    }

    # ✅ Valid → save & allow frontend to move forward
    # save_response(
    #     session_id,
    #     question_text,
    #     transcription,
    #     validation["mappedOption"]
    # )

    # return {
    #     "transcription": transcription,
    #     "mappedOption": validation["mappedOption"]
    # }