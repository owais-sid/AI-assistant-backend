import os
import io
import time
import json
import base64
import uuid
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import Response, JSONResponse
from pydub import AudioSegment
from io import BytesIO
import openai
import csv
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict,Any
from dotenv import load_dotenv


SESSIONS = {}

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

QUESTIONS_FILE = r"C:\Users\dossani\AI-assistant-backend\questions.csv"
RESPONSES_FILE = r"C:\Users\dossani\AI-assistant-backend\responses.csv"

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
    
@app.post("/start_session")
def start_session():
    session_id = str(uuid.uuid4())

    # 1. Initialize session
    SESSIONS[session_id] = {
        "current_index": 0,
        "answers": {},
        "status": "asking"
    }

    # 2. Greeting
    greeting_text = (
        "Hello! Welcome to the survey. "
        "I will ask you a few questions one by one. "
        "Please answer using your voice."
    )

    greeting_tts = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=greeting_text
    )
    greeting_audio = base64.b64encode(
        greeting_tts.content
    ).decode("utf-8")

    # 3. First question
    first_row = questions_df.loc[questions_df["id"] == 1].iloc[0]

    question_text = first_row["question"]
    options = (
        [o.strip() for o in str(first_row["options"]).split(",")]
        if pd.notna(first_row["options"])
        else []
    )

    question_tts = openai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=question_text,
        response_format="wav"
    )
    question_audio = base64.b64encode(
        question_tts.content
    ).decode("utf-8")

    return {
        "session_id": session_id,
        "messages": [
            {
                "role": "assistant",
                "type": "greeting",
                "text": greeting_text,
                "audio": greeting_audio
            },
            {
                "role": "assistant",
                "type": "question",
                "question_id": 1,
                "text": question_text,
                "options": options,
                "audio": question_audio
            }
        ]
    }
    
def build_message(text: str, options=None):
    return {
        "role": "assistant",
        "text": text,
        "audio": text_to_speech(text),
        "options": options or []
    }

def text_to_speech(text: str) -> str:
    tts = openai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        response_format="wav"
    )
    return base64.b64encode(tts.content).decode("utf-8")

def classify_intent_llm(question, options, user_text):
    prompt = f"""
Current Question:
{question}

Options:
{options}

User said:
"{user_text}"

Classify intent as ONE of:
- ANSWER
- QUERY
- CHANGE_REQUEST
- INVALID

If CHANGE_REQUEST, extract question number.

Return JSON only:
{{
  "intent": "ANSWER | QUERY | CHANGE_REQUEST | INVALID",
  "target_question": number or null
}}
"""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

def explain_question_llm(question):
    prompt = f"""
Explain this survey question in simple terms (1–2 sentences):

{question}
"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

async def transcribe_audio(file: UploadFile):
    """
    Converts uploaded audio to mono 16kHz WAV and transcribes with Whisper.
    """
    # audio_bytes = file.file.read()

    # # Convert to mono PCM 16kHz WAV
    # audio = AudioSegment.from_file(BytesIO(audio_bytes))
    # audio = audio.set_channels(1).set_frame_rate(16000)

    # buffer = BytesIO()
    # audio.export(buffer, format="wav")
    # buffer.seek(0)

    # transcription = openai.audio.transcriptions.create(
    #     model="whisper-1",
    #     file=("audio.wav", buffer, "audio/wav")
    # )
    # return transcription.text.strip()
    audio_bytes = await file.read()
    
    # Whisper API supports webm, mp3, wav, m4a, etc. directly - no conversion needed
    transcription = openai.audio.transcriptions.create(
        model="whisper-1",
        file=(file.filename or "audio.webm", audio_bytes, file.content_type or "audio/webm")
    )
    return transcription.text.strip()


@app.post("/process_user_input")
async def process_user_input(
    session_id: str = Form(...),
    audio: UploadFile = Form(...)
):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=400, detail="Invalid session")

    session = SESSIONS[session_id]
    current_index = session["current_index"]

    if current_index >= total_questions:
        return {
            "messages": [build_message("The survey is already complete.")],
            "ui_action": "end"
        }

    row = questions_df.iloc[current_index]
    question_text = row["question"]
    options = (
        [o.strip() for o in str(row["options"]).split(",")]
        if pd.notna(row["options"])
        else []
    )

    # user_text = transcribe_audio(audio)

    try:
        print("Transcribing audio: ", audio)
        # user_text = transcribe_audio(audio)
        audio_bytes = await audio.read()
        transcription = openai.audio.transcriptions.create(
            model="whisper-1",
            file=(audio.filename or "audio.webm", audio_bytes, audio.content_type or "audio/webm")
        )
        user_text = transcription.text.strip()
        print("User text: ", user_text)
    except Exception as e:
        print("Error transcribing audio: ", e)
        raise HTTPException(status_code=400, detail=f"Audio transcription failed: {str(e)}")


    intent_data = classify_intent_llm(
        question_text,
        options,
        user_text
    )

    intent = intent_data["intent"]
    target_q = intent_data.get("target_question")

    messages = []

    # ANSWER
    if intent == "ANSWER":
        validation = validate_with_llm(question_text, options, user_text)

        if not validation["valid"]:
            messages.append(build_message("Please answer using the given options."))
            messages.append(build_message(question_text, options))
            return {"messages": messages, "ui_action": "stay"}

        session["answers"][current_index + 1] = validation["mappedOption"]
        session["current_index"] += 1

        if session["current_index"] < total_questions:
            next_row = questions_df.iloc[session["current_index"]]
            next_options = (
                [o.strip() for o in str(next_row["options"]).split(",")]
                if pd.notna(next_row["options"])
                else []
            )
            messages.append(build_message(next_row["question"], next_options))
            return {"messages": messages, "ui_action": "next"}

        messages.append(build_message("Thank you. The survey is complete."))
        return {"messages": messages, "ui_action": "end"}

    # QUERY
    if intent == "QUERY":
        explanation = explain_question_llm(question_text)
        messages.append(build_message(explanation))
        messages.append(build_message(question_text, options))
        return {"messages": messages, "ui_action": "stay"}

    # CHANGE REQUEST
    if intent == "CHANGE_REQUEST" and target_q:
        if target_q < 1 or target_q > total_questions:
            messages.append(build_message("That question number is invalid."))
            messages.append(build_message(question_text, options))
            return {"messages": messages, "ui_action": "stay"}

        session["current_index"] = target_q - 1
        row = questions_df.iloc[session["current_index"]]
        opts = (
            [o.strip() for o in str(row["options"]).split(",")]
            if pd.notna(row["options"])
            else []
        )
        messages.append(build_message("Okay, let’s update your answer."))
        messages.append(build_message(row["question"], opts))
        return {"messages": messages, "ui_action": "reask"}

    # INVALID
    messages.append(build_message("I didn’t understand that. Please answer clearly."))
    messages.append(build_message(question_text, options))
    return {"messages": messages, "ui_action": "stay"}

