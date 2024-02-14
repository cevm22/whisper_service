
from typing import Union
import datetime
from pydantic import BaseModel
from fastapi import FastAPI
import test
import utils

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/translate")
def translate_file(filename: str):
    # PENDING TO ADD FUNCTION TO TRANSLATION
    # 1 - Get the full path of the .wav or .ogg file
    bucket_path = '/app/bucket/'
    file_path = str(bucket_path) + str(filename) 
    # 2 - Ask for transcription with test.lets_transcribe(file_path, "txt")
    res = test.lets_transcribe(file_path, "txt")
    # 3 - Return the transcription
    
    
    # Translate the content
    translated_text = "DUMMY TEXT"

    return {"filename": filename, "transcription": res, "filepath":file_path}