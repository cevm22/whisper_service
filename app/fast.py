
from typing import Union
import datetime
from pydantic import BaseModel
from fastapi import FastAPI
import test
import utils
import re

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
    
    # 3 - get transcription data
    # path of transcription .txt
    transcription_path = res[1][0]
    transcription_data = utils.read_txt_file(f'./{transcription_path}')
    
    # 4 - get file name without extension after dot
    match = re.match(r'.*/([^/.]+)\.', file_path)
    only_file_name = match.group(1)
    
    # 5 - move txt file to bucket
    utils.move_and_rename_file(f'./{transcription_path}', f'{bucket_path}', f'{only_file_name}.txt')
    
    # 6 - Return the transcription
    return {"filename": filename, "transcription": transcription_data, "audio_filepath":file_path}