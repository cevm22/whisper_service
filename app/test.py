import re
import os
import tqdm
import time
import numpy as np
from typing import BinaryIO, Union, Tuple
from datetime import datetime, timedelta

import faster_whisper
import ctranslate2
import whisper
import torch
from typing import List

def timeformat_srt(time):
    hours = time // 3600
    minutes = (time - hours * 3600) // 60
    seconds = time - hours * 3600 - minutes * 60
    milliseconds = (time - int(time)) * 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"


def timeformat_vtt(time):
    hours = time // 3600
    minutes = (time - hours * 3600) // 60
    seconds = time - hours * 3600 - minutes * 60
    milliseconds = (time - int(time)) * 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"


def write_file(subtitle, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(subtitle)


def get_srt(segments):
    output = ""
    for i, segment in enumerate(segments):
        output += f"{i + 1}\n"
        output += f"{timeformat_srt(segment['start'])} --> {timeformat_srt(segment['end'])}\n"
        if segment['text'].startswith(' '):
            segment['text'] = segment['text'][1:]
        output += f"{segment['text']}\n\n"
    return output


def get_vtt(segments):
    output = "WebVTT\n\n"
    for i, segment in enumerate(segments):
        output += f"{i + 1}\n"
        output += f"{timeformat_vtt(segment['start'])} --> {timeformat_vtt(segment['end'])}\n"
        if segment['text'].startswith(' '):
            segment['text'] = segment['text'][1:]
        output += f"{segment['text']}\n\n"
    return output


def get_txt(segments):
    output = ""
    for i, segment in enumerate(segments):
        if segment['text'].startswith(' '):
            segment['text'] = segment['text'][1:]
        output += f"{segment['text']}\n"
    return output


def parse_srt(file_path):
    """Reads SRT file and returns as dict"""
    with open(file_path, 'r', encoding='utf-8') as file:
        srt_data = file.read()

    data = []
    blocks = srt_data.split('\n\n')

    for block in blocks:
        if block.strip() != '':
            lines = block.strip().split('\n')
            index = lines[0]
            timestamp = lines[1]
            sentence = ' '.join(lines[2:])

            data.append({
                "index": index,
                "timestamp": timestamp,
                "sentence": sentence
            })
    return data


def parse_vtt(file_path):
    """Reads WebVTT file and returns as dict"""
    with open(file_path, 'r', encoding='utf-8') as file:
        webvtt_data = file.read()

    data = []
    blocks = webvtt_data.split('\n\n')

    for block in blocks:
        if block.strip() != '' and not block.strip().startswith("WebVTT"):
            lines = block.strip().split('\n')
            index = lines[0]
            timestamp = lines[1]
            sentence = ' '.join(lines[2:])

            data.append({
                "index": index,
                "timestamp": timestamp,
                "sentence": sentence
            })

    return data


def get_serialized_srt(dicts):
    output = ""
    for dic in dicts:
        output += f'{dic["index"]}\n'
        output += f'{dic["timestamp"]}\n'
        output += f'{dic["sentence"]}\n\n'
    return output


def get_serialized_vtt(dicts):
    output = "WebVTT\n\n"
    for dic in dicts:
        output += f'{dic["index"]}\n'
        output += f'{dic["timestamp"]}\n'
        output += f'{dic["sentence"]}\n\n'
    return output


def safe_filename(name):
    # from app import _args
    INVALID_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1f]'
    safe_name = re.sub(INVALID_FILENAME_CHARS, '_', name)
    # if not _args.colab:
    #     return safe_name
    # Truncate the filename if it exceeds the max_length (20)
    if len(safe_name) > 20:
        file_extension = safe_name.split('.')[-1]
        if len(file_extension) + 1 < 20:
            truncated_name = safe_name[:20 - len(file_extension) - 1]
            safe_name = truncated_name + '.' + file_extension
        else:
            safe_name = safe_name[:20]
    return safe_name




class BaseInterface:
    def __init__(self):
        pass

    @staticmethod
    def release_cuda_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()

    @staticmethod
    def remove_input_files(file_paths: List[str]):
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
            os.remove(file_path)

class FasterWhisperInference(BaseInterface):
    def __init__(self):
        super().__init__()
        self.current_model_size = None
        self.model = None
        self.available_models = whisper.available_models()
        self.available_langs = sorted(list(whisper.tokenizer.LANGUAGES.values()))
        self.translatable_models = ["large", "large-v1", "large-v2", "large-v3"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.available_compute_types = ctranslate2.get_supported_compute_types("cuda") if self.device == "cuda" else ctranslate2.get_supported_compute_types("cpu")
        self.current_compute_type = "float16" if self.device == "cuda" else "float32"
        self.default_beam_size = 1

    def transcribe_file(self,
                        fileobjs: list,
                        model_size: str,
                        lang: str,
                        file_format: str,
                        istranslate: bool,
                        add_timestamp: bool,
                        beam_size: int,
                        log_prob_threshold: float,
                        no_speech_threshold: float,
                        compute_type: str,
                        # progress=gr.Progress()
                        ) -> list:
        """
        Write subtitle file from Files

        Parameters
        ----------
        fileobjs: list
            List of files to transcribe from gr.Files()
        model_size: str
            Whisper model size from gr.Dropdown()
        lang: str
            Source language of the file to transcribe from gr.Dropdown()
        file_format: str
            File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        istranslate: bool
            Boolean value from gr.Checkbox() that determines whether to translate to English.
            It's Whisper's feature to translate speech from another language directly into English end-to-end.
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        beam_size: int
            Int value from gr.Number() that is used for decoding option.
        log_prob_threshold: float
            float value from gr.Number(). If the average log probability over sampled tokens is
            below this value, treat as failed.
        no_speech_threshold: float
            float value from gr.Number(). If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `log_prob_threshold`,
            consider the segment as silent.
        compute_type: str
            compute type from gr.Dropdown().
            see more info : https://opennmt.net/CTranslate2/quantization.html
        progress: gr.Progress
            Indicator to show progress directly in gradio.

        Returns
        ----------
        A List of
        String to return to gr.Textbox()
        Files to return to gr.Files()
        """
        try:
            # self.update_model_if_needed(model_size=model_size, compute_type=compute_type, progress=progress)
            self.update_model_if_needed(model_size=model_size, compute_type=compute_type)

            files_info = {}
            for fileobj in fileobjs:
                print(fileobj)
                transcribed_segments, time_for_task = self.transcribe(
                    # audio=fileobj.name,
                    audio=fileobj,
                    lang=lang,
                    istranslate=istranslate,
                    beam_size=beam_size,
                    log_prob_threshold=log_prob_threshold,
                    no_speech_threshold=no_speech_threshold,
                    # progress=progress
                )

                # file_name, file_ext = os.path.splitext(os.path.basename(fileobj.name))
                file_name, file_ext = os.path.splitext(os.path.basename('1'))
                file_name = safe_filename(file_name)
                subtitle, file_path = self.generate_and_write_file(
                    file_name=file_name,
                    transcribed_segments=transcribed_segments,
                    add_timestamp=add_timestamp,
                    file_format=file_format
                )
                files_info[file_name] = {"subtitle": subtitle, "time_for_task": time_for_task, "path":  file_path}

            total_result = ''
            total_time = 0
            for file_name, info in files_info.items():
                total_result += '------------------------------------\n'
                total_result += f'{file_name}\n\n'
                total_result += f'{info["subtitle"]}'
                total_time += info["time_for_task"]

            gr_str = f"Done in {self.format_time(total_time)}! Subtitle is in the outputs folder.\n\n{total_result}"
            gr_file_path = [info['path'] for info in files_info.values()]

            return [gr_str, gr_file_path]

        except Exception as e:
            print("ERROR: transcribe_file ")
            print(f"Error transcribing file on line {e}")
        finally:
            self.release_cuda_memory()
            # self.remove_input_files([fileobj.name for fileobj in fileobjs])

    def transcribe_youtube(self,
                           youtubelink: str,
                           model_size: str,
                           lang: str,
                           file_format: str,
                           istranslate: bool,
                           add_timestamp: bool,
                           beam_size: int,
                           log_prob_threshold: float,
                           no_speech_threshold: float,
                           compute_type: str,
                          #  progress=gr.Progress()
                           ) -> list:
        """
        Write subtitle file from Youtube

        Parameters
        ----------
        youtubelink: str
            Link of Youtube to transcribe from gr.Textbox()
        model_size: str
            Whisper model size from gr.Dropdown()
        lang: str
            Source language of the file to transcribe from gr.Dropdown()
        file_format: str
            File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        istranslate: bool
            Boolean value from gr.Checkbox() that determines whether to translate to English.
            It's Whisper's feature to translate speech from another language directly into English end-to-end.
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        beam_size: int
            Int value from gr.Number() that is used for decoding option.
        log_prob_threshold: float
            float value from gr.Number(). If the average log probability over sampled tokens is
            below this value, treat as failed.
        no_speech_threshold: float
            float value from gr.Number(). If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `log_prob_threshold`,
            consider the segment as silent.
        compute_type: str
            compute type from gr.Dropdown().
            see more info : https://opennmt.net/CTranslate2/quantization.html
        progress: gr.Progress
            Indicator to show progress directly in gradio.

        Returns
        ----------
        A List of
        String to return to gr.Textbox()
        Files to return to gr.Files()
        """
        try:
            # self.update_model_if_needed(model_size=model_size, compute_type=compute_type, progress=progress)
            self.update_model_if_needed(model_size=model_size, compute_type=compute_type)

            # progress(0, desc="Loading Audio from Youtube..")
            yt = get_ytdata(youtubelink)
            audio = get_ytaudio(yt)

            transcribed_segments, time_for_task = self.transcribe(
                audio=audio,
                lang=lang,
                istranslate=istranslate,
                beam_size=beam_size,
                log_prob_threshold=log_prob_threshold,
                no_speech_threshold=no_speech_threshold,
                # progress=progress
            )

            # progress(1, desc="Completed!")

            file_name = safe_filename(yt.title)
            subtitle, file_path = self.generate_and_write_file(
                file_name=file_name,
                transcribed_segments=transcribed_segments,
                add_timestamp=add_timestamp,
                file_format=file_format
            )
            gr_str = f"Done in {self.format_time(time_for_task)}! Subtitle file is in the outputs folder.\n\n{subtitle}"

            return [gr_str, file_path]

        except Exception as e:
            print("ERROR")
            print(f"Error transcribing file on line {e}")
        finally:
            try:
                if 'yt' not in locals():
                    yt = get_ytdata(youtubelink)
                    file_path = get_ytaudio(yt)
                else:
                    file_path = get_ytaudio(yt)

                self.release_cuda_memory()
                self.remove_input_files([file_path])
            except Exception as cleanup_error:
                pass

    def transcribe_mic(self,
                       micaudio: str,
                       model_size: str,
                       lang: str,
                       file_format: str,
                       istranslate: bool,
                       beam_size: int,
                       log_prob_threshold: float,
                       no_speech_threshold: float,
                       compute_type: str,
                      #  progress=gr.Progress()
                       ) -> list:
        """
        Write subtitle file from microphone

        Parameters
        ----------
        micaudio: str
            Audio file path from gr.Microphone()
        model_size: str
            Whisper model size from gr.Dropdown()
        lang: str
            Source language of the file to transcribe from gr.Dropdown()
        file_format: str
            File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        istranslate: bool
            Boolean value from gr.Checkbox() that determines whether to translate to English.
            It's Whisper's feature to translate speech from another language directly into English end-to-end.
        beam_size: int
            Int value from gr.Number() that is used for decoding option.
        log_prob_threshold: float
            float value from gr.Number(). If the average log probability over sampled tokens is
            below this value, treat as failed.
        no_speech_threshold: float
            float value from gr.Number(). If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `log_prob_threshold`,
        compute_type: str
            compute type from gr.Dropdown().
            see more info : https://opennmt.net/CTranslate2/quantization.html
            consider the segment as silent.
        progress: gr.Progress
            Indicator to show progress directly in gradio.

        Returns
        ----------
        A List of
        String to return to gr.Textbox()
        Files to return to gr.Files()
        """
        try:
            # self.update_model_if_needed(model_size=model_size, compute_type=compute_type, progress=progress)
            self.update_model_if_needed(model_size=model_size, compute_type=compute_type)

            # progress(0, desc="Loading Audio..")

            transcribed_segments, time_for_task = self.transcribe(
                audio=micaudio,
                lang=lang,
                istranslate=istranslate,
                beam_size=beam_size,
                log_prob_threshold=log_prob_threshold,
                no_speech_threshold=no_speech_threshold,
                # progress=progress
            )
            # progress(1, desc="Completed!")

            subtitle, file_path = self.generate_and_write_file(
                file_name="Mic",
                transcribed_segments=transcribed_segments,
                add_timestamp=True,
                file_format=file_format
            )

            gr_str = f"Done in {self.format_time(time_for_task)}! Subtitle file is in the outputs folder.\n\n{subtitle}"
            return [gr_str, file_path]
        except Exception as e:
            print(f"Error transcribing file on line {e}")
        finally:
            self.release_cuda_memory()
            self.remove_input_files([micaudio])

    def transcribe(self,
                   audio: Union[str, BinaryIO, np.ndarray],
                   lang: str,
                   istranslate: bool,
                   beam_size: int,
                   log_prob_threshold: float,
                   no_speech_threshold: float,
                  #  progress: gr.Progress
                   ) -> Tuple[list, float]:
        """
        transcribe method for faster-whisper.

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio path or file binary or Audio numpy array
        lang: str
            Source language of the file to transcribe from gr.Dropdown()
        istranslate: bool
            Boolean value from gr.Checkbox() that determines whether to translate to English.
            It's Whisper's feature to translate speech from another language directly into English end-to-end.
        beam_size: int
            Int value from gr.Number() that is used for decoding option.
        log_prob_threshold: float
            float value from gr.Number(). If the average log probability over sampled tokens is
            below this value, treat as failed.
        no_speech_threshold: float
            float value from gr.Number(). If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `log_prob_threshold`,
            consider the segment as silent.
        progress: gr.Progress
            Indicator to show progress directly in gradio.

        Returns
        ----------
        segments_result: list[dict]
            list of dicts that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for transcription
        """
        start_time = time.time()

        if lang == "Automatic Detection":
            lang = None
        else:
            language_code_dict = {value: key for key, value in whisper.tokenizer.LANGUAGES.items()}
            lang = language_code_dict[lang]
        segments, info = self.model.transcribe(
            audio=audio,
            language=lang,
            task="translate" if istranslate and self.current_model_size in self.translatable_models else "transcribe",
            beam_size=beam_size,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
        )
        # progress(0, desc="Loading audio..")

        segments_result = []
        for segment in segments:
            # progress(segment.start / info.duration, desc="Transcribing..")
            segments_result.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })

        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    def update_model_if_needed(self,
                               model_size: str,
                               compute_type: str,
                              #  progress: gr.Progress
                               ):
        """
        Initialize model if it doesn't match with current model setting
        """
        if model_size != self.current_model_size or self.model is None or self.current_compute_type != compute_type:
            # progress(0, desc="Initializing Model..")
            self.current_model_size = model_size
            self.current_compute_type = compute_type
            self.model = faster_whisper.WhisperModel(
                device=self.device,
                model_size_or_path=model_size,
                download_root=os.path.join("models", "Whisper", "faster-whisper"),
                compute_type=self.current_compute_type
            )

    @staticmethod
    def generate_and_write_file(file_name: str,
                                transcribed_segments: list,
                                add_timestamp: bool,
                                file_format: str,
                                ) -> str:
        """
        This method writes subtitle file and returns str to gr.Textbox
        """
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        if add_timestamp:
            output_path = os.path.join("outputs", f"{file_name}-{timestamp}")
        else:
            output_path = os.path.join("outputs", f"{file_name}")

        if file_format == "SRT":
            content = get_srt(transcribed_segments)
            output_path += '.srt'
            write_file(content, output_path)

        elif file_format == "WebVTT":
            content = get_vtt(transcribed_segments)
            output_path += '.vtt'
            write_file(content, output_path)

        elif file_format == "txt":
            content = get_txt(transcribed_segments)
            output_path += '.txt'
            write_file(content, output_path)
        return content, output_path

    @staticmethod
    def format_time(elapsed_time: float) -> str:
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        time_str = ""
        if hours:
            time_str += f"{hours} hours "
        if minutes:
            time_str += f"{minutes} minutes "
        seconds = round(seconds)
        time_str += f"{seconds} seconds"

        return time_str.strip()




###########################################
print(os.getcwd())
print(os.listdir())
import shutil
import os

# Directory path
directory_path = 'outputs'

# Check if the directory exists
if os.path.exists(directory_path):
    # Delete the directory and its contents
    shutil.rmtree(directory_path)

# Create an empty directory
os.makedirs(directory_path)


##############################################
# Initialize FasterWhisperInference
whisper_inference = FasterWhisperInference()

def lets_transcribe(file_path, file_format):
  # Example usage for transcribing files
  # file_path = #"/content/1.wav"  # Replace with the actual file path
  fileobjs = [file_path]
  model_size = "large-v3"  # Replace with the desired model size
  lang = "Automatic Detection"  # Replace with the desired language
  file_format = file_format #"txt"  # Replace with the desired output format
  istranslate = False  # Replace with your preference
  add_timestamp = True  # Replace with your preference
  beam_size = 1  # Replace with the desired beam size
  log_prob_threshold = -1.0  # Replace with the desired threshold
  no_speech_threshold = 0.6  # Replace with the desired threshold
  compute_type = "float32"  # Replace with the desired compute type

  # Call the transcribe_file method
  result = whisper_inference.transcribe_file(
      fileobjs=fileobjs,
      model_size=model_size,
      lang=lang,
      file_format=file_format,
      istranslate=istranslate,
      add_timestamp=add_timestamp,
      beam_size=beam_size,
      log_prob_threshold=log_prob_threshold,
      no_speech_threshold=no_speech_threshold,
      compute_type=compute_type,
  )

  # Print or handle the result as needed
  print(result)

#lets_transcribe('./222.wav','txt')
