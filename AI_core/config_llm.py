import sys
from os import environ, path
from dotenv import load_dotenv

# Load dotenv file
try:
    load_dotenv(path.abspath(path.dirname(sys.argv[0]))+'/.env_llm')
except FileNotFoundError:
    print ('No such config file')

class Config_LLM:
    MODEL_ID = environ.get('MODEL_ID')  # Model name
    VOICE_PRESET = environ.get('VOICE_PRESET')  # Path to the voice preset for TTS
    STOPPER = environ.get('STOPPER')  # Stop of the dialogue
    FINAL_PHRASE = environ.get('FINAL_PHRASE')  # Phrase to finish the dialogue

    NOT_ENGLISH_ANSWER = environ.get('NOT_ENGLISH_ANSWER')  # Suggestion to use English
    NOT_ENGLISH_ANSWER_RUS = environ.get('NOT_ENGLISH_ANSWER_RUS') # Suggestion to use English Russian

    NOT_HEARD_ANSWER = environ.get('NOT_HEARD_ANSWER')  # Suggestion to repeat what was said
    NOT_HEARD_ANSWER_RUS = environ.get('NOT_HEARD_ANSWER_RUS') # Suggestion to repeat what was said Russian

    NOT_ENGLISH_GENERATION = environ.get('NOT_ENGLISH_GENERATION')  # When generation is not in English
    NOT_ENGLISH_GENERATION_RUS = environ.get('NOT_ENGLISH_GENERATION_RUS')  # When generation is not in English Russian

    IF_TOXIC_ANSWER = environ.get('IF_TOXIC_ANSWER')  # Asking for changing the theme of the dialogue
    IF_TOXIC_ANSWER_RUS = environ.get('IF_TOXIC_ANSWER_RUS')  # Asking for changing the theme of the dialogue Russian

    LLM_ERROR_ANSWER = environ.get('LLM_ERROR_ANSWER')  # Answer when problems with LLM

    FIRST_MSG_TRIGGERS = environ.get('FIRST_MSG_TRIGGERS')  # Phrases for initializing the dialogue
    FIRST_MSG_TRIGGERS_RUS = environ.get('FIRST_MSG_TRIGGERS_RUS')  # Phrases for initializing the dialogue Russian

    USER_INPUT_BLOCK = environ.get('USER_INPUT_BLOCK') # Frame for user's input
    BASIC_ANSWER_PROMPT = environ.get('BASIC_ANSWER_PROMPT')  # Prompt for generating answer to user's input
    CONSIDER_LEVEL_ANSWER_PROMPT = environ.get('CONSIDER_LEVEL_ANSWER_PROMPT')  # Prompt for generating answer to user's input considering language level

    CHECK_MISTAKES_ANSWER_PROMPT = environ.get('CHECK_MISTAKES_ANSWER_PROMPT')  # Prompt for analyzing mistakes in user's input
    SHOW_ANSWERS_PROMPT = environ.get('SHOW_ANSWERS_PROMPT')  # Prompt for generating answers to model's output

    FOLDER_PATH = path.abspath(path.dirname(sys.argv[0])) # Absolute path where the program executes
    TEXT_DETOX_PATH = environ.get('TEXT_DETOX_PATH')  # Path to textdetox model
    SPEECH_TEXT_PATH = environ.get('SPEECH_TEXT_PATH')  # Path to speech-text model
    TEXT_SPEECH_PATH = environ.get('TEXT_SPEECH_PATH')  # Path to text-speech model
    TRANSLATION_PATH = environ.get('TRANSLATION_PATH')  # Path to translation model