import torch
import numpy as np
import ollama

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSpeechSeq2Seq, AutoProcessor, BarkModel, pipeline, T5ForConditionalGeneration, T5Tokenizer, AutoModelForAudioClassification, AutoFeatureExtractor

from config_llm import Config_LLM

import langdetect
import re
import random

import io
from scipy.io.wavfile import write

from pydub import AudioSegment
from pydub.silence import split_on_silence

#Toxicity model
tokenizer_toxicity = AutoTokenizer.from_pretrained(Config_LLM.FOLDER_PATH+Config_LLM.TEXT_DETOX_PATH)
model_toxicity = AutoModelForSequenceClassification.from_pretrained(Config_LLM.FOLDER_PATH+Config_LLM.TEXT_DETOX_PATH, use_safetensors=True)

#STT model
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_speechtext = AutoModelForSpeechSeq2Seq.from_pretrained(Config_LLM.FOLDER_PATH+Config_LLM.SPEECH_TEXT_PATH, torch_dtype=torch_dtype, low_cpu_mem_usage=False, use_safetensors=True)
processor_speechtext = AutoProcessor.from_pretrained(Config_LLM.FOLDER_PATH+Config_LLM.SPEECH_TEXT_PATH, language='en')

#TTS model
processor_textspeech = AutoProcessor.from_pretrained(Config_LLM.FOLDER_PATH+Config_LLM.TEXT_SPEECH_PATH)
model_textspeech = BarkModel.from_pretrained(Config_LLM.FOLDER_PATH+Config_LLM.TEXT_SPEECH_PATH)
voice_preset = Config_LLM.VOICE_PRESET

#Translation model
model_translator = T5ForConditionalGeneration.from_pretrained(Config_LLM.FOLDER_PATH+Config_LLM.TRANSLATION_PATH)
model_translator.to('cpu')
tokenizer_translator = T5Tokenizer.from_pretrained(Config_LLM.FOLDER_PATH+Config_LLM.TRANSLATION_PATH)


async def generate_audio_answer (text: str): #Audio answer generation
    sample_rate = model_textspeech.generation_config.sample_rate

    def generate_audio_phrase (text: str): #Generate phrase
        inputs = processor_textspeech (text, voice_preset = voice_preset)
        audio_array = model_textspeech.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()

        audio_array = audio_array [int(sample_rate/2):] #Remove first half min to remove strange sounds
        return audio_array

    #Logic if text is too long
    if len (text)>40:
        audio_array = []
        phrases = re.findall(r"[\w']+|[.!?;]", text) #Split by syntax rules
        for phrase in phrases:
            if len (phrase)>40:
                sub_phrases = re.findall(r"[\w']+|[,]", phrase) #Split by comma
                for sub_phrase in sub_phrases:
                    if len (sub_phrase)>40:
                        small_phrases = []
                        for symbol in range(0, len(sub_phrase), 40): #Split by symbols
                            if (len(sub_phrase)-symbol) < 40:
                                small_phrases.append (sub_phrase[symbol:])
                                break
                            else:
                                small_phrases.append (sub_phrase[symbol:symbol+40])
                        for small_phrase in small_phrases:
                            if len(audio_array) > 0:
                                audio_array = np.concatenate ((audio_array, generate_audio_phrase (small_phrase)))
                            else:
                                audio_array = generate_audio_phrase (small_phrase)
                    else:
                        if len(audio_array) > 0:
                            audio_array = np.concatenate ((audio_array, generate_audio_phrase (sub_phrase)))
                        else:
                            audio_array = generate_audio_phrase (sub_phrase)
            else:
                if len(audio_array) > 0:
                    audio_array = np.concatenate ((audio_array, generate_audio_phrase (phrase)))
                else:
                    audio_array = generate_audio_phrase (phrase)
    else:
        audio_array = generate_audio_phrase (text)

    #Convert into bytes
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    write(byte_io, sample_rate, audio_array)
    result_bytes = byte_io.read()

    #Remove silence
    audio = AudioSegment.from_file(io.BytesIO(result_bytes), format='wav')  
    audio_chunks = split_on_silence(audio
                            ,min_silence_len = 2000
                            ,silence_thresh = -45
                            ,keep_silence = False
                        )

    audio = AudioSegment.empty()

    for chunk in audio_chunks:
        audio += chunk

    audio_numpy = np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (1 << (8 * audio.sample_width - 1))
    write(byte_io, sample_rate, audio_numpy)
    result_bytes = byte_io.read()

    return result_bytes, sample_rate, audio_numpy #Return bytes, sample rate and audio as array numpy

async def generate_answer (prompt: str): #LLM generation
    try:
        response = await ollama.AsyncClient ().chat(model=Config_LLM.MODEL_ID, messages=[{'role': 'user', 'content': prompt,},])
    except ollama.ResponseError as e:
        print('Error:', e.error)
        return Config_LLM.LLM_ERROR_ANSWER
    response = response['message']['content']
    return response

async def choose_first_msg (): #Choose first msg for trigger
    triggers = (Config_LLM.FIRST_MSG_TRIGGERS).split ('#')
    rus_s = (Config_LLM.FIRST_MSG_TRIGGERS_RUS).split ('#')
    ind = random.randint(0, (len (triggers)-1))
    response = triggers [ind]
    rus = rus_s [ind]
    return response, rus

async def process_input (user_input, params: dict, input_audio: bool): #Main func for processing logic
    def recognize_language (text_user_input: str): #Lang recognition
        try:
            langs = langdetect.detect_langs (text_user_input)
        except langdetect.lang_detect_exception.LangDetectException:
            return False
        if len(langs) > 1:
            if ((str((langs [0]))[:2]) == 'en') or ((str((langs [1]))[:2]) == 'en'):
                return True
        else:
            if (str((langs [0]))[:2]) == 'en':
                return True
        return False
    
    def is_toxic (text: str): #Toxic check
        checked_text_tokens = tokenizer_toxicity.encode(text, return_tensors='pt')
        checked_text = model_toxicity(checked_text_tokens)
        if checked_text ['logits'].detach().numpy() [0][0] >= checked_text ['logits'].detach().numpy() [0][1]:
            return False
        else:
            return True

    def clean_answer (llm_answer: str): #Removing thoughts from LLM output
        start_idx = llm_answer.index('<think>')
        end_idx = llm_answer.index('</think>')
        llm_answer = llm_answer[:start_idx] + llm_answer[end_idx+8:]
        return llm_answer
    
    def recognize_audio (audio): #Audio input recognition
        to_process = pipeline(
            'automatic-speech-recognition',
            model=model_speechtext,
            tokenizer=processor_speechtext.tokenizer,
            feature_extractor=processor_speechtext.feature_extractor,
            torch_dtype=torch_dtype,
        )
        return to_process (audio) ['text']

    async def translate (text: str): #Translation output
        input_ids = tokenizer_translator('translate to ru: '+text, return_tensors='pt')
        generated_tokens = model_translator.generate(**input_ids.to('cpu'))
        return (tokenizer_translator.batch_decode(generated_tokens, skip_special_tokens=True)) [0]

    async def prompt_logic (user_input: str, params: dict): #Logic of prompts if additional params
        async def ask_with_check_mistakes (basic_answer: str, user_input: str): #Concat fragments with original output and input mistakes check
            check_mistakes = await generate_answer (Config_LLM.CHECK_MISTAKES_ANSWER_PROMPT+Config_LLM.USER_INPUT_BLOCK+user_input+Config_LLM.USER_INPUT_BLOCK)
            if check_mistakes != Config_LLM.LLM_ERROR_ANSWER:
                check_mistakes = clean_answer (check_mistakes)
            else:
                check_mistakes = ''
            return basic_answer+'   About the errors: '+check_mistakes

        #Just generate answer
        if params ['ch_level'] != 'Auto':
            answer = await generate_answer (Config_LLM.CONSIDER_LEVEL_ANSWER_PROMPT+params ['ch_level']+'. '+Config_LLM.USER_INPUT_BLOCK+user_input+Config_LLM.USER_INPUT_BLOCK)
        else:
            answer = await generate_answer (Config_LLM.BASIC_ANSWER_PROMPT+Config_LLM.USER_INPUT_BLOCK+user_input+Config_LLM.USER_INPUT_BLOCK)
        if answer != Config_LLM.LLM_ERROR_ANSWER:
            answer = clean_answer (answer)

        if params ['ch_mist_input']: #Add mistakes check
            answer = await ask_with_check_mistakes (answer, user_input)

        return answer
    
    async def translation_logic (ch_rus, llm_answer):
        if ch_rus:
            rus = await translate (llm_answer)
            if is_toxic (rus):
                rus = False
        else:
            rus = False
        return rus

    #Logic for input
    if input_audio: #Recognize audio input
        user_input = recognize_audio (user_input)

    if not recognize_language (user_input): #Lang recognize error
        if input_audio:
            return Config_LLM.NOT_HEARD_ANSWER, Config_LLM.NOT_HEARD_ANSWER_RUS, False
        else:
            return Config_LLM.NOT_ENGLISH_ANSWER, Config_LLM.NOT_ENGLISH_ANSWER_RUS, False
    
    if is_toxic (user_input): #Toxic recognize error
        return Config_LLM.IF_TOXIC_ANSWER, Config_LLM.IF_TOXIC_ANSWER_RUS, False
    
    #Logic for output
    llm_answer = await prompt_logic (user_input, params) #Generate response

    if not recognize_language (llm_answer): #Check output lang error
        return Config_LLM.NOT_ENGLISH_GENERATION, Config_LLM.NOT_ENGLISH_GENERATION_RUS, False

    if is_toxic (llm_answer): #Check output toxic error
        return Config_LLM.IF_TOXIC_ANSWER, Config_LLM.IF_TOXIC_ANSWER_RUS, False
    
    #Logic for translation:
    rus = await translation_logic (params ['ch_rus'], llm_answer)

    #Logic for generating possible user's answers
    if params ['ch_gen_answers']: #Add answers
        prompt = Config_LLM.SHOW_ANSWERS_PROMPT+Config_LLM.USER_INPUT_BLOCK+llm_answer+Config_LLM.USER_INPUT_BLOCK
        add_answer = await generate_answer (prompt)

        if add_answer != Config_LLM.LLM_ERROR_ANSWER:
            add_answer = clean_answer (add_answer)
        
        if not recognize_language (add_answer): #Check output lang error
            add_answer = False

        if is_toxic (add_answer):
            add_answer = False

        add_answer = add_answer.replace ('#', '')
    else:
        add_answer = False

    #Logic for audio output
    if (params ['ch_audio_output']) and (llm_answer != Config_LLM.LLM_ERROR_ANSWER):

        llm_answer, sample_rate, audio_array = await generate_audio_answer (llm_answer) #Generate audio output

        transcribed_llm_answer = recognize_audio (llm_answer) #Describe audio output for additional safety checks
        if not recognize_language (transcribed_llm_answer): #If not English in audio output
            return Config_LLM.NOT_ENGLISH_GENERATION, Config_LLM.NOT_ENGLISH_GENERATION_RUS, False
        
        if is_toxic (transcribed_llm_answer): #If audio output is toxic
            return Config_LLM.IF_TOXIC_ANSWER, Config_LLM.IF_TOXIC_ANSWER_RUS, False

    return llm_answer, rus, add_answer