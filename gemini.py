import json
import google.generativeai as genai
from google.generativeai.types.generation_types import StopCandidateException
from google.api_core.exceptions import ResourceExhausted
import logging
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_KEY = os.getenv('GEMINI_KEY')


log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
log = logging.getLogger()
log.setLevel(logging.DEBUG)

log = logging.getLogger('app')


class GeminiService:
    def __init__(self, SYSTEM_PROMPT) -> None:
        genai.configure(api_key=GEMINI_KEY)
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            system_instruction=[SYSTEM_PROMPT],
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
        )

    @property
    def generation_config(self):
        return {
            "temperature": 1,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }

    @property
    def safety_settings(self):
        return [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
        ]

    def generate_completion(self, USER_PROMPT):
        try:
            log.info('Sending request to gemini api.')
            conversation = self.model.start_chat()
            conversation.send_message(USER_PROMPT)

            response = conversation.last.text
            cleaned_data_string = response.replace('```json', '').replace('```', '').replace("'", "\"").strip()
            json_data = json.loads(cleaned_data_string)
            # json_data = json.loads(json.dumps(cleaned_data_string))

            required_keys = ['statement', 'hint']

            if set(required_keys) - set(json_data.keys()):
                log.debug(f'Gemini api sent invalid keys: {json_data}')
                return None

            log.info('Request successfull.')
            return json_data

        except StopCandidateException as e:
            log.error(f'Response error: {str(e)}')
            return None
        except ResourceExhausted as e:
            log.error(f'Resource exhausted: {str(e)}')
            return None
        except Exception as e:
            log.error(f'Error Gemini: {str(e)}')
            return None
