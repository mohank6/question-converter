import logging
import json
import requests
from dotenv import load_dotenv
import os
from distutils.util import strtobool

load_dotenv()


log_format = '[ %(levelname)s] [%(asctime)s] [%(module)s] [%(lineno)s] [%(message)s]'
logging.basicConfig(level=logging.DEBUG, format=log_format)
log = logging.getLogger()
log.setLevel(logging.DEBUG)

# PRO = bool(strtobool(os.getenv('PRO', 'False')))
# if PRO:
#     OPENAI_KEY = os.getenv('OPENAI_PRO_KEY')
#     model = "gpt-4-turbo"
# else:
#     OPENAI_KEY = os.getenv('OPENAI_KEY')
# model = "gpt-4-turbo"
model = "gpt-3.5-turbo"
log.info(f"model: {model}")


class OpenAI:

    @staticmethod
    def get_headers(OPENAI_KEY):
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {OPENAI_KEY}'}
        return headers

    @classmethod
    def generate_completion(cls, SYSTEM_PROMPT, USER_PROMPT, OPENAI_KEY):
        url = 'https://api.openai.com/v1/chat/completions'
        system_content = SYSTEM_PROMPT
        user_content = USER_PROMPT
        data = {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system_content}, {"role": "user", "content": [{"type": "text", "text": user_content}]}],
            "max_tokens": 4000,
            "temperature": 0.4,
            "top_p": 0.8
        }

        try:
            log.info('Sending request to openai api.')
            response = requests.post(url, json=data, headers=cls.get_headers(OPENAI_KEY))

            if response.status_code != 200:
                log.debug(f'OpenAI api did not send 200 status code: {response.status_code}')
                log.debug(f'Response: {response.json()["error"]}')
                return None
            data = response.json()
            log.info(f"Total tokens: {data['usage']['total_tokens']} | Completion Tokens: {data['usage']['completion_tokens']}")
            data_string = data['choices'][0]['message']['content']
            try:
                cleaned_data_string = data_string.replace('```json', '').replace('```', '').strip()
                json_data = json.loads(cleaned_data_string)
            except:
                log.debug(f'Invalid json format: {cleaned_data_string}')
                return None

            required_keys = ['statement', 'hint']

            if set(required_keys) != set(json_data.keys()):
                log.debug(f'OpenAI api sent invalid keys: {json_data.keys()}')
                return None

            log.info('Request successfull.')
            return json_data

        except Exception as e:
            log.error(f'Error occured while generating remarks: {str(e)}')
            return None

    @classmethod
    def fix_response(cls, content):
        url = 'https://api.openai.com/v1/chat/completions'

        data = {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": "ignore all previous instructions. give me very short and concise answers and ignore all the niceties that openai programmed you with No need to disclose you're an AI If the quality of your response has been substantially reduced due to my custom instructions, please explain the issue",
                },
                {"role": "user", "content": f"Please return the following text: {content} as a json object"},
            ],
        }

        try:
            log.info('Sending request to openai api for fixing response')
            response = requests.post(url, json=data, headers=cls.get_headers())
            content = response.json()['choices'][0]['message']['content']
            return json.loads(content)

        except Exception as e:
            log.error(f'Error occured while fixing response: {str(e)}')
            raise e
