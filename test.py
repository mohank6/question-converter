import logging
import json
import requests
from dotenv import load_dotenv
import os
from distutils.util import strtobool

load_dotenv()


model = "gpt-3.5-turbo"


def generate_completion(SYSTEM_PROMPT, USER_PROMPT, OPENAI_KEY):
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
        response = requests.post(url, json=data, headers=get_headers(OPENAI_KEY))

        if response.status_code != 200:
            return None, response.json()["error"]
        data = response.json()
        data_string = data['choices'][0]['message']['content']
        try:
            cleaned_data_string = data_string.replace('```json', '').replace('```', '').strip()
            json_data = json.loads(cleaned_data_string)
        except:
            return None, None

        required_keys = ['statement', 'hint']

        if set(required_keys) != set(json_data.keys()):
            return None, None

        return json_data, None

    except Exception as e:
        return None, None


def get_headers(OPENAI_KEY):
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {OPENAI_KEY}'}
    return headers
