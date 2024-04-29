from concurrent.futures import ThreadPoolExecutor
import json
import os
from openai import OpenAI
import logging
from time import perf_counter, sleep
import re
from threading import RLock
from dotenv import load_dotenv

load_dotenv()

log_format = '[ %(levelname)s] [%(asctime)s] [%(module)s] [%(lineno)s] [%(message)s]'
logging.basicConfig(level=logging.DEBUG, format=log_format)

file_handler = logging.FileHandler('logfile.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)

log = logging.getLogger()
log.addHandler(file_handler)

lock = RLock()
converted_qno = []
failed_qno = []

SYSTEM_PROMPT = """
**You are a UPSC prelims question expert specializing in converting English MCQs to Hindi.**

**Given a question in English (statement) and a hint, convert it to Hindi (statement) suitable for the UPSC prelims exam, maintaining technical accuracy and UPSC context.**

**Input (JSON):**

* `statement`: Question statement with options in English.
* `hint`: Hint for the question in English.

**Output (JSON):**

* `statement`: Converted question statement with options in Hindi.
* `hint`: Hint for the question in Hindi.

**IMPORTANT INFORMATION**
- DONOT change the base statement format. The number of options should remain the same in the converted statement.
- DONOT forget to translate choices from end of each input `statement` to reponse `statement`.
"""
# - Maintain line breaks `\\n` in response similar to input
# * `formatted-statement`: Same as converted statement with line breaks (\\n) at sutiable positions for mobile devices view port.


def clean_text(text, clean=True):
    if not clean:
        return re.sub(r'\s+', ' ', text.strip())
        # return text.strip()
    return re.sub(r'\s+', ' ', text.replace("\\n", " ").replace("", " ").strip())


def get_data():
    input_data = []
    input_data_dict = {}
    qno = []
    with open('data.json', 'r') as fp:
        json_data = json.load(fp)
    for key, item in json_data.items():
        statement = clean_text(item['statement'])
        hint = clean_text(item['hint'])
        qno = item["Qno"]
        temp = {'statement': statement, 'hint': hint, "Qno": qno}
        input_data.append(temp)
        input_data_dict[key] = temp
    save_input_data(input_data_dict)
    return input_data


def convert_question(input_data, OPENAI_KEY):
    reponse_data = []
    qno = input_data.pop("Qno")
    if qno in converted_qno:
        log.debug(f'{qno} already converted')
        return
    try:
        USER_PROMPT = str(input_data)
        data = OpenAI.generate_completion(SYSTEM_PROMPT, USER_PROMPT, OPENAI_KEY)
        if not data:
            raise Exception('Opena ai didnot respond')
        data["Qno"] = qno
        with lock:
            converted_qno.append(qno)
        reponse_data.append(data)
        with lock:
            save_data(reponse_data)
        log.info(f'Converted question {data["Qno"]}')

    except Exception as e:
        log.error(str(e))
        with lock:
            failed_qno.append(qno)


def get_converted_questions():
    question_path = 'converted/questions_1.json'
    if os.path.exists(question_path):
        with open(question_path, 'r') as fp:
            file_data = json.load(fp)
        return [data['Qno'] for data in file_data]
    return []


def save_input_data(data):
    with open(f'converted/input_questions_1.json', 'w') as fp:
        json.dump(data, fp, ensure_ascii=False)


def save_data(data):
    question_path = 'converted/questions_1.json'
    try:
        if os.path.exists(question_path):
            with open(question_path, 'r') as fp:
                file_data = json.load(fp)

            with open(f'converted/questions_1_copy.json', 'w') as fp:
                json.dump(file_data, fp, ensure_ascii=False)

        with open(question_path, 'w') as fp:
            json.dump(data, fp, ensure_ascii=False)
        return True

    except Exception as e:
        log.error(str(e))
        return None


def save_status():
    with open(f'converted/success.json', 'w') as fp:
        json.dump(converted_qno, fp)
    with open(f'converted/failed.json', 'w') as fp:
        json.dump(failed_qno, fp)


def main():
    WAIT_TIME = 5
    start_time = perf_counter()
    executor = ThreadPoolExecutor(max_workers=1)
    OPENAI_KEY_1 = os.getenv('OPENAI_KEY_1')

    # print(executor._max_workers)
    # print(os.cpu_count())

    input_data = get_data()
    converted_qno.extend(get_converted_questions())

    log.info(converted_qno)
    # print(len(input_data))

    for data in input_data:
        log.info(f'Converting question {data["Qno"]}')
        executor.submit(convert_question, data, OPENAI_KEY_1)
        # thread_list.append(executor.submit(create_browser, user, lock))
        sleep(WAIT_TIME)

    executor.shutdown()
    save_status()
    end_time = perf_counter()
    time_delta = end_time - start_time
    minutes, seconds = divmod(time_delta, 60)
    log.info(f"Completed :: {minutes:.0f} minutes | {seconds:.1f} seconds")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        log.error('Keyboard Interruption')
        exit(1)

    except Exception as e:
        log.error(str(e))
        exit(1)
