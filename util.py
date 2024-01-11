import os
from dotenv import load_dotenv, find_dotenv
from termcolor import colored
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import requests

load_dotenv(find_dotenv())


def get_chat_response(system_prompt, user_prompt):
    # By default, use the local LLM
    llm_type = os.environ.get('LLM_TYPE', "local")
    if llm_type == "local":
        return get_local_llm_response(system_prompt, user_prompt)
    else:
        return get_openai_response(system_prompt, user_prompt)


def get_local_llm_response(system_prompt, user_prompt, model="openchat_3.5", temperature=0.9):
    api_path = os.environ.get('LOCAL_OPENAI_API', 'http://localhost:18888/v1/chat/completions')
    model_name = os.environ.get('LOCAL_MODEL_NAME', model)
    openai_api_key = os.environ.get('OPENAI_API_KEY', 'null')
    headers = {
        'Authorization': f'Bearer {openai_api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': model_name,
        'temperature': temperature,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
    }
    response = requests.post(api_path, headers=headers, json=data)
    response_json = response.json()
    print(response_json)
    return response_json['choices'][0]['message']['content']


def get_openai_response(system_prompt, user_prompt, model="gpt-3.5-turbo", temperature=0):
    chat = ChatOpenAI(model_name=model, temperature=temperature)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    response = chat(messages)
    print(response)
    return response.content
