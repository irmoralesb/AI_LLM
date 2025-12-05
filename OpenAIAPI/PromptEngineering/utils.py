from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def print_response(prompt: str):
    print(get_response(prompt))


def get_response(prompt: str):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content


def get_response(system_prompt: str, user_prompt: str):
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content
