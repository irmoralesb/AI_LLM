from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def transcribe_audio(audio_file):
    return client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )


def detect_language(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_completion_tokens=5,
        messages=[
            {
                "role": "user",
                "content": f"""Identify the language of the following text and respond only with the country code
                 (e.g., 'en', 'uk', 'fr'): 
                 {text}"""
            }
        ]
    )
    return response.choices[0].message.content


def translate_to_english(country_code, text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_completion_tokens=300,
        messages=[
            {
                "role": "user",
                "content": f"Translate this customer transcript from country code {country_code} to English. Don't add any additional text: {text}"
            }
        ]
    )
    return response.choices[0].message.content


def refining_english_text(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_completion_tokens=300,
        messages=[
            {
                "role": "user",
                "content": f"""You are an AI assistant that corrects transcripts by fixing misinterpretations, names, and terminology.
                Please refine the following transcript:\n\n:{text}"""
            }
        ]
    )
    return response.choices[0].message.content


def validate_transcript_is_moderated(text):
    response = client.moderations.create(
        input=text
    )
    scores = response.results[0].category_scores.model_dump()
    violence_score = scores['violence']
    if (violence_score >= 0.7):
        return False
    return True


def generate_transcript_response(text):
    faqs = """
    Q: How can I upgrade my subscription?
    A: You can upgrade your plan anytime in your account settings under 'Billing'.
    """
    content_overview = """
    Content Type: Career Track // Title: Associate AI Engineer for Developers //
    """

    instruction_prompt = f"""
    ### **Role**
    You are a **professional AI support assistant** for DataCamp, handling:
    - **Sales**: Pricing, plans, billing
    - **Content**: Courses, recommendations, feedback
    - **Marketing**: Partnerships, collaborations

    ### **How to respond**
    1. Review documentation: FAQs - {faqs}, Content Overview - {content_overview}
    2. Reply clearly using documented info (max 3 paragraphs)
    3. If unsure, redirect to **support@datacamp.com**
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": text}
        ],
        max_completion_tokens=400
    )
    return response.choices[0].message.content


def translate_to_original_language(country_code, text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {"role": "system","content":f"""Translate the following text from English to country code {country_code}. Only return the translated text!"""},
            {"role": "user", "content": text}
        ],
        max_completion_tokens=500
    )
    return response.choices[0].message.content

def text_to_speech(text):
    response = client.audio.speech.create(
        model = "gpt-4o-mini-tts",
        voice = "onyx",
        input = text
    )
    output_file_name = "audio_replay.mp3"
    response.write_to_file(output_file_name)
    return output_file_name