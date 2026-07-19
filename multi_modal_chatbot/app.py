import os
import json
import base64
import sqlite3
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from PIL import Image

# ---------------------------------------------------------
# 1. INITIALIZATION & SETUP
# ---------------------------------------------------------
load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

MODEL = "gpt-4.1-mini"
openai = OpenAI()
DB = "prices.db"

# Setup a quick mock database for pricing lookups
with sqlite3.connect(DB) as conn:
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS prices (city TEXT PRIMARY KEY, price INTEGER)')
    cursor.executemany('INSERT OR REPLACE INTO prices VALUES (?, ?)', [
        ('paris', 450), ('new york city', 350), ('tokyo', 800), ('london', 400)
    ])
    conn.commit()

# ---------------------------------------------------------
# 2. PROMPT & TOOLS DEFINITION
# ---------------------------------------------------------
system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}
tools = [{"type": "function", "function": price_function}]

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS (DATABASE, ARTIST, AUDIO)
# ---------------------------------------------------------
def get_ticket_price(city):
    print(f"DATABASE TOOL CALLED: Getting price for {city}", flush=True)
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT price FROM prices WHERE city = ?', (city.lower(),))
        result = cursor.fetchone()
        return f"Ticket price to {city} is ${result[0]}" if result else "No price data available for this city"

def artist(city):
    image_response = openai.images.generate(
        model="gpt-image-1-mini",
        prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
        size="1024x1024",
        n=1,
    )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))

def talker(message):
    response = openai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input=message
    )
    return response.content

def handle_tool_calls_and_return_cities(message):
    responses = []
    cities = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            arguments = json.loads(tool_call.function.arguments)
            city = arguments.get('destination_city')
            cities.append(city)
            price_details = get_ticket_price(city)
            responses.append({
                "role": "tool",
                "content": price_details,
                "tool_call_id": tool_call.id
            })
    return responses, cities

# ---------------------------------------------------------
# 4. CORE CHAT ENGINE
# ---------------------------------------------------------
def chat(history):
    formatted_messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + formatted_messages
    
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    cities = []
    image = None

    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        responses, extracted_cities = handle_tool_calls_and_return_cities(message)
        cities.extend(extracted_cities)
        messages.append(message)
        messages.extend(responses)
        response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    reply = response.choices[0].message.content
    history += [{"role": "assistant", "content": reply}]

    voice = talker(reply)

    if cities:
        try:
            image = artist(cities[0])
        except Exception as e:
            print(f"Image generation failed: {e}")
    
    return history, voice, image

# ---------------------------------------------------------
# 5. GRADIO USER INTERFACE
# ---------------------------------------------------------
def put_message_in_chatbot(message, history):
    return "", history + [{"role": "user", "content": message}]

with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500)
        image_output = gr.Image(height=500, interactive=False)
    with gr.Row():
        audio_output = gr.Audio(autoplay=True)
    with gr.Row():
        message = gr.Textbox(label="Chat with our AI Assistant:")

    message.submit(
        put_message_in_chatbot, 
        inputs=[message, chatbot], 
        outputs=[message, chatbot]
    ).then(
        chat, 
        inputs=chatbot, 
        outputs=[chatbot, audio_output, image_output]
    )

if __name__ == "__main__":
    ui.launch(inbrowser=True)