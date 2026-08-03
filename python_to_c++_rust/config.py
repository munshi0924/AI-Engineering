"""
config.py

API key + client setup. OpenAI only.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

# Models available through the OpenAI API. Add/remove as needed.
AVAILABLE_MODELS = ["gpt-5", "gpt-5-mini", "gpt-4o"]

# Reasoning models accept a reasoning_effort param; others don't.
REASONING_MODELS = {"gpt-5", "gpt-5-mini"}

_client = None


def get_client(model: str):
    global _client
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
    if _client is None:
        _client = OpenAI(api_key=openai_api_key)
    return _client
