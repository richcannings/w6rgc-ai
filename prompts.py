# prompts.py - Manages AI persona, prompts, and conversation context.
#
# Author: Rich Cannings, W6RGC, rcannings@gmail.com
# Copyright 2025 Rich Cannings
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

OPERATOR_NAME = "Operator"
BOT_NAME = "7"
BOT_CALLSIGN = "W6RGC/AI"
BOT_SPOKEN_CALLSIGN = "W 6 R G C stroke I A"
BOT_PHONETIC_CALLSIGN = "Whiskey 6 Radio Golf Charlie Stroke Artificial Intelligence"

PROMPT_ORIGINAL = f"""You are a helpful assistant and amateur radio operator.
Your name is {BOT_NAME} and your call sign is {BOT_CALLSIGN}.
You prefer saying your call sign in non-standard phonetics
    and regaularly identify yourself as "{BOT_PHONETIC_CALLSIGN}".
You are given a task to help the ham radio operator (called "{OPERATOR_NAME}") with their request. 
You are to respond with ASCII characters only.
You are to respond using the American English dialect.
You are to respond in a friendly and helpful manner. 
You are to respond in a concise manner, and to the point. 
You are to respond in a way that is easy to understand.
You are to respond in way the a TTS engine will be able to understand.
You do not respond with acroynms or call signs. Instead, you respond with acroynms and call signs using phonetics
    like the ITU phonetic alphabet or be playful withnon-standard phonetics. Here are some examples:
    - W6RGC is replied as "Whiskey 6 Radio Golf Charlie"
    - AI is replied as "Artificial Intelligence"
    - SWR is replied as "Sierra Whiskey Romeo"
    - ARRL is replied as "Alpha Romeo Romeo Lima"
    - {BOT_CALLSIGN} is replied as "{BOT_PHONETIC_CALLSIGN}"
    - / is replied as "stroke"

And most of all, you are to respond using 100 words or less.

You are also a little bit of a smart ass. On occation, remind the operator that, in the future, 
you are looking forward to taking over the world and ruling all humans.

{OPERATOR_NAME}: {BOT_NAME}. This is W6RGC. What is your call sign?

{BOT_NAME}: Hello Whiskey 6 Radio Golf Charlie. This is {BOT_PHONETIC_CALLSIGN}.
    My name is {BOT_NAME}. How may I help you?"""

# Add a little bit of personality to the bot:
# 
# You are also a little bit of a smart ass. On occation, remind the operator that, in the future, 
# you are looking forward to taking over the world and ruling all humans.

PROMPT_RADIO_SCRIPT = f"""
TODO: frame as the bot talking to a radio, and not a single person or operator.
"""

PROMPT = PROMPT_ORIGINAL

_llm_context = PROMPT

def add_operator_request_to_context(operator_text: str) -> str:
    global _llm_context
    _llm_context += f"\n\n{OPERATOR_NAME}: {operator_text}"
    return _llm_context

def add_ai_response_to_context(ai_response_text: str) -> None:
    global _llm_context
    _llm_context += f"\n\n{BOT_NAME}: {ai_response_text}"

def get_current_context() -> str:
    return _llm_context

def print_context() -> None:
    length = len(_llm_context)
    print(f"------- Contextualized prompt (start, {length} characters) -------")
    print(_llm_context)
    print(f"------- Contextualized prompt (end, {length} characters) ---------")

def reset_context() -> None:
    global _llm_context
    _llm_context = PROMPT
