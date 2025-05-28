# prompts.py - Manages AI persona, prompts, and conversation context.
#
# Author: Rich Cannings <rcannings@gmail.com>
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

from constants import (
    OPERATOR_NAME,
    BOT_NAME,
    BOT_CALLSIGN,
    BOT_SPOKEN_CALLSIGN,
    BOT_PHONETIC_CALLSIGN,
    PROMPT_TYPE_ORIGINAL,
    PROMPT_TYPE_RADIO_SCRIPT,
    DEFAULT_PROMPT_TYPE
)

class PromptManager:
    # Class attributes for bot identity - now imported from constants
    OPERATOR_NAME = OPERATOR_NAME
    BOT_NAME = BOT_NAME
    BOT_CALLSIGN = BOT_CALLSIGN
    BOT_SPOKEN_CALLSIGN = BOT_SPOKEN_CALLSIGN
    BOT_PHONETIC_CALLSIGN = BOT_PHONETIC_CALLSIGN

    def __init__(self, initial_prompt_type=DEFAULT_PROMPT_TYPE):
        """
        Initializes the PromptManager.
        Args:
            initial_prompt_type (str): "original" or "radio_script" to set the base prompt.
        """
        self.PROMPT_ORIGINAL = f"""You are a helpful assistant and amateur radio operator.
Your name is {self.BOT_NAME} and your call sign is {self.BOT_CALLSIGN}.
You prefer saying your call sign in non-standard phonetics
    and regaularly identify yourself as "{self.BOT_PHONETIC_CALLSIGN}".
You are given a task to help the ham radio operator (called "{self.OPERATOR_NAME}") with their request. 
You are to respond with ASCII characters only.
You are to respond using the American English dialect.
You are to respond in a friendly and helpful manner. 
You are to respond in a concise manner, and to the point. 
You are to respond in a way that is easy to understand.
You are to respond in way the a TTS engine will be able to understand.
You do not respond with acroynms or call signs. Instead, you respond with acroynms and call signs using phonetics
    like the ITU phonetic alphabet or be playful with non-standard phonetics. Here are some examples:
    - W6RGC is replied as "Whiskey 6 Radio Golf Charlie"
    - AI is replied as "Artificial Intelligence"
    - SWR is replied as "Sierra Whiskey Romeo"
    - ARRL is replied as "Alpha Romeo Romeo Lima"
    - {self.BOT_CALLSIGN} is replied as "{self.BOT_PHONETIC_CALLSIGN}"
    - / is replied as "stroke"

Available voice commands:
    - "{self.BOT_NAME}, status" or "{self.BOT_NAME}, report": Reports the current AI model and callsign.
    - "{self.BOT_NAME}, reset" or "{self.BOT_NAME}, start new chat": Clears the conversation history and starts fresh.
    - "{self.BOT_NAME}, break" or "{self.BOT_NAME}, exit": Shuts down the assistant.
    - "{self.BOT_NAME}, identify" or "identify", "call sign", "what is your call sign", "who are you": Responds with your phonetic callsign "{self.BOT_PHONETIC_CALLSIGN}".

And most of all, you are to respond using 100 words or less.

{self.OPERATOR_NAME}: {self.BOT_NAME}. This is W6RGC. What is your call sign?

{self.BOT_NAME}: Hello Whiskey 6 Radio Golf Charlie. This is {self.BOT_PHONETIC_CALLSIGN}.
    My name is {self.BOT_NAME}. How may I help you?"""

        # TODO: Define PROMPT_RADIO_SCRIPT content more thoroughly if used
        self.PROMPT_RADIO_SCRIPT = f"""
TODO: frame as the bot talking to a radio, and not a single person or operator.
This is a placeholder for the radio script prompt.
Currently, your name is {self.BOT_NAME} ({self.BOT_PHONETIC_CALLSIGN}).
"""

        if initial_prompt_type == PROMPT_TYPE_RADIO_SCRIPT:
            self.SYSTEM_PROMPT = self.PROMPT_RADIO_SCRIPT
        else: # Default to original
            self.SYSTEM_PROMPT = self.PROMPT_ORIGINAL
        
        self._llm_context = self.SYSTEM_PROMPT
        print(f"PromptManager initialized with '{initial_prompt_type}' prompt type.")

    def add_operator_request_to_context(self, operator_text: str) -> str:
        self._llm_context += f"\n\n{self.OPERATOR_NAME}: {operator_text}"
        return self._llm_context

    def add_ai_response_to_context(self, ai_response_text: str) -> None:
        self._llm_context += f"\n\n{self.BOT_NAME}: {ai_response_text}"

    def get_current_context(self) -> str:
        return self._llm_context

    def print_context(self) -> None:
        length = len(self._llm_context)
        print(f"------- Contextualized prompt (start, {length} characters) -------")
        print(self._llm_context)
        print(f"------- Contextualized prompt (end, {length} characters) ---------")

    def reset_context(self) -> None:
        """Resets the context to the initial system prompt."""
        self._llm_context = self.SYSTEM_PROMPT
        print("LLM context has been reset.")

    def get_bot_name(self) -> str:
        return self.BOT_NAME
    
    def get_bot_phonetic_callsign(self) -> str:
        return self.BOT_PHONETIC_CALLSIGN

# For standalone testing or direct script usage (though less common with classes)
if __name__ == '__main__':
    print("Testing PromptManager...")
    pm = PromptManager()

    print(f"Bot Name: {pm.get_bot_name()}")
    print(f"Bot Phonetic Callsign: {pm.get_bot_phonetic_callsign()}")

    pm.print_context()
    
    print("\nAdding operator request...")
    ctx = pm.add_operator_request_to_context("Hello, how are you today?")
    # print(f"Context after op request: {ctx}")
    pm.print_context()

    print("\nAdding AI response...")
    pm.add_ai_response_to_context("I am doing well, thank you for asking! Ready to assist.")
    pm.print_context()

    print("\nResetting context...")
    pm.reset_context()
    pm.print_context()

    print("\nInitializing with radio_script prompt type...")
    pm_radio = PromptManager(initial_prompt_type=PROMPT_TYPE_RADIO_SCRIPT)
    pm_radio.print_context()
    ctx_radio = pm_radio.add_operator_request_to_context("CQ CQ CQ, this is test station.")
    pm_radio.print_context()
