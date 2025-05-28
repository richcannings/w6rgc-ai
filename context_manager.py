# context_manager.py - Manages AI persona, prompts, and conversation context.
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
    DEFAULT_PROMPT_TYPE
)
from prompt import PROMPT

class ContextManager:
    # Class attributes for bot identity - now imported from constants
    OPERATOR_NAME = OPERATOR_NAME
    BOT_NAME = BOT_NAME
    BOT_CALLSIGN = BOT_CALLSIGN
    BOT_SPOKEN_CALLSIGN = BOT_SPOKEN_CALLSIGN
    BOT_PHONETIC_CALLSIGN = BOT_PHONETIC_CALLSIGN

    def __init__(self, initial_prompt=PROMPT):
        """
        Initializes the ContextManager.
        Args:
            initial_prompt (str): The initial prompt to use
        """
        self.PROMPT_ORIGINAL = initial_prompt
        self.SYSTEM_PROMPT = self.PROMPT_ORIGINAL
        self._llm_context = self.SYSTEM_PROMPT
        print(f"ContextManager initialized.")

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
    print("Testing ContextManager...")
    cm = ContextManager()

    print(f"Bot Name: {cm.get_bot_name()}")
    print(f"Bot Phonetic Callsign: {cm.get_bot_phonetic_callsign()}")

    cm.print_context()
    
    print("\nAdding operator request...")
    ctx = cm.add_operator_request_to_context("Hello, how are you today?")
    # print(f"Context after op request: {ctx}")
    cm.print_context()

    print("\nAdding AI response...")
    cm.add_ai_response_to_context("I am doing well, thank you for asking! Ready to assist.")
    cm.print_context()

    print("\nResetting context...")
    cm.reset_context()
    cm.print_context() 