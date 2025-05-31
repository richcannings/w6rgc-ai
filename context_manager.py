#!/usr/bin/env python3
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

import logging
import os
from datetime import datetime

from constants import (
    OPERATOR_NAME,
    BOT_NAME,
)
#from prompt import PROMPT
from prompt_gemini_generated import PROMPT

class ContextManager:
    def __init__(self, initial_prompt=PROMPT, model_tracks_context=False, log_file_path=None):
        """
        Initializes the ContextManager.

        Args:
            initial_prompt (str): The base system prompt.
            model_tracks_context (bool): If True, indicates the LLM handles its own
                conversation history. `add_operator_request_to_context` will then
                return a minimal prompt (system prompt + first user message, then
                only the current user message). The ContextManager still maintains
                the full history internally. Defaults to False.
            log_file_path (str): Path to the log file where context will be written.
                If None, defaults to "chatbot-script-{time}-{date}.log".
        """
        self._llm_context = initial_prompt
        self.model_tracks_context = model_tracks_context
        self._is_first_run = True  # Flag for the first operator turn after init or reset
        
        # Generate default log file name if not provided
        if log_file_path is None:
            now = datetime.now()
            time_str = now.strftime("%H-%M-%S")
            date_str = now.strftime("%Y-%m-%d")
            log_file_path = f"chatbot-script-{time_str}-{date_str}.log"
        
        # Set up logging
        self.log_file_path = log_file_path
        self._setup_context_logger()
        
        # Log initial context
        self._log_context("INIT", "Initial context setup")
        
        print(f"ContextManager initialized. Context logging to: {self.log_file_path}")

    def _setup_context_logger(self):
        """Sets up a dedicated logger for context changes."""
        # Create a logger specifically for context
        self.context_logger = logging.getLogger('llm_context')
        self.context_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.context_logger.handlers[:]:
            self.context_logger.removeHandler(handler)
        
        # Create file handler with immediate flushing
        file_handler = logging.FileHandler(self.log_file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter that includes timestamp
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.context_logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        self.context_logger.propagate = False

    def _log_context(self, action, description=""):
        """
        Logs the current context to the file.
        
        Args:
            action (str): The action that triggered this log (e.g., "OPERATOR_MSG", "AI_RESPONSE")
            description (str): Optional description of the action
        """
        separator = "=" * 80
        context_length = len(self._llm_context)
        
        log_entry = f"\n{separator}\n"
        log_entry += f"ACTION: {action}\n"
        if description:
            log_entry += f"DESCRIPTION: {description}\n"
        log_entry += f"CONTEXT_LENGTH: {context_length} characters\n"
        log_entry += f"{separator}\n"
        log_entry += self._llm_context
        log_entry += f"\n{separator}\n"
        
        self.context_logger.info(log_entry)
        
        # Force flush to ensure immediate writing
        for handler in self.context_logger.handlers:
            handler.flush()

    def add_operator_request_to_context(self, operator_text: str) -> str:
        """
        Adds the operator's text to the internal conversation history and returns
        the appropriate prompt for the LLM.

        Args:
            operator_text (str): The transcribed text from the operator.

        Returns:
            str: The prompt to be sent to the LLM. This varies based on
                 `model_tracks_context`.
        """
        if self.model_tracks_context:
            # Only give the initial prompt once (and the operator text)
            if self._is_first_run:
                self._llm_context += f"\n\n{OPERATOR_NAME}'s first request is: {operator_text}"
                self._is_first_run = False
                self._log_context("OPERATOR_FIRST_MSG", f"First operator message: {operator_text[:100]}...")
            else:
                self._llm_context = operator_text
                self._log_context("OPERATOR_MSG", f"Operator message (model tracks context): {operator_text[:100]}...")
        else:
            self._llm_context += f"\n\n{OPERATOR_NAME}: {operator_text}"
            self._log_context("OPERATOR_MSG", f"Operator message: {operator_text[:100]}...")
        
        return self._llm_context

    def add_ai_response_to_context(self, ai_response_text: str) -> None:
        """Adds the AI's response to the internal conversation history."""
        if not self.model_tracks_context:
            self._llm_context += f"\n\n{BOT_NAME}: {ai_response_text}"
            self._log_context("AI_RESPONSE", f"AI response: {ai_response_text[:100]}...")

    def get_current_context(self) -> str:
        return self._llm_context

    def print_context(self) -> None:
        """Prints the full internal conversation history to the console."""
        length = len(self._llm_context)
        print(f"------- Contextualized prompt (start, {length} characters) -------")
        print(self._llm_context)
        print(f"------- Contextualized prompt (end, {length} characters) ---------")

    def reset_context(self) -> None:
        """Resets the internal context to the initial system prompt."""
        self._llm_context = PROMPT
        self._is_first_run = True  # Reset for model_tracks_context logic
        self._log_context("RESET", "Context reset to initial prompt")
        print("LLM context has been reset.")

# For standalone testing or direct script usage (though less common with classes)
if __name__ == '__main__':
    print("--- Testing ContextManager (model_tracks_context=False) ---")
    cm_false = ContextManager(model_tracks_context=False)

    # Directly use constants for bot identity checks here, as methods are removed
    print(f"Bot Name (from constants): {BOT_NAME}")

    print("\nInitial full context (should be SYSTEM_PROMPT):")
    cm_false.print_context()
    
    print("\nTurn 1: Operator says 'Hello'. Prompt for LLM:")
    prompt1_false = cm_false.add_operator_request_to_context("Hello")
    print(f"LLM receives: {prompt1_false}")
    print("\nFull context after Turn 1 operator message:")
    cm_false.print_context()
    cm_false.add_ai_response_to_context("Hi there! I'm good.")
    print("\nFull context after Turn 1 AI response:")
    cm_false.print_context()

    print("\nTurn 2: Operator says 'How are you?'. Prompt for LLM:")
    prompt2_false = cm_false.add_operator_request_to_context("How are you?")
    print(f"LLM receives: {prompt2_false}")
    print("\nFull context after Turn 2 operator message:")
    cm_false.print_context()
    cm_false.add_ai_response_to_context("I am doing great, thanks!")
    print("\nFull context after Turn 2 AI response:")
    cm_false.print_context()

    print("\nResetting context for cm_false...")
    cm_false.reset_context()
    print("\nFull context after reset:")
    cm_false.print_context()

    print("\n\n--- Testing ContextManager (model_tracks_context=True) ---")
    cm_true = ContextManager(model_tracks_context=True)

    # Bot identity is global, no need to test per instance of ContextManager here
    # if it were instance-specific, we would test cm_true.get_bot_name() etc.

    print("\nInitial full context (should be SYSTEM_PROMPT):")
    cm_true.print_context()

    print("\nTurn 1: Operator says 'Greetings'. Prompt for LLM:")
    prompt1_true = cm_true.add_operator_request_to_context("Greetings")
    print(f"LLM receives: {prompt1_true}")
    print("\nFull context after Turn 1 operator message (should be full still):")
    cm_true.print_context()
    cm_true.add_ai_response_to_context("Hello to you too!")
    print("\nFull context after Turn 1 AI response:")
    cm_true.print_context()

    print("\nTurn 2: Operator says 'What is the weather?'. Prompt for LLM:")
    prompt2_true = cm_true.add_operator_request_to_context("What is the weather?")
    print(f"LLM receives: {prompt2_true}") # Should be only current operator msg
    print("\nFull context after Turn 2 operator message (should be full still):")
    cm_true.print_context()
    cm_true.add_ai_response_to_context("It is sunny today.")
    print("\nFull context after Turn 2 AI response:")
    cm_true.print_context()

    print("\nResetting context for cm_true...")
    cm_true.reset_context()
    print("\nFull context after reset:")
    cm_true.print_context()
    print("\nTurn 1 after reset: Operator says 'Testing reset'. Prompt for LLM:")
    prompt_reset_true = cm_true.add_operator_request_to_context("Testing reset")
    print(f"LLM receives: {prompt_reset_true}") # Should be system prompt + operator msg
    print("\nFull context after Turn 1 post-reset:")
    cm_true.print_context() 