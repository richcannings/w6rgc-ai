#!/usr/bin/env python3
# llm_openclaw.py - OpenClaw Local Gateway Integration
#
# This module provides integration with a local OpenClaw Gateway using its
# OpenAI-compatible Chat Completions endpoint.
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

import json
import os
import requests

from constants import (
    BOT_CALLSIGN,
    OPENCLAW_GATEWAY_URL,
    OPENCLAW_AGENT_ID,
    OPENCLAW_TOKEN_FILE,
    REQUEST_TIMEOUT
)


class OpenClawAPIError(Exception):
    """Custom exception for OpenClaw API related errors."""
    pass

requires_internet = False


def load_openclaw_token() -> str:
    """
    Load the OpenClaw gateway token from environment or file.

    Priority:
    1. OPENCLAW_GATEWAY_TOKEN environment variable
    2. OPENCLAW_TOKEN_FILE file content
    """
    env_token = os.getenv("OPENCLAW_GATEWAY_TOKEN")
    if env_token:
        return env_token.strip()

    if not os.path.exists(OPENCLAW_TOKEN_FILE):
        raise OpenClawAPIError(
            f"OpenClaw token file '{OPENCLAW_TOKEN_FILE}' not found and "
            "OPENCLAW_GATEWAY_TOKEN is not set."
        )

    with open(OPENCLAW_TOKEN_FILE, "r") as f:
        token = f.read().strip()

    if not token:
        raise OpenClawAPIError(
            f"OpenClaw token file '{OPENCLAW_TOKEN_FILE}' is empty."
        )

    return token


def ask_openclaw(prompt: str) -> str:
    """
    Send a prompt to the OpenClaw Gateway and return the response text.
    """
    if not prompt or not prompt.strip():
        raise OpenClawAPIError("Prompt cannot be empty")

    token = load_openclaw_token()
    url = f"{OPENCLAW_GATEWAY_URL}/v1/chat/completions"

    payload = {
        "model": f"openclaw:{OPENCLAW_AGENT_ID}",
        "messages": [{"role": "user", "content": prompt}],
        "user": BOT_CALLSIGN
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-openclaw-agent-id": OPENCLAW_AGENT_ID
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OpenClawAPIError(f"OpenClaw request failed: {exc}") from exc

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise OpenClawAPIError("OpenClaw returned invalid JSON") from exc

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError) as exc:
        raise OpenClawAPIError(
            f"Unexpected OpenClaw response shape: {data}"
        ) from exc
