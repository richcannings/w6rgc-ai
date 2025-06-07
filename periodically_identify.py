#!/usr/bin/env python3
# identify_periodically.py - Periodic identification announcements for W6RGC/AI
#
# This module handles automatic periodic identification transmissions every 10 minutes
# to inform other operators about the AI assistant and how to interact with it.
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

import threading
import time
from constants import (
    BOT_NAME,
    BOT_PHONETIC_CALLSIGN,
    PERIODIC_ID_INTERVAL_MINUTES
)

class PeriodicIdentifier:
    def __init__(self, tts_engine, radio_interface, play_tts_function, interval_minutes=PERIODIC_ID_INTERVAL_MINUTES):
        """
        Initializes the periodic identifier.
        
        Args:
            tts_engine: The CoquiTTS engine instance
            radio_interface: The radio interface layer instance
            play_tts_function: Function to call for TTS playback (e.g., play_tts_audio)
            interval_minutes (int): Minutes between identification announcements
        """
        self.tts_engine = tts_engine
        self.radio_interface = radio_interface
        self.play_tts_function = play_tts_function
        self.interval_seconds = interval_minutes * 60
        self.running = False
        self.thread = None
        
        # Create the identification message
        self.identification_message = (
            f"This is {BOT_NAME}. Call sign {BOT_PHONETIC_CALLSIGN}. "
            f"I am an off grid Artificial Intelligence assistant. "
            f"Ask me anything by using my wake word {BOT_NAME}. "
            f"For example, {BOT_NAME}. {BOT_NAME}, How shall I prepare for a power outage at my house in Santa Cruz, California. "
            f"To turn {BOT_NAME} off, repeat {BOT_NAME} break a few times. Ask {BOT_NAME} about more commands."
        )
        
        print(f"📡 Periodic identifier initialized: {interval_minutes} minute intervals")
        print(f"📡 Identification message: {self.identification_message}")

    def start(self):
        """Starts the periodic identification announcements."""
        if self.running:
            print("⚠️ Periodic identifier is already running.")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._identification_loop, daemon=True)
        self.thread.start()
        print(f"📡 Periodic identifier started - announcing every {self.interval_seconds//60} minutes")

    def stop(self):
        """Stops the periodic identification announcements."""
        if not self.running:
            print("⚠️ Periodic identifier is not running.")
            return
            
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        print("📡 Periodic identifier stopped")

    def _identification_loop(self):
        """Main loop for periodic identification announcements."""
        while self.running:
            try:
                # Wait for the interval (checking every second for stop signal)
                for _ in range(self.interval_seconds):
                    if not self.running:
                        return
                    time.sleep(1)
                
                if self.running:  # Double-check we're still supposed to be running
                    print("📡 Transmitting periodic identification...")
                    self.play_tts_function(
                        self.identification_message,
                        self.tts_engine,
                        self.radio_interface
                    )
                    print("📡 Periodic identification transmission complete")
                    
            except Exception as e:
                print(f"❌ Error in periodic identification: {e}")
                # Continue running even if there's an error
                time.sleep(60)  # Wait a minute before trying again

    def transmit_now(self):
        """Immediately transmit the identification message (for testing or manual trigger)."""
        try:
            print("📡 Manual identification transmission...")
            self.play_tts_function(
                self.identification_message,
                self.tts_engine,
                self.radio_interface
            )
            print("📡 Manual identification transmission complete")
        except Exception as e:
            print(f"❌ Error in manual identification: {e}")

    def get_identification_message(self):
        """Returns the current identification message."""
        return self.identification_message

    def update_interval(self, interval_minutes):
        """Updates the identification interval."""
        self.interval_seconds = interval_minutes * 60
        print(f"📡 Periodic identifier interval updated to {interval_minutes} minutes")

    def reset_timer(self):
        """Resets the periodic identification timer to start counting from zero again."""
        if not self.running:
            print("⚠️ Periodic identifier is not running, cannot reset timer.")
            return
            
        # Stop the current thread
        old_running_state = self.running
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        # Restart if it was running
        if old_running_state:
            self.running = True
            self.thread = threading.Thread(target=self._identification_loop, daemon=True)
            self.thread.start()
            print(f"📡 Periodic identifier timer reset - next announcement in {self.interval_seconds//60} minutes")

    def restart_timer(self):
        """Alias for reset_timer() - restarts the periodic identification timer."""
        self.reset_timer()

# For standalone testing
if __name__ == '__main__':
    print("Testing PeriodicIdentifier...")
    
    # Mock functions for testing
    def mock_tts_function(text, tts_engine, aioc_interface):
        print(f"🔊 [MOCK TTS] Would transmit: {text}")
    
    class MockTTSEngine:
        pass
    
    class MockAIOCInterface:
        pass
    
    # Create test instance
    identifier = PeriodicIdentifier(
        tts_engine=MockTTSEngine(),
        aioc_interface=MockAIOCInterface(),
        play_tts_function=mock_tts_function,
        interval_minutes=1  # 1 minute for testing
    )
    
    print("\nTesting manual transmission...")
    identifier.transmit_now()
    
    print("\nTesting periodic transmission (will run for 3 minutes)...")
    identifier.start()
    
    try:
        print("Waiting 90 seconds, then resetting timer...")
        time.sleep(90)  # Wait 1.5 minutes
        
        print("Resetting timer now...")
        identifier.reset_timer()
        
        print("Waiting another 90 seconds...")
        time.sleep(90)  # Wait another 1.5 minutes
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    identifier.stop()
    print("Test complete.") 