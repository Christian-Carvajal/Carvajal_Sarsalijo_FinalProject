"""
speech_engine.py
Decoupled Speech I/O Pipeline for Perpetual Bistro AI Virtual Waiter
Course: BSCS 3112 - Artificial Intelligence (Lesson 7 Final Project)
University of Perpetual Help System DALTA

Architectural Role:
Decouples physical audio hardware and speech processing from the core FSM Agent.
Converts spoken customer audio into plain text strings (STT) and agent responses
into natural vocal audio (TTS).

Supported Audio Engines:
- STT: speech_recognition (local microphone), Web Speech API bridge, or console fallback.
- TTS: edge_tts (high-fidelity Filipino Taglish neural voice), pyttsx3 (local SAPI5 offline), or wave output.
"""

import os
import sys
import re
import asyncio
from typing import Optional, Callable

# Check available speech engines
try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False


class SpeechEngine:
    """Decoupled Speech-to-Text and Text-to-Speech Controller."""

    def __init__(self, voice_name: str = "en-PH-JamesNeural", fallback_voice_id: Optional[str] = None):
        self.voice_name = voice_name
        self.fallback_voice_id = fallback_voice_id
        self.recognizer = sr.Recognizer() if HAS_SR else None

        # Initialize offline pyttsx3 engine if available
        self.pyttsx_engine = None
        if HAS_PYTTSX3:
            try:
                self.pyttsx_engine = pyttsx3.init()
                self.pyttsx_engine.setProperty("rate", 170)
                self.pyttsx_engine.setProperty("volume", 0.95)
            except Exception as e:
                print(f"[SpeechEngine] Warning: pyttsx3 init failed: {e}")

    def normalize_taglish_text(self, text: str) -> str:
        """Phonetic normalization for natural Taglish pronunciation of prices, acronyms, and greetings."""
        clean = text
        # Clean currency symbols for phonetic speech
        clean = re.sub(r"₱\s*(\d+(?:\.\d{2})?)", r"\1 pesos", clean)
        clean = re.sub(r"Php\s*(\d+(?:\.\d{2})?)", r"\1 pesos", clean, flags=re.IGNORECASE)
        # Normalize campus acronyms
        clean = clean.replace("UPHSD", "Perpetual Help System DALTA")
        clean = clean.replace("BistroBot", "Bistro Bot")
        # Strip code blocks or markdown asterisks/brackets
        clean = re.sub(r"[*_#`]", "", clean)
        clean = re.sub(r"\[.*?\]", "", clean)
        # Strip multiple whitespaces
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    def listen_from_microphone(self, timeout: int = 5, phrase_time_limit: int = 10, allow_console_input: bool = False) -> str:
        """Captures microphone audio from the physical room and converts it to text string."""
        if not HAS_SR or not self.recognizer:
            if allow_console_input and sys.stdin and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                try:
                    return input("\n[Customer Voice Input Prompt] Speak/Type: ").strip()
                except (EOFError, OSError):
                    return ""
            return ""

        try:
            with sr.Microphone() as source:
                print("[SpeechEngine] 🎙️ Adjusting for ambient noise... Speak now.")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.8)
                print(f"[SpeechEngine] 👂 Listening (timeout={timeout}s)...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                print("[SpeechEngine] ⚡ Transcribing audio...")
                transcription = self.recognizer.recognize_google(audio)
                print(f"[SpeechEngine] [STT Recognized]: \"{transcription}\"")
                return transcription.strip()
        except sr.WaitTimeoutError:
            print("[SpeechEngine] [STT]: No speech detected within timeout.")
            return ""
        except sr.UnknownValueError:
            print("[SpeechEngine] [STT]: Speech was unintelligible.")
            return ""
        except Exception as e:
            print(f"[SpeechEngine] [STT Error]: {e}")
            return ""

    async def _generate_edge_tts(self, text: str, output_path: str) -> bool:
        """Synthesizes speech to an MP3 file using Edge-TTS."""
        if not HAS_EDGE_TTS:
            return False
        clean_text = self.normalize_taglish_text(text)
        try:
            communicate = edge_tts.Communicate(clean_text, self.voice_name)
            await communicate.save(output_path)
            return True
        except Exception as e:
            print(f"[SpeechEngine] Edge-TTS error: {e}")
            return False

    def synthesize_to_file(self, text: str, output_path: str) -> bool:
        """Generates an audio file from text using available TTS backend."""
        clean_text = self.normalize_taglish_text(text)
        if HAS_EDGE_TTS:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success = loop.run_until_complete(self._generate_edge_tts(clean_text, output_path))
                loop.close()
                if success and os.path.exists(output_path):
                    return True
            except Exception as e:
                print(f"[SpeechEngine] Async TTS file generation failed: {e}")

        if HAS_PYTTSX3 and self.pyttsx_engine:
            try:
                self.pyttsx_engine.save_to_file(clean_text, output_path)
                self.pyttsx_engine.runAndWait()
                return os.path.exists(output_path)
            except Exception as e:
                print(f"[SpeechEngine] pyttsx3 save_to_file failed: {e}")

        return False

    def speak(self, text: str) -> None:
        """Speaks the response string aloud through physical speakers."""
        clean_text = self.normalize_taglish_text(text)
        print(f"[BistroBot Vocal Feedback]: {clean_text}")

        # Attempt offline pyttsx3 first for zero-latency local speech
        if HAS_PYTTSX3 and self.pyttsx_engine:
            try:
                self.pyttsx_engine.say(clean_text)
                self.pyttsx_engine.runAndWait()
                return
            except Exception as e:
                print(f"[SpeechEngine] pyttsx3 playback failed: {e}")

        # Fallback to Edge-TTS temporary playback
        if HAS_EDGE_TTS:
            try:
                import tempfile
                import subprocess
                temp_audio = os.path.join(tempfile.gettempdir(), "bistrobot_speech.mp3")
                if self.synthesize_to_file(clean_text, temp_audio):
                    # Play asynchronously or via system media
                    if sys.platform == "win32":
                        # Use Windows PowerShell media player command for local audio output
                        subprocess.Popen(
                            ["powershell", "-c", f'(New-Object Media.SoundPlayer "{temp_audio}").PlaySync()'],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    return
            except Exception as e:
                print(f"[SpeechEngine] Audio playback failed: {e}")


# Global Singleton Instance
speech_engine = SpeechEngine()


if __name__ == "__main__":
    print("==============================================================")
    print(" [TEST] Perpetual Bistro Speech Engine Diagnostic")
    print("==============================================================")
    print(f" • speech_recognition available: {HAS_SR}")
    print(f" • pyttsx3 offline TTS available: {HAS_PYTTSX3}")
    print(f" • edge_tts neural voice available: {HAS_EDGE_TTS}")
    print("==============================================================")

    test_msg = "Mabuhay! Welcome to Perpetual Bistro Table 1. Our Perpetual Classic Burger is only ₱150."
    print(f"\nOriginal Text:  {test_msg}")
    print(f"Normalized:     {speech_engine.normalize_taglish_text(test_msg)}")

    print("\n[SpeechEngine] Testing vocal synthesis...")
    speech_engine.speak(test_msg)
    print("[SpeechEngine] Diagnostic complete.")
