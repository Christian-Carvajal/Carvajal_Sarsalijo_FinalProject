import os
import sys
import wave
import tempfile
import unittest

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from speech_engine import speech_engine, SpeechEngine


class TestVoiceSTT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a valid 16kHz mono PCM WAV byte buffer (0.5s silence)
        cls.temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        cls.temp_wav.close()
        with wave.open(cls.temp_wav.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 8000)
        
        with open(cls.temp_wav.name, "rb") as f:
            cls.silent_wav_bytes = f.read()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_wav.name):
            try:
                os.remove(cls.temp_wav.name)
            except Exception:
                pass

    def test_01_speech_engine_has_transcribe_wav_bytes(self):
        """
        SpeechEngine must expose transcribe_wav_bytes to decode browser-recorded PCM WAV.
        """
        self.assertTrue(
            hasattr(speech_engine, "transcribe_wav_bytes"),
            "SpeechEngine missing required method transcribe_wav_bytes"
        )

    def test_02_transcribe_wav_bytes_handles_silent_audio_gracefully(self):
        """
        Transcribing silent audio must return empty string or None without crashing.
        """
        result = speech_engine.transcribe_wav_bytes(self.silent_wav_bytes, language="en-US")
        # Silence has no recognizable words, so result should be empty string or None, not raise exception
        self.assertIn(result, ["", None])

    def test_03_transcribe_wav_bytes_rejects_corrupted_data(self):
        """
        Corrupted audio bytes must return empty string or None without unhandled crash.
        """
        corrupted = b"NOT_A_REAL_WAV_HEADER_DATA"
        result = speech_engine.transcribe_wav_bytes(corrupted, language="en-US")
        self.assertIn(result, ["", None])

    def test_04_api_voice_transcribe_endpoint_handles_empty_payload(self):
        """
        Endpoint /api/voice/transcribe must respond with status error if audio payload is missing.
        """
        import json
        import urllib.request
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/voice/transcribe",
            data=json.dumps({"audio_base64": ""}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data.get("status"), "error")
        except urllib.error.URLError:
            # If server not running in test runner, skip HTTP test
            pass


if __name__ == "__main__":
    unittest.main()
