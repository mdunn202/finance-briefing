from __future__ import annotations

import logging
import httpx
from pathlib import Path
from datetime import datetime

from config import get_settings

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "data"


class TTSAgent:

    BASE_URL = "https://api.elevenlabs.io/v1"

    # Rachel — natural, professional, works well for news briefings
    DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

    def __init__(self) -> None:
        self.api_key = get_settings().elevenlabs_api_key
        OUTPUT_DIR.mkdir(exist_ok=True)

    def _generate_filename(self, profile_name: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = profile_name.lower().replace(" ", "_")
        return OUTPUT_DIR / f"briefing_{safe_name}_{timestamp}.mp3"

    async def run(self, briefing_text: str, profile_name: str) -> Path:
        """Convert briefing text to audio and save to disk. Returns the file path."""
        if not briefing_text:
            raise ValueError("TTSAgent received empty briefing text")

        logger.info(f"TTSAgent generating audio for {profile_name} ({len(briefing_text.split())} words)")

        output_path = self._generate_filename(profile_name)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/text-to-speech/{self.DEFAULT_VOICE_ID}",
                headers={
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": briefing_text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                    }
                },
                timeout=60.0,
            )
            response.raise_for_status()

            output_path.write_bytes(response.content)

        logger.info(f"TTSAgent saved audio to {output_path}")
        return output_path