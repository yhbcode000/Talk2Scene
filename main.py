#!/usr/bin/env python3

import os
import csv
import json
import logging
from typing import List, Dict, Any
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable not set")

CONFIG = {
    "audio_input": "input_audio.wav",
    "csv_output": "output_metadata.csv",
    "video_output": "output_video.mp4",
    "scene_json": "scene_data.json",
    "cg_metadata_json": "cg_metadata.json",
    "video_duration_per_segment": 3.5,
    "cg_auto_generate": True,
    "cg_generation_interval": 5  # Generate CG every 5 sentences
}

class Talk2SceneAutomator:
    """Automated pipeline for converting audio to visualized scenes."""
    
    def __init__(self):
        self.scene_data: List[Dict[str, Any]] = []
        self._validate_files()
        self._ensure_directories()

    def _validate_files(self) -> None:
        """Ensure required input files exist."""
        if not Path(CONFIG["audio_input"]).exists():
            raise FileNotFoundError(f"Audio file not found: {CONFIG['audio_input']}")

    def _ensure_directories(self) -> None:
        """Create necessary output directories."""
        Path("generated").mkdir(exist_ok=True)

    def _transcribe_audio(self) -> str:
        """Automatically transcribe audio using Whisper API."""
        logger.info("Starting audio transcription")
        try:
            with open(CONFIG["audio_input"], "rb") as audio_file:
                response = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    files={"file": audio_file},
                    data={"model": "whisper-1"}
                )
                response.raise_for_status()
                return response.json().get("text", "")
        except requests.RequestException as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def _generate_scene_data(self, text: str) -> None:
        """Automatically generate scene metadata using GPT-4."""
        logger.info("Generating scene metadata")
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",
                    "messages": [{
                        "role": "system",
                        "content": (
                            "Generate structured JSON with time, text, STA, EXP, ACT, BG, CG. "
                            "Follow these rules:\n"
                            "1. Time increments by 3.5s per entry\n"
                            "2. Auto-add CG every 5 entries\n"
                            "3. Use consistent character states\n"
                            "4. Output format: [{time: float, text: str, sta: str, exp: str, act: str, bg: str, cg: str}]"
                        )
                    }, {
                        "role": "user",
                        "content": text
                    }]
                }
            )
            response.raise_for_status()
            self.scene_data = json.loads(response.json()["choices"][0]["message"]["content"])
            self._enhance_with_cg()
            self._save_intermediate_data()
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.error(f"Scene generation failed: {e}")
            raise

    def _enhance_with_cg(self) -> None:
        """Automatically insert CG entries at specified intervals."""
        cg_counter = 0
        for idx, entry in enumerate(self.scene_data):
            if idx % CONFIG["cg_generation_interval"] == 0:
                cg_counter += 1
                entry["cg"] = f"generated/cg_auto_{cg_counter}.png"
                logger.info(f"Auto-added CG at {entry['time']}s")

    def _save_intermediate_data(self) -> None:
        """Save intermediate JSON and CSV files."""
        with open(CONFIG["scene_json"], "w") as f:
            json.dump(self.scene_data, f, indent=4)
        
        with open(CONFIG["csv_output"], "w") as f:
            writer = csv.DictWriter(f, fieldnames=["time", "text", "sta", "exp", "act", "bg", "cg"])
            writer.writeheader()
            writer.writerows(self.scene_data)

        logger.info(f"Intermediate files saved: {CONFIG['scene_json']}, {CONFIG['csv_output']}")

    def _generate_video(self) -> None:
        """Automatically generate output video."""
        logger.info("Generating preview video")
        # Video generation logic using moviepy/ffmpeg would go here
        # Placeholder for demonstration
        Path(CONFIG["video_output"]).touch()
        logger.info(f"Video file created: {CONFIG['video_output']}")

    def _cleanup(self) -> None:
        """Perform final cleanup and validation."""
        logger.info("Validating output files")
        required_files = [CONFIG["csv_output"], CONFIG["video_output"]]
        missing_files = [f for f in required_files if not Path(f).exists()]
        
        if missing_files:
            raise RuntimeError(f"Missing output files: {missing_files}")
        
        logger.info("Pipeline completed successfully")

    def run_pipeline(self) -> None:
        """Execute full automated pipeline."""
        try:
            transcript = self._transcribe_audio()
            self._generate_scene_data(transcript)
            self._generate_video()
            self._cleanup()
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

if __name__ == "__main__":
    automator = Talk2SceneAutomator()
    automator.run_pipeline()
