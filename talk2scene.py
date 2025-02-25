import os
import json
import time
import asyncio
import logging
import psutil
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import yaml
from moviepy.editor import *
from moviepy.config import change_settings
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError

# Load environment variables from .env file
load_dotenv()

# Configure FFmpeg path
change_settings({"FFMPEG_BINARY": "/usr/bin/ffmpeg"})

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("talk2scene.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ======================== Interface Definitions ========================
class ITranscriber(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        pass

class ILLMService(ABC):
    @abstractmethod
    def generate_scenes(self, text: str) -> List[Dict]:
        pass

# ======================== Core Implementations ========================
class WhisperTranscriber(ITranscriber):
    """OpenAI Whisper transcription implementation."""
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
    def transcribe(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            f.seek(0)  # 确保文件指针位置正确
            try:
                response = requests.post(
                    os.getenv('WHISPER_URL', 'https://api.openai.com/v1/audio/transcriptions'),
                    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
                    files={"file": f},
                    data={"model": os.getenv('WHISPER_MODEL', 'whisper-1')},
                    timeout=30  # 超时设置已存在
                )
                response.raise_for_status()
                return response.json()["text"]
            except requests.exceptions.Timeout as te:
                logger.error(f"Transcription timed out: {str(te)}")
                raise ValueError("请求超时，请稍后重试") from te
            except requests.exceptions.RequestException as e:
                logger.error(f"Transcription failed: {str(e)}")
                raise

class GPTSceneGenerator(ILLMService):
    """GPT-4 scene generation implementation."""
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
    def generate_scenes(self, text: str) -> List[Dict]:
        try:
            response = requests.post(
                os.getenv('CHAT_URL', 'https://api.openai.com/v1/chat/completions'),
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",
                    "messages": [{
                        "role": "system",
                        "content": self._build_system_prompt()
                    }, {
                        "role": "user", 
                        "content": text
                    }]
                },
                timeout=30  # Add timeout to prevent indefinite hanging
            )
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"])
        except requests.exceptions.Timeout as te:
            logger.error(f"Scene generation timed out: {str(te)}")
            raise ValueError("请求超时，请稍后重试") from te
        except requests.exceptions.RequestException as e:
            logger.error(f"Scene generation failed: {str(e)}")
            raise

    def _build_system_prompt(self) -> str:
        return """
        You are a video scene generation AI. Process the text according to the following rules:
        1. Split the dialogue into 3-5 second scene segments.
        2. For each scene, select components: STA (pose), EXP (expression), ACT (action), BG (background), CG (illustration).
        3. Follow the combination rules.
        Output format: JSON array containing time (seconds), text, sta, exp, act, bg, cg.
        """

class AssetManager:
    """Asset manager for loading and managing resources."""
    def __init__(self, config_path: str = "config/character_config.yaml"):
        self.config = self._load_config(config_path)
        self.assets = defaultdict(dict)
        self._preload_assets()

    def _load_config(self, path: str) -> Dict:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _preload_assets(self):
        for category in ["STA", "EXP", "BG", "CG"]:
            asset_dir = Path(self.config["paths"][f"{category.lower()}_dir"])
            for file in asset_dir.glob("*.*"):
                self.assets[category][file.stem] = str(file)

        act_dir = Path(self.config["paths"]["act_dir"])
        for act_folder in act_dir.iterdir():
            if act_folder.is_dir():
                self.assets["ACT"][act_folder.name] = [
                    str(f) for f in sorted(act_folder.glob("*.png"))
                ]

    def get_asset(self, category: str, code: str) -> Optional[Any]:
        return self.assets.get(category, {}).get(code)

class SceneStateManager:
    """Manages scene state transitions."""
    def __init__(self):
        self._state = {
            'current_sta': 'STA_Stand_Default',
            'current_exp': 'EXP_Neutral',
            'current_bg': 'BG_Default',
            'transition_history': []
        }
        self.transition_rules = {
            'STA': {
                'STA_Stand_Default': ['STA_Sit_Normal', 'STA_ArmsCrossed'],
                'STA_Sit_Normal': ['STA_Stand_Default', 'STA_LeanForward'],
                'STA_ArmsCrossed': ['STA_Stand_Default']
            },
            'EXP': {
                'EXP_Neutral': ['EXP_Smile', 'EXP_Thinking'],
                'EXP_Thinking': ['EXP_Neutral', 'EXP_Surprised'],
                'EXP_Surprised': ['EXP_Neutral']
            }
        }
    
    def apply_transition(self, new_sta: str, new_exp: str) -> bool:
        if self._validate_transition(new_sta, new_exp):
            self._state['transition_history'].append({
                'timestamp': time.time(),
                'from': (self._state['current_sta'], self._state['current_exp']),
                'to': (new_sta, new_exp)
            })
            self._state.update({
                'current_sta': new_sta,
                'current_exp': new_exp
            })
            return True
        return False

    def _validate_transition(self, sta: str, exp: str) -> bool:
        valid_sta = sta in self.transition_rules['STA'].get(
            self._state['current_sta'], []
        ) or sta == self._state['current_sta']
        valid_exp = exp in self.transition_rules['EXP'].get(
            self._state['current_exp'], []
        ) or exp == self._state['current_exp']
        return valid_sta and valid_exp

class IncrementalRenderer:
    """Renders video incrementally to save memory."""
    def __init__(self, output_path: str):
        self.clips = []
        self.output = output_path
        self.audio_clips = []
        self.current_duration = 0.0
        
    def append_clip(self, clip: VideoClip):
        clip = clip.fx(vfx.fadein(0.5)).fx(vfx.fadeout(0.5))
        self.clips.append(clip)
        self.audio_clips.append(clip.audio)
        self.current_duration += clip.duration
        
        if len(self.clips) % 5 == 0 or self.current_duration >= 30:
            self._partial_render()
    
    def finalize(self):
        if len(self.clips) > 0:
            self._partial_render()
        
        final_audio = CompositeAudioClip(self.audio_clips)
        final_video = VideoFileClip("temp_final.mp4")
        final_video = final_video.set_audio(final_audio)
        final_video.write_videofile(
            self.output,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            threads=4
        )
        
        for f in Path(".").glob("temp_*.mp4"):
            f.unlink()

    def _partial_render(self):
        temp_video = concatenate_videoclips(self.clips, method="compose")
        temp_audio = CompositeAudioClip(self.audio_clips)
        temp_video = temp_video.set_audio(temp_audio)
        
        temp_filename = f"temp_{int(time.time())}.mp4"
        temp_video.write_videofile(
            temp_filename,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp.aac"
        )
        
        self.clips = [VideoFileClip(temp_filename)]
        self.audio_clips = [AudioFileClip("temp.aac")]
        self.current_duration = 0.0

class PipelineProcessor:
    """Processes scenes in parallel."""
    def __init__(self, asset_manager: AssetManager, state_manager: SceneStateManager, 
                 renderer: IncrementalRenderer, max_workers=4):
        self.asset_manager = asset_manager
        self.state_manager = state_manager
        self.renderer = renderer
        self.executor = ThreadPoolExecutor(max_workers)
        self.loop = asyncio.get_event_loop()
        
    async def process_scenes(self, scenes: List[Dict]):
        futures = []
        for scene in scenes:
            future = self.loop.run_in_executor(
                self.executor,
                self._process_single_scene,
                scene
            )
            futures.append(future)
        
        await asyncio.gather(*futures)
        self.executor.shutdown(wait=True)

    def _process_single_scene(self, scene: Dict):
        start_time = time.time()
        
        bg_path = self.asset_manager.get_asset("BG", scene['bg'])
        sta_image = self.asset_manager.get_asset("STA", scene['sta'])
        act_frames = self.asset_manager.get_asset("ACT", scene['act'])
        cg_path = self.asset_manager.get_asset("CG", scene['cg']) if scene['cg'] else None
        
        duration = scene.get('duration', 5.0)
        base_clip = ImageClip(bg_path).set_duration(duration)
        
        if act_frames and len(act_frames) > 0:
            character_clip = ImageSequenceClip(act_frames, fps=24)
            character_clip = character_clip.resize(height=300).set_position(('center', 'bottom'))
            base_clip = CompositeVideoClip([base_clip, character_clip])
        
        if cg_path:
            cg_clip = ImageClip(cg_path).set_duration(3).set_position(('right', 'top')).crossfadein(1)
            base_clip = CompositeVideoClip([base_clip, cg_clip])
        
        txt_clip = TextClip(
            scene['text'],
            fontsize=24,
            color='white',
            stroke_color='black',
            stroke_width=1
        ).set_position(('center', 'top')).set_duration(duration)
        base_clip = CompositeVideoClip([base_clip, txt_clip])
        
        if self.state_manager.apply_transition(scene['sta'], scene['exp']):
            logger.info(f"State transition successful: {scene['sta']} {scene['exp']}")
        else:
            logger.warning(f"Invalid state transition: {scene['sta']} {scene['exp']}")
        
        self.renderer.append_clip(base_clip)
        logger.info(f"Scene processed: {scene['time']}s, took: {time.time()-start_time:.2f}s")

class TelemetryMonitor:
    """Monitors system performance."""
    def __init__(self):
        self.metrics = defaultdict(list)
        self.process = psutil.Process()
        
    def track_metric(self, name: str, value: float):
        self.metrics[name].append({
            'timestamp': datetime.now().isoformat(),
            'value': value
        })
        
    def generate_report(self) -> Dict:
        render_times = [m['value'] for m in self.metrics['render_time']]
        mem_usage = [m['value'] for m in self.metrics['memory_usage']]
        
        return {
            'performance': {
                'total_scenes': len(self.metrics['scene_time']),
                'avg_render_time': np.mean(render_times) if render_times else 0,
                'max_memory': max(mem_usage) if mem_usage else 0,
                'cpu_usage': np.mean([m['value'] for m in self.metrics['cpu_usage']])
            },
            'system': {
                'component_usage': Counter(
                    [m['component'] for m in self.metrics['component_usage']]
                )
            }
        }
    
    def track_system_metrics(self):
        self.track_metric('memory_usage', self.process.memory_info().rss / 1024**2)
        self.track_metric('cpu_usage', psutil.cpu_percent())

async def main():
    try:
        transcriber = WhisperTranscriber()
        llm = GPTSceneGenerator()
        asset_manager = AssetManager()
        state_manager = SceneStateManager()
        renderer = IncrementalRenderer("output/output_video.mp4")
        monitor = TelemetryMonitor()
        
        Path("output").mkdir(exist_ok=True)
        Path("temp").mkdir(exist_ok=True)

        start_time = time.time()
        logger.info("Starting audio transcription...")
        text = transcriber.transcribe("input/audio.wav")
        logger.info(f"Transcription complete, text length: {len(text)} characters")
        
        logger.info("Generating scene data...")
        scenes = llm.generate_scenes(text)
        logger.info(f"Generated {len(scenes)} scenes")
        
        processor = PipelineProcessor(asset_manager, state_manager, renderer)
        
        logger.info("Processing scenes in parallel...")
        await processor.process_scenes(scenes)
        
        logger.info("Generating final video...")
        renderer.finalize()
        
        monitor.track_metric("total_time", time.time() - start_time)
        report = monitor.generate_report()
        with open("output/system_report.json", 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Processing complete! Total time: {report['performance']['total_time']:.2f}s")

    except Exception as e:
        logger.error(f"System failure: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())