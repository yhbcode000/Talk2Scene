"""
Talk2Scene - 基于小人表情动作的互动式视频生成系统
版本：1.0
作者：AI助手
日期：2024-05-20
"""

import os
import json
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any
import requests
from moviepy.editor import *
from moviepy.config import change_settings

# 配置FFmpeg路径（根据实际情况修改）
change_settings({"FFMPEG_BINARY": "/usr/bin/ffmpeg"})

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("talk2scene.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AssetManager:
    """资源管理器：加载和管理所有素材资源"""
    def __init__(self, config_path: str = "config/character_config.yaml"):
        self.config = self._load_config(config_path)
        self.assets = {
            "STA": {},
            "EXP": {},
            "ACT": {},
            "BG": {},
            "CG": {}
        }
        self._load_assets()

    def _load_config(self, path: str) -> Dict:
        """加载角色配置文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _load_assets(self):
        """预加载所有素材资源"""
        # 加载静态资源
        for category in ["STA", "EXP", "BG", "CG"]:
            asset_dir = Path(self.config["paths"][f"{category.lower()}_dir"])
            for file in asset_dir.glob("*.*"):
                code = file.stem
                self.assets[category][code] = str(file)
                logger.debug(f"Loaded {category} asset: {code}")

        # 加载动态动作资源
        act_dir = Path(self.config["paths"]["act_dir"])
        for act_folder in act_dir.iterdir():
            if act_folder.is_dir():
                code = act_folder.name
                self.assets["ACT"][code] = [
                    str(f) for f in sorted(act_folder.glob("*.png"))
                ]
                logger.debug(f"Loaded ACT sequence: {code} ({len(self.assets['ACT'][code])} frames)")

    def get_asset(self, category: str, code: str) -> Any:
        """获取指定资源"""
        return self.assets.get(category, {}).get(code)

class SceneGenerator:
    """场景生成器：处理逻辑和AI集成"""
    def __init__(self, asset_manager: AssetManager):
        self.am = asset_manager
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict:
        """加载组合规则"""
        with open("config/composition_rules.json", 'r') as f:
            return json.load(f)

    def transcribe_audio(self, audio_path: str) -> str:
        """语音转文本"""
        logger.info(f"开始音频转录: {audio_path}")
        try:
            with open(audio_path, "rb") as f:
                response = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
                    files={"file": f},
                    data={"model": "whisper-1"}
                )
                response.raise_for_status()
                return response.json()["text"]
        except Exception as e:
            logger.error(f"转录失败: {str(e)}")
            raise

    def generate_scene_data(self, text: str) -> List[Dict]:
        """生成场景数据"""
        logger.info("正在生成场景数据...")
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
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
                }
            )
            response.raise_for_status()
            return self._parse_response(response.json())
        except Exception as e:
            logger.error(f"场景生成失败: {str(e)}")
            raise

    def _build_system_prompt(self) -> str:
        """构建AI系统提示"""
        return f"""
        你是一个视频场景生成AI，请根据以下规则处理文本：
        1. 将对话分割为每3-5秒的场景片段
        2. 为每个场景选择组件：
           - STA（姿态）: {list(self.am.assets['STA'].keys())}
           - EXP（表情）: {list(self.am.assets['EXP'].keys())}
           - ACT（动作）: {list(self.am.assets['ACT'].keys())}
           - BG（背景）: {list(self.am.assets['BG'].keys())}
           - CG（插画）: {list(self.am.assets['CG'].keys())}
        3. 遵守组合规则：{json.dumps(self.rules, indent=2)}
        输出格式：JSON数组，包含time(秒), text, sta, exp, act, bg, cg
        """

    def _parse_response(self, response: Dict) -> List[Dict]:
        """解析AI响应"""
        try:
            data = json.loads(response["choices"][0]["message"]["content"])
            return self._validate_scenes(data)
        except json.JSONDecodeError:
            logger.error("AI响应解析失败")
            raise

    def _validate_scenes(self, scenes: List[Dict]) -> List[Dict]:
        """验证场景数据有效性"""
        valid_scenes = []
        for idx, scene in enumerate(scenes):
            # 时间戳自动填充
            if idx > 0 and "time" not in scene:
                scene["time"] = valid_scenes[-1]["time"] + 3.5
            
            # 组件存在性检查
            valid = True
            for category in ["sta", "exp", "act", "bg"]:
                if not self.am.get_asset(category.upper(), scene[category]):
                    logger.warning(f"场景{idx}包含无效{category}: {scene[category]}")
                    valid = False
            if valid:
                valid_scenes.append(scene)
        return valid_scenes

class VideoRenderer:
    """视频渲染引擎"""
    def __init__(self, asset_manager: AssetManager):
        self.am = asset_manager
        self.base_fps = 24

    def render_scene(self, scene: Dict, duration: float) -> CompositeVideoClip:
        """渲染单个场景"""
        clips = []

        # 背景层
        bg_path = self.am.get_asset("BG", scene["bg"])
        bg_clip = ImageClip(bg_path).set_duration(duration)
        clips.append(bg_clip)

        # 角色层
        character_clip = self._build_character_clip(scene, duration)
        clips.append(character_clip)

        # CG层
        if scene.get("cg"):
            cg_path = self.am.get_asset("CG", scene["cg"])
            cg_clip = ImageClip(cg_path).set_duration(2).set_start(duration-2)
            clips.append(cg_clip)

        return CompositeVideoClip(clips)

    def _build_character_clip(self, scene: Dict, duration: float) -> CompositeVideoClip:
        """构建角色动画"""
        components = []

        # 姿态基础
        sta_path = self.am.get_asset("STA", scene["sta"])
        base_image = ImageClip(sta_path).set_duration(duration)
        components.append(base_image)

        # 表情叠加
        exp_path = self.am.get_asset("EXP", scene["exp"])
        exp_clip = ImageClip(exp_path).set_duration(duration)
        components.append(exp_clip)

        # 动作动画
        if scene["act"] != "ACT_None":
            act_frames = self.am.get_asset("ACT", scene["act"])
            act_clip = ImageSequenceClip(act_frames, fps=self.base_fps)
            act_clip = act_clip.loop(duration=duration)
            components.append(act_clip)

        return CompositeVideoClip(components)

def main():
    """主工作流程"""
    try:
        # 初始化系统
        am = AssetManager()
        generator = SceneGenerator(am)
        renderer = VideoRenderer(am)

        # 处理输入
        audio_path = "input/audio.wav"
        transcript = generator.transcribe_audio(audio_path)
        scenes = generator.generate_scene_data(transcript)

        # 渲染视频
        final_clips = []
        for scene in scenes:
            clip = renderer.render_scene(scene, duration=3.5)
            final_clips.append(clip.set_start(scene["time"]))

        final_video = CompositeVideoClip(final_clips)
        final_video.write_videofile(
            "output/output_video.mp4",
            fps=24,
            codec="libx264",
            audio_codec="aac"
        )
        logger.info("视频生成完成")

    except Exception as e:
        logger.error(f"系统运行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()