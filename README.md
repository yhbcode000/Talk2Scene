# Talk2Scene - 语音转场景终端应用 - README

## 📌 项目概述

Talk2Scene 是一个 **终端应用（Terminal App）**，用于将 **音频杂谈文件** 自动解析为 **可视化场景**。该工具能够 **识别文本内容及时间节点**，并智能匹配 **STA（姿态）、EXP（表情）、ACT（动作）、BG（背景）、CG（CG 插画）**，最终生成 **CSV 文件** 和 **预览视频**。

## 🚀 功能介绍

### 1️⃣ **音频输入**
- 读取 `input_audio.wav` 文件。
- 通过 **Google Speech API** 进行语音转录，生成 **文本 + 时间戳**。
- 可选：支持 **本地 JSON 存储**，缓存转录结果以减少重复计算。

### 2️⃣ **文本解析与匹配**
- **利用 LLM 进行 In-Context Learning**，从音频文本中推理合适的 **STA、EXP、ACT、BG、CG**。
- **自动选择 CG 插入点** → 通过 GPT 识别重点内容，每 **5 句** 自动标记 **CG 插画** 插入位置。
- **动态角色匹配** → 基于 LLM 的风格分析，自动为不同角色分配合适的表现方式。
- **本地状态存储** → 解析过程中，所有文本分析及匹配信息会实时写入 `parsed_data.json`，防止进程中断丢失。

### 3️⃣ **CSV 数据导出**
- 记录 **时间戳、文本、STA、EXP、ACT、BG、CG**。
- 生成 **可点击（Clickable）CSV 文件** 方便查看和调整。
- **支持本地 CSV 状态存储**，可以中断后续恢复处理。

### 4️⃣ **预览视频生成**
- 结合 **文本 + 角色动作 + 背景 + CG 插画** 生成 **视频** `output_video.mp4`。
- 通过 Function Calling 直接调用视频生成 API，无需手动匹配动画参数。
- **支持本地缓存**，在 `video_cache/` 目录下存储部分关键帧，便于快速回溯与调整。

### 5️⃣ **CG 插画自动处理**
- 终端自动检测 **CG 插入点**，并通过 **Function Calling** 请求 AI 推荐 **最佳 CG 插画**。
- 用户可选择自动匹配或手动上传 CG 图片。
- **支持本地存储 CG 资源**，已使用的 CG 记录存储在 `cg_metadata.json`。

## 🛠️ **安装与运行**

### 1️⃣ **环境要求**
- **Python 3.8+**
- **依赖库**
  ```sh
  pip install speechrecognition moviepy ffmpeg-python openai pandas
  ```

### 2️⃣ **运行步骤**

```sh
python talk2scene.py
```

### 3️⃣ **CSV 数据示例**

| 时间戳 | 文本 | STA（姿态） | EXP（表情） | ACT（动作） | BG（背景） | CG（CG 插画） |
|--------|------|------------|------------|------------|------------|----------------|
| 0.00s  | 你好，欢迎来到本次讨论。 | STA_Stand_Default | EXP_Neutral | ACT_RaiseHand | BG_CoffeeShop | CG_Intro |
| 3.50s  | 今天我们要探讨人工智能的影响。 | STA_Sit_Normal | EXP_Thinking | ACT_Shrug | BG_Lab | CG_AI_Impact |
| 7.00s  | 许多人认为 AI 既有利也有弊。 | STA_ArmsCrossed | EXP_Surprised | ACT_SlamTable | BG_NewsStudio | CG_Pros_Cons |

### 4️⃣ **自动 CG 插画输入示例**

```sh
等待 AI 推荐 CG 插画路径 (输入 'exit' 结束)...
系统建议的 CG: /generated/cg_impact_of_ai.png @ 14.00s
是否接受？(Y/N): Y
CG 已自动添加: /generated/cg_impact_of_ai.png @ 14.00s
```

## 🎥 **输出内容**
✅ `output_metadata.csv`  → **记录时间戳 + 文本 + STA + EXP + ACT + BG + CG**
✅ `output_video.mp4`  → **带动画预览的 AI 角色视频**
✅ `parsed_data.json`  → **存储当前解析状态，可随时恢复**
✅ `video_cache/`  → **关键帧缓存，便于视频调整**
✅ `cg_metadata.json`  → **存储已匹配的 CG 资源**

## 📝 **后续改进计划**
1. **LLM 自动优化匹配**（GPT 解析对话风格，动态调整角色表现）
2. **API 驱动的 CG 生成**（自动绘制 AI 访谈关键场景）
3. **多角色 AI 互动支持**（基于音频识别不同说话人）
4. **支持情绪识别**（分析语音情感，调整表情 & 动作匹配）
5. **增强可视化编辑**（提供 GUI 交互界面以调整动画匹配结果）
6. **本地缓存优化**（减少重复计算，提高处理效率）

🎯 **Talk2Scene 适用于 AI 访谈、教学演示、虚拟主播、AI 动态内容创作等多种场景。**

---
**© 2025 Talk2Scene 语音驱动场景生成工具**

