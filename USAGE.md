# MuseTalk 口型同步服务 — 使用说明

项目路径：`/home/ubuntu/raoyonghui/MuseTalk`  
当前机器：NVIDIA L20（约 46GB 显存）

---

## 1. Conda 环境

| 项 | 值 |
|----|----|
| 环境名 | **`heng`** |
| Python | 3.10 |
| 路径 | `/home/ubuntu/miniconda3/envs/heng` |

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate heng
cd /home/ubuntu/raoyonghui/MuseTalk
```

依赖见 `requirements.txt`（含 FastAPI、uvicorn、torch、mmcv/mmdet/mmpose 等）。

---

## 2. 模型路径

所有权重位于项目下的 `models/` 与 `third_party/`：

| 用途 | 路径 |
|------|------|
| MuseTalk v1.5 UNet | `models/musetalkV15/unet.pth` |
| MuseTalk v1.5 配置 | `models/musetalkV15/musetalk.json` |
| MuseTalk v1.0（备用） | `models/musetalk/` |
| VAE | `models/sd-vae/` |
| Whisper | `models/whisper/` |
| DWPose 人脸/姿态 | `models/dwpose/dw-ll_ucoco_384.pth` |
| Face Parsing | `models/face-parse-bisent/` |
| SyncNet | `models/syncnet/latentsync_syncnet.pt` |
| LR-ASD 说话人检测 | `third_party/LR-ASD/weight/finetuning_TalkSet.model` |
| CodeFormer 人脸修复 | `models/codeformer/codeformer.pth` |

若缺少 CodeFormer 权重，可从官方 Release 下载（约 360MB）：

```bash
mkdir -p models/codeformer
aria2c -x 16 -s 16 -k 1M -d models/codeformer -o codeformer.pth \
  "https://ghproxy.net/https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
```

源码位于 `third_party/CodeFormer/`（仅对人脸 crop 推理，无需再跑官方检测流程）。

HTTP 服务默认使用 **v1.5**，对应配置见 `musetalk/service/engine.py` 中的 `ServiceConfig`。

---

## 3. 依赖的系统工具

| 工具 | 路径 / 说明 |
|------|-------------|
| ffmpeg / ffprobe | `/usr/local/ffmpeg/bin/` |
| 动态库 | `/usr/local/ffmpeg/lib`（需设置 `LD_LIBRARY_PATH`） |

systemd 服务已配置：

```ini
Environment=PATH=...:/usr/local/ffmpeg/bin:...
Environment=LD_LIBRARY_PATH=/usr/local/ffmpeg/lib
```

手动启动时请同样设置，否则会出现 `ffprobe` 退出码 127（找不到 `libavdevice.so.59`）。

---

## 4. 启动方式

### 方式 A：systemd（推荐，生产）

服务单元：`muse.service`（已安装到 `/etc/systemd/system/muse.service`）

```bash
# 启动 / 停止 / 重启 / 状态
sudo systemctl start muse
sudo systemctl stop muse
sudo systemctl restart muse
sudo systemctl status muse

# 开机自启（已 enable）
sudo systemctl enable muse

# 查看日志
sudo journalctl -u muse -f
# 或业务日志
tail -f /home/ubuntu/raoyonghui/MuseTalk/logs/musetalk_service.log
```

| 项 | 值 |
|----|----|
| 监听地址 | `0.0.0.0:8765` |
| WorkingDirectory | `/home/ubuntu/raoyonghui/MuseTalk` |
| ExecStart | `heng` 环境的 `uvicorn server:app --host 0.0.0.0 --port 8765` |
| GPU | `CUDA_VISIBLE_DEVICES=0` |
| 入口文件 | `server.py` |

修改代码或配置后需：

```bash
sudo systemctl restart muse
```

### 方式 B：前台手动启动（调试）

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate heng
cd /home/ubuntu/raoyonghui/MuseTalk
export LD_LIBRARY_PATH=/usr/local/ffmpeg/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/ffmpeg/bin:$PATH

uvicorn server:app --host 0.0.0.0 --port 8765
```

可选环境变量：

| 变量 | 含义 | 默认 |
|------|------|------|
| `MUSETALK_MAX_CONCURRENT` | 最大并发推理数 | `1` |
| `MUSETALK_GPU_IDS` | 各引擎槽位 GPU，如 `0,1` | 未设置则用 `gpu_id=0` |
| `MUSETALK_USE_CODEFORMER` | 是否对人脸生成结果做 CodeFormer 修复 | `true` |
| `MUSETALK_CODEFORMER_FIDELITY` | 保真度 `0~1`（越高越接近原脸） | `0.7` |
| `MUSETALK_CODEFORMER_MODEL` | CodeFormer 权重路径 | `./models/codeformer/codeformer.pth` |
| `MUSETALK_ASD_MASK_DILATE` | speaking mask 前后各膨胀帧数 | `8` |
| `MUSETALK_ASD_THRESHOLD` | LR-ASD 分数阈值（logit，越低越宽松） | `0.0` |
| `MUSETALK_CODEFORMER_STRIDE` | CodeFormer 每隔 N 个说话帧修复一次 | `2` |

单卡建议保持 `MUSETALK_MAX_CONCURRENT=1`；短视频可尝试 `2`，长视频不建议。

CodeFormer 仅作用于 **LR-ASD 判定为说话、且已生成口型的人脸 crop**；失败时自动回退到未修复人脸。权重缺失时服务仍可启动，但会跳过修复。

---

## 5. API 说明

### 健康检查

```bash
curl http://127.0.0.1:8765/health
```

示例返回：

```json
{
  "status": "ok",
  "model_loaded": true,
  "max_concurrent": 1,
  "total_engines": 1,
  "available_engines": 1,
  "busy_engines": 0
}
```

### 口型同步

```http
POST /lipsync
Content-Type: application/json
```

| 字段 | 必填 | 说明 |
|------|------|------|
| `video_path` | 是 | 输入视频本地路径 |
| `audio_path` | 是 | **仅用于驱动口型**；成片音轨仍用原视频声音 |
| `output_path` | 是 | 输出视频路径 |
| `force_chunk` | 否 | 强制分段（默认 `false`） |
| `chunk_duration_sec` | 否 | 分段时长秒数（默认服务内 60） |

行为要点：

- **画面**：按 `audio_path` 做口型同步  
- **声音**：保留**原视频音轨**（不替换成 `audio_path`）  
- **长视频**：有效时长 > 120s 时自动按帧精确分段（默认每段 60s），再校验帧数并拼接  
- **说话人检测**：默认开启 LR-ASD，仅对判定为说话的人脸做口型同步  

示例：

```bash
curl -X POST http://127.0.0.1:8765/lipsync \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/home/ubuntu/raoyonghui/MuseTalk/test_video.mp4",
    "audio_path": "/home/ubuntu/raoyonghui/MuseTalk/translated_voice.wav",
    "output_path": "/home/ubuntu/raoyonghui/MuseTalk/test_output.mp4"
  }'
```

---

## 6. 测试脚本

```bash
conda activate heng
cd /home/ubuntu/raoyonghui/MuseTalk
python scripts/test_lipsync_api.py
```

脚本内默认路径（可自行改 `scripts/test_lipsync_api.py`）：

| 变量 | 路径 |
|------|------|
| video | `/home/ubuntu/raoyonghui/MuseTalk/test_video.mp4` |
| audio | `/home/ubuntu/raoyonghui/MuseTalk/translated_voice.wav` |
| output | `/home/ubuntu/raoyonghui/MuseTalk/test_output.mp4` |
| base_url | `http://127.0.0.1:8765` |

也可用 curl 或任意 HTTP 客户端调用 `/lipsync`。

---

## 7. 离线 CLI 推理（不用 HTTP）

```bash
conda activate heng
cd /home/ubuntu/raoyonghui/MuseTalk

python -m scripts.inference \
  --inference_config configs/inference/custom_test.yaml \
  --result_dir results/test \
  --unet_model_path models/musetalkV15/unet.pth \
  --unet_config models/musetalkV15/musetalk.json \
  --version v15 \
  --use_float16
```

配置文件示例：`configs/inference/custom_test.yaml`。

---

## 8. 日志

| 来源 | 位置 |
|------|------|
| 业务日志 | `logs/musetalk_service.log`（轮转，单文件约 10MB，保留 2 个） |
| systemd | `sudo journalctl -u muse -f` |

---

## 9. 常见问题

**1. `ffprobe` exit 127 / `libavdevice.so.59`**  
未设置 `LD_LIBRARY_PATH=/usr/local/ffmpeg/lib`。用 systemd 启动一般已配置；手动启动请 export。

**2. 服务启动慢 / TimeoutStartSec**  
首次加载模型约十几秒到一两分钟，属正常。`TimeoutStartSec=300`。

**3. 输出没有翻译配音**  
当前设计是：**口型跟 `audio_path`，声音保留原视频**。若要成片也换成配音，需改合成逻辑。

**4. 长视频 OOM**  
依赖自动分段（>120s）。单卡并发建议保持 `1`。

**5. 改代码不生效**  

```bash
sudo systemctl restart muse
```

---

## 10. 目录速查

```
MuseTalk/
├── server.py                 # HTTP 入口
├── muse.service              # systemd 单元（仓库内副本）
├── scripts/
│   ├── test_lipsync_api.py   # API 测试脚本
│   └── inference.py          # CLI 推理
├── musetalk/service/
│   ├── engine.py             # 推理引擎与配置
│   ├── engine_pool.py        # 并发引擎池
│   ├── long_video.py         # 长视频帧精确分段/拼接
│   └── logging_setup.py
├── models/                   # 模型权重
├── third_party/LR-ASD/       # 说话人检测
├── configs/inference/        # CLI 配置
└── logs/                     # 服务日志
```
