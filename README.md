## 项目简介

D_create 是一个命令行工具，读取 PDF/DOCX/TXT/MD 等文件，通过大语言模型生成问答对数据集，并导出只有 `input, ground_truth` 两列的 CSV。工具目前支持：

- OpenAI 系列模型（默认）
- 通义千问 3.x（DashScope API

## 环境准备

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## LLM 配置

### OpenAI

```bash
OPENAI_API_KEY=sk-xxxx
OPENAI_MODEL=gpt-4o-mini   # 可覆盖 CLI 的 --openai-model
```

### 通义千问（DashScope）

```bash
DASHSCOPE_API_KEY=sk-aliyun-xxxx   # 或 QWEN_API_KEY
QWEN_MODEL=qwen-plus               # 可覆盖 CLI 的 --qwen-model
```

> DashScope 需要单独申请 AK/SK，并保证账号开通了所需模型的调用权限。

### 自动重试 / 限流参数（OpenAI 专用，可选）

```bash
# 最大重试次数（默认 5）
OPENAI_MAX_RETRIES=5
# 初始重试等待秒数（默认 20）
OPENAI_RETRY_INITIAL_DELAY=20
# 指数退避因子（默认 2）
OPENAI_BACKOFF_FACTOR=2
# 单次等待的最大秒数（默认 60）
OPENAI_RETRY_MAX_DELAY=60
# 相邻请求的最小时间间隔（默认 0，不限流）
OPENAI_RATELIMIT_MIN_INTERVAL=25
```

## 命令示例

```bash
# OpenAI（默认）
python -m d_create.cli ^
  --paths Journey_to_the_West.pdf ^
  --provider openai ^
  --output dataset.csv ^
  --chunk-size 900 ^
  --chunk-overlap 200 ^
  --per-chunk 2

# 通义千问（DashScope）
python -m d_create.cli ^
  --paths Journey_to_the_West.pdf ^
  --provider qwen ^
  --qwen-model qwen-plus ^
  --output dataset_qwen.csv

# 无 Key 测试
python -m d_create.cli --paths Journey_to_the_West.pdf --provider dummy --output demo.csv
```

命令执行后会在控制台展示进度（依赖 `tqdm`），并在目标路径生成两列 CSV。


## 目录结构（简要）
```
D_create/
├─ d_create/
│  ├─ cli.py            # 命令行入口
│  ├─ ingest.py         # 文档读取
│  ├─ chunking.py       # 文本分块
│  ├─ llm.py            # Provider：Qwen/OpenAI/Dummy + 解析
│  ├─ qa_generator.py   # 生成主流程与导出
│  └─ utils_text.py     # 文本清洗与提示辅助
├─ datas/               # 输入数据集
├─ output/              # 实验输出
├─ requirements.txt
├─ README.md
```