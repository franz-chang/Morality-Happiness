# Morality Happiness Multi-Agent Simulation

这个工程实现了三份设计文档中的“微缩狩猎部落”Multi-Agent 仿真：六种资源/制度情境、四类道德类型、狩猎产出、分配、极度疲劳救济、幸福感倒 U 型评估，以及面向前端的参数化运行接口。

## 结构

- `config.example.yaml`：项目统一配置模板，包含服务端端口、仿真默认参数、模型 provider 与 API 占位值。
- `morality_happiness/`：Python 仿真核心、CLI、HTTP API、Camel-AI provider 适配层。
- `frontend/`：React + Vite 可视化控制台。
- `scripts/smoke.py`：六情境 smoke 检查，输出 `outputs/smoke.json`。
- `docs/frontend_design_philosophy.md`：前端视觉原则。

## 配置文件

所有主要默认参数和 API 设置模板都集中在项目根目录：

[config.example.yaml](/Users/chongzhang/Morality_Happiness/config.example.yaml)

本地运行时请保留你自己的 `config.yaml`。仓库默认忽略真实 `config.yaml`，避免把 API key 推到远端。

可直接修改：

```yaml
simulation:
  population_size: 120
  ticks: 200
  food_scale: 1.6

llm:
  default_provider: "rule"
  providers:
    openai:
      model: "gpt-4o-mini"
      api_key: "replace-me"
      base_url: ""
    deepseek:
      model: "deepseek-v4-flash"
      api_key: "replace-me"
      base_url: "https://api.deepseek.com/v1"
    qwen3:
      model: "qwen3:8b"
      api_key: "EMPTY"
      base_url: "http://your-server:11434"
```

环境变量仍可临时覆盖 `config.yaml` 中的 API 设置，例如 `OPENAI_API_KEY`、`DEEPSEEK_API_KEY`、`QWEN3_BASE_URL`。

## 快速运行

```bash
cd /Users/chongzhang/Morality_Happiness
./setup.sh
python3 -m morality_happiness --scenario all --ticks 60 --population 120 --out outputs/run.json
python3 scripts/smoke.py
```

## 启动 API 与前端

一键启动服务端和已构建前端：

```bash
cd /Users/chongzhang/Morality_Happiness
./setup.sh
./start.sh
```

脚本会后台启动服务，自动选择 `8000` 或 `8010-8020` 中的空闲端口，并把 PID、端口和日志写入 `outputs/`。也可以手动指定端口：

```bash
PORT=8010 ./start.sh
```

如果希望前台运行并直接观察日志：

```bash
FOREGROUND=1 ./start.sh
```

关闭本项目服务并清理 PID/端口占用：

```bash
./Stop.sh
```

`Stop.sh` 只会处理 `/Users/chongzhang/Morality_Happiness` 项目进程实际占用的监听端口，不会清理其它项目或系统服务端口。

后端不依赖 FastAPI，使用 Python 标准库 HTTP server：

```bash
cd /Users/chongzhang/Morality_Happiness
python3 -m morality_happiness.server --port 8000
```

前端：

```bash
cd /Users/chongzhang/Morality_Happiness/frontend
npm install
npm run dev
```

浏览器打开 `http://127.0.0.1:5173`。前端会通过 Vite proxy 调用 `http://127.0.0.1:8000/api/run`。
如果 8000 被占用，可改用例如：

```bash
python3 -m morality_happiness.server --port 8010
VITE_API_BASE=http://127.0.0.1:8010 npm run dev
```

## 模型接口

仿真通过 `morality_happiness/llm/providers.py` 组织模型驱动。OpenAI/DeepSeek/本地DeepSeek/阿里云Qwen 使用 Camel-AI provider，Qwen3 使用 Ollama 原生 `/api/chat`。实际 API、模型名和 base_url 从 `config.yaml` 读取。页面参数面板提供模型下拉菜单，可选择：

- `rule`：本地规则默认，不调用外部 API
- `openai`：OpenAI API
- `deepseek`：DeepSeek-V4-flash API
- `local-deepseek`：本地DeepSeek，中国科技云 Uni-API OpenAI-Compatible 接口，默认模型 `deepseek-v4-flash`
- `qwen3`：Qwen3 Ollama 原生 API，可指向远程服务器
- `qwen-bailian`：阿里云Qwen，百炼 OpenAI-Compatible Chat Completions 接口，默认模型 `qwen-plus`

缺省情况下 `rule` provider 不调用外部 API。选择 `openai`、`deepseek`、`local-deepseek`、`qwen3` 或 `qwen-bailian` 时，主仿真会按 Agent 顺序调用模型 provider 生成行为决策；如果 provider 不可用，模型行为驱动会明确失败，不会静默回退为规则决策。安装依赖：

```bash
./setup.sh
```

DeepSeek 模式默认使用 `deepseek-v4-flash`。每个 Agent 在同一轮内按 id 顺序独立判断 `effort`、`compliance_action`、`share_rate` 和 `restraint_rate`，Prompt 会注入情境、道德类型、疲劳/储备/幸福状态，以及同轮已决策 Agent 的摘要。

本地DeepSeek 模式读取 `config.yaml` 中 `llm.providers.local-deepseek`：

```yaml
local-deepseek:
  label: "本地DeepSeek"
  model: "deepseek-v4-flash"
  api_key: "这里是API"
  base_url: "https://uni-api.cstcloud.cn/v1"
  response_mode: "instruct"
  thinking: false
  chat_template_kwargs:
    thinking: false
```

把 `api_key` 替换为你自己的中国科技云 Uni-API token 后即可使用。`deepseek-v4-flash` 支持通过 `chat_template_kwargs.thinking` 控制 thinking；当前配置显式关闭 thinking，作为 instruct/直接回答模式使用。若要开启思考过程，把 `thinking` 改为 `true`。

Qwen3 模式会向 `${base_url}/api/chat` 发送 Ollama 原生请求。远程部署时把 `config.yaml` 中 `llm.providers.qwen3.base_url` 改成你的服务器地址，例如 `http://192.168.1.20:11434`；如果你的服务器前面有鉴权代理，可以把 token 写到 `api_key`，系统会作为 Bearer token 发送。

阿里云Qwen 模式读取 `config.yaml` 中 `llm.providers.qwen-bailian`：

```yaml
qwen-bailian:
  label: "阿里云Qwen"
  model: "qwen-plus"
  api_key: "这里是API"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  response_mode: "instruct"
  max_tokens: 256
```

把 `api_key` 替换为百炼 API Key；如果你使用北京地域业务空间专属域名，可把 `base_url` 改为 `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`，并按控制台模型页把 `model` 改成对应 Qwen 模型 ID。

## 运行日志

每次运行都会返回结构化日志：

- `system_logs`：模型 provider 状态、每轮总产出/平均幸福/最低保障率摘要。
- `agent_logs`：每个 Agent 每一轮的行动、努力值、分配肉量、储备、疲劳、状态、幸福感和结果。

前端“系统运行日志”栏会按当前情境展示这些日志，下载 JSON 时也会保留完整日志。

## 已实现的设计点

- 120 人默认人口、青年/中年/老年能力与需求差异。
- 四类道德权重：自我发展、自我克制、发展型利他、克制型利他。
- S1-S6 情境适配矩阵。
- 努力、服从、分享、节约函数。
- 强组织、自由抱团、各自行动三套产出和分配规则。
- 小/中/大型猎物刷新与 `P(success)=C_team/(C_team+C_p)` 捕猎判定。
- `Extreme_Fatigue` 人道救济，不死亡。
- 个体/道德分组、群体、社会层面的指标汇总。
