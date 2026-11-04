# Morality Happiness Multi-Agent Simulation

这个工程实现了“微缩狩猎部落”Multi-Agent 仿真：六种资源/制度情境、四类道德类型、狩猎产出、分配、极度疲劳救济、幸福感倒 U 型评估，以及扩展后的自由交易和三级身份制度。

## 结构

- `config.example.yaml`：项目统一配置模板，包含服务端端口、仿真默认参数、模型 provider 与 API 占位值。
- `morality_happiness/`：自主实现的 Python 仿真核心、CLI、HTTP API、无状态直连模型适配层。
- `frontend/`：React + Vite 可视化控制台。
- `scripts/smoke.py`：六情境 smoke 检查，输出 `outputs/smoke.json`。
- `docs/frontend_design_philosophy.md`：前端视觉原则。

## 配置文件

所有主要默认参数和 API 设置模板都集中在项目根目录：

[config.example.yaml](config.example.yaml)

本地运行时请保留你自己的 `config.yaml`。仓库默认忽略真实 `config.yaml`，避免把 API key 推到远端。

可直接修改：

```yaml
simulation:
  population_size: 120
  ticks: 560
  replications: 10
  scarcity_resource_multiplier: 1.21094
  abundance_resource_multiplier: 0.92285
  food_scale: 1.6
  trade_enabled: true
  rank_enabled: true
  rank_split_enabled: true
  promotion_enabled: true
  initial_shells_distribution: "uniform" # uniform | equal (robustness control)
  initial_shells_min: 20.0
  initial_shells_max: 40.0
  burn_in_ticks: 28
  relief_source: "external" # external | public | none
  public_food_reserve: 0.0
  chieftain_slots: 7
  hunters_per_chieftain: 16
  promotion_period: 28
  promotion_shell_weight: 0.80
  promotion_hunting_weight: 0.20
  chief_leadership_share: 0.0307692308 # 4 / 130; aligns S2/S5 with S1/S4
  idio_sigma: 0.3
  trade_safety_buffer: 1.5
  market_price_rule: "demand_supply" # demand_supply | fixed
  shell_min_price: 0.2
  shell_max_price: 5.0
  buyer_budget_fraction: 0.5

llm:
  default_provider: "rule"
  providers:
    openai:
      model: "gpt-4o"
      api_key: "replace-me"
      base_url: "https://api.rcouyi.com/v1"
      response_mode: "instruct"
    deepseek:
      model: "deepseek-flash"
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
cd /path/to/Morality-Happiness
python -m morality_happiness --scenario all --ticks 560 --replications 10 --population 120 --out outputs/run.json
python scripts/smoke.py
```

## 启动 API 与前端

启动后端 API：

```bash
cd /path/to/Morality-Happiness
./C1_start.sh
```

脚本会后台启动服务，将 PID、端口和日志写入 `outputs/`。默认端口为 `38471`，可在 `config.yaml` 中的 `server.port` 修改。也可以手动指定端口：

```bash
PORT=8010 ./C1_start.sh
```

如果希望前台运行并直接观察日志：

```bash
FOREGROUND=1 ./C1_start.sh
```

关闭本项目服务并清理 PID/端口占用：

```bash
./C2_stop.sh
```

`C2_stop.sh` 只会处理本项目实际占用的监听端口，不会清理其它项目或系统服务端口。

后端不依赖 FastAPI，使用 Python 标准库 HTTP server：

```bash
cd /path/to/Morality-Happiness
python -m morality_happiness.server --port 38471
```

前端：

```bash
cd /path/to/Morality-Happiness/frontend
npm install
npm run dev
```

浏览器打开 `http://127.0.0.1:5173`。前端会通过 Vite proxy 调用 `http://127.0.0.1:38471/api/run`。服务端默认监听 `38471`；可在 `config.yaml` 的 `server.port` 中修改。
如果 38471 被占用，可改用例如：

```bash
python3 -m morality_happiness.server --port 8010
VITE_API_BASE=http://127.0.0.1:8010 npm run dev
```

## 模型接口

仿真通过 `morality_happiness/llm/providers.py` 组织模型驱动。OpenAI/DeepSeek/本地DeepSeek/阿里云Qwen 直接使用 OpenAI-compatible SDK，Qwen3 使用 Ollama 原生 `/api/chat`，不再依赖 CAMEL-AI。实际 API、模型名和 base_url 从 `config.yaml` 读取。客户端在仿真内复用，每次只发送当前决策上下文，不积累其他个体的聊天历史。狩猎、交易、身份评选与幸福感更新均由本地仿真环境执行，原有决策顺序不变。`response_mode: instruct` 仅以提示词要求 JSON；其他模式使用 JSON-object 参数。重试仍由 `max_retries` 控制，不再在框架失败后额外调用一次 API。网络延迟与逐个体调用量仍会影响速度；移除框架不等于消除这些成本。页面参数面板提供模型下拉菜单，可选择：

- `rule`：本地规则默认，不调用外部 API
- `openai`：NO.1 API 的 OpenAI 兼容中转接口；需使用该站控制台提供的 API Key 和 Base URL
- `deepseek`：界面显示为 DeepSeek-V4.1-Flash-1017，API 调用 ID 为 `deepseek-flash`
- `local-deepseek`：本地DeepSeek，中国科技云 Uni-API OpenAI-Compatible 接口，默认模型 `deepseek-v4-flash`
- `qwen3`：Qwen3 Ollama 原生 API，可指向远程服务器
- `qwen-bailian`：阿里云Qwen，百炼 OpenAI-Compatible Chat Completions 接口，默认模型 `qwen-plus`

缺省情况下 `rule` provider 不调用外部 API。选择 `openai`、`deepseek`、`local-deepseek`、`qwen3` 或 `qwen-bailian` 时，主仿真会按 Agent 顺序调用模型 provider 生成行为决策；如果 provider 不可用，模型行为驱动会明确失败，不会静默回退为规则决策。安装依赖：

NO.1 API 文档给出的聊天接口为 `https://api.rcouyi.com/v1/chat/completions`，因此项目的 `base_url` 配为 `https://api.rcouyi.com/v1`；认证使用该站控制台令牌。示例模型设为文档请求示例中的 `gpt-4o`，若控制台模型列表提供不同 ID，请相应调整。`response_mode: instruct` 使用标准 Chat Completions 请求，不附加 JSON-mode 扩展参数。也可用 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `OPENAI_MODEL` 环境变量覆盖。参见[平台前言](https://platform.rcouyi.com/)与[OpenAI 官方库使用教程](https://platform.rcouyi.com/5312630m0)。

```bash
pip install -e .
```

## 长周期实验口径

长周期实验默认使用 560 轮、10 个重复。匮乏/丰裕档的目标参考供给比为 0.80/1.025；当前资源乘数（1.21094/0.92285）是 S3/S6、560 轮、seed 42 的单种子起始校准值，正式实验仍需检查各重复的实测供给比。可用 `python scripts/calibrate_resource_supply.py` 重新标定。S2/S5 的组织成本按小队人数折算（每 `typical_team_size` 人计一个成本单位），首领从每支小队的实际产出中取得一次领导份额，默认 `4/130`，保证与 S1/S4 的首领/猎人人均分成比对齐。集中市场每轮使用 `p*=clip(D/S,0.2,5.0)`（或固定价稳健性对照），买方最多动用 50% 贝壳。救济来源必须显式选择：`external` 计入外部输入，`public` 从公共粮仓扣除，`none` 不发放救济。结果 JSON 包含逐轮食物守恒误差、贝壳守恒误差、实测供给比和救济总量。多重复运行保留聚合统计和首轮样本记录，不保留海量逐 Agent 日志；将重复数设为 1 可取得完整逐 Agent 日志。

## 身份与交易修订版规则

- 初始贝壳默认按 `U(20,40)` 抽样；前 28 tick 是 burn-in，不进入正式统计，首轮身份评选在 burn-in 结束时进行。
- 组织狩猎的身份分成后会施加 `exp(N(0, idio_sigma))` 个体扰动，再在团队内重归一化，因此不改变团队总产出，也允许同一轮中同时出现买方与卖方。
- 身份评选使用 `shell_weight·z(log(1+shell_balance)) + hunting_weight·z(period_hunting_contribution)`；默认权重为 `0.8/0.2`，并按贝壳、贡献、极度疲劳率和固定随机种子处理平局。
- 交易参与率超过 30% 的情境被标记为“市场有效”；低于该阈值的情境仍保留记录，但不应用于交易效果比较。
- 结果额外提供 `hierarchy_metrics`（晋升率、上下流动率、首领/酋长留任率及跨身份不平等）和完整的交易、评选事件元数据。

DeepSeek 模式在界面显示为 `DeepSeek-V4.1-Flash-1017`，实际 API 调用 ID 使用服务端公布的 `deepseek-flash`。每个 Agent 在同一轮内按 id 顺序独立判断 `effort`、`compliance_action`、`share_rate` 和 `restraint_rate`，Prompt 会注入情境、道德类型、疲劳/储备/幸福状态，以及同轮已决策 Agent 的摘要。

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

扩展结果还包含 `trade_metrics`、`trade_records`、`role_metrics`、`promotion_events` 和 `hierarchy`。交易发生在狩猎分成之后、原有食物结算之前；身份在评选周期边界重新生效。

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
- 首领/酋长/猎人三级身份树，默认 1 + 7 + 7×16 = 120 人。
- 群体狩猎身份分成权重：首领 4、酋长 2、猎人 1；各自行动场景不强行制造群体分成。
- 食物—贝壳自由交易、市场清算价格、交易参与率和贝壳财富基尼系数。
- 默认每 28 Tick 评选身份，贝壳数量占 80%，周期狩猎贡献占 20%。
- 交易、身份、身份分成和周期评选均可关闭，用于原始系统基线和消融实验。
