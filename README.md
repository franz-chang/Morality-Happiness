# Morality Happiness: Multi-Agent Hunting Society Simulation

This project implements a small-scale hunting-society agent-based model for studying the relationship between moral behavior, resource allocation, social hierarchy, trade, and well-being. It includes six resource-and-organization scenarios, four moral orientations, hunting production, fatigue and relief, a food--shell market, and a three-level hierarchy.

The simulation environment is implemented directly in Python. LLMs may drive individual behavioral decisions, but hunting, allocation, trade, promotion, accounting, and well-being updates remain local and reproducible; the project does not depend on CAMEL-AI.

## Project layout

- `config.example.yaml`: configuration template, including simulation defaults and API-key placeholders.
- `morality_happiness/`: simulation engine, command-line interface, HTTP API, and direct LLM providers.
- `frontend/`: React + Vite experiment console.
- `scripts/smoke.py`: smoke runner for the six scenarios.
- `scripts/calibrate_resource_supply.py`: resource-supply calibration helper.
- `tests/`: simulation and provider tests.
- `docs/`: UML and frontend-design materials.

## Installation

```bash
git clone https://github.com/franz-chang/Morality-Happiness.git
cd Morality-Happiness
python -m venv .venv
source .venv/bin/activate
pip install -e .

cd frontend
npm install
```

Copy `config.example.yaml` to `config.yaml` before using an external model provider. `config.yaml` is ignored by Git so that local API credentials are not committed.

## Quick run

Run all scenarios from the command line:

```bash
python -m morality_happiness \
  --scenario all --ticks 560 --replications 10 --population 120 \
  --out outputs/run.json

python scripts/smoke.py
```

Run the test suite:

```bash
pytest -q
```

## API and frontend

Start the backend API:

```bash
./C1_start.sh
```

The script records the PID, port, and log in `outputs/`. The default API endpoint is `http://127.0.0.1:38471/`; change `server.port` in `config.yaml`, or set `PORT` explicitly:

```bash
PORT=8010 ./C1_start.sh
FOREGROUND=1 ./C1_start.sh
```

Stop only this project's backend process:

```bash
./C2_stop.sh
```

Start the development frontend in a second terminal:

```bash
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173/`. The Vite development server proxies `/api` requests to port `38471` by default.

## Simulation design

### Scenarios

The six scenarios cross resource regime and hunting organization:

| Scenario | Resource regime | Organization |
| --- | --- | --- |
| S1 / S4 | Scarce / abundant | Tribe-wide organized hunt |
| S2 / S5 | Scarce / abundant | Chieftain-led squad hunt |
| S3 / S6 | Scarce / abundant | Individual hunt |

S3 and S6 do not create an artificial shared hunting pool. All new hierarchy and market modules can be disabled through `trade_enabled`, `rank_enabled`, `rank_split_enabled`, and `promotion_enabled`, preserving the original-system baseline for ablation studies.

### Three-level hierarchy

The default population is 120 agents:

- one Chief;
- seven Chieftains;
- sixteen Hunters per Chieftain.

Roles determine organizational relationships and post-hunt shares, but do not directly change physical ability, exploration, coordination, or hunting success. The default group-share weights are Chief `4`, Chieftain `2`, and Hunter `1`.

For S2/S5, the Chief receives a leadership share of `4/130 ≈ 0.03077` from each squad's realized output. The remaining output stays with that squad and is distributed by the Chieftain/Hunter weights. This aligns the default Chief-to-Hunter per-capita food ratio with S1/S4.

### Revised trade and promotion protocol

The current design implements the revised identity-and-trade protocol:

- Initial shell balances are sampled from `U(20, 40)` by default. Set `initial_shells_distribution: equal` for the equal-balance robustness control.
- The first 28 ticks are a burn-in period. They are retained in raw logs but excluded from formal aggregate metrics; the first identity review occurs when burn-in ends.
- After a shared hunting allocation, individual food realizations are multiplied by `exp(N(0, idio_sigma))`, with default `idio_sigma = 0.3`, and then renormalized within the team. Team production is therefore unchanged while buyers and sellers can coexist in the same tick.
- The centralized market clears after hunting allocation and before food-survival settlement. Its default price is `p* = clip(D / S, 0.2, 5.0)` shells per food unit. A fixed-price robustness control is also available.
- Each buyer may spend no more than 50% of its current shell balance in one clearing round. Demand is budget-adjusted and both sides are rationed pro rata.
- The default safety buffer is `1.5 × need`; the recommended sensitivity sweep is `{1.0, 1.5, 2.0}`.
- Promotion uses `0.8 × z(log(1 + shell_balance)) + 0.2 × z(period_hunting_contribution)`, rather than a percentile score. Ties are resolved by shell balance, contribution, lower extreme-fatigue burden, and a seed-stable tie-breaker.
- A market is marked effective only when the analysis-period trade participation rate exceeds 30%. Lower-participation runs are retained but should not be used for trade-effect comparisons.

## Configuration

The main simulation settings live under `simulation:` in `config.yaml`:

```yaml
simulation:
  population_size: 120
  ticks: 560
  replications: 10

  trade_enabled: true
  rank_enabled: true
  rank_split_enabled: true
  promotion_enabled: true

  initial_shells_distribution: "uniform" # uniform | equal
  initial_shells_min: 20.0
  initial_shells_max: 40.0
  burn_in_ticks: 28

  chief_share_weight: 4.0
  chieftain_share_weight: 2.0
  hunter_share_weight: 1.0
  chief_leadership_share: 0.0307692308 # 4 / 130
  chieftain_slots: 7
  hunters_per_chieftain: 16

  idio_sigma: 0.3
  trade_safety_buffer: 1.5
  market_price_rule: "demand_supply" # demand_supply | fixed
  shell_base_price: 1.0
  shell_min_price: 0.2
  shell_max_price: 5.0
  buyer_budget_fraction: 0.5

  promotion_period: 28
  promotion_shell_weight: 0.80
  promotion_hunting_weight: 0.20
```

Recommended robustness sweeps are:

- `trade_safety_buffer`: `{1.0, 1.5, 2.0}`;
- `idio_sigma`: `{0.2, 0.3, 0.5}`;
- `market_price_rule`: demand/supply clearing versus fixed `p = 1.0`;
- `promotion_shell_weight`: `{0.60, 0.80, 1.00}`;
- initial shells: uniform `U(20,40)` versus equal-balance control.

## LLM decision providers

The provider configuration is under `llm.providers` in `config.yaml`. The supported providers are:

- `rule`: deterministic local policy; no external API calls.
- `openai`: OpenAI-compatible provider.
- `deepseek`: DeepSeek-compatible provider; the UI name is `DeepSeek-V4.1-Flash-1017` and the configured call ID is `deepseek-flash`.
- `local-deepseek`: China Science and Technology Cloud Uni-API compatible provider.
- `qwen3`: Ollama-native `/api/chat` provider.
- `qwen-bailian`: Alibaba Cloud Bailian OpenAI-compatible provider.

Example placeholder configuration:

```yaml
llm:
  default_provider: "rule"
  providers:
    openai:
      model: "gpt-4o"
      api_key: "replace-me"
      base_url: "https://api.rcouyi.com/v1"
      response_mode: "instruct"
```

When a non-rule provider is selected, the simulation calls the provider once per agent decision. A failed provider does not silently fall back to the local rule policy. API keys and endpoint values may also be supplied through environment variables such as `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `DEEPSEEK_API_KEY`.

## Outputs and accounting

Each scenario result contains:

- `system_logs` and optional `agent_logs`;
- `society_metrics`, `moral_type_metrics`, `group_metrics`, and `role_metrics`;
- `trade_metrics`, `trade_records`, and shell-wealth inequality;
- `hierarchy_metrics`, `promotion_events`, `promotion_reviews`, and the current hierarchy;
- food and shell accounting ledgers;
- `experiment_protocol`, including burn-in, price rule, and idiosyncratic-noise metadata.

Food and shells are conserved within their respective accounting rules. External relief is explicitly recorded as an external food input; public relief is deducted from the public granary. A run raises an error when food or shell ledger checks fail.

## Implemented capabilities

- Four moral orientations: self-development, self-restraint, altruistic development, and altruistic restraint.
- Age-dependent abilities and food needs for young, middle-aged, and old agents.
- Effort, compliance, sharing, restraint, fatigue, relief, and well-being dynamics.
- Organized, group, and individual hunting modes with prey encounters and success probability `P(success) = C_team / (C_team + C_p)`.
- Food--shell trade, wealth Gini metrics, and transaction records.
- Chief--Chieftain--Hunter organization, periodic reassignment, and mobility metrics.
- JSON export through the frontend dashboard.
