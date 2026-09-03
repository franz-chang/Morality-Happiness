import {
  Activity,
  CircleStop,
  Download,
  ArrowLeft,
  Languages,
  Network,
  Play,
  RefreshCw,
  RotateCcw,
  SlidersHorizontal,
  TerminalSquare,
} from "lucide-react";
import { Fragment, useEffect, useMemo, useRef, useState } from "react";

type Parameters = Record<string, number | string | boolean>;

type Scenario = {
  id: string;
  name: string;
  mode: string;
  mode_label: string;
  resource_label: string;
  base_resource: number;
};

type MetricMap = Record<string, number | string>;

type MoralMetric = MetricMap & { label: string };

type TimePoint = {
  tick: number;
  avg_food_allocated: number;
  avg_wellbeing: number;
  social_gini_index: number;
  minimum_survival_rate: number;
  total_food: number;
  by_moral_type: Record<string, MoralMetric>;
};

type ScenarioResult = {
  scenario: Scenario;
  model_status: ModelStatus;
  society_metrics: MetricMap;
  moral_type_metrics: Record<string, MoralMetric>;
  group_metrics: MetricMap;
  time_series: TimePoint[];
  system_logs: LogEntry[];
  agent_logs: LogEntry[];
  group_records: Array<Record<string, number | string>>;
  trade_metrics?: MetricMap;
  role_metrics?: Record<string, MetricMap>;
  hierarchy?: {
    chief_id?: number;
    chieftain_ids?: number[];
    hunter_assignments?: Record<string, number[]>;
  };
  promotion_events?: Array<Record<string, number | string>>;
};

type BatchResult = {
  parameters: Parameters;
  scenarios: Scenario[];
  results: ScenarioResult[];
  comparison: Array<Record<string, number | string>>;
  model_status: ModelStatus;
};

type ModelOption = {
  id: string;
  label: string;
  model: string;
  base_url: string;
  description: string;
};

type ModelStatus = {
  framework: string;
  provider: string;
  model: string;
  base_url: string;
  active: boolean;
  using_fallback: boolean;
  reason: string;
};

type LogEntry = {
  tick: number;
  scenario_id: string;
  sequence?: number;
  received_at?: string;
  level?: string;
  event?: string;
  agent_id?: number;
  moral_label?: string;
  action?: string;
  result?: string;
  decision_source?: string;
  decision_reason?: string;
  allocated_food?: number;
  effective_need?: number;
  fatigue_after?: number;
  status_after?: string;
  wellbeing_after?: number;
  message: string;
};

type Language = "zh" | "en";

type StatusKey = "ready" | "apiOffline" | "running" | "complete" | "error" | "reset" | "stopped";
type ModelTestStatus = "idle" | "testing" | "ok" | "fail";

type ModelTestState = {
  provider: string;
  status: ModelTestStatus;
  message: string;
  latency_ms?: number;
};

type StreamEvent = {
  type: string;
  parameters?: Parameters;
  scenarios?: Scenario[];
  model_status?: ModelStatus;
  scenario?: Scenario;
  scenario_id?: string;
  target?: "system_logs" | "agent_logs";
  log?: LogEntry;
  result?: BatchResult | ScenarioResult;
  error?: string;
  logged?: boolean;
};

const defaultParams: Parameters = {
  population_size: 120,
  ticks: 56,
  replications: 1,
  seed: 42,
  food_scale: 1.6,
  alpha: 1.15,
  alpha_individual: 1.35,
  lambda_c: 0.2,
  lambda_g: 0.25,
  environment_volatility: 0.1,
  organized_equal_weight: 0.8,
  individual_transfer_strength: 0.18,
  relationship_memory_weight: 0.15,
  moral_consistency_weight: 0.2,
  autonomy_weight: 0.1,
  enable_participation_choice: true,
  moral_adaptation_rate: 0.01,
  trade_enabled: true,
  rank_enabled: true,
  rank_split_enabled: true,
  promotion_enabled: true,
  initial_shells: 8,
  shell_base_price: 1,
  trade_safety_buffer: 1,
  chief_share_weight: 4,
  chieftain_share_weight: 2,
  hunter_share_weight: 1,
  chief_leadership_share: 0.1,
  chieftain_slots: 7,
  hunters_per_chieftain: 16,
  promotion_period: 28,
  promotion_shell_weight: 0.8,
  promotion_hunting_weight: 0.2,
  llm_provider: "rule",
  llm_sample_rate: 0,
};

const fallbackScenarios: Scenario[] = [
  { id: "S1", name: "匮乏强组织总体行动", mode: "organized", mode_label: "全体高度有组织行动", resource_label: "高度匮乏", base_resource: 0.35 },
  { id: "S2", name: "匮乏弱组织抱团行动", mode: "group", mode_label: "自由抱团行动", resource_label: "高度匮乏", base_resource: 0.35 },
  { id: "S3", name: "匮乏弱组织各自行动", mode: "individual", mode_label: "各自行动", resource_label: "高度匮乏", base_resource: 0.35 },
  { id: "S4", name: "丰裕强组织总体行动", mode: "organized", mode_label: "全体高度有组织行动", resource_label: "基础丰裕", base_resource: 0.75 },
  { id: "S5", name: "丰裕弱组织抱团行动", mode: "group", mode_label: "自由抱团行动", resource_label: "基础丰裕", base_resource: 0.75 },
  { id: "S6", name: "丰裕弱组织各自行动", mode: "individual", mode_label: "各自行动", resource_label: "基础丰裕", base_resource: 0.75 },
];

const moralOrder = [
  "self_development",
  "self_restraint",
  "altruistic_development",
  "altruistic_restraint",
];

const moralColors: Record<string, string> = {
  self_development: "#2f6f73",
  self_restraint: "#bd8b2f",
  altruistic_development: "#7b5aa6",
  altruistic_restraint: "#c95f4a",
};

const apiBase = import.meta.env.VITE_API_BASE ?? "";

const copy = {
  zh: {
    title: "部落狩猎模拟仿真",
    params: "参数",
	    model: "模型",
	    run: "运行",
	    reset: "重置",
	    stop: "停止",
    downloadJson: "导出实验数据",
    exportButton: "导出",
	    switchLanguage: "切换语言",
	    analysisButton: "实验结果分析",
	    analysisTitle: "实验结果分析",
	    exportResult: "导出结果",
	    back: "返回",
	    ticks: "轮数",
    population: "人口",
    foodScale: "食物尺度",
    lambdaC: "强组织增益",
    lambdaG: "抱团增益",
    envVolatility: "环境波动",
    moralWeight: "道德一致权重",
    autonomyWeight: "自主权重",
    relationMemory: "关系记忆权重",
    moralAdaptation: "道德适应率",
    economy: "交易与身份制度",
    tradeEnabled: "开放自由交易",
    rankEnabled: "启用三级身份",
    rankSplitEnabled: "身份差额分成",
    promotionEnabled: "周期身份评选",
    initialShells: "初始贝壳/人",
    shellPrice: "基础食物价格",
    tradeSafety: "交易储备倍数",
    chiefWeight: "首领分成权重",
    chieftainWeight: "酋长分成权重",
    hunterWeight: "猎人分成权重",
    chiefLeadershipShare: "首领领导份额",
    chieftainSlots: "酋长名额",
    huntersPerChieftain: "酋长辖下猎人",
    promotionPeriod: "身份评选周期",
    shellPromotionWeight: "贝壳评选权重",
    huntingPromotionWeight: "狩猎评选权重",
    economySummary: "交易与身份结果",
    shellBalance: "平均贝壳",
    shellGini: "贝壳基尼",
    tradeVolume: "交易食物量",
    participationRate: "交易参与率",
    promotions: "身份变动",
    chief: "首领",
    chieftain: "酋长",
    hunter: "猎人",
    populationCount: "人数",
    roleFood: "平均分得肉量",
    roleWellbeing: "平均幸福",
    roleShells: "平均贝壳",
    economyComparison: "财富与交易对比",
    roleComparison: "三级身份分层",
    roleMobility: "身份流动与评选",
    marketPrice: "市场价格",
    currentChief: "当前首领",
    quota: "辖额",
    eventTick: "轮次",
    eventAgent: "Agent",
    eventChange: "身份变化",
    eventShells: "贝壳",
    noPromotion: "本次实验暂无身份变动",
    tradeProfile: "交易结构画像",
    shellEquality: "贝壳均衡",
    selfSufficiency: "食物自给",
    tradeRounds: "交易轮次",
    huntingContribution: "狩猎贡献",
    bestWellbeing: "最高幸福",
    bestFood: "最高肉量",
    fairest: "最低不平等",
    minimumSurvival: "最低保障",
    state: "状态",
    waiting: "等待仿真结果",
    systemLog: "系统运行日志",
    notRun: "未运行",
    entries: "条",
    comparison: "六情境对比",
    moralFood: "道德类型收益",
    moralWellbeing: "道德类型幸福",
    wellbeingTrend: "幸福感动态",
    fitHeatmap: "适配热力",
    groupNetwork: "抱团结构",
    wellbeing: "幸福",
    food: "肉量",
    avgWellbeing: "平均幸福",
    survival: "最低保障",
    meat: "肉",
    fatigue: "疲劳",
    logWellbeing: "幸福",
    decision: "驱动",
    testRoute: "检测模型通路",
    modelTestIdle: "未测",
    modelTestTesting: "检测中",
    modelTestOk: "通畅",
    modelTestFail: "失败",
    ready: "Ready",
    apiOffline: "API offline",
    running: "Running",
	    complete: "Complete",
	    error: "Run failed",
	    resetDone: "已清空",
	    stopped: "已停止",
	  },
  en: {
    title: "Micro Hunting Tribe Simulation",
    params: "Parameters",
	    model: "Model",
	    run: "Run",
	    reset: "Reset",
	    stop: "Stop",
    downloadJson: "Export experiment data",
    exportButton: "Export",
	    switchLanguage: "Switch language",
	    analysisButton: "Analysis",
	    analysisTitle: "Experiment Analysis",
	    exportResult: "Export Image",
	    back: "Back",
	    ticks: "Ticks",
    population: "Population",
    foodScale: "Food scale",
    lambdaC: "Organized gain",
    lambdaG: "Group gain",
    envVolatility: "Environment volatility",
    moralWeight: "Moral consistency",
    autonomyWeight: "Autonomy weight",
    relationMemory: "Relationship memory",
    moralAdaptation: "Moral adaptation",
    economy: "Trade & hierarchy",
    tradeEnabled: "Enable free trade",
    rankEnabled: "Enable three-tier roles",
    rankSplitEnabled: "Role-based shares",
    promotionEnabled: "Periodic promotion",
    initialShells: "Initial shells / agent",
    shellPrice: "Base food price",
    tradeSafety: "Trade reserve buffer",
    chiefWeight: "Chief share weight",
    chieftainWeight: "Chieftain share weight",
    hunterWeight: "Hunter share weight",
    chiefLeadershipShare: "Chief leadership share",
    chieftainSlots: "Chieftain slots",
    huntersPerChieftain: "Hunters per chieftain",
    promotionPeriod: "Promotion period",
    shellPromotionWeight: "Shell ranking weight",
    huntingPromotionWeight: "Hunting ranking weight",
    economySummary: "Trade & hierarchy result",
    shellBalance: "Avg shells",
    shellGini: "Shell Gini",
    tradeVolume: "Food traded",
    participationRate: "Trade participation",
    promotions: "Role changes",
    chief: "Chief",
    chieftain: "Chieftain",
    hunter: "Hunter",
    populationCount: "Population",
    roleFood: "Avg allocated food",
    roleWellbeing: "Avg wellbeing",
    roleShells: "Avg shells",
    economyComparison: "Wealth & trade comparison",
    roleComparison: "Three-tier role breakdown",
    roleMobility: "Role mobility & promotion",
    marketPrice: "Market price",
    currentChief: "Current chief",
    quota: "Quota",
    eventTick: "Tick",
    eventAgent: "Agent",
    eventChange: "Role change",
    eventShells: "Shells",
    noPromotion: "No role changes in this experiment",
    tradeProfile: "Trade profile",
    shellEquality: "Shell equality",
    selfSufficiency: "Food self-sufficiency",
    tradeRounds: "Trade rounds",
    huntingContribution: "Hunting contribution",
    bestWellbeing: "Best wellbeing",
    bestFood: "Most food",
    fairest: "Lowest inequality",
    minimumSurvival: "Survival floor",
    state: "Status",
    waiting: "Waiting for result",
    systemLog: "System Run Log",
    notRun: "Not run",
    entries: "entries",
    comparison: "Scenario Comparison",
    moralFood: "Moral-Type Food",
    moralWellbeing: "Moral-Type Wellbeing",
    wellbeingTrend: "Wellbeing Trend",
    fitHeatmap: "Fit Heatmap",
    groupNetwork: "Group Structure",
    wellbeing: "Wellbeing",
    food: "Food",
    avgWellbeing: "Avg wellbeing",
    survival: "Survival floor",
    meat: "Food",
    fatigue: "Fatigue",
    logWellbeing: "Wellbeing",
    decision: "Driver",
    testRoute: "Test model route",
    modelTestIdle: "Untested",
    modelTestTesting: "Testing",
    modelTestOk: "OK",
    modelTestFail: "Failed",
    ready: "Ready",
    apiOffline: "API offline",
    running: "Running",
	    complete: "Complete",
	    error: "Run failed",
	    resetDone: "Reset",
	    stopped: "Stopped",
	  },
} satisfies Record<Language, Record<string, string>>;

const scenarioNameEn: Record<string, string> = {
  S1: "Scarce · Organized",
  S2: "Scarce · Groups",
  S3: "Scarce · Individual",
  S4: "Abundant · Organized",
  S5: "Abundant · Groups",
  S6: "Abundant · Individual",
};

const moralLabelEn: Record<string, string> = {
  self_development: "Self development",
  self_restraint: "Self restraint",
  altruistic_development: "Altruistic development",
  altruistic_restraint: "Altruistic restraint",
  自我发展: "Self development",
  自我克制: "Self restraint",
  发展型利他: "Altruistic development",
  克制型利他: "Altruistic restraint",
};

const resultLabelEn: Record<string, string> = {
  触发极度疲劳救济: "Triggered extreme-fatigue relief",
  狩猎受挫并完成结算: "Hunt failed, settlement completed",
  完成狩猎并完成结算: "Hunt completed, settlement completed",
};

const statusLabelEn: Record<string, string> = {
  Normal: "Normal",
  Extreme_Fatigue: "Extreme fatigue",
};

function numberValue(value: number | string | boolean | undefined): number {
  if (typeof value === "number") return value;
  if (typeof value === "string") return Number(value) || 0;
  if (typeof value === "boolean") return value ? 1 : 0;
  return 0;
}

function isAbortError(error: unknown) {
  return error instanceof DOMException && error.name === "AbortError";
}

function App() {
  const [params, setParams] = useState<Parameters>(defaultParams);
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([
    { id: "rule", label: "规则默认", model: "deterministic-wellbeing-engine", base_url: "", description: "本地规则评分" },
    { id: "openai", label: "OpenAI", model: "gpt-4o-mini", base_url: "", description: "Camel-AI + OpenAI" },
    { id: "deepseek", label: "DeepSeek-V4-flash", model: "deepseek-v4-flash", base_url: "https://api.deepseek.com/v1", description: "OpenAI-Compatible direct-json 行为驱动" },
    { id: "local-deepseek", label: "本地DeepSeek", model: "deepseek-v4-flash", base_url: "https://uni-api.cstcloud.cn/v1", description: "中国科技云 Uni-API direct-json；优先国内入口并自动备用 Global" },
  ]);
  const [scenarios, setScenarios] = useState<Scenario[]>(fallbackScenarios);
  const [selectedScenarioIds, setSelectedScenarioIds] = useState<string[]>(fallbackScenarios.map((s) => s.id));
  const [activeScenarioId, setActiveScenarioId] = useState("S1");
  const [result, setResult] = useState<BatchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [statusKey, setStatusKey] = useState<StatusKey>("ready");
  const [errorMessage, setErrorMessage] = useState("");
  const [language, setLanguage] = useState<Language>(() => {
    if (typeof window === "undefined") return "zh";
    return window.localStorage.getItem("morality_happiness_language") === "en" ? "en" : "zh";
  });
  const [view, setView] = useState<"home" | "analysis">("home");
  const analysisRef = useRef<HTMLDivElement>(null);
  const controlPanelRef = useRef<HTMLElement>(null);
  const runAbortRef = useRef<AbortController | null>(null);
  const abortReasonRef = useRef<"stop" | "reset" | null>(null);
  const [controlPanelHeight, setControlPanelHeight] = useState<number | null>(null);
  const t = copy[language];
  const selectedProviderId = String(params.llm_provider ?? "rule");
  const [modelTest, setModelTest] = useState<ModelTestState>({
    provider: selectedProviderId,
    status: "idle",
    message: "",
  });

  useEffect(() => {
    Promise.all([
      fetch(`${apiBase}/api/defaults`).then((res) => res.json()),
      fetch(`${apiBase}/api/scenarios`).then((res) => res.json()),
    ])
      .then(([defaults, scenarioPayload]) => {
        setParams({ ...defaultParams, ...defaults.parameters });
        if (defaults.model_options?.length) setModelOptions(defaults.model_options);
        setScenarios(scenarioPayload.scenarios);
        setSelectedScenarioIds(scenarioPayload.scenarios.map((scenario: Scenario) => scenario.id));
      })
      .catch(() => {
        setStatusKey("apiOffline");
      });
  }, []);

  useEffect(() => {
    setModelTest({ provider: selectedProviderId, status: "idle", message: "" });
  }, [selectedProviderId]);

  useEffect(() => {
    const node = controlPanelRef.current;
    if (!node) return;
    const panel = node;

    function syncControlPanelHeight() {
      const next = Math.ceil(panel.getBoundingClientRect().height);
      setControlPanelHeight((current) => (current === next ? current : next));
    }

    syncControlPanelHeight();
    window.addEventListener("resize", syncControlPanelHeight);
    if (typeof ResizeObserver === "undefined") {
      return () => window.removeEventListener("resize", syncControlPanelHeight);
    }

    const observer = new ResizeObserver(syncControlPanelHeight);
    observer.observe(panel);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", syncControlPanelHeight);
    };
  }, []);

  const activeResult = useMemo(() => {
    return result?.results.find((item) => item.scenario.id === activeScenarioId) ?? result?.results[0] ?? null;
  }, [activeScenarioId, result]);

  const kpis = useMemo(() => {
    if (!result?.comparison?.length) return [];
    const bestWellbeing = [...result.comparison].sort((a, b) => numberValue(b.avg_wellbeing) - numberValue(a.avg_wellbeing))[0];
    const bestFood = [...result.comparison].sort((a, b) => numberValue(b.avg_food_allocated) - numberValue(a.avg_food_allocated))[0];
    const fairest = [...result.comparison].sort((a, b) => numberValue(a.social_gini_index) - numberValue(b.social_gini_index))[0];
    return [
      [t.bestWellbeing, `${bestWellbeing.scenario_id} ${numberValue(bestWellbeing.avg_wellbeing).toFixed(1)}`],
      [t.bestFood, `${bestFood.scenario_id} ${numberValue(bestFood.avg_food_allocated).toFixed(2)}`],
      [t.fairest, `${fairest.scenario_id} ${numberValue(fairest.social_gini_index).toFixed(3)}`],
      [t.minimumSurvival, `${numberValue(result.comparison[0]?.minimum_survival_rate).toFixed(2)}-${Math.max(...result.comparison.map((row) => numberValue(row.minimum_survival_rate))).toFixed(2)}`],
      [t.shellBalance, numberValue(result.comparison[0]?.avg_shell_balance).toFixed(2)],
      [t.tradeVolume, numberValue(result.comparison[0]?.total_trade_volume_food).toFixed(1)],
    ];
  }, [result, t]);

  function updateParam(key: string, value: number) {
    setParams((current) => ({ ...current, [key]: value }));
  }

  function updateParamValue(key: string, value: number | string | boolean) {
    setParams((current) => ({ ...current, [key]: value }));
  }

  function toggleScenario(id: string) {
    setSelectedScenarioIds((current) => {
      const next = current.includes(id) ? current.filter((item) => item !== id) : [...current, id];
      return next.length ? next : current;
    });
  }

  async function testModelRoute() {
    const providerId = selectedProviderId;
    setModelTest({ provider: providerId, status: "testing", message: "" });
    try {
      const response = await fetch(`${apiBase}/api/model-test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: providerId, parameters: params }),
      });
      const payload = await response.json();
      const message = String(payload.message || payload.error || "");
      if (!response.ok || !payload.ok) {
        setModelTest({
          provider: providerId,
          status: "fail",
          message: message || "model route test failed",
          latency_ms: Number(payload.latency_ms) || undefined,
        });
        return;
      }
      setModelTest({
        provider: providerId,
        status: "ok",
        message: `${message || "model route is healthy"}${payload.latency_ms ? ` (${payload.latency_ms}ms)` : ""}`,
        latency_ms: Number(payload.latency_ms) || undefined,
      });
    } catch (error) {
      setModelTest({
        provider: providerId,
        status: "fail",
        message: error instanceof Error ? error.message : "model route test failed",
      });
    }
  }

  async function run() {
    runAbortRef.current?.abort();
    const controller = new AbortController();
    runAbortRef.current = controller;
    abortReasonRef.current = null;
    setLoading(true);
    setErrorMessage("");
    setStatusKey("running");
    setView("home");
    try {
      const response = await fetch(`${apiBase}/api/run-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({ scenario_ids: selectedScenarioIds, parameters: params }),
      });
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.error || "run failed");
      }
      if (!response.body) throw new Error("Streaming is not supported by this browser.");
      await readRunStream(response.body);
      setStatusKey("complete");
    } catch (error) {
      if (abortReasonRef.current || isAbortError(error)) {
        setErrorMessage("");
        setStatusKey(abortReasonRef.current === "reset" ? "reset" : "stopped");
      } else {
        setErrorMessage(error instanceof Error ? error.message : "");
        setStatusKey("error");
      }
    } finally {
      if (runAbortRef.current === controller) {
        runAbortRef.current = null;
      }
      abortReasonRef.current = null;
      setLoading(false);
    }
  }

  function stopSimulation() {
    abortReasonRef.current = "stop";
    runAbortRef.current?.abort();
    runAbortRef.current = null;
    void fetch(`${apiBase}/api/stop`, { method: "POST" }).catch(() => undefined);
    setErrorMessage("");
    setStatusKey("stopped");
    setLoading(false);
    setView("home");
  }

  async function readRunStream(body: ReadableStream<Uint8Array>) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (value) {
        buffer += decoder.decode(value, { stream: !done });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line) as StreamEvent;
          if (event.type === "error") {
            applyStreamEvent(event);
            throw new Error(event.error || "run failed");
          }
          applyStreamEvent(event);
        }
      }
      if (done) break;
    }
    const tail = buffer.trim();
    if (tail) {
      const event = JSON.parse(tail) as StreamEvent;
      if (event.type === "error") {
        applyStreamEvent(event);
        throw new Error(event.error || "run failed");
      }
      applyStreamEvent(event);
    }
  }

  function applyStreamEvent(event: StreamEvent) {
    if (event.type === "batch_start") {
      setResult({
        parameters: event.parameters ?? params,
        scenarios: event.scenarios ?? scenarios,
        results: [],
        comparison: [],
        model_status: event.model_status ?? emptyModelStatus(String(params.llm_provider ?? "rule")),
      });
      return;
    }
    if (event.type === "scenario_start" && event.scenario) {
      setActiveScenarioId(event.scenario.id);
      setResult((current) => upsertScenario(current, event.scenario!, emptyScenarioResult(event.scenario!, current?.model_status ?? emptyModelStatus(String(params.llm_provider ?? "rule")))));
      return;
    }
    if (event.type === "log" && event.log && event.target) {
      setResult((current) => appendScenarioLog(current, event.log!, event.target!, scenarios, params));
      return;
    }
    if (event.type === "error" && !event.logged) {
      const scenarioId = event.scenario_id ?? activeScenarioId ?? selectedScenarioIds[0] ?? "S1";
      setResult((current) => appendScenarioLog(
        current,
        {
          tick: 0,
          scenario_id: scenarioId,
          level: "error",
          event: "simulation_error",
          message: `运行错误: ${event.error || "run failed"}`,
        },
        "system_logs",
        scenarios,
        params,
      ));
      return;
    }
    if (event.type === "scenario_result" && event.result && "scenario" in event.result) {
      const scenarioResult = event.result as ScenarioResult;
      setResult((current) => upsertScenario(current, scenarioResult.scenario, scenarioResult));
      return;
    }
    if (event.type === "complete" && event.result && "comparison" in event.result) {
      const batchResult = event.result as BatchResult;
      setResult(batchResult);
      setActiveScenarioId(batchResult.results[0]?.scenario.id ?? "S1");
    }
  }

  function resetSimulation() {
    runAbortRef.current?.abort();
    runAbortRef.current = null;
    abortReasonRef.current = "reset";
    void fetch(`${apiBase}/api/stop`, { method: "POST" }).catch(() => undefined);
    setResult(null);
    setActiveScenarioId(selectedScenarioIds[0] ?? scenarios[0]?.id ?? "S1");
    setErrorMessage("");
    setStatusKey("reset");
    setView("home");
  }

  function toggleLanguage() {
    setLanguage((current) => {
      const next = current === "zh" ? "en" : "zh";
      if (typeof window !== "undefined") {
        window.localStorage.setItem("morality_happiness_language", next);
      }
      return next;
    });
  }

  function downloadJson() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "morality_happiness_experiment_data.json";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function exportAnalysisImage() {
    if (!analysisRef.current) return;
    const node = analysisRef.current;
    const width = Math.max(1, Math.ceil(node.scrollWidth));
    const height = Math.max(1, Math.ceil(node.scrollHeight));
    const clone = node.cloneNode(true) as HTMLElement;
    clone.setAttribute("xmlns", "http://www.w3.org/1999/xhtml");
    clone.style.width = `${width}px`;
    clone.style.minHeight = `${height}px`;
    clone.style.background = "#f4f6f2";
    const serialized = new XMLSerializer().serializeToString(clone);
    const css = collectCssText();
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><foreignObject width="100%" height="100%"><div xmlns="http://www.w3.org/1999/xhtml"><style>${css}</style>${serialized}</div></foreignObject></svg>`;
    const svgBlob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
    const svgUrl = URL.createObjectURL(svgBlob);
    try {
      const image = await loadImage(svgUrl);
      const scale = Math.min(2, window.devicePixelRatio || 1);
      const canvas = document.createElement("canvas");
      canvas.width = Math.ceil(width * scale);
      canvas.height = Math.ceil(height * scale);
      const context = canvas.getContext("2d");
      if (!context) throw new Error("canvas unavailable");
      context.fillStyle = "#f4f6f2";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      downloadUrl(canvas.toDataURL("image/png"), "morality_happiness_analysis.png");
    } catch {
      downloadUrl(svgUrl, "morality_happiness_analysis.svg");
    } finally {
      URL.revokeObjectURL(svgUrl);
    }
  }

  const statusLabel = errorMessage || (statusKey === "reset" ? t.resetDone : t[statusKey]);
  const canOpenAnalysis = statusKey === "complete" && !loading && Boolean(result?.comparison?.length);
  const currentModelTestStatus = modelTest.provider === selectedProviderId ? modelTest.status : "idle";
  const modelTestTitle = modelTest.provider === selectedProviderId && modelTest.message ? modelTest.message : t.testRoute;

  if (view === "analysis" && result) {
    return (
      <main className="app-shell">
        <header className="topbar">
          <div>
            <p className="eyebrow">Morality Happiness Lab</p>
            <h1>{t.analysisTitle}</h1>
          </div>
          <div className="run-cluster">
            <button className="icon-button ghost language-button" onClick={toggleLanguage} title={t.switchLanguage}>
              <Languages size={18} />
              <span>{language === "zh" ? "EN" : "中"}</span>
            </button>
            <button className="icon-button ghost text-button" onClick={exportAnalysisImage} title={t.exportResult}>
              <Download size={18} />
              <span>{t.exportResult}</span>
            </button>
            <button className="icon-button" onClick={() => setView("home")} title={t.back}>
              <ArrowLeft size={18} />
              <span>{t.back}</span>
            </button>
          </div>
        </header>

        <section className="analysis-page" ref={analysisRef}>
          <div className="chart-grid analysis-grid">
            <section className="surface large">
              <div className="section-heading">
                <Activity size={18} />
                <h2>{t.comparison}</h2>
              </div>
              <ComparisonBars result={result} language={language} />
            </section>

            <section className="surface analysis-wide">
              <div className="section-heading">
                <h2>{t.economyComparison}</h2>
              </div>
              <EconomyComparison result={result} activeResult={activeResult} language={language} />
            </section>

            <section className="surface">
              <div className="section-heading">
                <h2>{t.roleComparison}</h2>
              </div>
              <RoleComparison result={activeResult} language={language} />
            </section>

            <section className="surface">
              <div className="section-heading">
                <h2>{t.roleMobility}</h2>
              </div>
              <PromotionEvents result={activeResult} language={language} />
            </section>

            <section className="surface">
              <div className="section-heading">
                <h2>{t.moralFood}</h2>
              </div>
              <MoralBars result={result} metric="avg_food_allocated" />
            </section>

            <section className="surface">
              <div className="section-heading">
                <h2>{t.moralWellbeing}</h2>
              </div>
              <MoralBars result={result} metric="avg_wellbeing" />
            </section>

            <section className="surface wide">
              <div className="section-heading with-tabs">
                <h2>{t.wellbeingTrend}</h2>
                <div className="tabs">
                  {(result?.results ?? []).map((item) => (
                    <button
                      key={item.scenario.id}
                      className={activeScenarioId === item.scenario.id ? "tab active" : "tab"}
                      onClick={() => setActiveScenarioId(item.scenario.id)}
                    >
                      {item.scenario.id}
                    </button>
                  ))}
                </div>
              </div>
              <TimeSeries result={activeResult} language={language} />
            </section>

            <section className="surface">
              <div className="section-heading">
                <h2>{t.fitHeatmap}</h2>
              </div>
              <MoralHeatmap result={result} language={language} />
            </section>

            <section className="surface network-surface">
              <div className="section-heading">
                <Network size={18} />
                <h2>{t.groupNetwork}</h2>
              </div>
              <GroupNetwork result={activeResult} />
            </section>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Morality Happiness Lab</p>
          <h1>{t.title}</h1>
        </div>
        <div className="run-cluster">
          <span className="status-pill">{statusLabel}</span>
          <button className="icon-button ghost reset-button" onClick={resetSimulation} title={t.reset}>
            <RotateCcw size={18} />
            <span>{t.reset}</span>
          </button>
          <button className="icon-button ghost stop-button" onClick={stopSimulation} disabled={!loading} title={t.stop}>
            <CircleStop size={18} />
            <span>{t.stop}</span>
          </button>
          <button className="icon-button ghost language-button" onClick={toggleLanguage} title={t.switchLanguage}>
            <Languages size={18} />
            <span>{language === "zh" ? "EN" : "中"}</span>
          </button>
          <button className="icon-button" onClick={run} disabled={loading} title={t.run}>
            {loading ? <RefreshCw className="spin" size={18} /> : <Play size={18} />}
            <span>{t.run}</span>
          </button>
          <button className="icon-button ghost text-button export-data-button" onClick={downloadJson} disabled={!result} title={t.downloadJson}>
            <Download size={18} />
            <span>{t.exportButton}</span>
          </button>
        </div>
      </header>

      <section className="workspace">
        <aside className="control-panel" ref={controlPanelRef}>
          <div className="panel-title">
            <SlidersHorizontal size={18} />
            <span>{t.params}</span>
          </div>
          <div className="select-control">
            <div className="model-control-header">
              <span>{t.model}</span>
              <button
                type="button"
                className={`model-test-button ${currentModelTestStatus}`}
                onClick={testModelRoute}
                disabled={currentModelTestStatus === "testing"}
                title={modelTestTitle}
              >
                <span className="model-test-light" />
                <span>{modelTestLabel(currentModelTestStatus, t)}</span>
              </button>
            </div>
            <select
              value={selectedProviderId}
              onChange={(event) => updateParamValue("llm_provider", event.target.value)}
            >
              {modelOptions.map((option) => (
                <option key={option.id} value={option.id}>
                  {modelOptionTitle(option)}
                </option>
              ))}
            </select>
          </div>
          <div className="model-note">
            {result?.model_status
              ? `${result.model_status.framework}/${result.model_status.provider}: ${result.model_status.reason}`
              : selectedModelDescription(modelOptions, String(params.llm_provider ?? "rule"))}
          </div>
          <ParamControl label={t.ticks} value={numberValue(params.ticks)} min={10} max={300} step={10} onChange={(value) => updateParam("ticks", value)} />
          <ParamControl label={t.population} value={numberValue(params.population_size)} min={24} max={240} step={12} onChange={(value) => updateParam("population_size", value)} />
          <ParamControl label={t.foodScale} value={numberValue(params.food_scale)} min={0.8} max={3.2} step={0.05} onChange={(value) => updateParam("food_scale", value)} />
          <ParamControl label={t.lambdaC} value={numberValue(params.lambda_c)} min={0} max={0.8} step={0.02} onChange={(value) => updateParam("lambda_c", value)} />
          <ParamControl label={t.lambdaG} value={numberValue(params.lambda_g)} min={0} max={0.8} step={0.02} onChange={(value) => updateParam("lambda_g", value)} />
          <ParamControl label={t.envVolatility} value={numberValue(params.environment_volatility)} min={0} max={0.5} step={0.01} onChange={(value) => updateParam("environment_volatility", value)} />
          <ParamControl label={t.moralWeight} value={numberValue(params.moral_consistency_weight)} min={0.05} max={0.45} step={0.01} onChange={(value) => updateParam("moral_consistency_weight", value)} />
          <ParamControl label={t.autonomyWeight} value={numberValue(params.autonomy_weight)} min={0.05} max={0.35} step={0.01} onChange={(value) => updateParam("autonomy_weight", value)} />
          <ParamControl label={t.relationMemory} value={numberValue(params.relationship_memory_weight)} min={0} max={1} step={0.01} onChange={(value) => updateParam("relationship_memory_weight", value)} />
          <ParamControl label={t.moralAdaptation} value={numberValue(params.moral_adaptation_rate)} min={0} max={0.2} step={0.005} onChange={(value) => updateParam("moral_adaptation_rate", value)} />
          <div className="scenario-section-title">{t.economy}</div>
          <ToggleControl label={t.tradeEnabled} value={Boolean(params.trade_enabled)} onChange={(value) => updateParamValue("trade_enabled", value)} />
          <ToggleControl label={t.rankEnabled} value={Boolean(params.rank_enabled)} onChange={(value) => updateParamValue("rank_enabled", value)} />
          <ToggleControl label={t.rankSplitEnabled} value={Boolean(params.rank_split_enabled)} onChange={(value) => updateParamValue("rank_split_enabled", value)} />
          <ToggleControl label={t.promotionEnabled} value={Boolean(params.promotion_enabled)} onChange={(value) => updateParamValue("promotion_enabled", value)} />
          <ParamControl label={t.initialShells} value={numberValue(params.initial_shells)} min={0} max={100} step={1} onChange={(value) => updateParam("initial_shells", value)} />
          <ParamControl label={t.shellPrice} value={numberValue(params.shell_base_price)} min={0.1} max={10} step={0.1} onChange={(value) => updateParam("shell_base_price", value)} />
          <ParamControl label={t.tradeSafety} value={numberValue(params.trade_safety_buffer)} min={0.5} max={4} step={0.1} onChange={(value) => updateParam("trade_safety_buffer", value)} />
          <ParamControl label={t.chiefWeight} value={numberValue(params.chief_share_weight)} min={0} max={10} step={0.5} onChange={(value) => updateParam("chief_share_weight", value)} />
          <ParamControl label={t.chieftainWeight} value={numberValue(params.chieftain_share_weight)} min={0} max={10} step={0.5} onChange={(value) => updateParam("chieftain_share_weight", value)} />
          <ParamControl label={t.hunterWeight} value={numberValue(params.hunter_share_weight)} min={0} max={10} step={0.5} onChange={(value) => updateParam("hunter_share_weight", value)} />
          <ParamControl label={t.chiefLeadershipShare} value={numberValue(params.chief_leadership_share)} min={0} max={0.5} step={0.01} onChange={(value) => updateParam("chief_leadership_share", value)} />
          <ParamControl label={t.chieftainSlots} value={numberValue(params.chieftain_slots)} min={1} max={20} step={1} onChange={(value) => updateParam("chieftain_slots", value)} />
          <ParamControl label={t.huntersPerChieftain} value={numberValue(params.hunters_per_chieftain)} min={1} max={40} step={1} onChange={(value) => updateParam("hunters_per_chieftain", value)} />
          <ParamControl label={t.promotionPeriod} value={numberValue(params.promotion_period)} min={1} max={200} step={1} onChange={(value) => updateParam("promotion_period", value)} />
          <ParamControl label={t.shellPromotionWeight} value={numberValue(params.promotion_shell_weight)} min={0} max={1} step={0.05} onChange={(value) => updateParam("promotion_shell_weight", value)} />
          <ParamControl label={t.huntingPromotionWeight} value={numberValue(params.promotion_hunting_weight)} min={0} max={1} step={0.05} onChange={(value) => updateParam("promotion_hunting_weight", value)} />
          <div className="scenario-section-title">实验情景</div>
          <div className="scenario-grid">
            {scenarios.map((scenario) => (
              <button
                key={scenario.id}
                className={selectedScenarioIds.includes(scenario.id) ? "scenario-chip active" : "scenario-chip"}
                onClick={() => toggleScenario(scenario.id)}
              >
                <strong>{scenario.id}</strong>
                <span>{language === "en" ? scenarioNameEn[scenario.id] ?? scenario.name : scenario.name}</span>
              </button>
            ))}
          </div>
        </aside>

        <section className="dashboard" style={controlPanelHeight ? { height: controlPanelHeight } : undefined}>
          <div className="kpi-row">
            {kpis.length ? kpis.map(([label, value]) => (
              <div className="kpi" key={label}>
                <span>{label}</span>
                <KpiValue value={value} />
              </div>
            )) : (
              <div className="kpi wide">
                <span>{t.state}</span>
                <strong>{t.waiting}</strong>
              </div>
            )}
          </div>

          <EconomySummary result={activeResult} language={language} />

          <section className="surface log-surface">
            <div className="section-heading with-tabs">
              <div className="heading-left">
                <TerminalSquare size={18} />
                <h2>{t.systemLog}</h2>
              </div>
              <div className="log-meta">
                {activeResult ? `${activeResult.scenario.id} · ${(activeResult.system_logs?.length ?? 0) + (activeResult.agent_logs?.length ?? 0)} ${t.entries}` : t.notRun}
              </div>
            </div>
            <SystemLog result={activeResult} language={language} />
          </section>
        </section>

        <div className="analysis-cta-row">
          <button
            className="icon-button analysis-button"
            onClick={() => setView("analysis")}
            disabled={!canOpenAnalysis}
            title={t.analysisButton}
          >
            <Activity size={18} />
            <span>{t.analysisButton}</span>
          </button>
        </div>
      </section>
    </main>
  );
}

function selectedModelDescription(options: ModelOption[], selectedId: string) {
  const selected = options.find((option) => option.id === selectedId) ?? options[0];
  return `${selected.label}: ${selected.description}`;
}

function KpiValue({ value }: { value: string }) {
  const match = value.match(/^(S\d+)\s+(.+)$/);
  if (!match) return <strong className="kpi-value">{value}</strong>;
  return (
    <strong className="kpi-value">
      <span className="kpi-scenario">{match[1]}</span>
      <span className="kpi-number">{match[2]}</span>
    </strong>
  );
}

function ToggleControl({ label, value, onChange }: {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className="toggle-control">
      <span>{label}</span>
      <input type="checkbox" checked={value} onChange={(event) => onChange(event.target.checked)} />
      <span className="toggle-track" aria-hidden="true"><span /></span>
    </label>
  );
}

function EconomySummary({ result, language }: { result: ScenarioResult | null; language: Language }) {
  const t = copy[language];
  const trade = result?.trade_metrics ?? {};
  const roles = result?.role_metrics ?? {};
  const roleOrder = ["chief", "chieftain", "hunter"];
  return (
    <section className="surface economy-surface">
      <div className="section-heading">
        <h2>{t.economySummary}</h2>
        {result?.hierarchy?.chief_id !== undefined && (
          <span className="log-meta">{t.chief} A{result.hierarchy.chief_id}</span>
        )}
      </div>
      <div className="economy-kpis">
        <div><span>{t.shellBalance}</span><strong>{numberValue(trade.avg_shell_balance).toFixed(2)}</strong></div>
        <div><span>{t.shellGini}</span><strong>{numberValue(trade.shell_gini_index).toFixed(3)}</strong></div>
        <div><span>{t.tradeVolume}</span><strong>{numberValue(trade.total_trade_volume_food).toFixed(1)}</strong></div>
        <div><span>{t.participationRate}</span><strong>{(numberValue(trade.trade_participation_rate) * 100).toFixed(0)}%</strong></div>
        <div><span>{t.promotions}</span><strong>{result?.promotion_events?.length ?? 0}</strong></div>
      </div>
      <div className="role-grid">
        {roleOrder.map((role) => {
          const metric = roles[role] ?? {};
          const roleLabel = role === "chief" ? t.chief : role === "chieftain" ? t.chieftain : t.hunter;
          return (
            <div className={`role-card role-${role}`} key={role}>
              <div className="role-card-title"><strong>{roleLabel}</strong><span>{numberValue(metric.population).toFixed(0)} {t.populationCount}</span></div>
              <span>{t.roleShells} {numberValue(metric.avg_shell_balance).toFixed(2)}</span>
              <span>{t.roleFood} {numberValue(metric.avg_food_allocated).toFixed(2)}</span>
              <span>{t.roleWellbeing} {numberValue(metric.avg_wellbeing).toFixed(1)}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function modelTestLabel(status: ModelTestStatus, t: Record<string, string>) {
  if (status === "testing") return t.modelTestTesting;
  if (status === "ok") return t.modelTestOk;
  if (status === "fail") return t.modelTestFail;
  return t.modelTestIdle;
}

function emptyModelStatus(provider: string): ModelStatus {
  return {
    framework: "",
    provider,
    model: "",
    base_url: "",
    active: true,
    using_fallback: false,
    reason: "",
  };
}

function emptyScenarioResult(scenario: Scenario, modelStatus: ModelStatus): ScenarioResult {
  return {
    scenario,
    model_status: modelStatus,
    society_metrics: {},
    moral_type_metrics: {},
    group_metrics: {},
    time_series: [],
    system_logs: [],
    agent_logs: [],
    group_records: [],
  };
}

function upsertScenario(current: BatchResult | null, scenario: Scenario, scenarioResult: ScenarioResult): BatchResult {
  const base: BatchResult = current ?? {
    parameters: {},
    scenarios: [scenario],
    results: [],
    comparison: [],
    model_status: scenarioResult.model_status,
  };
  const nextScenarios = base.scenarios.some((item) => item.id === scenario.id)
    ? base.scenarios
    : [...base.scenarios, scenario];
  const index = base.results.findIndex((item) => item.scenario.id === scenario.id);
  const nextResults = index >= 0
    ? base.results.map((item, itemIndex) => (itemIndex === index ? scenarioResult : item))
    : [...base.results, scenarioResult];
  return { ...base, scenarios: nextScenarios, results: nextResults };
}

function appendScenarioLog(
  current: BatchResult | null,
  log: LogEntry,
  target: "system_logs" | "agent_logs",
  fallbackScenarios: Scenario[],
  params: Parameters,
): BatchResult | null {
  const scenario = current?.scenarios.find((item) => item.id === log.scenario_id)
    ?? fallbackScenarios.find((item) => item.id === log.scenario_id);
  if (!scenario) return current;
  const modelStatus = current?.model_status ?? emptyModelStatus(String(params.llm_provider ?? "rule"));
  const existing = current?.results.find((item) => item.scenario.id === log.scenario_id)
    ?? emptyScenarioResult(scenario, modelStatus);
  const nextScenarioResult = {
    ...existing,
    [target]: [...existing[target], { ...log, received_at: log.received_at ?? new Date().toISOString() }],
  };
  return upsertScenario(current, scenario, nextScenarioResult);
}

function collectCssText() {
  const chunks: string[] = [];
  for (const sheet of Array.from(document.styleSheets)) {
    try {
      chunks.push(Array.from(sheet.cssRules).map((rule) => rule.cssText).join("\n"));
    } catch {
      // Cross-origin stylesheets are ignored; this app uses local CSS for export.
    }
  }
  return chunks.join("\n");
}

function loadImage(url: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("image export failed"));
    image.src = url;
  });
}

function downloadUrl(url: string, filename: string) {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
}

function modelOptionTitle(option: ModelOption) {
  if (option.id === "deepseek" || option.id === "local-deepseek") return option.label;
  return `${option.label} · ${option.model}`;
}

function moralLabel(label: string | undefined, fallback: string, language: Language) {
  const raw = label || fallback;
  return language === "en" ? moralLabelEn[fallback] ?? moralLabelEn[raw] ?? raw : raw;
}

function actionLabel(action: string | undefined, language: Language) {
  if (!action || language === "zh") return action;
  if (action === "休息恢复/等待救济") return "Rest / wait for relief";
  if (action === "服从强组织分工并参与集体狩猎") return "Follow organized division and join collective hunt";
  if (action === "独自探索并个人狩猎") return "Explore alone and hunt individually";
  const groupMatch = action.match(/^加入群体 (.+) 进行抱团狩猎$/);
  if (groupMatch) {
    const groupId = groupMatch[1] === "未分组" ? "ungrouped" : groupMatch[1];
    return `Join group ${groupId} for group hunt`;
  }
  return action;
}

function resultLabel(result: string | undefined, language: Language) {
  if (!result || language === "zh") return result;
  return resultLabelEn[result] ?? result;
}

function agentStatusLabel(status: string | undefined, language: Language) {
  if (!status || language === "zh") return status;
  return statusLabelEn[status] ?? status.replace(/_/g, " ");
}

function ParamControl({ label, value, min, max, step, onChange }: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  const precision = Math.max(decimalPlaces(step), decimalPlaces(min));
  const [draft, setDraft] = useState(formatParamValue(value, precision));

  useEffect(() => {
    setDraft(formatParamValue(value, precision));
  }, [precision, value]);

  function commit(raw: string) {
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) {
      setDraft(formatParamValue(value, precision));
      return;
    }
    const bounded = Math.min(max, Math.max(min, parsed));
    const next = precision === 0 ? Math.round(bounded) : Number(bounded.toFixed(precision));
    onChange(next);
    setDraft(formatParamValue(next, precision));
  }

  return (
    <label className="param-control">
      <span>{label}</span>
      <input
        type="number"
        value={draft}
        min={min}
        max={max}
        step={step}
        inputMode={precision === 0 ? "numeric" : "decimal"}
        onChange={(event) => {
          const next = event.target.value;
          setDraft(next);
          const parsed = Number(next);
          if (next !== "" && Number.isFinite(parsed) && parsed >= min && parsed <= max) {
            onChange(precision === 0 ? Math.round(parsed) : parsed);
          }
        }}
        onBlur={(event) => commit(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.currentTarget.blur();
          }
        }}
      />
    </label>
  );
}

function decimalPlaces(value: number) {
  const text = String(value);
  if (!text.includes(".")) return 0;
  return text.split(".")[1]?.length ?? 0;
}

function formatParamValue(value: number, precision: number) {
  return precision === 0 ? String(Math.round(value)) : value.toFixed(precision);
}

function EconomyComparison({ result, activeResult, language }: { result: BatchResult | null; activeResult: ScenarioResult | null; language: Language }) {
  const rows = result?.comparison ?? [];
  const t = copy[language];
  if (!rows.length) return <EmptyChart />;
  const maxTrade = Math.max(1, ...rows.map((row) => numberValue(row.total_trade_volume_food)));
  return (
    <div className="economy-analysis-layout">
      <div className="analysis-table-wrap">
        <table className="analysis-table">
          <thead>
            <tr>
              <th>{language === "zh" ? "情境" : "Scenario"}</th>
              <th>{t.shellBalance}</th>
              <th>{t.shellGini}</th>
              <th>{t.tradeVolume}</th>
              <th>{t.participationRate}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const tradeVolume = numberValue(row.total_trade_volume_food);
              return (
                <tr key={String(row.scenario_id)}>
                  <th>{String(row.scenario_id)}</th>
                  <td>{numberValue(row.avg_shell_balance).toFixed(2)}</td>
                  <td>{numberValue(row.shell_gini_index).toFixed(3)}</td>
                  <td>
                    <div className="metric-bar-cell">
                      <span>{tradeVolume.toFixed(1)}</span>
                      <i style={{ width: `${Math.max(2, (tradeVolume / maxTrade) * 100)}%` }} />
                    </div>
                  </td>
                  <td>{(numberValue(row.trade_participation_rate) * 100).toFixed(0)}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <TradeProfileRadar result={result} activeResult={activeResult} language={language} />
    </div>
  );
}

function TradeProfileRadar({ result, activeResult, language }: { result: BatchResult | null; activeResult: ScenarioResult | null; language: Language }) {
  const t = copy[language];
  const trade = activeResult?.trade_metrics ?? {};
  if (!activeResult) return <div className="radar-empty">{t.waiting}</div>;
  const rows = result?.comparison ?? [];
  const maxTrade = Math.max(1, ...rows.map((row) => numberValue(row.total_trade_volume_food)));
  const maxRounds = Math.max(1, ...rows.map((row) => numberValue(row.trade_rounds)));
  const maxPrice = Math.max(1, ...rows.map((row) => numberValue(row.avg_market_clearing_price)));
  return (
    <div className="radar-panel">
      <div className="radar-panel-heading">
        <strong>{t.tradeProfile}</strong>
        <span>{activeResult.scenario.id}</span>
      </div>
      <RadarChart
        axes={[t.participationRate, t.selfSufficiency, t.tradeVolume, t.tradeRounds, t.marketPrice]}
        series={[{
          label: activeResult.scenario.id,
          color: "var(--teal)",
          values: [
            numberValue(trade.trade_participation_rate) * 100,
            numberValue(trade.food_self_sufficiency_rate) * 100,
            (numberValue(trade.total_trade_volume_food) / maxTrade) * 100,
            (numberValue(trade.trade_rounds) / maxRounds) * 100,
            (numberValue(trade.avg_market_clearing_price) / maxPrice) * 100,
          ],
        }]}
      />
    </div>
  );
}

function RoleComparison({ result, language }: { result: ScenarioResult | null; language: Language }) {
  const t = copy[language];
  const roles = result?.role_metrics ?? {};
  const order = ["chief", "chieftain", "hunter"];
  if (!result || !Object.keys(roles).length) return <EmptyChart />;
  return (
    <div className="role-analysis-layout">
      <div className="radar-panel">
        <div className="radar-panel-heading">
          <strong>{t.roleComparison}</strong>
          <span>{result.scenario.id}</span>
        </div>
        <RadarChart
          axes={[t.roleShells, t.roleFood, t.roleWellbeing, t.huntingContribution]}
          series={order.map((role) => {
            const metric = roles[role] ?? {};
            return {
              label: role === "chief" ? t.chief : role === "chieftain" ? t.chieftain : t.hunter,
              color: role === "chief" ? "var(--gold)" : role === "chieftain" ? "var(--violet)" : "var(--teal)",
              values: [
                numberValue(metric.avg_shell_balance),
                numberValue(metric.avg_food_allocated),
                numberValue(metric.avg_wellbeing),
                numberValue(metric.avg_hunting_contribution),
              ],
            };
          })}
        />
      </div>
      <div className="role-analysis-list">
        <div className="role-analysis-meta">
          <span>{t.currentChief}: A{result.hierarchy?.chief_id ?? "-"}</span>
          <span>{t.quota}: {result.hierarchy?.chieftain_ids?.length ?? 0} / {result.hierarchy?.hunter_assignments ? Object.values(result.hierarchy.hunter_assignments).reduce((total, ids) => total + ids.length, 0) : 0}</span>
        </div>
        {order.map((role) => {
          const metric = roles[role] ?? {};
          const label = role === "chief" ? t.chief : role === "chieftain" ? t.chieftain : t.hunter;
          return (
            <div className={`role-analysis-row role-${role}`} key={role}>
              <div className="role-analysis-title"><strong>{label}</strong><span>{numberValue(metric.population).toFixed(0)} {t.populationCount}</span></div>
              <div><span>{t.roleShells}</span><strong>{numberValue(metric.avg_shell_balance).toFixed(2)}</strong></div>
              <div><span>{t.roleFood}</span><strong>{numberValue(metric.avg_food_allocated).toFixed(2)}</strong></div>
              <div><span>{t.roleWellbeing}</span><strong>{numberValue(metric.avg_wellbeing).toFixed(1)}</strong></div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

type RadarSeries = { label: string; color: string; values: number[] };

function RadarChart({ axes, series }: { axes: string[]; series: RadarSeries[] }) {
  const size = 310;
  const center = size / 2;
  const radius = 98;
  const angleStep = (Math.PI * 2) / axes.length;
  const pointsAt = (ratio: number, index: number) => {
    const angle = -Math.PI / 2 + index * angleStep;
    return { x: center + Math.cos(angle) * radius * ratio, y: center + Math.sin(angle) * radius * ratio };
  };
  const axisPoints = axes.map((_, index) => pointsAt(1, index));
  const maxima = axes.map((_, index) => Math.max(1, ...series.map((entry) => numberValue(entry.values[index]))));
  const polygonPoints = (values: number[]) => values.map((value, index) => {
    const point = pointsAt(Math.min(1, numberValue(value) / maxima[index]), index);
    return `${point.x},${point.y}`;
  }).join(" ");
  return (
    <div className="radar-chart-shell">
      <svg viewBox={`0 0 ${size} ${size}`} className="radar-chart" role="img" aria-label={axes.join(", ")}>
        {[0.25, 0.5, 0.75, 1].map((ratio) => (
          <polygon key={ratio} points={axisPoints.map((_, index) => {
            const point = pointsAt(ratio, index);
            return `${point.x},${point.y}`;
          }).join(" ")} className="radar-grid" />
        ))}
        {axisPoints.map((point, index) => (
          <line key={axes[index]} x1={center} y1={center} x2={point.x} y2={point.y} className="radar-axis" />
        ))}
        {series.map((entry) => (
          <g key={entry.label}>
            <polygon points={polygonPoints(entry.values)} fill={entry.color} fillOpacity="0.16" stroke={entry.color} strokeWidth="2" className="radar-series" />
            {entry.values.map((value, index) => {
              const point = pointsAt(Math.min(1, numberValue(value) / maxima[index]), index);
              return <circle key={`${entry.label}-${index}`} cx={point.x} cy={point.y} r="3" fill={entry.color} />;
            })}
          </g>
        ))}
        {axes.map((label, index) => {
          const point = pointsAt(1.2, index);
          const anchor = point.x < center - 5 ? "end" : point.x > center + 5 ? "start" : "middle";
          return <text key={label} x={point.x} y={point.y} textAnchor={anchor} dominantBaseline="middle" className="radar-label">{label}</text>;
        })}
      </svg>
      <div className="radar-legend">
        {series.map((entry) => <span key={entry.label}><i style={{ background: entry.color }} />{entry.label}</span>)}
      </div>
    </div>
  );
}

function PromotionEvents({ result, language }: { result: ScenarioResult | null; language: Language }) {
  const t = copy[language];
  const events = (result?.promotion_events ?? []).slice(-10).reverse();
  if (!result || !events.length) {
    return <div className="no-events">{t.noPromotion}</div>;
  }
  return (
    <div className="promotion-list">
      {events.map((event, index) => (
        <div className="promotion-row" key={`${String(event.tick)}-${String(event.agent_id)}-${index}`}>
          <span className="promotion-tick">T{String(event.tick)}</span>
          <strong>A{String(event.agent_id)}</strong>
          <span>{String(event.old_role_label ?? event.old_role)} → {String(event.new_role_label ?? event.new_role)}</span>
          <span className="promotion-shells">{t.eventShells} {numberValue(event.shell_balance).toFixed(1)}</span>
        </div>
      ))}
    </div>
  );
}

function ComparisonBars({ result, language }: { result: BatchResult | null; language: Language }) {
  const rows = result?.comparison ?? [];
  if (!rows.length) return <EmptyChart />;
  const t = copy[language];
  const width = 760;
  const height = 260;
  const maxWellbeing = 100;
  const maxFood = Math.max(1, ...rows.map((row) => numberValue(row.avg_food_allocated)));
  const band = width / rows.length;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="chart">
      <line x1="34" y1="212" x2={width - 20} y2="212" className="axis" />
      {rows.map((row, index) => {
        const x = 42 + index * band;
        const wellbeingH = (numberValue(row.avg_wellbeing) / maxWellbeing) * 170;
        const foodH = (numberValue(row.avg_food_allocated) / maxFood) * 170;
        return (
          <g key={String(row.scenario_id)}>
            <rect x={x} y={212 - wellbeingH} width="24" height={wellbeingH} rx="3" className="bar wellbeing" />
            <rect x={x + 30} y={212 - foodH} width="24" height={foodH} rx="3" className="bar food" />
            <text x={x + 27} y="236" textAnchor="middle" className="tick-label">{row.scenario_id}</text>
          </g>
        );
      })}
      <text x="44" y="24" className="legend wellbeing-text">{t.wellbeing}</text>
      <text x="104" y="24" className="legend food-text">{t.food}</text>
    </svg>
  );
}

function MoralBars({ result, metric }: { result: BatchResult | null; metric: string }) {
  const rows = result?.results ?? [];
  if (!rows.length) return <EmptyChart />;
  const width = 520;
  const height = 260;
  const values = rows.flatMap((scenario) => moralOrder.map((moral) => numberValue(scenario.moral_type_metrics[moral]?.[metric])));
  const maxValue = Math.max(metric === "avg_wellbeing" ? 100 : 1, ...values);
  const scenarioBand = width / rows.length;
  const barWidth = Math.max(6, Math.min(14, scenarioBand / 6));
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="chart">
      <line x1="24" y1="210" x2={width - 10} y2="210" className="axis" />
      {rows.map((scenario, sIndex) => (
        <g key={scenario.scenario.id}>
          {moralOrder.map((moral, mIndex) => {
            const value = numberValue(scenario.moral_type_metrics[moral]?.[metric]);
            const h = (value / maxValue) * 164;
            const x = 34 + sIndex * scenarioBand + mIndex * (barWidth + 3);
            return <rect key={moral} x={x} y={210 - h} width={barWidth} height={h} rx="3" fill={moralColors[moral]} />;
          })}
          <text x={42 + sIndex * scenarioBand} y="234" className="tick-label">{scenario.scenario.id}</text>
        </g>
      ))}
    </svg>
  );
}

function TimeSeries({ result, language }: { result: ScenarioResult | null; language: Language }) {
  if (!result?.time_series?.length) return <EmptyChart />;
  const t = copy[language];
  const width = 780;
  const height = 260;
  const points = result.time_series;
  const maxTick = Math.max(1, points[points.length - 1].tick);
  const toX = (tick: number) => 36 + (tick / maxTick) * (width - 66);
  const toY = (value: number) => 210 - (value / 100) * 170;
  const wellbeingPath = points.map((point, index) => `${index ? "L" : "M"} ${toX(point.tick)} ${toY(point.avg_wellbeing)}`).join(" ");
  const survivalPath = points.map((point, index) => `${index ? "L" : "M"} ${toX(point.tick)} ${toY(point.minimum_survival_rate * 100)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="chart">
      <line x1="34" y1="210" x2={width - 20} y2="210" className="axis" />
      <line x1="34" y1="40" x2="34" y2="210" className="axis" />
      <path d={wellbeingPath} className="line wellbeing-line" />
      <path d={survivalPath} className="line survival-line" />
      <text x="48" y="28" className="legend wellbeing-text">{t.avgWellbeing}</text>
      <text x="142" y="28" className="legend survival-text">{t.survival}</text>
    </svg>
  );
}

function MoralHeatmap({ result, language }: { result: BatchResult | null; language: Language }) {
  const rows = result?.results ?? [];
  if (!rows.length) return <EmptyChart />;
  return (
    <div className="heatmap">
      <div />
      {moralOrder.map((moral) => <span key={moral}>{moralLabel(rows[0].moral_type_metrics[moral]?.label, moral, language)}</span>)}
      {rows.map((scenario) => (
        <Fragment key={scenario.scenario.id}>
          <strong key={`${scenario.scenario.id}-label`}>{scenario.scenario.id}</strong>
          {moralOrder.map((moral) => {
            const latest = scenario.time_series[scenario.time_series.length - 1];
            const value = numberValue(latest?.by_moral_type[moral]?.moral_fit);
            return (
              <div
                key={`${scenario.scenario.id}-${moral}`}
                className="heat-cell"
                style={{ background: heatColor(value) }}
              >
                {value.toFixed(2)}
              </div>
            );
          })}
        </Fragment>
      ))}
    </div>
  );
}

function heatColor(value: number) {
  const v = Math.max(0, Math.min(0.7, value)) / 0.7;
  const r = Math.round(232 - 74 * v);
  const g = Math.round(238 - 82 * v);
  const b = Math.round(232 - 74 * v);
  return `rgb(${r}, ${g}, ${b})`;
}

function GroupNetwork({ result }: { result: ScenarioResult | null }) {
  const records = result?.group_records ?? [];
  const width = 520;
  const height = 260;
  if (!records.length) return <EmptyChart />;
  const lastTick = Math.max(...records.map((record) => numberValue(record.tick)));
  const last = records.filter((record) => numberValue(record.tick) === lastTick).slice(-24);
  const maxSize = Math.max(1, ...last.map((record) => numberValue(record.size)));
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="chart network-chart">
      {last.map((record, index) => {
        const angle = (Math.PI * 2 * index) / Math.max(1, last.length);
        const radius = 78;
        const cx = width / 2 + Math.cos(angle) * radius;
        const cy = height / 2 + Math.sin(angle) * radius;
        const size = 12 + (numberValue(record.size) / maxSize) * 24;
        const stability = numberValue(record.group_stability);
        return (
          <g key={`${record.group_id}-${index}`}>
            <line x1={width / 2} y1={height / 2} x2={cx} y2={cy} className="network-link" />
            <circle cx={cx} cy={cy} r={size} fill={moralColors[String(record.dominant_moral_type)] ?? "#83908b"} opacity={0.82} />
            <circle cx={cx} cy={cy} r={size * stability} fill="none" className="stability-ring" />
            <text x={cx} y={cy + 4} textAnchor="middle" className="node-label">{record.group_id}</text>
          </g>
        );
      })}
      <circle cx={width / 2} cy={height / 2} r="14" className="network-core" />
    </svg>
  );
}

function SystemLog({ result, language }: { result: ScenarioResult | null; language: Language }) {
  const t = copy[language];
  const listRef = useRef<HTMLDivElement>(null);
  const logs = result ? [...(result.system_logs ?? []), ...(result.agent_logs ?? [])].sort((a, b) => {
    if (a.tick !== b.tick) return a.tick - b.tick;
    if ((a.sequence ?? 0) !== (b.sequence ?? 0)) return (a.sequence ?? 0) - (b.sequence ?? 0);
    return (a.agent_id ?? -1) - (b.agent_id ?? -1);
  }) : [];
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [logs.length]);
  if (!result) return <EmptyChart />;
  if (!logs.length) return <EmptyChart />;
  return (
    <div className="log-list" ref={listRef}>
      {logs.map((log, index) => (
        <div className={logRowClass(log)} key={`${log.tick}-${log.agent_id ?? "system"}-${index}`}>
          <div className="log-stamp">
            <strong>T{log.tick}</strong>
            <span>{log.agent_id === undefined ? "SYS" : `A${log.agent_id}`}</span>
          </div>
          <time className="log-time" dateTime={log.received_at}>
            {formatLogTime(log.received_at)}
          </time>
          <div className="log-body">
            <div className="log-line">{log.message}</div>
            {shouldRenderAgentFields(log) && (
              <div className="log-fields">
                <span>{moralLabel(log.moral_label, log.moral_label ?? "", language)}</span>
                <span>{actionLabel(log.action, language)}</span>
                <span>{t.meat} {formatLogNumber(log.allocated_food)}/{formatLogNumber(log.effective_need)}</span>
                <span>{t.fatigue} {formatLogNumber(log.fatigue_after)}</span>
                <span>{t.logWellbeing} {formatLogNumber(log.wellbeing_after)}</span>
                {log.decision_source && log.decision_source !== "rule" && (
                  <span>{t.decision} {log.decision_source}{log.decision_reason ? ` · ${log.decision_reason}` : ""}</span>
                )}
                <span>{agentStatusLabel(log.status_after, language)}</span>
                <span>{resultLabel(log.result, language)}</span>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function shouldRenderAgentFields(log: LogEntry) {
  return Boolean(
    log.moral_label
    || log.action
    || log.result
    || log.decision_source
    || log.allocated_food !== undefined
    || log.effective_need !== undefined
    || log.fatigue_after !== undefined
    || log.status_after !== undefined
    || log.wellbeing_after !== undefined
  );
}

function logRowClass(log: LogEntry) {
  const classes = ["log-row"];
  if (log.agent_id === undefined) classes.push("system");
  if (log.level === "error") classes.push("error");
  return classes.join(" ");
}

function formatLogNumber(value: number | undefined) {
  if (value === undefined) return "-";
  return value.toFixed(2);
}

function formatLogTime(value: string | undefined) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "--";
  const pad = (part: number) => String(part).padStart(2, "0");
  return [
    date.getFullYear(),
    pad(date.getMonth() + 1),
    pad(date.getDate()),
  ].join("-") + " " + [
    pad(date.getHours()),
    pad(date.getMinutes()),
    pad(date.getSeconds()),
  ].join(":");
}

function EmptyChart() {
  return (
    <div className="empty-chart">
      <div className="empty-mark" />
    </div>
  );
}

export default App;
