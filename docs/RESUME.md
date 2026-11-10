# 简历材料 · tokbench

> 所有数字均可从仓库中复现验证。引用时请保留"mock 模式"这一前提（见文末"诚实边界"）。

---

## 一、一句话定位

**tokbench** — 面向 LLM 编码 Agent 的 Token 经济性基准测试平台：把「哪个模型 + 哪套提示词策略最划算」从经验判断变成可复现的量化实验。

---

## 二、简历条目（三种长度，按版面选一个）

### A. 标准版（3–4 行，最推荐）

> **tokbench — LLM Token 经济性基准测试平台**（个人项目，Python / Typer / Streamlit）
> - 设计并实现模型无关的基准测试框架：将 5 模型 × 3 提示策略 × 50 任务 × 3 重复展开为 **2,250 个实验单元**，自动记录每次调用的输入/输出/缓存 token、时延与成败，落地为 append-only JSONL 运行目录（1.99M token / $1.96 / 79.5s）。
> - 用**成本-质量帕累托前沿**替代单一指标选型：在 15 个候选方案中识别出 8 个非支配解，并定位拐点（`deepseek-chat/baseline`，$0.000676/task，quality 0.763）——拐点之后每质量点成本上升 **9.1 倍**，把"加预算换质量"的收益衰减量化到具体数字。
> - 建立**失败重试成本核算**与**缓存敏感度模型**：测得 3.75% 的支出（$0.0734 / 53,381 token）为纯重试浪费且集中在个别组合（最高 7.11%）；缓存命中率每提升 10pp 可省 $0.0148，使缓存从"优化项"变为可预算的成本项。
> - 采用适配器模式解耦模型接入与计费（adapters 只上报 token，定价集中在单一模块），支持 OpenAI / Anthropic / DeepSeek / Ollama / mock 五种后端；199 个测试覆盖，mock 模式两次独立运行**逐字节一致**，CI 全流程离线可跑。

### B. 精简版（2 行，适合空间紧张）

> **tokbench — LLM Token 经济性基准测试平台**（Python, Typer/Streamlit, 个人项目）
> 构建模型×提示策略的自动化基准框架（2,250 单元 / 1.99M token / $1.96），以成本-质量帕累托前沿与失败重试成本核算量化选型：识别 8/15 非支配方案，测得拐点后边际收益衰减 9.1×、重试浪费占总支出 3.75%、缓存每 +10pp 省 $0.0148。适配器模式支持 5 种后端，199 测试，mock 运行逐字节可复现。

### C. 极简版（1 行，适合技术栈列表）

> **tokbench** — LLM Token 成本基准平台：2,250 单元实验、帕累托前沿选型（边际收益衰减 9.1×）、重试成本核算（3.75% 浪费）、缓存敏感度建模；Python/Typer/Streamlit，199 测试，离线确定性可复现。

---

## 三、面试可展开的技术点（每点都对应真实实现）

### 1. 为什么用帕累托前沿，而不是"性价比排名"

单一排名会误导：便宜模型单位成本低但质量差，贵模型质量高但溢价。帕累托前沿直接回答**"给定质量下限，最便宜的达标方案是哪个"**。

实现：严格支配判定（两轴均不差且至少一轴更优）→ 非支配集 → 按成本排序成前沿曲线 → 用「到首尾连线的最远垂直距离」定位拐点。

**可讲的数据**：前沿在靠近原点处极陡（便宜模型花几分钱就能快速提升质量），拐点之后边际收益骤降 **9.1 倍**（每美元质量点从 ~504 降到 ~55）。这意味着超过 `deepseek-chat` 之后的预算基本买不到成比例的质量。

### 2. 为什么把"重试"当成一等公民成本

大多数成本分析只看成功调用，把失败调用当噪声丢掉——这会系统性低估真实账单 10–15%。tokbench 让每次尝试都产生独立的 `CallRecord`，`TaskResult.usage_total` 汇总全部尝试，`retry_overhead_usage` 单独隔离浪费。

**可讲的数据**：3.75% 的支出（$0.0734）没有产生任何额外价值；且高度集中——`gpt-4o-mini/fewshot-compress` 高达 7.11%，而最省的组合仅 1.68%。错误类型分布（rate_limit 46 / timeout 19 / server_error 17）直接指向修复方向：这是并发问题，不是模型问题。

### 3. 为什么适配器只上报 token、不计算成本

这是整个设计的核心约束。成本计算集中在 `pricing.py`，适配器只返回 provider 上报的 token 数。

**收益**：历史运行可以用新的价格表重新定价（`tokbench report --prices my-rates.yaml`），**零次 API 调用**；成本模型可审计（一个文件）；mock 与真实运行共用同一套成本逻辑。

### 4. 缓存敏感度为什么能建模成一条直线

固定总输入/输出 token，只改变命中率 h：

```
cost(h) = ((1−h)·input·P_in + h·input·P_cache_read + output·P_out) / 1e6
```

对 h 是**仿射函数**，所以曲线是直线、斜率固定。这个性质让缓存从"优化玄学"变成预算项：**每 +10pp 命中率 = 固定金额**（本实验 $0.0148）。

命中率分母用 `input − cache_write`：cache write 本质是一次"未命中但写入"，把它算成未命中会让积极预热缓存的系统显得更差。

### 5. 一个数字只能有一个出处（工程纪律）

项目里踩过真实坑：同一份报告曾同时出现 **45.4% / 48.9% / 第三种**三个缓存命中率——因为汇总 KPI、时间序列、敏感度曲线各自重算了一遍。

修复方式不是打补丁，而是建立契约：**所有派生量计算一次后发布到 report payload，渲染层只读不算**。帕累托拐点从"分析和渲染器各算一次"改为 `ParetoFrontier.knee_label` 单一权威字段。这条纪律写进了 `CONTRIBUTING.md`。

### 6. 确定性复现

mock 适配器的所有 token 数由 `sha256(cache_key)` 派生，两次独立运行的聚合值**逐字节一致**（已验证：input 907,007 / output 1,083,783 / cache_read 411,543 / cost $1.9557020575 完全相同）。这让 CI 无需 API key、无需网络，且回归测试不会因随机性抖动。

---

## 四、简历里可以写的量化结果汇总

| 指标 | 数值 | 来源 |
|---|---|---|
| 实验单元 | 2,250（5 模型 × 3 策略 × 50 任务 × 3 重复） | `examples/experiment.yaml` |
| 实际调用次数 | 2,332（含 82 次失败重试） | run manifest |
| 总 token / 计费 token | 1.99M / 1.51M | run manifest |
| 总成本 | $1.96（$0.00129 / 1k 计费 token） | report.json |
| 墙钟耗时 | 79.5s（并发 8） | run manifest |
| 缓存命中率 | 48.9% | report.json |
| 帕累托前沿 | 15 个候选中 8 个非支配 | report.json |
| 拐点后边际收益衰减 | 9.1× | 前沿斜率计算 |
| 重试浪费 | 3.75% 支出 = $0.0734 / 53,381 token | report.json |
| 缓存每 +10pp 价值 | $0.0148 | cache sensitivity |
| 测试数量 | 199（analysis 54 / engine 74 / render 30 / taskset 41） | `pytest --collect-only` |
| 代码规模 | 30 个包文件 / 13,078 行 + 4,119 行测试 | 仓库统计 |

> 以上数字全部由 `python scripts/verify_resume_claims.py` 自动校验（31 项断言，全部可复现）。
> 若后续修改了代码或重跑了实验，先跑这个脚本再更新简历。

---

## 五、诚实边界（面试被追问时主动说，反而加分）

1. **mock 模式的数字验证的是"管线与成本模型"，不是模型本身。** mock 适配器按 tier 模拟能力差异，所以"deepseek-reasoner 质量最高"这类结论是模拟器的设定，不是真实评测结果。真实结论需要 `adapter_mode: live` + API key。**主动说明这一点，比被面试官指出好得多。**
2. **质量评分是启发式**（`must_include` 子串 + 结构规则），是廉价代理指标，不是人工评测。分数可在同一次运行内横向比较，不是绝对值。
3. **3 次重复控制方差，不是显著性检验。** 要断言小的质量差异，需要 5 次以上重复。
4. **价格是 2026 公开列表价快照**，实际有折扣差异，引用金额前应传 `--prices` 覆盖。

---

## 六、GitHub 仓库简介（About 栏直接粘贴）

> A token-economics benchmark harness for LLM coding agents. Runs a 5-model × 3-prompt-strategy × 50-task matrix, prices every token, and reports a cost/quality Pareto frontier, retry-overhead ledger, and cache-sensitivity curve. Deterministic offline mode, no API key required.

**Topics（GitHub 标签，建议添加）**：
`llm` `benchmark` `token-cost` `llm-evaluation` `pareto-frontier` `prompt-engineering` `cost-optimization` `python` `typer` `streamlit` `observability` `llmops`

---

## 七、一句话电梯陈述（口头版）

> "我做了一个工具，回答一个问题：**我的 LLM 预算到底该怎么花**。它不是监控花了多少钱，而是通过跑对照实验，告诉你哪个模型配哪套提示词最划算。我用帕累托前沿找出了拐点——过了那个点，多花的钱换不到成比例的质量提升，边际收益掉 9 倍。另外我量化了重试浪费占 3.75%，缓存每提升 10 个百分点值多少钱。整个东西 199 个测试，离线确定性复现，不需要 API key 就能跑。"
