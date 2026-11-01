# prompt-augment-lens

- **项目**：prompt-augment-lens
- **状态**：demo 切片——fixture 对照台 + record-only OpenAI-compatible proxy；厂商专用 adapter 后置

## 目标与范围

可视化 **UI 提问 vs 模型侧输入** 的差异。首版用 OpenAI-compatible `chat.completions` 形态归一化；fixture 展示 coding-agent 常见拼装（system + rules + env + `<user_query>` + tools）。

## 架构

`capture JSON` → `normalize` → `AugmentedTurn` → 对照台。可选 `proxy` 将 POST `/v1/chat/completions` 记入 `captures/`（不持久化鉴权头）。

## 技术栈

- Python 3.11+，stdlib HTTP（无额外 Web 框架）
- 共享根 `.venv` / `requirements.txt`

## 交付物

- CLI：`serve` / `normalize` / `proxy`
- Fixture：`fixtures/sample_agent_turn.json`
- 单测：normalize + utterance 推断
