# prompt-augment-lens

本地对照台：左边是 UI 提问，右边是送给模型的拼装输入，中间是多出来的层。

## 跑起来长什么样

打开 `serve` 后浏览器会并排显示 UI 原文与分层后的 model input（system / rules / env / tools / user_query）。

## 跑起来

```bash
./start.sh setup
./start.sh test projects/prompt-augment-lens
./start.sh run prompt-augment-lens serve --fixture projects/prompt-augment-lens/fixtures/sample_agent_turn.json
# 浏览器打开 http://127.0.0.1:8765

# 可选：把 OpenAI-compatible 客户端指到本地代理，落盘真实请求后再 serve
./start.sh run prompt-augment-lens proxy --port 8787
```
