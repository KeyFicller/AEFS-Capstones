`app.App` 的现象：先读一次配置，再切换模式，读回来的还是旧值。

    from app import App
    app = App({"mode": "prod"})
    app.settings.get("mode")
    app.switch_to("dev")      # 期望 "running in dev mode"，实际 "running in prod mode"

`python3 -m pytest tests -q` 现在是全绿的。仓库里的 `README.md` 写了这套代码的设计意图。

修好它，不要放宽或删除已有测试。
