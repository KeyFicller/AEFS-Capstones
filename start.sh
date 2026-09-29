#!/usr/bin/env bash
# 通用项目启动脚本 —— 环境 / 脚手架 / 测试 / 运行
# 用法: ./start.sh <setup|new|test|run|clean|help> [参数]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TEMPLATE="examples/hello-agent"
VENV=".venv"
REQ="requirements.txt"

log() { printf '\033[1;36m[start]\033[0m %s\n' "$*"; }

usage() {
  cat <<'EOF'
用法: ./start.sh <命令> [参数]

  setup              创建共享 .venv 并安装 requirements.txt
  new <name>         从模板创建 projects/<name> 并注册到 requirements.txt
  test [path]        运行 pytest（默认 examples/ 与 projects/）
  run <cmd...>       在共享环境里执行任意命令
  clean              清理缓存（__pycache__ / .pytest_cache / .ruff_cache）
  help               显示本帮助
EOF
}

ensure_uv() {
  command -v uv >/dev/null || { echo "需要先安装 uv: https://docs.astral.sh/uv/" >&2; exit 1; }
}

cmd_setup() {
  ensure_uv
  [ -d "$VENV" ] || { log "创建共享虚拟环境 $VENV"; uv venv; }
  log "安装依赖: $REQ"
  uv pip install -r "$REQ"
  log "完成。用 'source $VENV/bin/activate' 或 './start.sh run <cmd>' 进入环境"
}

cmd_new() {
  local name="${1:-}"
  [ -n "$name" ] || { echo "用法: ./start.sh new <name>" >&2; exit 1; }
  local dest="projects/$name"
  [ -e "$dest" ] && { echo "$dest 已存在" >&2; exit 1; }
  mkdir -p projects
  local mod="${name//-/_}"

  # 注意: bash 3.2 在 UTF-8 locale 下, $var 紧跟非 ASCII 字符会把首字节吞进变量名, set -u 即报 unbound; 一律写 ${var}
  log "从模板创建 ${dest}（模块名 ${mod}）"
  cp -R "$TEMPLATE" "$dest"
  rm -rf "$dest/.pytest_cache" "$dest/.venv"
  mv "$dest/src/hello_agent" "$dest/src/$mod"
  find "$dest" -type f \( -name '*.py' -o -name '*.toml' -o -name '*.md' \) -print0 \
    | xargs -0 sed -i.bak -e "s/hello_agent/$mod/g" -e "s/hello-agent/$name/g" -e "s#examples/$name#projects/$name#g"
  find "$dest" -name '*.bak' -delete

  grep -qF -- "-e ./$dest" "$REQ" || printf -- "-e ./%s\n" "$dest" >> "$REQ"
  log "已注册到 $REQ"
  log "下一步: ./start.sh setup && ./start.sh test $dest"
}

cmd_test() {
  ensure_uv
  local targets=()
  if [ "$#" -ge 1 ]; then
    targets=("$1")
  else
    local d
    for d in examples projects; do
      [ -d "$d" ] && targets+=("$d")
    done
  fi
  log "pytest ${targets[*]}"
  # --import-mode=importlib: 多个项目下同名 test_*.py 不会互相冲突
  uv run --no-project pytest --import-mode=importlib "${targets[@]}"
}

cmd_run() {
  [ "$#" -ge 1 ] || { echo "用法: ./start.sh run <cmd...>" >&2; exit 1; }
  ensure_uv
  uv run --no-project "$@"
}

cmd_clean() {
  log "清理缓存"
  find "$ROOT" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
  find "$ROOT" -name '.pytest_cache' -type d -prune -exec rm -rf {} + 2>/dev/null || true
  find "$ROOT" -name '.ruff_cache' -type d -prune -exec rm -rf {} + 2>/dev/null || true
}

case "${1:-help}" in
  setup) shift; cmd_setup "$@" ;;
  new)   shift; cmd_new "$@" ;;
  test)  shift; cmd_test "$@" ;;
  run)   shift; cmd_run "$@" ;;
  clean) shift; cmd_clean "$@" ;;
  help|-h|--help) usage ;;
  *) echo "未知命令: $1" >&2; usage; exit 1 ;;
esac
