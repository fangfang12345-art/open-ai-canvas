#!/bin/bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")" && pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

for port in 3102 8080; do
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Port $port is already in use. Stop the existing service before starting again."
        exit 1
    fi
done

mkdir -p "$project_dir/.local/project-workbench-debug" "$project_dir/.local/logs"
export GOCACHE="$project_dir/.local/cache/go-build"
export GOMODCACHE="$project_dir/.local/cache/go-mod"
export CANVAS_BACKEND_ADDR=127.0.0.1:8080
export CANVAS_BACKEND_DATA_DIR="$project_dir/.local/project-workbench-debug"
export CANVAS_YTDLP_PATH="$project_dir/.local/cache/yt-dlp-runtime/bin/yt-dlp"
export CANVAS_DATABASE_DRIVER=sqlite
export CANVAS_REGISTRATION_ENABLED=false
unset DATABASE_URL REDIS_URL YINGCE_AGENT_URL
export CANVAS_CORS_ORIGINS=http://127.0.0.1:3102,http://localhost:3102
export CANVAS_ALLOWED_PRIVATE_UPSTREAM_HOSTS="${CANVAS_ALLOWED_PRIVATE_UPSTREAM_HOSTS:+$CANVAS_ALLOWED_PRIVATE_UPSTREAM_HOSTS,}8e8abb002517ffc1.copyvids.com"
export CANVAS_PI_RUNTIME_DIR="$project_dir/backend/agent-runtime/pi"
export VITE_API_PROXY_TARGET=http://127.0.0.1:8080
export GIN_MODE=release

cd "$project_dir/backend"
echo "Preparing backend..."
go build -o "$project_dir/.local/cache/yingce-server" ./cmd/server

backend_pid=""
frontend_pid=""
cleanup() {
    trap - EXIT
    if [ -n "$frontend_pid" ]; then kill "$frontend_pid" 2>/dev/null || true; fi
    if [ -n "$backend_pid" ]; then kill "$backend_pid" 2>/dev/null || true; fi
    wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$project_dir/.local/cache/yingce-server" >>"$project_dir/.local/logs/backend.log" 2>&1 &
backend_pid=$!
cd "$project_dir/web"
node ./node_modules/vite/bin/vite.js --host 127.0.0.1 --port 3102 --strictPort >>"$project_dir/.local/logs/frontend.log" 2>&1 &
frontend_pid=$!

echo "Starting Yingce at http://127.0.0.1:3102"
echo "Logs: $project_dir/.local/logs"
echo "Keep this terminal open. Press Ctrl+C to stop both services."
while kill -0 "$backend_pid" 2>/dev/null && kill -0 "$frontend_pid" 2>/dev/null; do
    sleep 2
done
echo "A service stopped. Check the logs above."
exit 1
