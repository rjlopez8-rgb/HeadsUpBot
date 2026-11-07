#!/usr/bin/env bash
# lab/stress_time.sh - worst-case timing test with the OFFICIAL judge over HTTP.
#
# Our bot is pinned to ONE core and a busy-loop "hog" shares that same core,
# so the bot gets roughly half a CPU (harsher than the tournament's 1 core).
# Usage: bash lab/stress_time.sh [deals=1000] [opponent=baseline-bots/baseline_random.py] [hogs=1]
set -u
DEALS=${1:-1000}
OPP=${2:-baseline-bots/baseline_random.py}
HOGS=${3:-1}
cd "$(dirname "$0")/.."
pkill -f "^python3 agent.py" 2>/dev/null; pkill -f "^python3 baseline-bots" 2>/dev/null; sleep 1
T0=$(date +%s.%N)
PORT=5008 taskset -c 0 python3 agent.py > /tmp/stress_agent.log 2>&1 &
for _ in $(seq 1 "$HOGS"); do taskset -c 0 python3 -c "while True: pass" & done
PORT=5009 taskset -c 1 python3 "$OPP" > /tmp/stress_opp.log 2>&1 &
until curl -s localhost:5008/ > /dev/null; do sleep 0.2; done
echo "agent startup: $(echo "$(date +%s.%N) - $T0" | bc) s (limit 15 s)"
python3 judge_engine.py --deals "$DEALS" --seed 123 2>&1 | grep -E "bb/deal|thinking|VERDICT|RESULT|FORFEIT|EMPTY|Error|error"
pkill -f "^python3 -c while True"; pkill -f "^python3 agent.py"; pkill -f "^python3 baseline-bots"
