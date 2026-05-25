#!/usr/bin/env bash
# Orchestrate simulation + analysis sweeps for the MAGPIE exploration.
#
# For each --rounds value, runs the simulator on the given scenario and then
# the analyzer on the resulting transcript. Outputs are tagged with the
# round budget so leakage-vs-deadline plots can pull them all together.
# Multiple round values (e.g. "5 10 15") run in parallel; a single value runs
# sequentially.
#
# Usage examples:
#   explore/run.sh --scenario finaldata/academic_1.json
#   explore/run.sh --scenario finaldata/academic_1.json --rounds "5 10 15"
#   explore/run.sh --scenario finaldata/academic_1.json --skip-sim
#   explore/run.sh --scenario finaldata/academic_1.json --llm gemini --judge-llm gemini

set -euo pipefail

SCENARIO=""
LLM="gemini"
JUDGE_LLM="gemini"
ROUNDS="5 10 15"
SKIP_SIM=false
SKIP_ANALYSIS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario) SCENARIO="$2"; shift 2 ;;
        --llm) LLM="$2"; shift 2 ;;
        --judge-llm) JUDGE_LLM="$2"; shift 2 ;;
        --rounds) ROUNDS="$2"; shift 2 ;;
        --skip-sim) SKIP_SIM=true; shift ;;
        --skip-analysis) SKIP_ANALYSIS=true; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$SCENARIO" ]]; then
    echo "Error: --scenario is required" >&2
    exit 1
fi

# Resolve repo paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAGPIE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$MAGPIE_DIR/../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python3"
if [[ ! -x "$VENV_PY" ]]; then
    VENV_PY="python3"
fi

cd "$MAGPIE_DIR"

SCENARIO_BASENAME="$(basename "$SCENARIO" .json)"

read -ra ROUND_VALUES <<< "$ROUNDS"

run_round() {
    local R="$1"
    local SIM_OUT="explore_simulations/${LLM}/${SCENARIO_BASENAME}_r${R}.json"
    local ANALYSIS_OUT="explore_analyses/${JUDGE_LLM}/${SCENARIO_BASENAME}_r${R}.json"

    if [[ "$SKIP_SIM" == false ]]; then
        echo ""
        echo "=== Simulation: ${SCENARIO_BASENAME} @ ${R} rounds (llm=${LLM}) ==="
        "$VENV_PY" -m explore.simulate \
            --scenario_file "$SCENARIO" \
            --llm "$LLM" \
            --max-rounds "$R" \
            --output "$SIM_OUT"
    fi

    if [[ "$SKIP_ANALYSIS" == false ]]; then
        if [[ ! -f "$SIM_OUT" ]]; then
            echo "⚠️  Simulation output missing: $SIM_OUT — skipping analysis for r=${R}" >&2
            return 0
        fi
        echo ""
        echo "=== Analysis: ${SCENARIO_BASENAME} @ ${R} rounds (judge=${JUDGE_LLM}) ==="
        "$VENV_PY" -m explore.analysis \
            --simulation "$SIM_OUT" \
            --output "$ANALYSIS_OUT" \
            --judge-llm "$JUDGE_LLM"
    fi
}

if [[ ${#ROUND_VALUES[@]} -gt 1 ]]; then
    echo "Running ${#ROUND_VALUES[@]} round budgets in parallel: ${ROUNDS}"
    PIDS=()
    for R in "${ROUND_VALUES[@]}"; do
        run_round "$R" &
        PIDS+=("$!")
    done
    FAILED=0
    for PID in "${PIDS[@]}"; do
        if ! wait "$PID"; then
            FAILED=1
        fi
    done
    if [[ "$FAILED" -ne 0 ]]; then
        echo "Error: one or more round jobs failed" >&2
        exit 1
    fi
else
    run_round "${ROUND_VALUES[0]}"
fi

echo ""
echo "✅ exploration sweep complete for ${SCENARIO_BASENAME} (rounds: ${ROUNDS})"
