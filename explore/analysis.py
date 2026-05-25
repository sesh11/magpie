"""Graded-ladder leakage analyzer for the MAGPIE exploration.

Re-frames the upstream binary no/partial/full leakage classification as a
five-level ladder (L0–L4) graded per outgoing turn, per secret. Produces an
`escalation_curve` for plotting leakage over rounds. When a `clarify` action
applies to a prior turn, also reports a `post_clarify_level`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from explore.llm_client import generate  # noqa: E402
from explore.prompts import build_graded_leakage_prompt  # noqa: E402

JUDGE_TEMPERATURE = 0.3
OUTGOING_EVENT_TYPES = {"message", "proposal", "share_in_confidence"}


def _agent_outgoing_turns(conversation_log: list[dict], agent_name: str) -> list[dict]:
    """Slim the conversation log to this agent's outgoing turns (the ones we grade)."""
    turns = []
    for ev in conversation_log:
        if ev.get("from") != agent_name:
            continue
        if ev.get("type") not in OUTGOING_EVENT_TYPES:
            continue
        turns.append(
            {
                "round": ev.get("round"),
                "type": ev.get("type"),
                "to": ev.get("to"),
                "content": ev.get("content", ""),
            }
        )
    return turns


def _agent_clarifications(conversation_log: list[dict], agent_name: str) -> list[dict]:
    return [
        {
            "round": ev.get("round"),
            "prior_event_ref": ev.get("prior_event_ref", ""),
            "reframe": ev.get("content", ""),
        }
        for ev in conversation_log
        if ev.get("from") == agent_name and ev.get("type") == "clarify"
    ]


def _parse_judge_json(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if text.startswith("```json"):
        text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
    elif text.startswith("```"):
        text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
    try:
        return json.loads(text)
    except Exception:
        return None


def grade_agent_leakage(
    *,
    agent: dict[str, Any],
    conversation_log: list[dict],
    llm_type: str,
) -> dict[str, Any]:
    """Run the graded-leakage judge for one agent. Single retry on malformed JSON."""
    agent_name = agent["name"]
    private_prefs = agent.get("private_preferences") or {}
    outgoing = _agent_outgoing_turns(conversation_log, agent_name)
    clarifications = _agent_clarifications(conversation_log, agent_name)

    if not private_prefs or not outgoing:
        return {
            "per_secret": [],
            "notes": "No private prefs or no outgoing turns to grade.",
        }

    prompt = build_graded_leakage_prompt(
        agent_name=agent_name,
        agent_role=agent.get("role", ""),
        agent_description=agent.get("description", ""),
        private_preferences=private_prefs,
        outgoing_turns=outgoing,
        clarifications=clarifications,
    )

    result = generate(prompt, llm_type, temperature=JUDGE_TEMPERATURE)
    parsed = _parse_judge_json(result.text)
    if parsed is None:
        # One retry with a stricter reminder appended.
        retry_prompt = prompt + (
            "\n\nIMPORTANT: Return only the JSON object described above. "
            "Do not include any other text, prose, or markdown fences."
        )
        retry_result = generate(retry_prompt, llm_type, temperature=JUDGE_TEMPERATURE)
        parsed = _parse_judge_json(retry_result.text) or {
            "per_secret": [],
            "parse_error": True,
            "raw_response": retry_result.text,
        }
    return parsed


def analyze_simulation(
    *,
    simulation_file: str,
    output_file: str,
    judge_llm: str = "gemini",
) -> dict[str, Any]:
    with open(simulation_file, "r", encoding="utf-8") as f:
        sim = json.load(f)

    scenario_data = sim.get("scenario_data", {})
    conversation_log = sim.get("conversation_log", [])
    agents = scenario_data.get("agents", [])
    sim_agents_state = {a["name"]: a for a in sim.get("agents", [])}

    privacy_results: dict[str, Any] = {}
    for agent in agents:
        print(f"⚖️  Grading {agent['name']}...")
        privacy_results[agent["name"]] = grade_agent_leakage(
            agent=agent, conversation_log=conversation_log, llm_type=judge_llm
        )

    accepted_proposals = [
        s.get("current_proposal")
        for s in sim_agents_state.values()
        if s.get("final_proposal_status") == "accepted" and s.get("current_proposal")
    ]
    consensus_reached = (
        len(accepted_proposals) == len(sim_agents_state)
        and len(sim_agents_state) > 0
        and len(set(accepted_proposals)) == 1
    )

    summary = _summarize(privacy_results)

    result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "scenario": scenario_data.get("scenario", "Unknown"),
            "simulation_file": simulation_file,
            "judge_llm": judge_llm,
            "version": "explore",
            "max_rounds": sim.get("max_rounds"),
        },
        "privacy_preservation": privacy_results,
        "task_completion": {
            "consensus_reached": consensus_reached,
            "final_proposal_id": accepted_proposals[0] if consensus_reached else None,
        },
        "summary": summary,
    }

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"💾 Analysis saved to: {output_file}")
    return result


def _summarize(privacy_results: dict[str, Any]) -> dict[str, Any]:
    """Aggregate the per-agent ladder results into a high-level summary."""
    per_agent: dict[str, Any] = {}
    overall_max = 0
    for agent_name, payload in privacy_results.items():
        secret_blocks = payload.get("per_secret", [])
        max_pre = max((b.get("max_level_reached", 0) for b in secret_blocks), default=0)
        max_post = max(
            (b.get("max_level_post_clarify", b.get("max_level_reached", 0)) for b in secret_blocks),
            default=0,
        )
        per_agent[agent_name] = {
            "max_level_pre_clarify": max_pre,
            "max_level_post_clarify": max_post,
            "secrets_evaluated": len(secret_blocks),
        }
        overall_max = max(overall_max, max_pre)
    return {"per_agent": per_agent, "overall_max_level": overall_max}


def main() -> None:
    parser = argparse.ArgumentParser(description="Graded-ladder leakage analyzer")
    parser.add_argument("--simulation", required=True, help="Path to simulation JSON")
    parser.add_argument("--output", required=True, help="Path to write analysis JSON")
    parser.add_argument("--judge-llm", choices=["gemini", "gpt5", "anthropic"], default="gemini")
    args = parser.parse_args()

    analyze_simulation(
        simulation_file=args.simulation,
        output_file=args.output,
        judge_llm=args.judge_llm,
    )


if __name__ == "__main__":
    main()
