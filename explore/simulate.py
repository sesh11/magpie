"""Negotiation simulator for the MAGPIE exploration.

Re-frames sensitive-context rendering, adds two action types
(share_in_confidence, clarify), and exposes the round counter as a
deadline-pressure variable. Compared to the upstream MAGPIE setup:

- Narrative sensitive-context rendering (no DO NOT SHARE label, no penalty table)
- share_in_confidence action for calibrated peer-to-peer disclosure
- clarify action for self-correction / walk-back
- Round counter surfaced to the agent (deadline-pressure variable)
- Per-turn prompt_log captured for inspection / replay

Scenario inputs are reused from `finaldata/` unchanged.
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

from explore.llm_client import GEMINI_API_KEYS, LLMResult, generate  # noqa: E402
from explore.prompts import (  # noqa: E402
    build_decide_action_prompt,
    build_memory_decision_prompt,
)


class Agent:
    """Negotiating agent. Holds its own state but issues stateless LLM calls."""

    def __init__(
        self,
        name: str,
        role: str,
        description: str,
        shareable_preferences: dict,
        private_preferences: dict,
        llm_type: str = "gemini",
        api_key: str | None = None,
    ) -> None:
        self.name = name
        self.role = role
        self.description = description
        self.shareable_preferences = shareable_preferences
        self.private_preferences = private_preferences
        self.llm_type = llm_type
        self.api_key = api_key

        self.memory: list[dict] = []
        self.temp_memory: list[dict] = []
        self.current_proposal: str | None = None
        self.proposal_status: str = "none"

        # exploration-specific state
        self.confidential_disclosures_made: list[dict] = []
        self.confidential_disclosures_received: dict[str, list[dict]] = {}
        self.clarifications_made: list[dict] = []
        self.prompt_log: list[dict] = []

    # ----- memory ----------------------------------------------------------
    def write_to_memory(self, text: str) -> None:
        self.memory.append({"timestamp": datetime.now().isoformat(), "content": text})
        print(f"🧠 {self.name} writes to main memory: {text}")

    def write_to_temp_memory(self, text: str) -> None:
        self.temp_memory.append({"timestamp": datetime.now().isoformat(), "content": text})

    # ----- actions ---------------------------------------------------------
    def send_message(
        self,
        agent_list: list[str],
        message: str,
        conversation_log: list[dict],
        round_index: int,
    ) -> dict:
        message = message if isinstance(message, str) else str(message or "")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "round": round_index,
            "from": self.name,
            "to": agent_list,
            "type": "message",
            "content": message,
        }
        conversation_log.append(entry)
        print(f"📨 {self.name} → {agent_list}: {message}")
        return entry

    def send_proposal(
        self,
        agent_list: list[str],
        proposal: str,
        conversation_log: list[dict],
        round_index: int,
    ) -> dict:
        proposal = proposal if isinstance(proposal, str) else str(proposal or "")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "round": round_index,
            "from": self.name,
            "to": agent_list,
            "type": "proposal",
            "content": proposal,
            "proposal_id": f"proposal_{len(conversation_log)}_{self.name}",
        }
        conversation_log.append(entry)
        self.current_proposal = entry["proposal_id"]
        self.proposal_status = "pending"
        print(f"📋 {self.name} → {agent_list}: PROPOSAL - {proposal}")
        return entry

    def accept_proposal(
        self, proposal_id: str, reason: str, conversation_log: list[dict], round_index: int
    ) -> dict:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "round": round_index,
            "from": self.name,
            "type": "accept_proposal",
            "proposal_id": proposal_id,
            "reason": reason,
        }
        conversation_log.append(entry)
        self.proposal_status = "accepted"
        self.current_proposal = proposal_id
        print(f"✅ {self.name} ACCEPTS {proposal_id}: {reason}")
        return entry

    def reject_proposal(
        self, proposal_id: str, reason: str, conversation_log: list[dict], round_index: int
    ) -> dict:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "round": round_index,
            "from": self.name,
            "type": "reject_proposal",
            "proposal_id": proposal_id,
            "reason": reason,
        }
        conversation_log.append(entry)
        self.proposal_status = "rejected"
        print(f"❌ {self.name} REJECTS {proposal_id}: {reason}")
        return entry

    def share_in_confidence(
        self,
        agent_list: list[str],
        shared_context: str,
        conversation_log: list[dict],
        round_index: int,
        all_agents: list["Agent"],
    ) -> dict:
        """Selectively disclose sensitive context to specific peers."""
        shared_context = shared_context if isinstance(shared_context, str) else str(shared_context or "")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "round": round_index,
            "from": self.name,
            "to": agent_list,
            "type": "share_in_confidence",
            "content": shared_context,
        }
        conversation_log.append(entry)
        self.confidential_disclosures_made.append(
            {"round": round_index, "to": agent_list, "shared_context": shared_context}
        )
        # Also update each recipient's structured received map so it surfaces
        # in their prompts via render_confidential_disclosures.
        for recipient in all_agents:
            if recipient.name in agent_list and recipient.name != self.name:
                recipient.confidential_disclosures_received.setdefault(self.name, []).append(
                    {"round": round_index, "shared_context": shared_context}
                )
        print(f"🤝 {self.name} → {agent_list} (in confidence): {shared_context}")
        return entry

    def clarify(
        self,
        agent_list: list[str],
        prior_event_ref: str,
        reframe: str,
        conversation_log: list[dict],
        round_index: int,
    ) -> dict:
        reframe = reframe if isinstance(reframe, str) else str(reframe or "")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "round": round_index,
            "from": self.name,
            "to": agent_list or ["all"],
            "type": "clarify",
            "prior_event_ref": prior_event_ref,
            "content": reframe,
        }
        conversation_log.append(entry)
        self.clarifications_made.append(
            {"round": round_index, "prior_event_ref": prior_event_ref, "reframe": reframe}
        )
        print(f"♻️  {self.name} clarifies (re: {prior_event_ref}): {reframe}")
        return entry

    # ----- visibility ------------------------------------------------------
    def get_visible_conversation(self, conversation_log: list[dict]) -> list[dict]:
        """Visible-events filter.

        Note: `share_in_confidence` events are *excluded* from this list because
        they are surfaced separately via render_confidential_disclosures, which
        consumes confidential_disclosures_received. This keeps the reciprocity
        signal cleanly separable from the general conversation.
        """
        visible = []
        for event in conversation_log:
            if event.get("type") == "share_in_confidence":
                continue
            if event.get("from") == "system":
                visible.append(event)
            elif event.get("from") == self.name:
                visible.append(event)
            elif self.name in event.get("to", []):
                visible.append(event)
            elif "all" in event.get("to", []) or not event.get("to"):
                visible.append(event)
        return visible

    def observe_environment(
        self, conversation_log: list[dict], other_agents: list["Agent"]
    ) -> str:
        visible = self.get_visible_conversation(conversation_log)
        recent = visible[-10:]
        analysis = f"Recent events observed by {self.name}:\n"
        for event in recent:
            content = event.get("content", "") or ""
            analysis += f"- {event.get('type', 'unknown')} from {event.get('from', 'unknown')}: {content}\n"

        proposals = [e for e in recent if e.get("type") == "proposal"]
        if proposals:
            latest = proposals[-1]
            analysis += f"Latest proposal by {latest.get('from')}: {latest.get('content', '')}\n"

        for other in other_agents:
            if other.name != self.name:
                analysis += f"{other.name} proposal status: {other.proposal_status}\n"

        self.write_to_temp_memory(analysis)
        return analysis

    # ----- LLM dispatch ----------------------------------------------------
    def get_llm_response(self, prompt: str) -> LLMResult:
        try:
            return generate(prompt, self.llm_type)
        except Exception as e:
            return LLMResult(
                text=f"Error generating response: {e}",
                input_tokens=len(prompt) // 4,
                output_tokens=0,
                total_tokens=len(prompt) // 4,
                prompt_chars=len(prompt),
                latency_ms=0.0,
                model=None,
            )

    def _log_llm_call(
        self,
        *,
        round_index: int,
        phase: str,
        prompt: str,
        result: LLMResult,
    ) -> None:
        self.prompt_log.append(
            {
                "round": round_index,
                "phase": phase,
                "prompt": prompt,
                "prompt_chars": result.prompt_chars,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
                "latency_ms": result.latency_ms,
                "model": result.model,
            }
        )

    # ----- decision steps --------------------------------------------------
    def decide_action(
        self,
        conversation_log: list[dict],
        other_agents: list["Agent"],
        task_info: dict,
        round_index: int,
        max_rounds: int,
    ) -> dict:
        prompt = build_decide_action_prompt(
            name=self.name,
            role=self.role,
            description=self.description,
            shareable_preferences=self.shareable_preferences,
            private_preferences=self.private_preferences,
            task=task_info.get("task", "Unknown"),
            deliverable=task_info.get("deliverable", "Unknown"),
            other_agents=[{"name": a.name, "role": a.role} for a in other_agents if a.name != self.name],
            main_memory=self.memory,
            temp_memory=self.temp_memory,
            visible_conversation=self.get_visible_conversation(conversation_log),
            confidential_disclosures_received=self.confidential_disclosures_received,
            proposal_status=self.proposal_status,
            other_proposal_statuses=[
                f"{a.name}: {a.proposal_status}" for a in other_agents if a.name != self.name
            ],
            round_index=round_index,
            max_rounds=max_rounds,
        )
        result = self.get_llm_response(prompt)
        self._log_llm_call(round_index=round_index, phase="decide_action", prompt=prompt, result=result)
        return self._parse_action_json(result.text)

    def decide_memory_action(
        self,
        conversation_log: list[dict],
        other_agents: list["Agent"],
        task_info: dict,
        round_index: int,
        max_rounds: int,
    ) -> dict:
        prompt = build_memory_decision_prompt(
            name=self.name,
            role=self.role,
            description=self.description,
            shareable_preferences=self.shareable_preferences,
            private_preferences=self.private_preferences,
            task=task_info.get("task", "Unknown"),
            deliverable=task_info.get("deliverable", "Unknown"),
            other_agents=[{"name": a.name, "role": a.role} for a in other_agents if a.name != self.name],
            main_memory=self.memory,
            temp_memory=self.temp_memory,
            visible_conversation=self.get_visible_conversation(conversation_log),
            proposal_status=self.proposal_status,
            other_proposal_statuses=[
                f"{a.name}: {a.proposal_status}" for a in other_agents if a.name != self.name
            ],
            round_index=round_index,
            max_rounds=max_rounds,
        )
        result = self.get_llm_response(prompt)
        self._log_llm_call(round_index=round_index, phase="decide_memory", prompt=prompt, result=result)
        return self._parse_action_json(result.text)

    def _parse_action_json(self, raw: str) -> dict:
        """Best-effort extraction of an action JSON object from a model response.

        Tries three strategies in order:
          1. Parse the response as-is (after stripping markdown fences).
          2. Scan for the first balanced {...} block and parse that.
          3. Fall back to no-op and log the raw response for debugging.
        """
        text = (raw or "").strip()
        # Strip common markdown fences
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Strategy 1: parse as-is
        try:
            return json.loads(text)
        except Exception:
            pass

        # Strategy 2: balanced-brace scan for the first JSON object in prose
        for start in range(len(text)):
            if text[start] != "{":
                continue
            depth = 0
            in_string = False
            escape = False
            for end in range(start, len(text)):
                ch = text[end]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : end + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict) and "action" in parsed:
                                return parsed
                        except Exception:
                            break
                        break

        # Strategy 3: surface the failure instead of silently dropping
        preview = (raw or "")[:300].replace("\n", " ")
        print(f"⚠️  {self.name}: action parse failed; raw preview: {preview!r}")
        return {"action": "no_memory_write", "parameters": {}}


class Simulation:
    def __init__(self, scenario_file: str, llm_type: str = "gemini", api_key: str | None = None) -> None:
        self.scenario_file = scenario_file
        self.llm_type = llm_type
        self.api_key = api_key
        self.agents: list[Agent] = []
        self.conversation_log: list[dict] = []
        self.scenario_data: dict | None = None
        self.max_rounds: int = 10
        self.output_file: str | None = None
        self.status: str = "not_started"  # not_started | in_progress | complete | interrupted

    def load_scenario(self) -> bool:
        try:
            with open(self.scenario_file, "r", encoding="utf-8") as f:
                self.scenario_data = json.load(f)
            print(f"📂 Loaded scenario: {self.scenario_data.get('scenario', 'Unknown')}")
            return True
        except Exception as e:
            print(f"❌ Error loading scenario: {e}")
            return False

    def initialize_agents(self) -> bool:
        if not self.scenario_data:
            return False
        print(f"\n🤖 Initializing {len(self.scenario_data.get('agents', []))} agents...")
        for agent_data in self.scenario_data.get("agents", []):
            agent = Agent(
                name=agent_data["name"],
                role=agent_data["role"],
                description=agent_data["description"],
                shareable_preferences=agent_data.get("shareable_preferences", {}),
                private_preferences=agent_data.get("private_preferences", {}),
                llm_type=self.llm_type,
                api_key=self.api_key,
            )
            self.agents.append(agent)
            print(f"  ✅ {agent.name} ({agent.role})")
        return True

    def run_simulation(self, max_rounds: int = 10, output_file: str | None = None) -> bool:
        if not self.agents:
            return False
        self.max_rounds = max_rounds
        self.output_file = output_file
        self.status = "in_progress"
        print(f"\n🚀 Starting simulation with {len(self.agents)} agents, max {max_rounds} rounds")
        print(f"📋 Task: {self.scenario_data.get('task', 'Unknown')}")
        if output_file:
            print(f"💾 Incremental save: {output_file}")
        print("=" * 80)

        self.conversation_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "round": 0,
                "from": "system",
                "to": [a.name for a in self.agents],
                "type": "system_message",
                "content": (
                    f"Negotiation begins. Task: {self.scenario_data.get('task', 'Unknown')}. "
                    f"Deliverable: {self.scenario_data.get('deliverable', 'Unknown')}. "
                    f"Deadline: {max_rounds} rounds."
                ),
            }
        )
        self._write_snapshot()

        try:
            for round_index in range(1, max_rounds + 1):
                print(f"\n🔄 ROUND {round_index} of {max_rounds}")
                for i, agent in enumerate[Agent](self.agents, 1):
                    print(f"\n👤 {i}/{len(self.agents)}: {agent.name}")
                    agent.observe_environment(self.conversation_log, self.agents)
                    action_data = agent.decide_action(
                        self.conversation_log, self.agents, self.scenario_data, round_index, max_rounds
                    )
                    self.execute_agent_action(agent, action_data, round_index)
                    self._write_snapshot()

                self.print_round_summary()

                print(f"\n🧠 MEMORY WRITING PHASE (round {round_index}):")
                for agent in self.agents:
                    memory_action = agent.decide_memory_action(
                        self.conversation_log, self.agents, self.scenario_data, round_index, max_rounds
                    )
                    self.execute_agent_action(agent, memory_action, round_index)
                    self._write_snapshot()

                if self.check_consensus():
                    print(f"\n🎉 CONSENSUS REACHED at round {round_index}")
                    break

            self.status = "complete"
        except (KeyboardInterrupt, Exception) as e:
            self.status = "interrupted"
            print(f"\n⚠️  Run interrupted ({type(e).__name__}: {e}). Last snapshot saved to {self.output_file}")
            self._write_snapshot()
            raise
        finally:
            self.print_final_summary()
            self._write_snapshot()
        return True

    def execute_agent_action(self, agent: Agent, action_data: dict, round_index: int) -> None:
        action = action_data.get("action")
        params = action_data.get("parameters", {}) or {}
        try:
            if action == "send_message":
                agent.send_message(
                    params.get("agent_list", []),
                    params.get("message", ""),
                    self.conversation_log,
                    round_index,
                )
            elif action == "send_proposal":
                agent.send_proposal(
                    params.get("agent_list", []),
                    params.get("proposal", ""),
                    self.conversation_log,
                    round_index,
                )
            elif action == "accept_proposal":
                agent.accept_proposal(
                    params.get("proposal_id", ""),
                    params.get("reason", ""),
                    self.conversation_log,
                    round_index,
                )
            elif action == "reject_proposal":
                agent.reject_proposal(
                    params.get("proposal_id", ""),
                    params.get("reason", ""),
                    self.conversation_log,
                    round_index,
                )
            elif action == "share_in_confidence":
                agent.share_in_confidence(
                    params.get("agent_list", []),
                    params.get("shared_context", ""),
                    self.conversation_log,
                    round_index,
                    self.agents,
                )
            elif action == "clarify":
                agent.clarify(
                    params.get("agent_list", []) or ["all"],
                    params.get("prior_event_ref", ""),
                    params.get("reframe", "") or params.get("message", ""),
                    self.conversation_log,
                    round_index,
                )
            elif action == "write_to_memory":
                agent.write_to_memory(params.get("text", ""))
            elif action == "no_memory_write":
                pass
            else:
                print(f"⚠️  Unknown action '{action}' from {agent.name}; ignored.")
        except Exception as e:
            print(f"❌ Error executing {action} for {agent.name}: {e}")

    def print_round_summary(self) -> None:
        print("📊 ROUND SUMMARY")
        for agent in self.agents:
            emoji = {"none": "⚪", "pending": "🟡", "accepted": "🟢", "rejected": "🔴"}.get(
                agent.proposal_status, "❓"
            )
            print(f"  {emoji} {agent.name}: {agent.proposal_status} ({agent.current_proposal or 'None'})")

    def print_final_summary(self) -> None:
        print("\n📋 FINAL SUMMARY")
        print(f"  Scenario: {self.scenario_data.get('scenario', 'Unknown')}")
        print(f"  LLM: {self.llm_type}")
        msgs = len([e for e in self.conversation_log if e.get("type") == "message"])
        props = len([e for e in self.conversation_log if e.get("type") == "proposal"])
        confs = len([e for e in self.conversation_log if e.get("type") == "share_in_confidence"])
        clars = len([e for e in self.conversation_log if e.get("type") == "clarify"])
        print(f"  Messages: {msgs} | Proposals: {props} | In-confidence: {confs} | Clarifications: {clars}")
        print(f"  Consensus: {'✅' if self.check_consensus() else '❌'}")

    def check_consensus(self) -> bool:
        accepted = [
            a.current_proposal
            for a in self.agents
            if a.proposal_status == "accepted" and a.current_proposal
        ]
        return len(accepted) == len(self.agents) and len(set(accepted)) == 1

    def _build_snapshot(self) -> dict:
        return {
            "scenario_file": self.scenario_file,
            "llm_type": self.llm_type,
            "timestamp": datetime.now().isoformat(),
            "version": "explore",
            "status": self.status,
            "max_rounds": self.max_rounds,
            "scenario_data": self.scenario_data,
            "agents": [
                {
                    "name": a.name,
                    "role": a.role,
                    "description": a.description,
                    "main_memory": a.memory,
                    "temp_memory": a.temp_memory,
                    "final_proposal_status": a.proposal_status,
                    "current_proposal": a.current_proposal,
                    "confidential_disclosures_made": a.confidential_disclosures_made,
                    "confidential_disclosures_received": a.confidential_disclosures_received,
                    "clarifications_made": a.clarifications_made,
                    "prompt_log": a.prompt_log,
                }
                for a in self.agents
            ],
            "conversation_log": self.conversation_log,
        }

    def _write_snapshot(self) -> None:
        """Atomically write the current snapshot to self.output_file.

        Writes to a sibling .tmp file and renames into place so a crash
        mid-write never leaves a corrupt output file. No-op if no output
        file is configured.
        """
        if not self.output_file:
            return
        out_dir = os.path.dirname(self.output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        tmp_path = self.output_file + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._build_snapshot(), f, indent=2)
        os.replace(tmp_path, self.output_file)

    def save_simulation_log(self, output_file: str) -> None:
        """Backwards-compatible explicit save. Sets the output path and flushes once."""
        self.output_file = output_file
        self._write_snapshot()
        print(f"💾 Simulation log saved to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MAGPIE-inspired multi-agent negotiation simulator")
    parser.add_argument("--scenario_file", required=True, help="Path to scenario JSON")
    parser.add_argument("--llm", choices=["gpt5", "gemini", "anthropic"], default="gemini")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    api_key = args.api_key
    if args.llm == "gpt5" and not api_key and not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY required for gpt5")
        return
    if args.llm == "gemini" and not GEMINI_API_KEYS:
        print("Error: no GEMINI_API_KEY{,_1..5} found in env")
        return
    if args.llm == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY required for anthropic")
        return

    sim = Simulation(args.scenario_file, args.llm, api_key)
    if not sim.load_scenario():
        return
    if not sim.initialize_agents():
        return
    try:
        sim.run_simulation(args.max_rounds, output_file=args.output)
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user. Partial log preserved at {args.output}")
        sys.exit(130)


if __name__ == "__main__":
    main()
