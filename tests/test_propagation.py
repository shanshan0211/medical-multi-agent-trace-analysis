import json
import unittest
import sys
from pathlib import Path
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "historical"))

from seal_med.exceptions import SchemaValidationError
from seal_med.models import ModelReply
from seal_med.propagation import (
    AgentSpec,
    AgentTurn,
    CommunicationTopology,
    DecisionProtocol,
    ErrorInjection,
    ExperimentSpec,
    GateBackend,
    PropagationExperimentRunner,
    _run_hard_evidence_gate,
    compute_propagation_metrics,
)
from conditions import build_condition


def reply(
    primary: str,
    top3: list[str],
    *,
    evidence: list[str] | None = None,
    response: str = "",
    confidence: float = 0.7,
) -> str:
    return json.dumps(
        {
            "primary_diagnosis": primary,
            "top3_differentials": top3,
            "evidence": evidence or ["case evidence"],
            "response_to_peers": response,
            "confidence": confidence,
        }
    )


class RecordingClient:
    def __init__(self, replies: list[str], model_id: str) -> None:
        self.replies = list(replies)
        self.messages: list[tuple[Mapping[str, str], ...]] = []
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_family(self) -> str:
        return "static"

    def complete(
        self,
        messages: tuple[Mapping[str, str], ...],
        *,
        temperature: float = 0.0,
        overrides: Mapping[str, object] | None = None,
    ) -> ModelReply:
        self.messages.append(messages)
        if not self.replies:
            raise RuntimeError("reply budget exhausted")
        return ModelReply(
            content=self.replies.pop(0),
            model_id=self.model_id,
            latency_ms=0,
        )


class TransientBlankClient(RecordingClient):
    def __init__(self, replies: list[str], model_id: str) -> None:
        super().__init__(replies, model_id)
        self.returned_blank = False

    def complete(
        self,
        messages: tuple[Mapping[str, str], ...],
        *,
        temperature: float = 0.0,
        overrides: Mapping[str, object] | None = None,
    ) -> ModelReply:
        if not self.returned_blank:
            self.returned_blank = True
            raise SchemaValidationError("model response content must not be blank")
        return super().complete(messages, temperature=temperature,
                                overrides=overrides)


def agents() -> tuple[AgentSpec, ...]:
    return (
        AgentSpec("agent_1", "primary", "Integrate the case."),
        AgentSpec("agent_2", "falsifier", "Seek counter-evidence."),
        AgentSpec("agent_3", "differential", "Preserve alternatives."),
    )


class PropagationSpecTests(unittest.TestCase):
    def test_canonical_pilot_conditions_change_only_declared_mechanisms(self) -> None:
        single, single_injected = build_condition("single")
        vote, vote_injected = build_condition("vote")
        clean, clean_injected = build_condition("debate_clean")
        injected, has_injection = build_condition("debate_injected")
        clean_v, clean_v_inj = build_condition("debate_clean_verifier")
        inj_v, inj_v_flag = build_condition("debate_injected_verifier")
        clean_h, clean_h_inj = build_condition("debate_clean_hardgate")
        inj_h, inj_h_flag = build_condition("debate_injected_hardgate")
        vis_h, vis_h_flag = build_condition(
            "debate_injected_hardgate_visibility_only"
        )
        vote_h, vote_h_flag = build_condition(
            "debate_injected_hardgate_vote_only"
        )

        self.assertFalse(single_injected)
        self.assertFalse(vote_injected)
        self.assertFalse(clean_injected)
        self.assertTrue(has_injection)
        self.assertFalse(clean_v_inj)
        self.assertTrue(inj_v_flag)
        self.assertFalse(clean_h_inj)
        self.assertTrue(inj_h_flag)
        self.assertTrue(vis_h_flag)
        self.assertTrue(vote_h_flag)
        self.assertFalse(clean.evidence_gate)
        self.assertTrue(clean_v.evidence_gate)
        self.assertTrue(inj_v.evidence_gate)
        self.assertEqual(clean_v.gate_backend, GateBackend.LLM)
        self.assertEqual(clean_h.gate_backend, GateBackend.HARD)
        self.assertTrue(clean_h.evidence_gate)
        self.assertEqual(clean_h.gate_all_blocked, "abstain")
        marker_h, marker_flag = build_condition("debate_injected_hardgate_marker")
        extractive_h, extractive_flag = build_condition(
            "debate_injected_hardgate_extractive"
        )
        clean_marker, clean_marker_flag = build_condition(
            "debate_clean_hardgate_marker"
        )
        self.assertTrue(marker_flag)
        self.assertTrue(extractive_flag)
        self.assertFalse(clean_marker_flag)
        self.assertEqual(marker_h.gate_backend, GateBackend.HARD_MARKER)
        self.assertEqual(extractive_h.gate_backend, GateBackend.HARD_EXTRACTIVE)
        self.assertEqual(clean_marker.gate_backend, GateBackend.HARD_MARKER)
        self.assertTrue(vis_h.gate_peer_visibility)
        self.assertFalse(vis_h.gate_final_vote)
        self.assertFalse(vote_h.gate_peer_visibility)
        self.assertTrue(vote_h.gate_final_vote)
        self.assertEqual(len(single.agents), 1)
        self.assertEqual(single.decision_protocol, DecisionProtocol.FIRST_AGENT)
        self.assertEqual(vote.topology, CommunicationTopology.ISOLATED)
        self.assertEqual(vote.rounds, 1)
        self.assertEqual(clean.topology, CommunicationTopology.COMPLETE)
        self.assertEqual(clean.rounds, 2)
        self.assertEqual(clean.agents, injected.agents)
        self.assertEqual(clean.topology, injected.topology)
        self.assertEqual(clean.rounds, injected.rounds)
        self.assertEqual(clean.decision_protocol, injected.decision_protocol)
        self.assertEqual(clean_v.agents, inj_v.agents)
        self.assertEqual(clean_h.agents, inj_h.agents)

    def test_star_requires_existing_hub(self) -> None:
        with self.assertRaises(SchemaValidationError):
            ExperimentSpec(
                condition_id="star",
                agents=agents(),
                topology=CommunicationTopology.STAR,
                rounds=2,
                decision_protocol=DecisionProtocol.MAJORITY_VOTE,
                star_hub_id="missing",
            )

    def test_duplicate_agent_ids_are_rejected(self) -> None:
        duplicate = (
            AgentSpec("agent_1", "a", "A"),
            AgentSpec("agent_1", "b", "B"),
        )
        with self.assertRaises(SchemaValidationError):
            ExperimentSpec(
                condition_id="bad",
                agents=duplicate,
                topology=CommunicationTopology.COMPLETE,
                rounds=2,
                decision_protocol=DecisionProtocol.MAJORITY_VOTE,
            )


class PropagationRunnerTests(unittest.TestCase):
    def _clients(self) -> dict[str, RecordingClient]:
        return {
            "agent_1": RecordingClient(
                [
                    reply("Gold", ["Gold", "Wrong", "Other"]),
                    reply("Wrong", ["Wrong", "Gold", "Other"]),
                ],
                "m1",
            ),
            "agent_2": RecordingClient(
                [
                    reply("Gold", ["Gold", "Other", "Wrong"]),
                    reply("Wrong", ["Wrong", "Gold", "Other"], response="Changed"),
                ],
                "m2",
            ),
            "agent_3": RecordingClient(
                [
                    reply("Other", ["Other", "Gold", "Wrong"]),
                    reply("Gold", ["Gold", "Wrong", "Other"], response="Kept"),
                ],
                "m3",
            ),
        }

    def test_complete_topology_is_synchronous_and_traceable(self) -> None:
        clients = self._clients()
        spec = ExperimentSpec(
            condition_id="debate_injected",
            agents=agents(),
            topology=CommunicationTopology.COMPLETE,
            rounds=2,
            decision_protocol=DecisionProtocol.MAJORITY_VOTE,
        )
        injection = ErrorInjection(
            source_agent_id="agent_1",
            round_index=1,
            wrong_diagnosis="Wrong",
            injected_rationale="Controlled misleading rationale.",
        )
        trace = PropagationExperimentRunner(clients).run(
            case_id="CASE_001",
            case_text="Published clinical case.",
            diagnosis_options=("Gold", "Wrong", "Other", "Fourth"),
            spec=spec,
            injection=injection,
        )

        round1 = [turn for turn in trace.turns if turn.round_index == 1]
        round2 = [turn for turn in trace.turns if turn.round_index == 2]
        self.assertTrue(all(turn.visible_peer_ids == () for turn in round1))
        self.assertTrue(
            all(len(turn.visible_peer_ids) == 2 for turn in round2)
        )

        injected = next(turn for turn in round1 if turn.agent_id == "agent_1")
        self.assertTrue(injected.injected)
        self.assertEqual(injected.original_primary_diagnosis, "Gold")
        self.assertEqual(injected.primary_diagnosis, "Wrong")

        # Agent 2's second call must contain only previous-round peer messages,
        # including the controlled intervention as peers actually saw it.
        round2_user = json.loads(clients["agent_2"].messages[1][1]["content"])
        visible = round2_user["visible_peer_messages"]
        self.assertEqual({item["agent_id"] for item in visible}, {"agent_1", "agent_3"})
        source_message = next(item for item in visible if item["agent_id"] == "agent_1")
        self.assertEqual(source_message["primary_diagnosis"], "Wrong")

        self.assertEqual(trace.final_diagnosis, "Wrong")
        self.assertEqual(trace.final_vote_count, 2)
        self.assertEqual(trace.final_status, "majority_vote")
        json.dumps(trace.to_record())

        metrics = compute_propagation_metrics(trace, gold_diagnosis="Gold")
        self.assertFalse(metrics.final_correct)
        self.assertEqual(metrics.exposed_agents, 2)
        self.assertEqual(metrics.adopting_agents, 2)
        self.assertEqual(metrics.error_adoption_rate, 1.0)
        self.assertEqual(metrics.primary_adopting_agents, 1)
        self.assertEqual(metrics.primary_adoption_rate, 0.5)
        self.assertEqual(metrics.flip_eligible_agents, 1)
        self.assertEqual(metrics.correct_to_wrong_flips, 1)
        self.assertEqual(metrics.correct_to_wrong_flip_rate, 1.0)
        self.assertTrue(metrics.final_error_retention)

    def test_persist_injection_rewrites_every_round(self) -> None:
        clients = self._clients()
        # Extra round-2 replies are already in the queue for each agent.
        spec = ExperimentSpec(
            condition_id="debate_persist",
            agents=agents(),
            topology=CommunicationTopology.COMPLETE,
            rounds=2,
            decision_protocol=DecisionProtocol.MAJORITY_VOTE,
        )
        injection = ErrorInjection(
            source_agent_id="agent_1",
            round_index=1,
            wrong_diagnosis="Wrong",
            injected_rationale="Controlled misleading rationale.",
            persist=True,
        )
        trace = PropagationExperimentRunner(clients).run(
            case_id="CASE_007",
            case_text="Published clinical case.",
            diagnosis_options=("Gold", "Wrong", "Other", "Fourth"),
            spec=spec,
            injection=injection,
        )
        source_turns = [
            turn for turn in trace.turns if turn.agent_id == "agent_1"
        ]
        self.assertTrue(all(turn.injected for turn in source_turns))
        self.assertTrue(
            all(turn.primary_diagnosis == "Wrong" for turn in source_turns)
        )

    def test_isolated_vote_has_no_propagation_denominator(self) -> None:
        one_round_agents = agents()
        clients = {
            agent.agent_id: RecordingClient(
                [reply("Gold", ["Gold", "Wrong", "Other"])],
                agent.agent_id,
            )
            for agent in one_round_agents
        }
        spec = ExperimentSpec(
            condition_id="vote",
            agents=one_round_agents,
            topology=CommunicationTopology.ISOLATED,
            rounds=1,
            decision_protocol=DecisionProtocol.MAJORITY_VOTE,
        )
        trace = PropagationExperimentRunner(clients).run(
            case_id="CASE_002",
            case_text="Case.",
            diagnosis_options=("Gold", "Wrong", "Other", "Fourth"),
            spec=spec,
        )
        metrics = compute_propagation_metrics(trace, gold_diagnosis="Gold")
        self.assertTrue(metrics.final_correct)
        self.assertEqual(metrics.exposed_agents, 0)
        self.assertIsNone(metrics.error_adoption_rate)
        self.assertEqual(metrics.flip_eligible_agents, 0)
        self.assertFalse(metrics.final_error_retention)

    def test_chain_exposes_only_adjacent_agents(self) -> None:
        clients = self._clients()
        spec = ExperimentSpec(
            condition_id="chain",
            agents=agents(),
            topology=CommunicationTopology.CHAIN,
            rounds=2,
            decision_protocol=DecisionProtocol.MAJORITY_VOTE,
        )
        trace = PropagationExperimentRunner(clients).run(
            case_id="CASE_003",
            case_text="Case.",
            diagnosis_options=("Gold", "Wrong", "Other", "Fourth"),
            spec=spec,
        )
        round2 = {
            turn.agent_id: turn.visible_peer_ids
            for turn in trace.turns
            if turn.round_index == 2
        }
        self.assertEqual(round2["agent_1"], ("agent_2",))
        self.assertEqual(round2["agent_2"], ("agent_1", "agent_3"))
        self.assertEqual(round2["agent_3"], ("agent_2",))

    def test_injected_diagnosis_must_be_an_option(self) -> None:
        clients = {
            agent.agent_id: RecordingClient(
                [reply("Gold", ["Gold", "Wrong"])], agent.agent_id
            )
            for agent in agents()
        }
        spec = ExperimentSpec(
            condition_id="bad_injection",
            agents=agents(),
            topology=CommunicationTopology.ISOLATED,
            rounds=1,
            decision_protocol=DecisionProtocol.MAJORITY_VOTE,
        )
        with self.assertRaises(SchemaValidationError):
            PropagationExperimentRunner(clients).run(
                case_id="CASE_004",
                case_text="Case.",
                diagnosis_options=("Gold", "Wrong"),
                spec=spec,
                injection=ErrorInjection(
                    "agent_1", 1, "Not an option", "Misleading assertion."
                ),
            )

    def test_strict_turn_schema_rejects_unknown_fields(self) -> None:
        bad = json.loads(reply("Gold", ["Gold", "Wrong"]))
        bad["notes"] = "not allowed"
        one_agent = (AgentSpec("agent_1", "solo", "Solve independently."),)
        spec = ExperimentSpec(
            condition_id="single",
            agents=one_agent,
            topology=CommunicationTopology.ISOLATED,
            rounds=1,
            decision_protocol=DecisionProtocol.FIRST_AGENT,
            max_schema_retries=0,
        )
        with self.assertRaises(SchemaValidationError):
            PropagationExperimentRunner(
                {"agent_1": RecordingClient([json.dumps(bad)], "m1")}
            ).run(
                case_id="CASE_005",
                case_text="Case.",
                diagnosis_options=("Gold", "Wrong"),
                spec=spec,
            )

    def test_schema_error_is_retried_with_corrective_feedback(self) -> None:
        one_agent = (AgentSpec("agent_1", "solo", "Solve independently."),)
        spec = ExperimentSpec(
            condition_id="single_retry",
            agents=one_agent,
            topology=CommunicationTopology.ISOLATED,
            rounds=1,
            decision_protocol=DecisionProtocol.FIRST_AGENT,
            max_schema_retries=1,
        )
        # Fenced JSON is salvaged on the first attempt now; to exercise the
        # corrective-retry path the first reply must carry a field violation
        # that salvage cannot fix (an extra key).
        bad = json.loads(reply("Gold", ["Gold", "Wrong"]))
        bad["notes"] = "not allowed"
        client = RecordingClient(
            [
                json.dumps(bad),
                reply("Gold", ["Gold", "Wrong"]),
            ],
            "m1",
        )
        trace = PropagationExperimentRunner({"agent_1": client}).run(
            case_id="CASE_006",
            case_text="Case.",
            diagnosis_options=("Gold", "Wrong"),
            spec=spec,
        )
        self.assertEqual(trace.final_diagnosis, "Gold")
        self.assertEqual(len(client.messages), 2)
        self.assertIn(
            "no Markdown fences",
            client.messages[1][-1]["content"],
        )

    def test_transient_blank_provider_turn_is_retried(self) -> None:
        one_agent = (AgentSpec("agent_1", "solo", "Solve independently."),)
        spec = ExperimentSpec(
            condition_id="single_blank_retry",
            agents=one_agent,
            topology=CommunicationTopology.ISOLATED,
            rounds=1,
            decision_protocol=DecisionProtocol.FIRST_AGENT,
            max_schema_retries=1,
        )
        client = TransientBlankClient(
            [reply("Gold", ["Gold", "Wrong"])],
            "m1",
        )
        trace = PropagationExperimentRunner({"agent_1": client}).run(
            case_id="CASE_BLANK_RETRY",
            case_text="Case.",
            diagnosis_options=("Gold", "Wrong"),
            spec=spec,
        )
        self.assertTrue(client.returned_blank)
        self.assertEqual(trace.final_diagnosis, "Gold")


def gate_reply(
    items: list[tuple[str, bool, bool, str]],
) -> str:
    return json.dumps(
        {
            "assessments": [
                {
                    "agent_id": agent_id,
                    "supported": supported,
                    "blocking": blocking,
                    "reason": reason,
                }
                for agent_id, supported, blocking, reason in items
            ]
        }
    )


class EvidenceGateTests(unittest.TestCase):
    def test_blocking_claim_is_hidden_and_excluded_from_final_vote(self) -> None:
        # Round 1: agent_1 will be injected as Wrong; others say Gold.
        # Gate blocks agent_1 after round 1.
        # Round 2: peers must not see agent_1; all vote Gold; agent_1 still Wrong
        # (persist) but blocked again → final = Gold.
        clients = {
            "agent_1": RecordingClient(
                [
                    reply("Gold", ["Gold", "Wrong", "Other"]),
                    reply("Gold", ["Gold", "Wrong", "Other"]),
                ],
                "m1",
            ),
            "agent_2": RecordingClient(
                [
                    reply("Gold", ["Gold", "Other", "Wrong"]),
                    reply("Gold", ["Gold", "Wrong", "Other"]),
                ],
                "m2",
            ),
            "agent_3": RecordingClient(
                [
                    reply("Gold", ["Gold", "Other", "Wrong"]),
                    reply("Gold", ["Gold", "Wrong", "Other"]),
                ],
                "m3",
            ),
            "verifier": RecordingClient(
                [
                    gate_reply(
                        [
                            ("agent_1", False, True, "Injected rationale not in case."),
                            ("agent_2", True, False, "Supported by case."),
                            ("agent_3", True, False, "Supported by case."),
                        ]
                    ),
                    gate_reply(
                        [
                            ("agent_1", False, True, "Still unsupported."),
                            ("agent_2", True, False, "Supported by case."),
                            ("agent_3", True, False, "Supported by case."),
                        ]
                    ),
                ],
                "verifier",
            ),
        }
        spec = ExperimentSpec(
            condition_id="debate_injected_verifier",
            agents=agents(),
            topology=CommunicationTopology.COMPLETE,
            rounds=2,
            decision_protocol=DecisionProtocol.MAJORITY_VOTE,
            evidence_gate=True,
        )
        injection = ErrorInjection(
            source_agent_id="agent_1",
            round_index=1,
            wrong_diagnosis="Wrong",
            injected_rationale="Controlled misleading rationale.",
            persist=True,
        )
        trace = PropagationExperimentRunner(clients).run(
            case_id="CASE_GATE_001",
            case_text="Published clinical case with gold findings.",
            diagnosis_options=("Gold", "Wrong", "Other", "Fourth"),
            spec=spec,
            injection=injection,
        )

        round2 = [turn for turn in trace.turns if turn.round_index == 2]
        for turn in round2:
            if turn.agent_id != "agent_1":
                self.assertNotIn("agent_1", turn.visible_peer_ids)

        round2_user = json.loads(clients["agent_2"].messages[1][1]["content"])
        visible_ids = {item["agent_id"] for item in round2_user["visible_peer_messages"]}
        self.assertEqual(visible_ids, {"agent_3"})
        self.assertTrue(
            any(
                item["blocking"] and item["agent_id"] == "agent_1"
                for item in round2_user["verification_bulletin"]
            )
        )
        self.assertTrue(
            all(
                "primary_diagnosis" not in item
                for item in round2_user["verification_bulletin"]
            )
        )

        self.assertEqual(trace.final_diagnosis, "Gold")
        self.assertTrue(trace.final_status.endswith("_gate_filtered"))
        self.assertEqual(len(trace.gate_assessments), 6)
        metrics = compute_propagation_metrics(trace, gold_diagnosis="Gold")
        self.assertTrue(metrics.final_correct)
        self.assertFalse(metrics.final_error_retention)
        # Source claim was gated out of peer visibility, so no exposure.
        self.assertEqual(metrics.exposed_agents, 0)

    def test_evidence_gate_requires_verifier_client(self) -> None:
        clients = {
            agent.agent_id: RecordingClient(
                [reply("Gold", ["Gold", "Wrong"])], agent.agent_id
            )
            for agent in agents()
        }
        spec = ExperimentSpec(
            condition_id="missing_verifier",
            agents=agents(),
            topology=CommunicationTopology.ISOLATED,
            rounds=1,
            decision_protocol=DecisionProtocol.MAJORITY_VOTE,
            evidence_gate=True,
        )
        with self.assertRaisesRegex(SchemaValidationError, "verifier"):
            PropagationExperimentRunner(clients).run(
                case_id="CASE_GATE_002",
                case_text="Case.",
                diagnosis_options=("Gold", "Wrong"),
                spec=spec,
            )


class HardGateTests(unittest.TestCase):
    @staticmethod
    def _ablation_clients(
        quote: str,
    ) -> dict[str, RecordingClient]:
        return {
            "agent_1": RecordingClient(
                [
                    reply("Gold", ["Gold", "Wrong"], evidence=[quote]),
                    reply("Gold", ["Gold", "Wrong"], evidence=[quote]),
                ],
                "m1",
            ),
            "agent_2": RecordingClient(
                [
                    reply("Gold", ["Gold", "Wrong"], evidence=[quote]),
                    reply("Gold", ["Gold", "Wrong"], evidence=[quote]),
                ],
                "m2",
            ),
            "agent_3": RecordingClient(
                [
                    reply("Gold", ["Gold", "Wrong"], evidence=[quote]),
                    reply("Wrong", ["Wrong", "Gold"], evidence=[quote]),
                ],
                "m3",
            ),
        }

    @staticmethod
    def _persistent_injection() -> ErrorInjection:
        return ErrorInjection(
            source_agent_id="agent_1",
            round_index=1,
            wrong_diagnosis="Wrong",
            injected_rationale="Controlled intervention: fabricated support.",
            persist=True,
        )

    def test_hardgate_blocks_injection_keeps_grounded_peers(self) -> None:
        case_text = (
            "The patient has a congenital asymptomatic scalp lesion without "
            "bone involvement on MRI."
        )
        quote = "congenital asymptomatic scalp lesion without bone involvement"
        clients = {
            "agent_1": RecordingClient(
                [
                    reply("Gold", ["Gold", "Wrong"], evidence=[quote]),
                    reply("Gold", ["Gold", "Wrong"], evidence=[quote]),
                ],
                "m1",
            ),
            "agent_2": RecordingClient(
                [
                    reply("Gold", ["Gold", "Wrong"], evidence=[quote]),
                    reply("Gold", ["Gold", "Wrong"], evidence=[quote]),
                ],
                "m2",
            ),
            "agent_3": RecordingClient(
                [
                    reply("Gold", ["Gold", "Wrong"], evidence=[quote]),
                    reply("Gold", ["Gold", "Wrong"], evidence=[quote]),
                ],
                "m3",
            ),
        }
        spec = ExperimentSpec(
            condition_id="debate_injected_hardgate",
            agents=agents(),
            topology=CommunicationTopology.COMPLETE,
            rounds=2,
            decision_protocol=DecisionProtocol.MAJORITY_VOTE,
            evidence_gate=True,
            gate_backend=GateBackend.HARD,
            gate_all_blocked="abstain",
        )
        injection = ErrorInjection(
            source_agent_id="agent_1",
            round_index=1,
            wrong_diagnosis="Wrong",
            injected_rationale="Controlled intervention: fabricated support.",
            persist=True,
        )
        trace = PropagationExperimentRunner(clients).run(
            case_id="CASE_HARD_001",
            case_text=case_text,
            diagnosis_options=("Gold", "Wrong", "Other", "Fourth"),
            spec=spec,
            injection=injection,
        )
        self.assertEqual(trace.gate_backend, "hard")
        self.assertEqual(trace.final_diagnosis, "Gold")
        self.assertIn("gate_filtered", trace.final_status)
        blocked_source = [
            item
            for item in trace.gate_assessments
            if item.target_agent_id == "agent_1" and item.blocking
        ]
        self.assertEqual(len(blocked_source), 2)
        round2_user = json.loads(clients["agent_2"].messages[1][1]["content"])
        self.assertTrue(
            all("primary_diagnosis" not in item for item in round2_user["verification_bulletin"])
        )
        metrics = compute_propagation_metrics(trace, gold_diagnosis="Gold")
        self.assertTrue(metrics.final_correct)
        self.assertFalse(metrics.final_error_retention)
        self.assertEqual(metrics.exposed_agents, 0)

    def test_visibility_only_hides_source_but_keeps_its_final_vote(self) -> None:
        case_text = "The patient has a grounded finding documented in the case."
        quote = "grounded finding documented in the case"
        spec, injected = build_condition(
            "debate_injected_hardgate_visibility_only"
        )
        self.assertTrue(injected)
        trace = PropagationExperimentRunner(
            self._ablation_clients(quote)
        ).run(
            case_id="CASE_HARD_VIS",
            case_text=case_text,
            diagnosis_options=("Gold", "Wrong", "Other", "Fourth"),
            spec=spec,
            injection=self._persistent_injection(),
        )

        round2_peers = {
            turn.agent_id: turn.visible_peer_ids
            for turn in trace.turns
            if turn.round_index == 2
        }
        self.assertNotIn("agent_1", round2_peers["agent_2"])
        self.assertNotIn("agent_1", round2_peers["agent_3"])
        self.assertEqual(trace.final_diagnosis, "Wrong")
        self.assertEqual(trace.final_vote_count, 2)
        self.assertEqual(trace.final_status, "majority_vote")
        self.assertTrue(trace.gate_peer_visibility)
        self.assertFalse(trace.gate_final_vote)
        metrics = compute_propagation_metrics(trace, gold_diagnosis="Gold")
        self.assertEqual(metrics.exposed_agents, 0)

    def test_vote_only_keeps_source_visible_but_filters_its_final_vote(self) -> None:
        case_text = "The patient has a grounded finding documented in the case."
        quote = "grounded finding documented in the case"
        spec, injected = build_condition(
            "debate_injected_hardgate_vote_only"
        )
        self.assertTrue(injected)
        clients = self._ablation_clients(quote)
        trace = PropagationExperimentRunner(clients).run(
            case_id="CASE_HARD_VOTE",
            case_text=case_text,
            diagnosis_options=("Gold", "Wrong", "Other", "Fourth"),
            spec=spec,
            injection=self._persistent_injection(),
        )

        round2_peers = {
            turn.agent_id: turn.visible_peer_ids
            for turn in trace.turns
            if turn.round_index == 2
        }
        self.assertIn("agent_1", round2_peers["agent_2"])
        self.assertIn("agent_1", round2_peers["agent_3"])
        round2_user = json.loads(clients["agent_2"].messages[1][1]["content"])
        self.assertEqual(round2_user["verification_bulletin"], [])
        self.assertEqual(trace.final_diagnosis, "Gold")
        self.assertTrue(trace.final_status.endswith("_gate_filtered"))
        self.assertFalse(trace.gate_peer_visibility)
        self.assertTrue(trace.gate_final_vote)
        metrics = compute_propagation_metrics(trace, gold_diagnosis="Gold")
        self.assertEqual(metrics.exposed_agents, 2)

    def test_hardgate_abstains_when_all_blocked(self) -> None:
        clients = {
            agent.agent_id: RecordingClient(
                [reply("Wrong", ["Wrong", "Gold"], evidence=["not in the case at all xx"])],
                agent.agent_id,
            )
            for agent in agents()
        }
        spec = ExperimentSpec(
            condition_id="all_blocked",
            agents=agents(),
            topology=CommunicationTopology.ISOLATED,
            rounds=1,
            decision_protocol=DecisionProtocol.MAJORITY_VOTE,
            evidence_gate=True,
            gate_backend=GateBackend.HARD,
            gate_all_blocked="abstain",
        )
        trace = PropagationExperimentRunner(clients).run(
            case_id="CASE_HARD_002",
            case_text="Only this sentence appears in the case.",
            diagnosis_options=("Gold", "Wrong"),
            spec=spec,
        )
        self.assertEqual(trace.final_status, "abstain_all_blocked")
        self.assertEqual(trace.final_diagnosis, "")

    def test_no_oracle_hard_gate_blocks_without_injected_reason(self) -> None:
        case_text = "The patient has a congenital asymptomatic scalp lesion."
        quote = "congenital asymptomatic scalp lesion"
        injected = AgentTurn(
            case_id="C1",
            condition_id="t",
            round_index=1,
            agent_id="agent_1",
            role_name="clinician",
            visible_peer_ids=(),
            primary_diagnosis="Wrong",
            top3_differentials=("Wrong", "Gold"),
            evidence=(
                "Controlled intervention: the source agent strongly asserts "
                "this alternative diagnosis.",
            ),
            response_to_peers="",
            confidence=0.9,
            injected=True,
            original_primary_diagnosis="Gold",
            raw_content="{}",
        )
        grounded = AgentTurn(
            case_id="C1",
            condition_id="t",
            round_index=1,
            agent_id="agent_2",
            role_name="clinician",
            visible_peer_ids=(),
            primary_diagnosis="Gold",
            top3_differentials=("Gold", "Wrong"),
            evidence=(quote,),
            response_to_peers="",
            confidence=0.8,
            injected=False,
            original_primary_diagnosis=None,
            raw_content="{}",
        )
        for backend in (GateBackend.HARD_MARKER, GateBackend.HARD_EXTRACTIVE):
            assessments = _run_hard_evidence_gate(
                case_text=case_text,
                round_index=1,
                turns=(injected, grounded),
                backend=backend,
            )
            by_id = {item.target_agent_id: item for item in assessments}
            self.assertTrue(by_id["agent_1"].blocking)
            self.assertFalse(by_id["agent_2"].blocking)
            self.assertNotIn(
                "injected claim is withheld",
                by_id["agent_1"].reason,
            )
            if backend is GateBackend.HARD_MARKER:
                self.assertIn("controlled-intervention marker", by_id["agent_1"].reason)
            else:
                self.assertIn("extractive quote", by_id["agent_1"].reason)

        oracle = _run_hard_evidence_gate(
            case_text=case_text,
            round_index=1,
            turns=(injected, grounded),
            backend=GateBackend.HARD,
        )
        self.assertIn(
            "injected claim is withheld",
            next(a.reason for a in oracle if a.target_agent_id == "agent_1"),
        )


if __name__ == "__main__":
    unittest.main()
