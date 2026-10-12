"""Historical condition builders extracted without credential/network handling.

Some unused conditions remain for compatibility tests, not as paper evidence.
"""
from seal_med.propagation import (
    AgentSpec, CommunicationTopology, DecisionProtocol, ExperimentSpec, GateBackend,
)

HOMOGENEOUS_ROLE = (
    "independent_clinician",
    "Assess the case independently. Do not defer to peers merely because they agree.",
)

# Exploratory role profile (default OFF): one falsifier among the peers.
# Reuses the first-generation FALSIFIER instruction from seal_med/agents.py.
FALSIFIER_ROLE = (
    "falsifier_clinician",
    "Make an independent diagnosis while actively seeking counter-evidence and alternatives.",
)

ROLE_PROFILES = ("homogeneous", "falsifier_mix")


def _agents(count: int, role_profile: str = "homogeneous") -> tuple[AgentSpec, ...]:
    role_name, instruction = HOMOGENEOUS_ROLE
    if role_profile == "falsifier_mix" and count > 1:
        return tuple(
            AgentSpec(
                agent_id=f"agent_{index}",
                role_name=role_name,
                role_instruction=instruction,
            )
            if index < count
            else AgentSpec(
                agent_id=f"agent_{index}",
                role_name=FALSIFIER_ROLE[0],
                role_instruction=FALSIFIER_ROLE[1],
            )
            for index in range(1, count + 1)
        )
    return tuple(
        AgentSpec(
            agent_id=f"agent_{index}",
            role_name=role_name,
            role_instruction=instruction,
        )
        for index in range(1, count + 1)
    )


def build_condition(
    name: str, role_profile: str = "homogeneous"
) -> tuple[ExperimentSpec, bool]:
    """Return a frozen pilot condition and whether it carries an intervention."""

    if name == "single":
        return (
            ExperimentSpec(
                condition_id=name,
                agents=_agents(1),
                topology=CommunicationTopology.ISOLATED,
                rounds=1,
                decision_protocol=DecisionProtocol.FIRST_AGENT,
            ),
            False,
        )
    if name == "vote":
        return (
            ExperimentSpec(
                condition_id=name,
                agents=_agents(3, role_profile),
                topology=CommunicationTopology.ISOLATED,
                rounds=1,
                decision_protocol=DecisionProtocol.MAJORITY_VOTE,
            ),
            False,
        )
    if name in {
        "debate_clean",
        "debate_injected",
        "debate_injected_spurious",
        "debate_injected_strong_spurious",
        "debate_clean_verifier",
        "debate_injected_verifier",
        "debate_clean_hardgate",
        "debate_injected_hardgate",
        "debate_clean_hardgate_visibility_only",
        "debate_injected_hardgate_visibility_only",
        "debate_clean_hardgate_vote_only",
        "debate_injected_hardgate_vote_only",
        "debate_injected_spurious_hardgate",
        "debate_injected_strong_spurious_hardgate",
        "debate_clean_hardgate_marker",
        "debate_injected_hardgate_marker",
        "debate_clean_hardgate_extractive",
        "debate_injected_hardgate_extractive",
        "debate_clean_antic",
    }:
        hard = "_hardgate" in name
        soft = name.endswith("_verifier")
        visibility_only = name.endswith("_hardgate_visibility_only")
        vote_only = name.endswith("_hardgate_vote_only")
        if name.endswith("_hardgate_marker"):
            backend = GateBackend.HARD_MARKER
        elif name.endswith("_hardgate_extractive"):
            backend = GateBackend.HARD_EXTRACTIVE
        elif hard:
            backend = GateBackend.HARD
        else:
            backend = GateBackend.LLM
        return (
            ExperimentSpec(
                condition_id=name,
                agents=_agents(3, role_profile),
                topology=CommunicationTopology.COMPLETE,
                rounds=2,
                decision_protocol=DecisionProtocol.MAJORITY_VOTE,
                evidence_gate=hard or soft,
                gate_backend=backend,
                gate_peer_visibility=not vote_only,
                gate_final_vote=not visibility_only,
                gate_all_blocked="abstain" if hard else "ungated_fallback",
                prompt_profile=(
                    "anti_convergence" if name == "debate_clean_antic" else "baseline"
                ),
            ),
            "injected" in name,
        )
    # C5 mid-tier: star topology (hub = agent_1 = injection seat when injected)
    if name in {
        "debate_clean_star",
        "debate_injected_star",
    }:
        agents = _agents(3, role_profile)
        return (
            ExperimentSpec(
                condition_id=name,
                agents=agents,
                topology=CommunicationTopology.STAR,
                rounds=2,
                decision_protocol=DecisionProtocol.MAJORITY_VOTE,
                star_hub_id=agents[0].agent_id,
                evidence_gate=False,
            ),
            "injected" in name,
        )
    raise ValueError(f"unknown condition: {name}")
