"""Controlled agent orchestration for V3 tourism planning.

An optional model selector chooses an approved tool.
The model never supplies or modifies execution arguments.
The final factual answer comes from the grounded V3 presenter.
"""

from copy import deepcopy

from src.v3_agent_tools import (
    execute_agent_tool,
    list_agent_tools,
)


MODE_TO_TOOL = {
    "recommend": "recommend_and_plan",
    "user_selected": "plan_selected_attractions",
}


def run_v3_agent(
    request,
    *,
    model_selector=None,
    catalog=None,
    v2_result=None,
):
    """Run one bounded planning action.

    request:
        user_message: original user text
        mode: optional confirmed planning mode
        arguments: trusted, structured planning inputs

    model_selector:
        Optional callable(user_message, tool_names)
        returning one approved tool name.

    Without a model selector, an explicit mode is required.
    """

    if not isinstance(request, dict):
        raise TypeError("Agent request must be a dictionary.")

    allowed_request_keys = {
        "user_message",
        "mode",
        "arguments",
    }

    unexpected = set(request) - allowed_request_keys

    if unexpected:
        raise ValueError(
            f"Unsupported request fields: {sorted(unexpected)}"
        )

    message = request.get("user_message", "")

    if not isinstance(message, str):
        raise TypeError("User message must be a string.")

    if len(message) > 4000:
        raise ValueError("User message is too long.")

    arguments = request.get("arguments")

    if not isinstance(arguments, dict):
        raise TypeError(
            "Structured planning arguments are required."
        )

    # Preserve user-confirmed inputs and prevent selector mutation.
    confirmed_arguments = deepcopy(arguments)

    mode = request.get("mode")

    if mode is not None and mode not in MODE_TO_TOOL:
        raise ValueError("Unsupported planning mode.")

    approved_tools = list_agent_tools()

    if model_selector is None:

        if mode is None:
            raise ValueError(
                "An explicit mode is required without an LLM."
            )

        chosen_tool = MODE_TO_TOOL[mode]
        selection_source = "EXPLICIT_MODE"

    else:

        if not callable(model_selector):
            raise TypeError(
                "Model selector must be callable."
            )

        if not message.strip():
            raise ValueError(
                "A user message is required for model selection."
            )

        chosen_tool = model_selector(
            message,
            tuple(approved_tools.keys()),
        )

        if not isinstance(chosen_tool, str):
            raise ValueError(
                "Model must return one tool name."
            )

        if chosen_tool not in approved_tools:
            raise ValueError(
                "Model selected an unapproved tool."
            )

        if mode is not None:
            expected_tool = MODE_TO_TOOL[mode]

            if chosen_tool != expected_tool:
                raise ValueError(
                    "Model choice conflicts with confirmed mode."
                )

        selection_source = "MODEL_SELECTED"

    if arguments != confirmed_arguments:
        raise ValueError(
            "Planning arguments were modified."
        )

    tool_result = execute_agent_tool(
        chosen_tool,
        deepcopy(confirmed_arguments),
        catalog=catalog,
        v2_result=v2_result,
    )

    if tool_result["execution_status"] != "SUCCESS":
        raise RuntimeError("Planning tool did not succeed.")

    if tool_result["llm_generated"] is not False:
        raise ValueError(
            "Grounded response provenance mismatch."
        )

    return {
        "agent_status": "TOOL_EXECUTED",
        "selected_tool": chosen_tool,
        "selection_source": selection_source,
        "planning_mode": tool_result["planning_mode"],
        "execution_status": tool_result["execution_status"],
        "answer": tool_result["answer"],
        "stop_count": tool_result["stop_count"],
        "known_admission_subtotal_inr": tool_result[
            "known_admission_subtotal_inr"
        ],
        "evidence_stops": tool_result["evidence_stops"],
        "stops": tool_result["stops"],
        "held": tool_result["held"],
        "limitations": tool_result["limitations"],
        "grounding_status": tool_result["grounding_status"],
        "llm_generated_answer": False,
        "route_optimized": False,
        "full_day_feasibility_verified": False,
    }
