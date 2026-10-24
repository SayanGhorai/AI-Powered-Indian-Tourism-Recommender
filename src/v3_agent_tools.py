"""Controlled V3 planning tools for future LLM orchestration.

Only approved planning operations are exposed.
The LLM cannot directly modify evidence, rankings or catalog data.
"""

from src.v3_planning_controller import run_v3_planning
from src.v3_itinerary_presenter import present_v3_itinerary
from src.v3_conservative_schedule import assess_conservative_schedule


TOOL_SCHEMAS = {
    "recommend_and_plan": {
        "required": {
            "query",
            "visit_date",
            "start_time",
            "visitors",
        },
        "optional": {
            "top_k",
        },
    },
    "plan_selected_attractions": {
        "required": {
            "city_canonical",
            "selections",
            "visit_date",
            "start_time",
        },
        "optional": set(),
    },
}


def list_agent_tools():
    """Return tool names and their accepted arguments."""

    return {
        name: {
            "required": sorted(schema["required"]),
            "optional": sorted(schema["optional"]),
        }
        for name, schema in TOOL_SCHEMAS.items()
    }


def execute_agent_tool(
    tool_name,
    arguments,
    *,
    catalog=None,
    v2_result=None,
):
    """Execute one allowlisted tool using existing V3 logic.

    catalog and v2_result are trusted application-side inputs,
    never parameters supplied by an LLM.
    """

    if tool_name not in TOOL_SCHEMAS:
        raise ValueError(
            f"Unapproved agent tool: {tool_name}"
        )

    if not isinstance(arguments, dict):
        raise TypeError(
            "Tool arguments must be a dictionary."
        )

    schema = TOOL_SCHEMAS[tool_name]

    required = schema["required"]
    allowed = required | schema["optional"]

    missing = required - set(arguments)
    unexpected = set(arguments) - allowed

    if missing:
        raise ValueError(
            f"Missing tool arguments: {sorted(missing)}"
        )

    if unexpected:
        raise ValueError(
            f"Unsupported tool arguments: {sorted(unexpected)}"
        )

    if tool_name == "recommend_and_plan":

        result = run_v3_planning(
            mode="recommend",
            query=arguments["query"],
            visit_date=arguments["visit_date"],
            start_time=arguments["start_time"],
            visitors=arguments["visitors"],
            top_k=arguments.get("top_k", 5),
            candidate_k=10,
            catalog=catalog,
            v2_result=v2_result,
        )

    else:

        result = run_v3_planning(
            mode="user_selected",
            city_canonical=arguments["city_canonical"],
            selections=arguments["selections"],
            visit_date=arguments["visit_date"],
            start_time=arguments["start_time"],
            catalog=catalog,
        )

    presented = present_v3_itinerary(result)
    schedule_assessment = assess_conservative_schedule(result)

    draft = result["draft"]

    return {
        "tool_name": tool_name,
        "execution_status": "SUCCESS",
        "planning_mode": result["mode"],
        "source": result["source"],
        "city": result["city"],
        "answer": presented["answer"],
        "stop_count": presented["stop_count"],
        "known_admission_subtotal_inr": (
            presented["known_admission_subtotal_inr"]
        ),
        "evidence_stops": draft["stops"],
        "schedule_assessment": schedule_assessment,
        "stops": presented["stops"],
        "held": presented["held"],
        "limitations": presented["limitations"],
        "grounding_status": presented["grounding_status"],
        "llm_generated": False,
        "route_optimized": False,
        "full_day_feasibility_verified": False,
    }
