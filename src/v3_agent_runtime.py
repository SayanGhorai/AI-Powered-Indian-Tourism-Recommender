"""V3 agent runtime with optional LLM tool selection.

The application owns all planning arguments.
LLM selection is optional; deterministic planning remains available.
Provider failures fall back only when an explicit planning mode exists.
"""

import os

from src.v3_agent_orchestrator import run_v3_agent
from src.v3_openai_agent_provider import make_openai_tool_selector


class _ProviderSelectionFailure(RuntimeError):
    """An LLM provider failed before an approved tool was executed."""


def run_v3_agent_runtime(
    agent_request,
    *,
    catalog=None,
    v2_result=None,
    use_llm=False,
    api_key=None,
    model=None,
    selector=None,
):
    """Execute grounded V3 planning with optional model tool selection."""

    if not isinstance(agent_request, dict):
        raise TypeError("Agent request must be a dictionary.")

    if not isinstance(use_llm, bool):
        raise TypeError("use_llm must be boolean.")

    def execute(model_selector=None):
        return run_v3_agent(
            agent_request,
            model_selector=model_selector,
            catalog=catalog,
            v2_result=v2_result,
        )

    if not use_llm:
        result = execute()
        return {
            **result,
            "llm_status": "OFFLINE_EXPLICIT_MODE",
            "live_llm_used": False,
            "provider_error_type": None,
        }

    if selector is not None:
        if not callable(selector):
            raise TypeError("Selector must be callable.")
        active_selector = selector
    else:
        configured_key = (
            os.environ.get("OPENAI_API_KEY", "")
            if api_key is None
            else api_key
        )

        if not isinstance(configured_key, str):
            raise TypeError("API key must be a string.")

        if not configured_key.strip():
            result = execute()
            return {
                **result,
                "llm_status": "NO_API_KEY_FALLBACK",
                "live_llm_used": False,
                "provider_error_type": None,
            }

        active_selector = make_openai_tool_selector(
            api_key=configured_key,
            model=model,
        )

    def guarded_selector(message, allowed_tools):
        try:
            return active_selector(message, allowed_tools)
        except (RuntimeError, ValueError, TimeoutError, OSError) as exc:
            raise _ProviderSelectionFailure(
                type(exc).__name__
            ) from None

    try:
        result = execute(model_selector=guarded_selector)
    except _ProviderSelectionFailure as exc:
        # Never silently switch modes if the request did not
        # explicitly confirm one.
        if agent_request.get("mode") is None:
            raise RuntimeError(
                "LLM selection failed and no explicit fallback "
                "planning mode was provided."
            ) from None

        result = execute()
        return {
            **result,
            "llm_status": "LLM_PROVIDER_ERROR_FALLBACK",
            "live_llm_used": False,
            "provider_error_type": str(exc),
        }

    return {
        **result,
        "llm_status": "LLM_TOOL_SELECTED",
        # A supplied selector may be simulated; this flag means
        # model-selection path was used, not verified live API access.
        "live_llm_used": selector is None,
        "provider_error_type": None,
    }
