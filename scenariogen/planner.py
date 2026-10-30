"""LLM-backed semantic planner.

Both DeepSeek and OpenAI use the Chat Completions protocol. The planner
receives a natural-language request, the scenario schema, and a resource
capability summary. It returns one ScenePlan with the five semantic factors.
The plan is normalized and validated before it is returned; rejected plans
are sent back to the model with the validation errors.

The model never produces scene ids, route ids, actor instances, weather
preset names, or TOML.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from scenariogen.models import ResourceSummary, ScenePlan, UserRequest
from scenariogen.schema import (
    SCENARIO_SCHEMA,
    normalize_scene_plan,
    validate_scene_plan,
)


DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

DEFAULT_TIMEOUT_S = 90.0
DEFAULT_MAX_ATTEMPTS = 3
MAX_OUTPUT_TOKENS = 1200


class PlannerError(RuntimeError):
    """Raised when the planner cannot produce an accepted plan."""


class PlanRejected(PlannerError):
    """Raised when a parsed plan violates the schema or the resource catalog."""

    def __init__(self, errors: List[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


class _ChatCompletionPlanner:
    provider_name = "openai-compatible"
    api_key_env = "OPENAI_API_KEY"
    model_env = "OPENAI_MODEL"
    base_url_env = "OPENAI_BASE_URL"
    default_model = OPENAI_DEFAULT_MODEL
    default_base_url = OPENAI_DEFAULT_BASE_URL
    reasoning_token_field = "max_tokens"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        temperature: float = 0.2,
    ):
        self.api_key = api_key or os.environ.get(self.api_key_env)
        self.model = model or os.environ.get(self.model_env) or self.default_model
        resolved_base_url = base_url or os.environ.get(self.base_url_env) or self.default_base_url
        self.base_url = resolved_base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.max_attempts = max(1, int(max_attempts))
        self.temperature = float(temperature)

    def plan(self, request: UserRequest, resource_summary: ResourceSummary) -> ScenePlan:
        if not self.api_key:
            raise PlannerError(f"{self.api_key_env} is not set")

        messages = _initial_messages(request, resource_summary)
        last_errors: List[str] = []

        for attempt in range(1, self.max_attempts + 1):
            try:
                content = self._chat(messages)
            except PlannerError as exc:
                last_errors = [str(exc)]
                continue

            try:
                return parse_scene_plan(
                    content,
                    request=request,
                    resource_summary=resource_summary,
                    planner_source=f"{self.provider_name}:{self.model}",
                )
            except PlanRejected as exc:
                last_errors = exc.errors
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The plan was rejected by the validator: "
                            + "; ".join(exc.errors)
                            + ". Return one corrected JSON object only."
                        ),
                    }
                )
            except PlannerError as exc:
                last_errors = [str(exc)]
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The response could not be parsed: {exc}. "
                            "Return one JSON object only, without markdown."
                        ),
                    }
                )

        raise PlannerError(
            f"{self.provider_name} planning failed after {self.max_attempts} attempts: {last_errors}"
        )

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if _is_reasoning_model(self.model):
            payload[self.reasoning_token_field] = MAX_OUTPUT_TOKENS
        else:
            payload["max_tokens"] = MAX_OUTPUT_TOKENS
            payload["temperature"] = self.temperature

        http_request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise PlannerError(f"{self.provider_name} HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise PlannerError(f"{self.provider_name} request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise PlannerError(f"{self.provider_name} returned invalid JSON: {exc}") from exc

        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise PlannerError(
                f"{self.provider_name} response has no message content: {body}"
            ) from exc


class DeepSeekPlanner(_ChatCompletionPlanner):
    provider_name = "deepseek"
    api_key_env = "DEEPSEEK_API_KEY"
    model_env = "DEEPSEEK_MODEL"
    base_url_env = "DEEPSEEK_BASE_URL"
    default_model = DEFAULT_MODEL
    default_base_url = DEFAULT_BASE_URL


class OpenAIPlanner(_ChatCompletionPlanner):
    provider_name = "openai"
    api_key_env = "OPENAI_API_KEY"
    model_env = "OPENAI_MODEL"
    base_url_env = "OPENAI_BASE_URL"
    default_model = OPENAI_DEFAULT_MODEL
    default_base_url = OPENAI_DEFAULT_BASE_URL
    reasoning_token_field = "max_completion_tokens"


def _is_reasoning_model(model: str) -> bool:
    lowered = model.lower()
    if "reasoner" in lowered:
        return True
    return lowered.startswith(("o1", "o3", "o4", "gpt-5"))


def parse_scene_plan(
    content: str,
    *,
    request: UserRequest,
    resource_summary: ResourceSummary,
    planner_source: str = "llm",
) -> ScenePlan:
    """Parse, normalize, and validate one planner response."""

    raw_plan = _extract_json_object(content)
    plan = normalize_scene_plan(raw_plan)

    if request.count is not None:
        plan.count = int(request.count)

    errors = validate_scene_plan(plan, resource_summary)
    if errors:
        raise PlanRejected(errors)

    plan.planner_source = planner_source
    plan.planner_error = None
    return plan


def _initial_messages(
    request: UserRequest,
    resource_summary: ResourceSummary,
) -> List[Dict[str, str]]:
    system_prompt = (
        "You are the semantic planner of a CARLA pedestrian-scenario generator. "
        "Translate the user request into one structured scene plan. "
        "Use only the values allowed by scenario_schema. "
        "Keep every condition that the user explicitly states. "
        "Use 'any' only for factors the user leaves unspecified. "
        "Do not output scene ids, route ids, actor counts, spawn times, "
        "weather preset names, or TOML. "
        "Return one JSON object only, without markdown."
    )

    user_payload = {
        "user_request": request.prompt,
        "requested_count": request.count,
        "scenario_schema": SCENARIO_SCHEMA,
        "resource_catalog": resource_summary.to_dict(),
        "output_example": {
            "count": 3,
            "description": "Rainy daytime crosswalk cases with grouped pedestrians.",
            "topology": {"scene_kind": "crossing", "route_types": ["crosswalk_cross"]},
            "environment": {
                "illumination": "day",
                "weather": "rain",
                "road_surface": "wet",
                "visibility": "any",
            },
            "traffic": {"vehicle_density": "medium"},
            "social": {
                "pedestrian_density": "medium",
                "group_pattern": "mixed",
                "flow": "two_way",
                "behavior_mix": "mixed",
            },
            "ego": {"behavior": "cautious_cross"},
        },
    }

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def _extract_json_object(content: str) -> Dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise PlannerError("planner response contains no JSON object")

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise PlannerError(f"planner response is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise PlannerError("planner response must be a JSON object")
    return parsed
