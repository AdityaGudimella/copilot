import typing as t

import pytest

from copilot.ai.openai_.function_calling import (
    AssistantToolMetadata,
    get_assistant_tool_metadata,
)


def get_current_temperature(
    location: t.Annotated[str, "The city and state, e.g., San Francisco, CA"],
    unit: t.Annotated[
        t.Literal["Celsius", "Fahrenheit"],
        "The temperature unit to use. Infer this from the user's location.",
    ],
) -> float:
    """Get the current temperature for a specific location."""
    return 0.0


def get_rain_probability(
    location: t.Annotated[str, "The city and state, e.g., San Francisco, CA"],
) -> float:
    """Get the probability of rain for a specific location."""
    return 0.0


@pytest.mark.parametrize(
    "function, expected_metadata",
    [
        (
            get_current_temperature,
            {
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "description": "Get the current temperature for a specific location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g., San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ("Celsius", "Fahrenheit"),
                                "description": "The temperature unit to use. Infer this from the user's location.",
                            },
                        },
                        "required": ["location", "unit"],
                    },
                },
            },
        ),
        (
            get_rain_probability,
            {
                "type": "function",
                "function": {
                    "name": "get_rain_probability",
                    "description": "Get the probability of rain for a specific location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g., San Francisco, CA",
                            }
                        },
                        "required": ["location"],
                    },
                },
            },
        ),
    ],
)
def test_get_assistant_tool_metadata(
    function: t.Callable[..., t.Any], expected_metadata: AssistantToolMetadata
):
    assert (
        get_assistant_tool_metadata(function).as_openai_tool_spec() == expected_metadata
    )
