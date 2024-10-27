import dataclasses
import inspect
import json
import textwrap
import traceback
import types
import typing as t

from openai.types.beta.function_tool_param import FunctionToolParam
from openai.types.beta.threads import RequiredActionFunctionToolCall
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from openai.types.shared_params.function_definition import FunctionDefinition

from copilot.ai.tools import TOOL_REGISTRY


class _NoDefaultValue:
    def __repr__(self) -> str:
        return "NoDefaultValue"


_NO_DEFAULT_VALUE = _NoDefaultValue()


@dataclasses.dataclass
class AssistantToolParameterMetadata:
    """Class to represent the metadata of an assistant tool parameter."""

    name: str
    annotation: t.Any
    json_schema: dict[str, t.Any]
    description: str
    is_required: bool
    default_value: t.Any = _NO_DEFAULT_VALUE

    def as_openai_tool_spec_property(self) -> dict[str, t.Any]:
        result: dict[str, t.Any] = {
            "description": self.description,
        } | self.json_schema
        if not self.is_required:
            result["optional"] = True
        if self.default_value is not _NO_DEFAULT_VALUE:
            result["default"] = self.default_value
        return result


@dataclasses.dataclass
class AssistantToolMetadata:
    """Class to represent the metadata of an assistant tool."""

    name: str
    description: str
    parameters: dict[str, AssistantToolParameterMetadata]
    return_type: str
    return_description: str
    return_required: bool

    def as_openai_tool_spec(self) -> FunctionToolParam:
        return FunctionToolParam(
            function=FunctionDefinition(
                name=self.name,
                description=self.description,
                parameters={
                    "type": "object",
                    "properties": {
                        p.name: p.as_openai_tool_spec_property()
                        for p in self.parameters.values()
                    },
                    "required": [
                        p.name for p in self.parameters.values() if p.is_required
                    ],
                },
            ),
            type="function",
        )


def get_assistant_tool_metadata(
    function: t.Callable[..., t.Any],
) -> AssistantToolMetadata:
    tool_name = function.__name__
    tool_description = function.__doc__
    if tool_description is None:
        raise ValueError("Function must have a docstring")
    function_signature = inspect.signature(function)
    annotations = {n: None for n, _ in function_signature.parameters.items()}
    annotations.update(inspect.get_annotations(function))
    for n, p in annotations.items():
        if n != "return":
            if t.get_origin(p) is not t.Annotated:
                raise ValueError(f"Parameter {n} is not annotated with Annotated")
            if len(args := t.get_args(p)) != 2:
                raise ValueError(
                    f"Parameter {n} has {len(args)} annotations, expected 2"
                )
            if not isinstance(args[1], str):
                raise ValueError(
                    f"Parameter {n} has {args[1]} as first annotation, expected str"
                )

    json_schemas = {n: _parse_json_schema(p) for n, p in annotations.items()}
    tool_parameter_metadata = {
        n: AssistantToolParameterMetadata(
            **_parse_parameter(n, annotation),
            json_schema=json_schemas[n],
            annotation=t.get_args(annotation)[0],
        )
        for n, annotation in annotations.items()
        if n != "return"
    }
    defaults = {
        n: p.default
        for n, p in function_signature.parameters.items()
        if p.default != p.empty
    }
    return_param_dict = {}
    for n, p in defaults.items():
        if "return" in annotations:
            return_param_dict = _parse_parameter("return", annotations["return"])
            if t.get_origin(annotations["return"]) is t.Annotated:
                return_param_dict["type_"] = t.get_args(annotations["return"])[0]
            else:
                return_param_dict["type_"] = annotations["return"]
    return AssistantToolMetadata(
        name=tool_name,
        description=tool_description,
        parameters=tool_parameter_metadata,
        return_type=return_param_dict.get("type_", "None"),
        return_description=return_param_dict.get("description", ""),
        return_required=return_param_dict.get("required", False),
    )


def _parse_json_schema(annotation: t.Any) -> dict[str, t.Any]:
    if t.get_origin(annotation) is t.Annotated:
        return _parse_json_schema(t.get_args(annotation)[0])
    type_mapping = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    if annotation in type_mapping:
        return {"type": type_mapping[annotation]}
    elif hasattr(annotation, "__origin__"):
        origin = annotation.__origin__
        args = annotation.__args__

        if origin is list:
            return {"type": "array", "items": _parse_json_schema(args[0])}
        elif origin is dict:
            return {
                "type": "object",
                "additionalProperties": _parse_json_schema(args[1]),
            }
        elif origin is t.Union:
            return {"anyOf": [_parse_json_schema(arg) for arg in args]}
        elif origin is t.Literal:
            return {"enum": args, "type": "string"}
        else:
            raise ValueError(
                f"Unsupported origin: {origin} for annotation: {annotation}."
            )
    elif isinstance(annotation, types.UnionType):
        return {"anyOf": [_parse_json_schema(arg) for arg in annotation.__args__]}
    else:
        raise ValueError(
            f"Unsupported type: {annotation}, with origin: {getattr(annotation, '__origin__', None)}, and type: {type(annotation)}."
        )


def _parse_parameter(name: str, annotation: t.Any) -> dict[str, t.Any]:
    result: dict[str, t.Any] = {"name": name}
    if not annotation and name != "return":
        raise ValueError(f"Parameter {name} has no annotation.")
    if isinstance(annotation, str):
        result["is_required"] = True
        return result
    if hasattr(annotation, "default"):
        result["default_value"] = annotation.default
        result["is_required"] = False
    else:
        result["is_required"] = True
    if hasattr(annotation, "__metadata__"):
        result["description"] = annotation.__metadata__[0]
    if hasattr(annotation, "__origin__"):
        result |= _parse_parameter(name, annotation.__origin__)
    if hasattr(annotation, "__args__"):
        args = []
        for arg in annotation.__args__:
            if isinstance(arg, type(None)):
                result["is_required"] = False
                result["default_value"] = None
                continue
            if isinstance(arg, t.ForwardRef):
                arg = arg.__forward_arg__
            args.append(_parse_parameter(name, arg))
    return result


def execute_tool(
    tool_call: RequiredActionFunctionToolCall,
) -> ToolOutput:
    def _execute_tool() -> str:
        try:
            kwargs = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return f"Could not json decode tool call arguments. Error: {e}. Traceback: {traceback.format_exc()}"

        try:
            tool = TOOL_REGISTRY[tool_call.function.name]
        except KeyError as e:
            return f"Tool {e} not found in tool registry. Available tools: {list(TOOL_REGISTRY)}"
        tool_metadata = get_assistant_tool_metadata(tool)
        tool_parameters = tool_metadata.parameters.values()
        required_parameters = [p for p in tool_parameters if p.is_required]
        missing_parameters = [
            p.name for p in required_parameters if p.name not in kwargs
        ]
        if missing_parameters:
            return textwrap.dedent(
                f"""
                Looks like you are missing some required parameters: {", ".join(missing_parameters)}.
                The required parameters are: {", ".join(p.name for p in required_parameters)}.
                """
            )
        try:
            return str(tool(**kwargs))
        except Exception as e:
            return (
                f"Error executing tool. Error: {e}. Traceback: {traceback.format_exc()}"
            )

    return {
        "output": _execute_tool(),
        "tool_call_id": tool_call.id,
    }


def execute_tools(
    tool_calls: list[RequiredActionFunctionToolCall],
) -> t.Generator[ToolOutput, None, None]:
    for tool_call in tool_calls:
        yield execute_tool(tool_call)
