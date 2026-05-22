import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any

from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# --- ANSI colours ---
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


# --- User input function ---
def get_user_message() -> Optional[str]:
    line = sys.stdin.readline()
    if not line:
        return None
    return line.rstrip("\n")


# --- Tool definition ---
@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]
    function: Callable[[Dict[str, Any]], str]


# --- read_file tool implementation ---
def read_file_tool_fn(input_obj: Dict[str, Any]) -> str:
    """
    Read the contents of a given *relative* file path (from the working directory).
    Safety: rejects absolute paths and path traversal outside the project folder.
    """
    if "path" not in input_obj or not isinstance(input_obj["path"], str):
        raise ValueError("read_file: missing required string field 'path'")

    raw_path = input_obj["path"]

    # Enforce relative paths only
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise ValueError("read_file: absolute paths are not allowed; provide a relative path")

    base_dir = Path.cwd().resolve()
    resolved = (base_dir / candidate).resolve()

    # Prevent path traversal
    if base_dir not in resolved.parents and resolved != base_dir:
        raise ValueError("read_file: path escapes the working directory")

    if not resolved.exists():
        raise FileNotFoundError(f"read_file: file not found: {raw_path}")
    if not resolved.is_file():
        raise ValueError("read_file: path must be a file, not a directory")

    max_bytes = 200_000  # 200 KB
    data = resolved.read_bytes()
    if len(data) > max_bytes:
        raise ValueError(f"read_file: file too large ({len(data)} bytes). Limit is {max_bytes} bytes.")

    return data.decode("utf-8", errors="replace")


READ_FILE_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "The relative path of a file in the working directory."
        }
    },
    "required": ["path"],
    "additionalProperties": False
}

READ_FILE_TOOL = ToolDefinition(
    name="read_file",
    description=(
        "Read the contents of a given relative file path. "
        "Use this when you want to see what's inside a file. "
        "Do not use this with directory names."
    ),
    input_schema=READ_FILE_INPUT_SCHEMA,
    function=read_file_tool_fn
)


# --- Agent definition ---
@dataclass
class Agent:
    client: AzureOpenAI
    deployment_name: str
    get_user_message: Callable[[], Optional[str]]
    tools: List[ToolDefinition] = field(default_factory=list)  # <-- IMPORTANT FIX 【3-890ae5】【4-2dcc5d】

    def run(self) -> None:
        system_prompt = os.getenv("SYSTEM_PROMPT", "You are a helpful AI assistant.")

        conversation: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

        print("Chat with the agent (use Ctrl+C to quit)")

        while True:
            try:
                print(f"{BLUE}You{RESET}: ", end="", flush=True)

                user_input = self.get_user_message()
                if user_input is None:
                    break

                conversation.append({"role": "user", "content": user_input})

                assistant_text = self.run_inference(conversation)

                conversation.append({"role": "assistant", "content": assistant_text})

                print(f"{YELLOW}Assistant{RESET}: {assistant_text}")

            except KeyboardInterrupt:
                print("\nExiting.")
                break

    def build_openai_tools(self) -> List[Dict[str, Any]]:
        openai_tools: List[Dict[str, Any]] = []
        for tool in self.tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            })
        return openai_tools

    def run_inference(self, conversation: List[Dict[str, Any]]) -> str:
        openai_tools = self.build_openai_tools()

        request_kwargs = {
            "model": self.deployment_name,
            "messages": conversation,
            "max_completion_tokens": 1024,
        }

        if openai_tools:
            request_kwargs["tools"] = openai_tools
            request_kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**request_kwargs)

        return response.choices[0].message.content or ""


# --- Factory function ---
def new_agent(
    client: AzureOpenAI,
    deployment_name: str,
    get_user_message_fn,
    tools: List[ToolDefinition],
) -> Agent:
    return Agent(
        client=client,
        deployment_name=deployment_name,
        get_user_message=get_user_message_fn,
        tools=tools,
    )


# --- Entry point ---
def main():
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )

    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not deployment_name:
        raise ValueError("Missing AZURE_OPENAI_DEPLOYMENT_NAME")

    # Tools are set explicitly here (clean + predictable)
    tools: List[ToolDefinition] = [READ_FILE_TOOL]

    agent = new_agent(
        client,
        deployment_name,
        get_user_message,
        tools
    )

    agent.run()


if __name__ == "__main__":
    main()