from argparse import ArgumentParser

from omegaconf import OmegaConf

from cse.agent.llm import LLMClient, resolve_native_tool_calling

# Minimal OpenAI-style specs exposing only each tool's semantic args
# (engine/base_dir stay bound by the agent loop, hidden from the model)
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Semantically search the indexed codebase for relevant snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language description of the code to find.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the repository root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the files and subdirectories of a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the repository root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents for a regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression to search for.",
                    }
                },
                "required": ["pattern"],
            },
        },
    },
]

SYSTEM_MSG = (
    "You are a coding assistant exploring a repository. Use the available "
    "tools to find the information needed to answer the user's question."
)

QUERIES = [
    "What does the resolve_device function do?",
    "Show me the contents of config/main_config.yaml",
    "Which files are in the src/cse/agent directory?",
    "Find every place that defines a function called search_code",
]


def main():
    # Whether to use native tool calling or the prompt+JSON fallback is
    # decided per-model by config/llm_config.yaml, not chosen manually here
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3.2:3b")
    parser.add_argument(
        "--llm-config", type=str, default="config/llm_config.yaml"
    )
    args = parser.parse_args()
    llm_config = OmegaConf.load(args.llm_config)

    mode = (
        "native"
        if resolve_native_tool_calling(args.model, llm_config)
        else "fallback (prompt+JSON)"
    )
    print(f"--- {mode} tool calling with {args.model} ---")
    llm = LLMClient(model_name=args.model)

    for query in QUERIES:
        print(f"\n=== User: {query} ===")
        messages = [("system", SYSTEM_MSG), ("user", query)]
        content, tool_calls = llm.call_with_tools_auto(
            messages, TOOL_SPECS, llm_config
        )

        if tool_calls:
            for call in tool_calls:
                print(f"  -> tool: {call.name}  args: {call.arguments}")
        else:
            print(f"  (no tool call) model said: {content!r}")


if __name__ == "__main__":
    main()
