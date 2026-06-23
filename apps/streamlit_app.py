from argparse import ArgumentParser

import streamlit as st

from cse.agent.presenter import describe_step
from cse.agent.setup import AgentOptions, build_agent


@st.cache_resource(show_spinner="Loading agent...")
def get_agent(options: AgentOptions):
    """Builds (and caches) the agent for a given combination of options."""
    for step in build_agent(options):
        if step.step_type == "ready":
            return step.agent
        elif step.step_type == "error":
            st.error(f"System failed to load: {step.content}")
            return None
    return None


# Parse arguments (paths only; model/agent type are chosen via the sidebar)
parser = ArgumentParser()
parser.add_argument("--config", type=str, default="config/main_config.yaml")
parser.add_argument("--llm-config", type=str, default="config/llm_config.yaml")
args = parser.parse_args()

# Set page configuration
st.set_page_config(
    page_title="AI Code Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set custom CSS styles
st.markdown(
    """
<style>
    .stChatMessage { font-family: 'Inter', sans-serif; }
    .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; }
    .agent-thought { color: #555; font-style: italic; font-size: 0.9em; border-left: 3px solid #ff4b4b; padding-left: 10px; margin: 5px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# Session State for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Configure sidebar
with st.sidebar:
    st.title("AI Code Assistant")
    st.markdown("---")

    st.markdown("### Agent Settings")
    agent_type = st.selectbox(
        "Agent",
        ["baseline", "tool-loop"],
        help=(
            "baseline: fixed plan->search->critique pipeline. "
            "tool-loop: LLM chooses tools (search/read/list/grep) in a loop."
        ),
    )
    model = st.text_input("Ollama model", value="phi3")
    finetuned = st.checkbox(
        "Use fine-tuned embedding model",
        value=False,
        help="Search using the fine-tuned model, otherwise uses the base model.",
    )

    options = AgentOptions(
        finetuned=finetuned,
        agent_type=agent_type,
        model=model,
        config_path=args.config,
        llm_config_path=args.llm_config,
    )
    agent = get_agent(options)

    st.markdown("---")
    st.markdown("### System Status")
    if agent:
        st.success("Agent Online")
        st.info(f"Backend: Qdrant + {model} (Ollama)")
    else:
        st.error("Agent Offline")

    st.markdown("### Debug Options")
    show_thoughts = st.toggle("Show Internal Monologue", value=True)

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("AI Self-Correcting Code Assistant")
st.caption(
    "I don't just search; I plan, retrieve, and critique my own answers."
)

# Render history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "source_code" in message:  # If there is saved code context, show it
            with st.expander("View Retrieved Code Context"):
                st.code(message["source_code"], language="python")

# --- Handle Input ---
if prompt := st.chat_input("How do I sort a list in Python?"):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response (the agent loop)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        sources_placeholder = st.container()

        # Use a container for the "Thinking" process
        thought_container = st.status(
            "Agent is thinking...", expanded=show_thoughts
        )

        final_answer = ""
        retrieved_code_display = ""

        # Run the agent stream
        try:
            for step in agent.solve(prompt):
                label, text = describe_step(step)

                if step.step_type == "search":
                    thought_container.write(f"**{label}:** {text}")
                    if step.data:
                        thought_container.markdown(
                            f"*Found {len(step.data)} candidates*"
                        )
                        # Format for saving later
                        for res in step.data:
                            src = res["payload"].get("source", "Unknown")
                            code = res["payload"].get("content", "") or res[
                                "payload"
                            ].get("code_content", "")
                            retrieved_code_display += (
                                f"# Source: {src}\n{code}\n\n"
                            )

                elif step.step_type == "critique":
                    thought_container.write(f"**{label}:** {text}")
                    if "Refining" in text:
                        thought_container.update(
                            label="Refining Search...", state="running"
                        )

                elif step.step_type == "answer":
                    thought_container.update(
                        label="Reasoning Complete",
                        state="complete",
                        expanded=False,
                    )
                    final_answer = text

                elif step.step_type == "error":
                    thought_container.update(label="Error", state="error")
                    st.error(text)

                else:  # start, plan, tool_call, tool_result
                    thought_container.write(f"**{label}:** {text}")

            # Render final answer
            response_placeholder.markdown(final_answer)

            # Show sources (if any)
            if retrieved_code_display:
                with sources_placeholder.expander(
                    "View Retrieved Code Context"
                ):
                    st.code(retrieved_code_display, language="python")

            # Save to history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_answer,
                    "source_code": retrieved_code_display
                    if retrieved_code_display
                    else None,
                }
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")
