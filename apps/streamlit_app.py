import streamlit as st
from cse.search_engine.engine import CodeSearchEngine
from cse.agent.llm import LLMClient
from cse.agent.core import CodingAgent
from argparse import ArgumentParser
from omegaconf import OmegaConf


@st.cache_resource
def initialize_system():
    """Lazy loads the heavy AI models only once."""
    print("Initializing System...")
    try:
        engine = CodeSearchEngine(
            model_name=config.finetuned_model_path,
            db_collection=config.qdrant.full_collection,
            db_path=config.qdrant.storage_path,
        )
        llm = LLMClient(model_name="phi3")
        return CodingAgent(engine, llm)
    except Exception as e:
        st.error(f"System failed to load: {e}")
        return None


# Parse arguments and config file
parser = ArgumentParser()
parser.add_argument("--config", type=str, default="config/main_config.yaml")
args = parser.parse_args()
config = OmegaConf.load(args.config)

# Set page configuration
st.set_page_config(page_title="AI Code Assistant", layout="wide", initial_sidebar_state="expanded")

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

# Initialize the AI Agent
agent = initialize_system()

# Configure sidebar
with st.sidebar:
    st.title("AI Code Assistant")
    st.markdown("---")
    st.markdown("### System Status")
    if agent:
        st.success("Agent Online")
        st.info("Backend: Qdrant + Phi-3 (Ollama)")
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
st.caption("I don't just search; I plan, retrieve, and critique my own answers.")

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
        thought_container = st.status("Agent is thinking...", expanded=show_thoughts)

        final_answer = ""
        retrieved_code_display = ""

        # Run the agent stream
        try:
            for step in agent.solve(prompt):
                # Update the thought container
                if step.step_type == "plan":
                    thought_container.write(f"**Plan:** {step.content}")

                elif step.step_type == "search":
                    thought_container.write(f"**Search:** {step.content}")
                    if step.data:
                        # Capture data for later display but don't clutter the thought stream too much
                        thought_container.markdown(f"*Found {len(step.data)} candidates*")

                        # Format for saving later
                        for res in step.data:
                            src = res["payload"].get("source", "Unknown")
                            code = res["payload"].get("content", "") or res["payload"].get("code_content", "")
                            retrieved_code_display += f"# Source: {src}\n{code}\n\n"

                elif step.step_type == "critique":
                    thought_container.write(f"**Critique:** {step.content}")
                    if "Refining" in step.content:
                        thought_container.update(label="Refining Search...", state="running")

                elif step.step_type == "answer":
                    thought_container.update(label="Reasoning Complete", state="complete", expanded=False)
                    final_answer = step.content

                elif step.step_type == "error":
                    thought_container.update(label="Error", state="error")
                    st.error(step.content)

            # Render final answer
            response_placeholder.markdown(final_answer)

            # Show sources (if any)
            if retrieved_code_display:
                with sources_placeholder.expander("View Retrieved Code Context"):
                    st.code(retrieved_code_display, language="python")

            # Save to history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_answer,
                    "source_code": retrieved_code_display if retrieved_code_display else None,
                }
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")
