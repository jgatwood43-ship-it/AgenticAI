"""
Local Streamlit frontend for the AgenticAI multi-agent workflow.

Run from the project root with:

    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

from typing import TypedDict

import streamlit as st

from core.application import run_agent_query

PAGE_TITLE = "AgenticAI Security Assistant"

WELCOME_MESSAGE = (
    "Hello. Ask me a question about user access activity, "
    "security controls, NIST or CIS guidance, or the connected databases."
)

CHAT_INPUT_PLACEHOLDER = "Ask a security, access activity, policy, or database question"


class ChatMessage(TypedDict):
    """One Streamlit chat-history message."""

    role: str
    content: str


def initialize_session_state() -> None:
    """Initialize conversation history for the current Streamlit session."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE,
            }
        ]


def reset_conversation() -> None:
    """Clear the current conversation and restore the welcome message."""
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
        }
    ]


def get_conversation_history() -> list[ChatMessage]:
    """
    Return prior user and assistant messages for QueryResolver.

    This function is called before the current user message is appended, so the
    current prompt is not duplicated in workflow history.
    """
    history: list[ChatMessage] = []

    for message in st.session_state.messages:
        role = str(message.get("role", "")).strip()

        content = str(message.get("content", "")).strip()

        if role in {"user", "assistant"} and content:
            history.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    return history


def append_message(
    role: str,
    content: str,
) -> None:
    """Append one normalized message to Streamlit session history."""
    cleaned_content = str(content or "").strip()

    if not cleaned_content:
        return

    st.session_state.messages.append(
        {
            "role": role,
            "content": cleaned_content,
        }
    )


def display_conversation() -> None:
    """Render all messages stored in the current session."""
    for message in st.session_state.messages:
        role = message.get(
            "role",
            "assistant",
        )

        content = message.get(
            "content",
            "",
        )

        with st.chat_message(role):
            st.markdown(content)


def display_sidebar() -> None:
    """Render session controls and workflow information."""
    with st.sidebar:
        st.header("Session")

        if st.button(
            "Clear conversation",
            use_container_width=True,
            type="secondary",
        ):
            reset_conversation()
            st.rerun()

        st.divider()

        st.subheader("Workflow")

        st.markdown("""
- Resolve conversational follow-up questions
- Route requests through the structured decision agent
- Retrieve NIST and CIS evidence from pgvector
- Retrieve authoritative schema documentation
- Plan and validate read-only MySQL queries
- Execute database queries through MCP
- Grade and refine the evidence
- Generate a final security-focused response
            """)

        st.divider()

        st.caption(
            "This application runs locally. Shared models and indexes are "
            "reused, while each prompt receives fresh workflow state and "
            "fresh agent instances."
        )


def process_prompt(
    prompt: str,
) -> None:
    """
    Submit one prompt to the workflow and display exactly one response.

    Conversation history is captured before the current user message is added.
    This permits QueryResolver to combine follow-up messages with prior context
    without duplicating the current prompt.
    """
    cleaned_prompt = str(prompt or "").strip()

    if not cleaned_prompt:
        return

    conversation_history = get_conversation_history()

    append_message(
        role="user",
        content=cleaned_prompt,
    )

    with st.chat_message("user"):
        st.markdown(cleaned_prompt)

    answer = ""

    with st.chat_message("assistant"):
        try:
            with st.status(
                "Running the AgenticAI workflow...",
                expanded=True,
            ) as status:
                st.write("Resolving conversation context...")
                st.write("Selecting the appropriate workflow route...")
                st.write("Retrieving and validating evidence when required...")

                # The workflow is called exactly once.
                answer = run_agent_query(
                    cleaned_prompt,
                    conversation_history=conversation_history,
                )

                status.update(
                    label="Workflow complete",
                    state="complete",
                    expanded=False,
                )

            st.markdown(answer)

        except Exception as exc:
            answer = (
                "The AgenticAI workflow encountered an error.\n\n"
                f"**{type(exc).__name__}:** `{exc}`"
            )

            st.error(answer)

    append_message(
        role="assistant",
        content=answer,
    )


def main() -> None:
    """Configure and run the Streamlit application."""
    st.set_page_config(
        page_title="AgenticAI",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()

    st.title(PAGE_TITLE)

    st.caption(
        "Local multi-agent assistant using structured routing, "
        "LlamaIndex, RAG, MCP, MySQL, PostgreSQL, and pgvector."
    )

    display_sidebar()
    display_conversation()

    prompt = st.chat_input(CHAT_INPUT_PLACEHOLDER)

    if prompt:
        process_prompt(prompt)


if __name__ == "__main__":
    main()
