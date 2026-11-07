"""Streamlit front end for the support agent. Run: streamlit run streamlit_app.py"""

import ast
import asyncio
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from agent import (
    DEFAULT_MODEL, MAX_TURNS, SAMPLE_QUESTIONS, SessionMemory, TooManySteps, run_turn,
)

HERE = Path(__file__).parent
CASES_FILE = HERE / "Supportcase.json"
OWNERS_FILE = HERE / "product_owners.json"
EXAMPLES = SAMPLE_QUESTIONS  # The same samples the command-line agent shows.


def load_json(path: Path, default):
    """Read a JSON file for the sidebar; a missing or broken file shows as empty."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def tool_output_text(output) -> str:
    """MCP results arrive as "{'type': 'text', 'text': ...}"; show just the JSON inside."""
    text = str(output)
    try:
        value = ast.literal_eval(text)
        if isinstance(value, dict) and "text" in value:
            text = value["text"]
    except (ValueError, SyntaxError):
        pass
    try:
        return json.dumps(json.loads(text), indent=2)
    except (json.JSONDecodeError, TypeError):
        return text


def tool_steps(result) -> list:
    """Pair each tool request in a run with its MCP result."""
    return [{**call, "output": tool_output_text(call["output"])}
            for call in result.tool_calls()]


def show_steps(steps: list):
    """Show tool calls under an answer, collapsed so the chat stays readable."""
    if not steps:
        return
    with st.expander(f"Tool calls ({len(steps)})"):
        for step in steps:
            st.markdown(f"**{step['name']}** `{step['arguments']}`")
            if step["output"]:
                st.code(step["output"], language="json")


# Workflow order for the status chart; any other status is listed after these.
STATUS_ORDER = ["New", "In Progress", "Waiting on Customer", "Escalated", "Resolved", "Closed"]
CLOSED_STATUSES = {"Resolved", "Closed"}


def chart_colors():
    """One series hue plus text ink, stepped for the viewer's light or dark theme."""
    if st.context.theme.type == "dark":
        return {"bar": "#3987e5", "text": "#f0efec"}
    return {"bar": "#2a78d6", "text": "#0b0b0b"}


def count_bar_chart(df: pd.DataFrame, field: str, label: str, order=None):
    """Horizontal bar chart of case counts per value of field, with the count at each tip."""
    counts = df[field].fillna("Unassigned").value_counts()
    if order:
        counts = counts.reindex([v for v in order if v in counts.index]
                                + [v for v in counts.index if v not in order])
    data = pd.DataFrame({label: counts.index, "Cases": counts.to_numpy()})
    colors = chart_colors()
    base = alt.Chart(data).encode(
        y=alt.Y(f"{label}:N", sort=list(data[label]), title=None,
                axis=alt.Axis(labelLimit=220, ticks=False, domain=False)),
        x=alt.X("Cases:Q", title=None,
                axis=alt.Axis(tickMinStep=1, format="d", domain=False),
                # Headroom past the longest bar so its count label is not clipped.
                scale=alt.Scale(domain=[0, int(counts.max()) + max(1, int(counts.max()) // 5)],
                                nice=False)),
    )
    bars = base.mark_bar(color=colors["bar"], size=20, cornerRadiusEnd=4).encode(
        tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip("Cases:Q")])
    values = base.mark_text(align="left", dx=6, color=colors["text"]).encode(text="Cases:Q")
    return (bars + values).properties(height=max(90, 34 * len(data)))


def show_dashboard(cases: dict):
    """Case counts by status, by case owner, and by company, with the cases as a table."""
    if not cases:
        st.info("No support cases yet. Create one in the Assistant tab.")
        return
    df = pd.DataFrame([{"Case ID": case_id, "Company": c.get("customer_name"),
                        "Product": c.get("product_name"), "Owner": c.get("case_owner"),
                        "Status": c.get("case_status"), "Created": c.get("created_date"),
                        "Title": c.get("title")} for case_id, c in cases.items()])
    closed = df["Status"].isin(CLOSED_STATUSES)
    tiles = st.columns(4)
    tiles[0].metric("Total cases", len(df))
    tiles[1].metric("Open", int((~closed).sum()))
    tiles[2].metric("Resolved or closed", int(closed.sum()))
    tiles[3].metric("Case owners", df["Owner"].nunique())

    left, right = st.columns(2)
    with left:
        st.subheader("Cases by status")
        st.altair_chart(count_bar_chart(df, "Status", "Status", STATUS_ORDER),
                        width="stretch")
    with right:
        st.subheader("Cases by case owner")
        st.altair_chart(count_bar_chart(df, "Owner", "Owner"), width="stretch")
    st.subheader("Cases by company")
    st.altair_chart(count_bar_chart(df, "Company", "Company"), width="stretch")

    st.subheader("All cases")
    st.dataframe(df, hide_index=True, width="stretch")


st.set_page_config(page_title="Support Case Assistant", page_icon=":material/support_agent:",
                   layout="wide")
st.session_state.setdefault("messages", [])  # What the chat shows.
st.session_state.setdefault("history", [])   # The conversation so far, sent to the agent.
st.session_state.setdefault("memory", SessionMemory())  # Short-term memory: current owner.

with st.sidebar:
    st.header("Settings")
    model = st.text_input("OpenAI model", DEFAULT_MODEL)
    if st.button("New conversation", icon=":material/refresh:", width="stretch"):
        st.session_state.messages, st.session_state.history = [], []
        st.session_state.memory = SessionMemory()
        st.rerun()
    st.caption("A new conversation clears the chat, the conversation history, and the "
               "current case owner.")
    owner = st.session_state.memory.case_owner
    st.markdown(f"**Current case owner:** {owner or 'not set'}")
    if not owner:
        st.caption('Say "my name is ..." to set it, then ask "show me cases".')

    st.header("Try asking")
    for example in EXAMPLES:
        if st.button(example, key=f"example-{example}", width="stretch"):
            st.session_state.pending = example

    st.header("Data")
    cases = load_json(CASES_FILE, {})
    with st.expander(f"Support cases ({len(cases)})"):
        st.dataframe([{"case_id": case_id, "customer": c.get("customer_name"),
                       "product": c.get("product_name"), "owner": c.get("case_owner"),
                       "status": c.get("case_status"), "title": c.get("title")}
                      for case_id, c in cases.items()], hide_index=True)
    owners = load_json(OWNERS_FILE, {})
    with st.expander(f"Product owners ({len(owners)})"):
        st.dataframe([{"product": name, "owner": info.get("owner"),
                       "keywords": ", ".join(info.get("keywords", []))}
                      for name, info in owners.items()], hide_index=True)

st.title("Support Case Assistant")
st.caption("Look up, create, and update support cases. New cases are assigned to the "
           "product owner automatically. Tools come from the support cases MCP server.")
chat_tab, dashboard_tab = st.tabs([":material/chat: Assistant", ":material/bar_chart: Dashboard"])

with dashboard_tab:
    show_dashboard(load_json(CASES_FILE, {}))

with chat_tab:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            show_steps(message.get("steps", []))

    prompt = (st.chat_input("Ask about a case, or create one")
              or st.session_state.pop("pending", None))
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            steps = []
            with st.spinner("Thinking and calling tools..."):
                try:
                    result = asyncio.run(run_turn(st.session_state.history, prompt, model,
                                                  st.session_state.memory))
                    answer = str(result.final_output or "The agent returned no answer.")
                    steps = tool_steps(result)
                    st.session_state.history = result.history
                except TooManySteps:
                    answer = f"Stopped after {MAX_TURNS} model turns. Try a simpler question."
                except Exception as exc:  # Show the problem in the chat, not a stack trace.
                    answer = (f"Could not complete the request: {exc}\n\nCheck that "
                              "OPENAI_API_KEY is set in MCP/.env.")
            st.markdown(answer)
            show_steps(steps)
        st.session_state.messages.append({"role": "assistant", "content": answer,
                                          "steps": steps})
        st.rerun()  # Refresh the sidebar and dashboard so new cases show up.
