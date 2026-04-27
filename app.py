"""
Upsell Discovery Agent — Streamlit Web UI
==========================================
A live, mobile-responsive interface for the LangGraph agent.

Key features:
- Account picker with at-a-glance health snapshot
- Quick-action prompt buttons + free-form chat
- Live streaming of the agent's reasoning (tool calls + LLM steps)
- Direct link to Phoenix traces for deep observability
"""

import os
import time
from typing import Iterable

import streamlit as st
from dotenv import load_dotenv

# Pull in the agent and its mock data from agent.py
from agent import build_agent, ACCOUNTS, USAGE

load_dotenv()

# ---------------------------------------------------------------------------
# Page config + styles
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Northstar - Disovery Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Tighten up Streamlit's default padding for a more app-like feel */
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

    /* Custom badge styling */
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
    }
    .badge-good   { background: #DCFCE7; color: #166534; }
    .badge-warn   { background: #FEF3C7; color: #92400E; }
    .badge-bad    { background: #FEE2E2; color: #991B1B; }
    .badge-accent { background: #FFE5DC; color: #C2410C; }

    /* Reasoning step cards */
    .step-card {
        background: #F8FAFC; border-left: 3px solid #FF6B47;
        padding: 12px 16px; margin: 8px 0; border-radius: 4px;
        font-size: 13px;
    }
    .step-tool { border-left-color: #FF6B47; }
    .step-think { border-left-color: #64748B; background: #F1F5F9; }
    .step-final { border-left-color: #10B981; background: #F0FDF4; }

    /* Code-style monospace for tool names */
    .mono { font-family: "SF Mono", Monaco, Consolas, monospace; font-size: 12px; }

    /* Hide Streamlit's default footer and menu for a cleaner look */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "agent" not in st.session_state:
    st.session_state.agent = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"query": str, "steps": [...], "final": str}


def get_agent():
    """Lazy-build the agent once per session."""
    if st.session_state.agent is None:
        with st.spinner("Initializing agent..."):
            st.session_state.agent = build_agent()
    return st.session_state.agent


# ---------------------------------------------------------------------------
# Helper: health-score badge
# ---------------------------------------------------------------------------
def health_badge(score: int) -> str:
    if score >= 85:
        return f'<span class="badge badge-good">HEALTHY · {score}</span>'
    if score >= 70:
        return f'<span class="badge badge-warn">WATCH · {score}</span>'
    return f'<span class="badge badge-bad">RISK · {score}</span>'


def utilization_badge(pct: int) -> str:
    if pct >= 110:
        return f'<span class="badge badge-accent">EXPANSION SIGNAL · {pct}% utilized</span>'
    if pct >= 90:
        return f'<span class="badge badge-warn">{pct}% utilized</span>'
    return f'<span class="badge badge-good">{pct}% utilized</span>'


# ---------------------------------------------------------------------------
# Sidebar — account picker + Phoenix link
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Book of Business")
    st.caption("Select an account to investigate")

    account_id = st.selectbox(
        "Account",
        options=list(ACCOUNTS.keys()),
        format_func=lambda aid: f"{ACCOUNTS[aid]['name']} ({aid})",
        label_visibility="collapsed",
    )

    acct = ACCOUNTS[account_id]
    usage = USAGE[account_id]

    # Account snapshot card
    st.markdown("---")
    st.markdown(f"#### {acct['name']}")
    st.markdown(
        f"<div style='margin: 4px 0'>{health_badge(acct['health_score'])}</div>"
        f"<div style='margin: 4px 0'>{utilization_badge(usage['license_utilization_pct'])}</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    col_a.metric("ARR", f"${acct['arr']:,}")
    col_b.metric("Tier", acct['tier'])
    col_c, col_d = st.columns(2)
    col_c.metric("Seats", f"{usage['active_seats_30d']}/{acct['seats_purchased']}")
    col_d.metric("Trend", usage['trend_30d'])

    st.caption(f"Renewal: **{acct['renewal_date']}**")
    st.caption(f"Industry: {acct['industry']}")

    # Phoenix link
    st.markdown("---")
    phoenix_url = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    st.markdown(
        f"""
        <a href="{phoenix_url}" target="_blank" style="text-decoration: none;">
          <div style="background: #1A1F2E; color: white; padding: 10px 14px;
                      border-radius: 6px; font-size: 13px; text-align: center;
                      border: 1px solid #2A3142;">
            🔍 <b>Open Phoenix Traces</b> ↗
          </div>
        </a>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"OTel collector: `{phoenix_url}`")

# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style='display:flex; align-items:baseline; gap:12px;'>
      <h1 style='margin:0; font-family: Georgia, serif;'>Northstar 💫</h1>
      <span class="badge badge-accent">LIVE</span>
    </div>
    <p style='color:#64748B; margin-top:4px;'>
      LangGraph copilot · Phoenix-traced · CSM-driven
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Quick-action prompts
# ---------------------------------------------------------------------------
st.markdown("##### 💡 Quick Plays")

quick_prompts = [
    f"What's the biggest expansion opportunity in {account_id}?",
    f"Build a full upsell motion for {account_id} including outreach.",
    f"Compare expansion potential between ACME-001 and GLOBEX-002.",
    f"Should I push for a co-term renewal on {account_id}?",
]

cols = st.columns(len(quick_prompts))
selected_quick = None
for i, prompt in enumerate(quick_prompts):
    label = ["🎯 Find opportunity", "✉️ Full motion + email", "⚖️ Compare accounts", "🤝 Co-term play"][i]
    if cols[i].button(label, use_container_width=True, key=f"quick_{i}"):
        selected_quick = prompt

# ---------------------------------------------------------------------------
# Free-form input
# ---------------------------------------------------------------------------
user_query = st.chat_input("Ask the agent anything about your accounts...")

# Resolve which query to actually run
query = selected_quick or user_query

# ---------------------------------------------------------------------------
# Streaming agent execution
# ---------------------------------------------------------------------------
def stream_agent_steps(agent, query: str) -> Iterable[dict]:
    """Yield events as the agent reasons through the problem.

    LangGraph's stream() emits a dict per node execution. We translate those
    into UI-friendly event dicts.
    """
    last_message_count = 0

    for chunk in agent.stream(
        {"messages": [("user", query)]},
        stream_mode="values",
    ):
        messages = chunk.get("messages", [])
        # Only process new messages we haven't seen yet
        new_messages = messages[last_message_count:]
        last_message_count = len(messages)

        for msg in new_messages:
            msg_type = type(msg).__name__

            if msg_type == "AIMessage":
                # Either a tool-call decision or a final answer
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        yield {
                            "kind": "tool_call",
                            "tool": tc["name"],
                            "args": tc.get("args", {}),
                        }
                elif msg.content:
                    yield {"kind": "final", "content": msg.content}

            elif msg_type == "ToolMessage":
                yield {
                    "kind": "tool_result",
                    "tool": getattr(msg, "name", "tool"),
                    "content": msg.content,
                }


if query:
    # Display the user's question
    with st.chat_message("user"):
        st.markdown(query)

    # Stream the agent's work
    with st.chat_message("assistant"):
        steps_container = st.container()
        final_container = st.container()

        agent = get_agent()

        steps_log = []
        final_answer = None
        start_time = time.time()

        with steps_container:
            with st.status("🤔 Agent is thinking...", expanded=True) as status:
                try:
                    for event in stream_agent_steps(agent, query):
                        if event["kind"] == "tool_call":
                            args_str = ", ".join(f"{k}={v!r}" for k, v in event["args"].items())
                            st.markdown(
                                f"""
                                <div class="step-card step-tool">
                                  🔧 <b>Calling tool:</b> <span class="mono">{event['tool']}({args_str})</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            steps_log.append(event)

                        elif event["kind"] == "tool_result":
                            preview = str(event["content"])[:200]
                            if len(str(event["content"])) > 200:
                                preview += "..."
                            st.markdown(
                                f"""
                                <div class="step-card step-think">
                                  📥 <b>Tool returned:</b>
                                  <div class="mono" style="margin-top:4px; color:#475569;">{preview}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            steps_log.append(event)

                        elif event["kind"] == "final":
                            final_answer = event["content"]

                    elapsed = time.time() - start_time
                    status.update(
                        label=f"✅ Done in {elapsed:.1f}s · {len(steps_log)} steps",
                        state="complete",
                        expanded=False,
                    )
                except Exception as e:
                    status.update(label=f"❌ Error: {e}", state="error")
                    raise

        if final_answer:
            with final_container:
                st.markdown("##### 📋 Recommendation")
                st.markdown(final_answer)

        st.session_state.history.append({
            "query": query,
            "steps": steps_log,
            "final": final_answer,
        })

# ---------------------------------------------------------------------------
# Footer / empty state
# ---------------------------------------------------------------------------
if not st.session_state.history:
    st.markdown(
        """
        <div style="text-align: center; padding: 40px; color: #94A3B8;">
          <p style="font-size: 15px;">👆 Pick an account from the sidebar, then click a quick play or type a question.</p>
          <p style="font-size: 13px;">Every run streams its reasoning here and emits a full trace to Phoenix.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
