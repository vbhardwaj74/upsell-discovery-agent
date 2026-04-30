"""
Northstar — Upsell Discovery Agent (Streamlit Web UI)
=====================================================
A live, mobile-responsive interface for the LangGraph agent.

Key features:
- Account picker with at-a-glance health snapshot
- Quick-action prompts (find opportunity, full motion, compare, co-term, health breakdown)
- Multi-account compare picker (no more hardcoded comparisons)
- Fuzzy account name resolution — type "compare Acme and Tyrell" and it just works
- Live streaming of the agent's reasoning (tool calls + LLM steps)
- Direct link to Phoenix traces for deep observability
"""

import os
import re
import time
from typing import Iterable, Optional

import streamlit as st
from dotenv import load_dotenv

# Pull in the agent and its mock data from agent.py
from agent import build_agent, ACCOUNTS, USAGE

load_dotenv()

# ---------------------------------------------------------------------------
# Page config + styles
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Northstar — Upsell Discovery",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
    }
    .badge-good   { background: #DCFCE7; color: #166534; }
    .badge-warn   { background: #FEF3C7; color: #92400E; }
    .badge-bad    { background: #FEE2E2; color: #991B1B; }
    .badge-accent { background: #FFE5DC; color: #C2410C; }

    .step-card {
        background: #F8FAFC; border-left: 3px solid #FF6B47;
        padding: 12px 16px; margin: 8px 0; border-radius: 4px;
        font-size: 13px;
    }
    .step-tool { border-left-color: #FF6B47; }
    .step-think { border-left-color: #64748B; background: #F1F5F9; }
    .step-final { border-left-color: #10B981; background: #F0FDF4; }

    .mono { font-family: "SF Mono", Monaco, Consolas, monospace; font-size: 12px; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "agent" not in st.session_state:
    st.session_state.agent = None
if "history" not in st.session_state:
    st.session_state.history = []
if "compare_mode" not in st.session_state:
    # When user clicks "Compare accounts", we surface a multiselect first
    st.session_state.compare_mode = False
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def get_agent():
    """Lazy-build the agent once per session."""
    if st.session_state.agent is None:
        with st.spinner("Initializing agent..."):
            st.session_state.agent = build_agent()
    return st.session_state.agent


# ---------------------------------------------------------------------------
# Fuzzy account-name resolver
# ---------------------------------------------------------------------------
# Build a lookup from various name forms → canonical account ID.
# We support:
#   - exact account ID (ACME-001)
#   - full company name ("Acme Corp")
#   - first-word shorthand ("acme", "tyrell", "pied piper")
def _build_name_index():
    index = {}
    for aid, record in ACCOUNTS.items():
        name = record["name"]
        # exact ID
        index[aid.lower()] = aid
        # full name
        index[name.lower()] = aid
        # first word of name (e.g. "Acme" from "Acme Corp")
        first_word = name.split()[0].lower()
        # avoid clobbering — "Wayne" should map to Wayne Enterprises, not get overwritten
        if first_word not in index:
            index[first_word] = aid
        # bare prefix of the ID (e.g. "ACME" from "ACME-001")
        prefix = aid.split("-")[0].lower()
        if prefix not in index:
            index[prefix] = aid
    return index


NAME_INDEX = _build_name_index()


def resolve_account_names(text: str) -> str:
    """Replace company-name mentions in free-form text with their canonical IDs.

    Conservative: we only swap in IDs when the match is unambiguous and the user
    didn't already use the canonical ID. The original phrasing is preserved by
    appending the ID in parentheses, so the agent gets both context and the key.

    Example:
        "compare acme and tyrell" → "compare acme (ACME-001) and tyrell (TYRELL-008)"
    """
    if not text:
        return text

    # Skip if the text already contains an account ID pattern
    # (we don't want to double-tag things)
    has_explicit_id = bool(re.search(r"\b[A-Z]+-\d{3}\b", text))

    # Find candidate name mentions, longest-first so "pied piper" wins over "pied"
    candidates = sorted(NAME_INDEX.keys(), key=len, reverse=True)

    result = text
    already_inserted_ids = set()
    if has_explicit_id:
        # Capture IDs already in the text so we don't re-tag them
        already_inserted_ids = set(
            m.upper() for m in re.findall(r"\b[A-Z]+-\d{3}\b", text)
        )

    for name in candidates:
        # Skip very-short ambiguous names (1-2 chars) to avoid false positives
        if len(name) < 3:
            continue
        canonical_id = NAME_INDEX[name]
        if canonical_id in already_inserted_ids:
            continue
        # Match whole-word, case-insensitive
        pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
        if pattern.search(result):
            # Replace first occurrence only — append the ID in parens
            def _sub(m, _id=canonical_id):
                return f"{m.group(0)} ({_id})"
            new_result, n = pattern.subn(_sub, result, count=1)
            if n > 0:
                result = new_result
                already_inserted_ids.add(canonical_id)

    return result


# ---------------------------------------------------------------------------
# Health-breakdown formatter
# ---------------------------------------------------------------------------
def health_breakdown_query(account_id: str) -> str:
    """Build a query that asks the agent for a comprehensive health readout."""
    name = ACCOUNTS[account_id]["name"]
    return (
        f"Give me a comprehensive account health breakdown for {name} ({account_id}). "
        f"Cover: 1) headline health summary, 2) usage and adoption signals, "
        f"3) renewal timing and risk factors, 4) recommended next action. "
        f"Be thorough — this is for my QBR prep."
    )


# ---------------------------------------------------------------------------
# Badge helpers
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
      <h1 style='margin:0; font-family: Georgia, serif;'>⭐ Northstar</h1>
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

quick_plays = [
    ("🎯 Find opportunity", lambda: f"What's the biggest expansion opportunity in {account_id}?"),
    ("✉️ Full motion + email", lambda: f"Build a full upsell motion for {account_id} including outreach."),
    ("⚖️ Compare accounts", "compare_picker"),  # special — opens multiselect
    ("🤝 Co-term play", lambda: f"Should I push for a co-term renewal on {account_id}?"),
    ("📊 Health breakdown", lambda: health_breakdown_query(account_id)),
]

cols = st.columns(len(quick_plays))
for i, (label, action) in enumerate(quick_plays):
    if cols[i].button(label, use_container_width=True, key=f"quick_{i}"):
        if action == "compare_picker":
            st.session_state.compare_mode = True
            st.session_state.pending_query = None
        else:
            st.session_state.compare_mode = False
            st.session_state.pending_query = action()

# ---------------------------------------------------------------------------
# Compare-accounts picker (only shown after the compare button is clicked)
# ---------------------------------------------------------------------------
if st.session_state.compare_mode:
    st.markdown("---")
    st.markdown("##### ⚖️ Pick accounts to compare")
    st.caption("Select 2-4 accounts. The agent will analyze each and recommend where to focus.")

    compare_col, button_col = st.columns([3, 1])
    with compare_col:
        compare_selection = st.multiselect(
            "Accounts to compare",
            options=list(ACCOUNTS.keys()),
            default=[account_id],  # seed with the currently-selected sidebar account
            format_func=lambda aid: f"{ACCOUNTS[aid]['name']} ({aid})",
            max_selections=4,
            label_visibility="collapsed",
        )
    with button_col:
        run_compare = st.button(
            "Run comparison",
            type="primary",
            use_container_width=True,
            disabled=len(compare_selection) < 2,
        )
        if len(compare_selection) < 2:
            st.caption("Pick 2+")

    if run_compare and len(compare_selection) >= 2:
        names = [f"{ACCOUNTS[aid]['name']} ({aid})" for aid in compare_selection]
        joined = ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else names[0]
        st.session_state.pending_query = (
            f"Compare expansion potential across these accounts: {joined}. "
            f"For each, identify the biggest signal and quantify the opportunity. "
            f"Then recommend which one I should prioritize this week and why."
        )
        st.session_state.compare_mode = False  # collapse the picker once submitted

# ---------------------------------------------------------------------------
# Free-form input
# ---------------------------------------------------------------------------
user_query = st.chat_input("Ask the agent anything — by company name or account ID...")

# Resolve which query to actually run
raw_query: Optional[str] = st.session_state.pending_query or user_query

# Apply fuzzy resolution to free-form input only (quick plays already use IDs)
query: Optional[str] = None
if raw_query:
    if user_query and raw_query == user_query:
        # Free-form text — resolve company names to IDs
        query = resolve_account_names(raw_query)
    else:
        # Quick play or compare picker — already has IDs baked in
        query = raw_query
    # Clear the pending query so it doesn't re-run on rerender
    st.session_state.pending_query = None

# ---------------------------------------------------------------------------
# Agent streaming
# ---------------------------------------------------------------------------
def stream_agent_steps(agent, query: str) -> Iterable[dict]:
    """Yield events as the agent reasons through the problem."""
    last_message_count = 0

    for chunk in agent.stream(
        {"messages": [("user", query)]},
        stream_mode="values",
    ):
        messages = chunk.get("messages", [])
        new_messages = messages[last_message_count:]
        last_message_count = len(messages)

        for msg in new_messages:
            msg_type = type(msg).__name__

            if msg_type == "AIMessage":
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
    with st.chat_message("user"):
        # If we expanded the original input with IDs, show that to the user
        # so they can see what the agent actually received
        if user_query and query != user_query:
            st.markdown(query)
            st.caption(f"_(resolved from: \"{user_query}\")_")
        else:
            st.markdown(query)

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
# Empty state
# ---------------------------------------------------------------------------
if not st.session_state.history and not st.session_state.compare_mode:
    st.markdown(
        """
        <div style="text-align: center; padding: 40px; color: #94A3B8;">
          <p style="font-size: 15px;">👆 Pick an account from the sidebar, then click a quick play or type a question.</p>
          <p style="font-size: 13px;">You can refer to accounts by company name — "compare Acme and Tyrell" works.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
