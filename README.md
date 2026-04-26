# Upsell Discovery Agent

A LangGraph agent that helps Enterprise CSMs identify expansion opportunities. Built for an Arize AI interview demo.

## What it does

Takes an account ID, investigates it across mock CRM / product usage / billing systems, and returns a quantified expansion thesis with a draft outreach email — fully traced in Arize Phoenix.

## Tech stack

- **LangGraph** (`create_react_agent`) — agent orchestration
- **OpenAI gpt-4o-mini** — reasoning model
- **Arize Phoenix + OpenInference** — distributed tracing
- **Mock data layer** — deterministic CRM/usage/billing tools so the demo is reproducible

## Setup

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add your OPENAI_API_KEY

# 3. Start Phoenix locally (separate terminal)
python -m phoenix.server.main serve
# UI at http://localhost:6006

# 4. Run the agent
python agent.py "Find expansion opportunities in ACME-001 and draft outreach"
```

## Demo accounts

| Account ID | Profile | Expected play |
|---|---|---|
| `ACME-001` | Mid-Enterprise, 124% license utilization, low Insider Threat adoption | Seat true-up + Insider Threat attach |
| `GLOBEX-002` | Mid-Market, healthy but flat, no premium SKUs | Insider Threat upsell on renewal |
| `INITECH-003` | Top-tier, 137% utilization, full SKU stack | Aggressive seat true-up + Advanced Threat Hunting |

## Demo script

```bash
# Run 1 — clean expansion case
python agent.py "What's the expansion opportunity in ACME-001?"

# Run 2 — show the agent comparing options
python agent.py "Compare expansion potential between ACME-001 and GLOBEX-002"

# Run 3 — show full output with email draft
python agent.py "Build me a full upsell motion for INITECH-003 including outreach"
```

Then open Phoenix at `http://localhost:6006` and walk through:
- The full agent trace tree (graph → LLM → tool → LLM → tool → ...)
- Token counts and latency per span
- Tool call inputs and outputs at every step
- Where the model decided to stop investigating

## File map

- `agent.py` — full app: tracing setup, mock data, tools, agent definition, CLI
- `requirements.txt` — pinned deps
- `.env.example` — config template
