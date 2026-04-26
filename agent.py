"""
Upsell Discovery Agent
======================
A LangGraph ReAct agent that helps Customer Success Managers identify
expansion opportunities across their book of business.

Tech stack:
- LangGraph (create_react_agent) for the agent loop
- OpenAI gpt-4o-mini as the reasoning model
- Arize Phoenix + OpenInference for tracing
- Mock CRM/usage/billing tools (deterministic, demo-friendly)
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# --- Phoenix tracing setup --------------------------------------------------
# This auto-instruments every LangChain / LangGraph call: graph nodes,
# LLM invocations, tool calls — all with proper parent/child span nesting.
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

load_dotenv()

tracer_provider = register(
    project_name="upsell-discovery-agent",
    auto_instrument=False,  # we instrument explicitly below
    protocol="http/protobuf",
)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)


# --- Mock data --------------------------------------------------------------
# Pretend this is a CRM + product-usage warehouse. In production, these tools
# would hit Salesforce, Snowflake, Stripe, etc.

ACCOUNTS = {
    "ACME-001": {
        "name": "Acme Corp",
        "tier": "Enterprise",
        "arr": 240_000,
        "seats_purchased": 200,
        "csm": "Vishaal",
        "industry": "Financial Services",
        "renewal_date": "2026-06-15",
        "health_score": 88,
    },
    "GLOBEX-002": {
        "name": "Globex Industries",
        "tier": "Mid-Market",
        "arr": 85_000,
        "seats_purchased": 75,
        "csm": "Vishaal",
        "industry": "Manufacturing",
        "renewal_date": "2026-09-01",
        "health_score": 72,
    },
    "INITECH-003": {
        "name": "Initech",
        "tier": "Enterprise",
        "arr": 410_000,
        "seats_purchased": 300,
        "csm": "Vishaal",
        "industry": "Technology",
        "renewal_date": "2026-04-30",
        "health_score": 94,
    },
}

USAGE = {
    "ACME-001": {
        "active_seats_30d": 247,            # 23% over license
        "license_utilization_pct": 124,
        "feature_adoption": {
            "core_dlp": 0.96,
            "insider_threat": 0.41,         # premium SKU — under-adopted
            "ai_assistant": 0.12,           # new SKU — green field
        },
        "trend_30d": "+18% MAU",
        "support_tickets_open": 2,
    },
    "GLOBEX-002": {
        "active_seats_30d": 64,
        "license_utilization_pct": 85,
        "feature_adoption": {
            "core_dlp": 0.78,
            "insider_threat": 0.0,
            "ai_assistant": 0.0,
        },
        "trend_30d": "+4% MAU",
        "support_tickets_open": 1,
    },
    "INITECH-003": {
        "active_seats_30d": 412,            # 37% over license
        "license_utilization_pct": 137,
        "feature_adoption": {
            "core_dlp": 0.99,
            "insider_threat": 0.88,
            "ai_assistant": 0.71,
        },
        "trend_30d": "+9% MAU",
        "support_tickets_open": 0,
    },
}

BILLING = {
    "ACME-001": {
        "current_skus": ["Core DLP"],
        "available_addons": [
            {"sku": "Insider Threat Module", "list_price_per_seat": 180},
            {"sku": "AI Assistant", "list_price_per_seat": 95},
            {"sku": "Premium Support", "list_price_flat": 25_000},
        ],
        "overage_rate_per_seat": 1_200,
    },
    "GLOBEX-002": {
        "current_skus": ["Core DLP"],
        "available_addons": [
            {"sku": "Insider Threat Module", "list_price_per_seat": 180},
            {"sku": "AI Assistant", "list_price_per_seat": 95},
        ],
        "overage_rate_per_seat": 950,
    },
    "INITECH-003": {
        "current_skus": ["Core DLP", "Insider Threat Module", "AI Assistant"],
        "available_addons": [
            {"sku": "Premium Support", "list_price_flat": 50_000},
            {"sku": "Advanced Threat Hunting", "list_price_per_seat": 220},
        ],
        "overage_rate_per_seat": 1_400,
    },
}


# --- Tools ------------------------------------------------------------------

@tool
def get_account_overview(account_id: str) -> dict:
    """Pull a customer's CRM record: name, tier, ARR, seats, renewal date, health score.
    Use this first when a CSM asks about an account."""
    record = ACCOUNTS.get(account_id.upper())
    if not record:
        return {"error": f"Account {account_id} not found. Try ACME-001, GLOBEX-002, or INITECH-003."}
    return {"account_id": account_id.upper(), **record}


@tool
def get_product_usage(account_id: str) -> dict:
    """Fetch product telemetry: active seats, license utilization, feature adoption,
    usage trend. Use this to spot over-utilization or under-adopted premium features."""
    usage = USAGE.get(account_id.upper())
    if not usage:
        return {"error": f"No usage data for {account_id}."}
    return {"account_id": account_id.upper(), **usage}


@tool
def get_billing_info(account_id: str) -> dict:
    """Get current SKUs, available add-ons with list pricing, and per-seat overage rates.
    Use this once you've identified an expansion signal to size the opportunity."""
    billing = BILLING.get(account_id.upper())
    if not billing:
        return {"error": f"No billing data for {account_id}."}
    return {"account_id": account_id.upper(), **billing}


@tool
def calculate_expansion_opportunity(
    account_id: str,
    seat_overage: int = 0,
    addon_sku: Optional[str] = None,
    addon_seat_count: int = 0,
) -> dict:
    """Quantify a potential expansion deal. Pass the seat overage you observed and/or
    a specific add-on SKU you want to attach. Returns annualized ARR uplift."""
    billing = BILLING.get(account_id.upper())
    if not billing:
        return {"error": f"No billing data for {account_id}."}

    components = []
    total = 0

    if seat_overage > 0:
        seat_uplift = seat_overage * billing["overage_rate_per_seat"]
        components.append({
            "type": "seat_true_up",
            "seats": seat_overage,
            "rate_per_seat": billing["overage_rate_per_seat"],
            "annual_uplift": seat_uplift,
        })
        total += seat_uplift

    if addon_sku:
        addon = next(
            (a for a in billing["available_addons"] if a["sku"].lower() == addon_sku.lower()),
            None,
        )
        if addon:
            if "list_price_per_seat" in addon:
                addon_uplift = addon["list_price_per_seat"] * (addon_seat_count or 0)
            else:
                addon_uplift = addon["list_price_flat"]
            components.append({
                "type": "addon_attach",
                "sku": addon["sku"],
                "annual_uplift": addon_uplift,
            })
            total += addon_uplift

    return {
        "account_id": account_id.upper(),
        "components": components,
        "total_annual_uplift": total,
    }


@tool
def draft_outreach_email(
    account_id: str,
    expansion_summary: str,
    annual_uplift: int,
) -> dict:
    """Draft a tailored outreach email to the customer's economic buyer based on the
    expansion thesis you've built. Use this as the FINAL step once you have a
    quantified opportunity."""
    record = ACCOUNTS.get(account_id.upper(), {})
    name = record.get("name", account_id)

    subject = f"Quick thought on {name}'s {record.get('tier', '').lower()} footprint"
    body = (
        f"Hi {{first_name}},\n\n"
        f"Reviewing {name}'s usage trends ahead of our next QBR and wanted to flag "
        f"something I think your team will want visibility into.\n\n"
        f"{expansion_summary}\n\n"
        f"Rough sizing: this lands around ${annual_uplift:,} in annualized value, "
        f"and we can structure it as a co-term to keep procurement clean.\n\n"
        f"Open to a 20-min walkthrough next week? Happy to bring product on the call.\n\n"
        f"Best,\nVishaal"
    )
    return {"account_id": account_id.upper(), "subject": subject, "body": body}


# --- Build the agent --------------------------------------------------------

TOOLS = [
    get_account_overview,
    get_product_usage,
    get_billing_info,
    calculate_expansion_opportunity,
    draft_outreach_email,
]

SYSTEM_PROMPT = """You are an Upsell Discovery copilot for an Enterprise Customer Success Manager.

Your job: take an account ID, investigate it, and return a quantified expansion thesis.

Workflow:
1. Pull the account overview to understand context (tier, ARR, renewal timing).
2. Pull product usage to find expansion signals — over-utilization, under-adopted premium
   features, strong growth trends.
3. Pull billing to see current SKUs and what add-ons are available.
4. Use calculate_expansion_opportunity to quantify the opportunity in dollars.
5. Draft an outreach email tying it together.

Be concise in your reasoning. Always end with a quantified recommendation:
the SKU(s) to push, the seat count, the dollar uplift, and the renewal-timing play.
"""


def build_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


# --- CLI entry point --------------------------------------------------------

if __name__ == "__main__":
    import sys

    agent = build_agent()
    query = " ".join(sys.argv[1:]) or (
        "Find me the biggest expansion opportunity in account ACME-001 and draft outreach."
    )

    print(f"\n>> {query}\n")
    print("-" * 70)

    result = agent.invoke({"messages": [("user", query)]})
    final = result["messages"][-1].content
    print(final)
    print("-" * 70)
    print(f"\nView traces: {os.getenv('PHOENIX_COLLECTOR_ENDPOINT', 'http://localhost:6006')}")
