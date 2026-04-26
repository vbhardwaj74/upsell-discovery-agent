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
#
# Generic SaaS SKUs:
#   - Core Platform        (base license)
#   - Advanced Analytics   (premium add-on, per-seat)
#   - AI Copilot           (newer add-on, per-seat)
#   - Integrations Suite   (per-seat add-on)
#   - Premium Support      (flat-fee add-on)
#   - Dedicated CSM        (flat-fee add-on)
#   - SSO + Audit Logs     (flat-fee add-on, enterprise compliance)

ACCOUNTS = {
    # --- Strong expansion signals (over-utilized) ---
    "ACME-001": {
        "name": "Acme Corp", "tier": "Enterprise", "arr": 240_000,
        "seats_purchased": 200, "industry": "Financial Services",
        "renewal_date": "2026-06-15", "health_score": 88,
    },
    "INITECH-003": {
        "name": "Initech", "tier": "Enterprise", "arr": 410_000,
        "seats_purchased": 300, "industry": "Technology",
        "renewal_date": "2026-04-30", "health_score": 94,
    },
    "STARK-007": {
        "name": "Stark Industries", "tier": "Enterprise", "arr": 580_000,
        "seats_purchased": 450, "industry": "Manufacturing",
        "renewal_date": "2026-08-20", "health_score": 91,
    },
    "WAYNE-012": {
        "name": "Wayne Enterprises", "tier": "Enterprise", "arr": 720_000,
        "seats_purchased": 600, "industry": "Conglomerate",
        "renewal_date": "2026-11-15", "health_score": 89,
    },
    "DUNDER-018": {
        "name": "Dunder Mifflin", "tier": "Mid-Market", "arr": 95_000,
        "seats_purchased": 80, "industry": "Distribution",
        "renewal_date": "2026-05-10", "health_score": 82,
    },

    # --- Dormant premium SKU (cross-sell opportunity) ---
    "GLOBEX-002": {
        "name": "Globex Industries", "tier": "Mid-Market", "arr": 85_000,
        "seats_purchased": 75, "industry": "Manufacturing",
        "renewal_date": "2026-09-01", "health_score": 72,
    },
    "OSCORP-009": {
        "name": "Oscorp Holdings", "tier": "Mid-Market", "arr": 125_000,
        "seats_purchased": 100, "industry": "Biotech",
        "renewal_date": "2026-07-22", "health_score": 78,
    },
    "PIEDPIPER-014": {
        "name": "Pied Piper", "tier": "Mid-Market", "arr": 68_000,
        "seats_purchased": 55, "industry": "Technology",
        "renewal_date": "2026-10-05", "health_score": 80,
    },
    "HOOLI-022": {
        "name": "Hooli", "tier": "Enterprise", "arr": 340_000,
        "seats_purchased": 280, "industry": "Technology",
        "renewal_date": "2026-12-01", "health_score": 85,
    },

    # --- Strong growth trend (preemptive upsell) ---
    "MASSIVE-005": {
        "name": "Massive Dynamic", "tier": "Mid-Market", "arr": 145_000,
        "seats_purchased": 120, "industry": "Research",
        "renewal_date": "2026-09-30", "health_score": 90,
    },
    "GEKKO-016": {
        "name": "Gekko Capital", "tier": "Mid-Market", "arr": 110_000,
        "seats_purchased": 90, "industry": "Financial Services",
        "renewal_date": "2026-11-30", "health_score": 86,
    },
    "VANDELAY-020": {
        "name": "Vandelay Industries", "tier": "SMB", "arr": 42_000,
        "seats_purchased": 40, "industry": "Import/Export",
        "renewal_date": "2026-06-01", "health_score": 84,
    },

    # --- At-risk renewals (retention before expansion) ---
    "TYRELL-008": {
        "name": "Tyrell Corporation", "tier": "Enterprise", "arr": 380_000,
        "seats_purchased": 320, "industry": "Technology",
        "renewal_date": "2026-05-25", "health_score": 58,
    },
    "WONKA-013": {
        "name": "Wonka Industries", "tier": "Mid-Market", "arr": 78_000,
        "seats_purchased": 70, "industry": "Consumer Goods",
        "renewal_date": "2026-05-15", "health_score": 54,
    },
    "SOYLENT-019": {
        "name": "Soylent Corp", "tier": "Mid-Market", "arr": 92_000,
        "seats_purchased": 85, "industry": "Food & Beverage",
        "renewal_date": "2026-04-20", "health_score": 49,
    },

    # --- Healthy but flat (renewal-timing play) ---
    "CYBERDYNE-006": {
        "name": "Cyberdyne Systems", "tier": "Enterprise", "arr": 295_000,
        "seats_purchased": 250, "industry": "Defense",
        "renewal_date": "2026-08-15", "health_score": 87,
    },
    "PRESTIGE-011": {
        "name": "Prestige Worldwide", "tier": "SMB", "arr": 38_000,
        "seats_purchased": 35, "industry": "Entertainment",
        "renewal_date": "2026-07-08", "health_score": 81,
    },
    "BLUTH-015": {
        "name": "Bluth Company", "tier": "SMB", "arr": 28_000,
        "seats_purchased": 25, "industry": "Real Estate",
        "renewal_date": "2026-06-30", "health_score": 79,
    },

    # --- Compliance-driven enterprise upsell ---
    "UMBRELLA-004": {
        "name": "Umbrella Pharmaceuticals", "tier": "Mid-Market", "arr": 165_000,
        "seats_purchased": 140, "industry": "Pharmaceuticals",
        "renewal_date": "2026-10-15", "health_score": 83,
    },
    "WEYLAND-010": {
        "name": "Weyland Logistics", "tier": "Mid-Market", "arr": 135_000,
        "seats_purchased": 110, "industry": "Logistics",
        "renewal_date": "2026-09-12", "health_score": 84,
    },

    # --- Low adoption / recovery play ---
    "NAKATOMI-017": {
        "name": "Nakatomi Trading", "tier": "Mid-Market", "arr": 98_000,
        "seats_purchased": 90, "industry": "Trading",
        "renewal_date": "2026-12-20", "health_score": 65,
    },
    "MONARCH-021": {
        "name": "Monarch Solutions", "tier": "Enterprise", "arr": 215_000,
        "seats_purchased": 200, "industry": "Consulting",
        "renewal_date": "2026-11-08", "health_score": 70,
    },
    "BLACKMESA-023": {
        "name": "Black Mesa Research", "tier": "Mid-Market", "arr": 88_000,
        "seats_purchased": 80, "industry": "Research",
        "renewal_date": "2026-07-15", "health_score": 68,
    },
}

USAGE = {
    # --- Strong over-utilization (seat true-up plays) ---
    "ACME-001": {
        "active_seats_30d": 247, "license_utilization_pct": 124,
        "feature_adoption": {"core_platform": 0.96, "advanced_analytics": 0.41, "ai_copilot": 0.12, "integrations_suite": 0.55},
        "trend_30d": "+18% MAU", "support_tickets_open": 2,
    },
    "INITECH-003": {
        "active_seats_30d": 412, "license_utilization_pct": 137,
        "feature_adoption": {"core_platform": 0.99, "advanced_analytics": 0.88, "ai_copilot": 0.71, "integrations_suite": 0.82},
        "trend_30d": "+9% MAU", "support_tickets_open": 0,
    },
    "STARK-007": {
        "active_seats_30d": 540, "license_utilization_pct": 120,
        "feature_adoption": {"core_platform": 0.94, "advanced_analytics": 0.62, "ai_copilot": 0.38, "integrations_suite": 0.71},
        "trend_30d": "+22% MAU", "support_tickets_open": 1,
    },
    "WAYNE-012": {
        "active_seats_30d": 798, "license_utilization_pct": 133,
        "feature_adoption": {"core_platform": 0.97, "advanced_analytics": 0.74, "ai_copilot": 0.21, "integrations_suite": 0.65},
        "trend_30d": "+14% MAU", "support_tickets_open": 3,
    },
    "DUNDER-018": {
        "active_seats_30d": 96, "license_utilization_pct": 120,
        "feature_adoption": {"core_platform": 0.92, "advanced_analytics": 0.18, "ai_copilot": 0.05, "integrations_suite": 0.31},
        "trend_30d": "+11% MAU", "support_tickets_open": 1,
    },

    # --- Dormant premium SKU ---
    "GLOBEX-002": {
        "active_seats_30d": 64, "license_utilization_pct": 85,
        "feature_adoption": {"core_platform": 0.78, "advanced_analytics": 0.0, "ai_copilot": 0.0, "integrations_suite": 0.12},
        "trend_30d": "+4% MAU", "support_tickets_open": 1,
    },
    "OSCORP-009": {
        "active_seats_30d": 88, "license_utilization_pct": 88,
        "feature_adoption": {"core_platform": 0.84, "advanced_analytics": 0.05, "ai_copilot": 0.0, "integrations_suite": 0.22},
        "trend_30d": "+3% MAU", "support_tickets_open": 0,
    },
    "PIEDPIPER-014": {
        "active_seats_30d": 52, "license_utilization_pct": 95,
        "feature_adoption": {"core_platform": 0.91, "advanced_analytics": 0.08, "ai_copilot": 0.0, "integrations_suite": 0.45},
        "trend_30d": "+6% MAU", "support_tickets_open": 0,
    },
    "HOOLI-022": {
        "active_seats_30d": 261, "license_utilization_pct": 93,
        "feature_adoption": {"core_platform": 0.89, "advanced_analytics": 0.12, "ai_copilot": 0.0, "integrations_suite": 0.51},
        "trend_30d": "+5% MAU", "support_tickets_open": 1,
    },

    # --- Strong growth (preemptive expansion) ---
    "MASSIVE-005": {
        "active_seats_30d": 118, "license_utilization_pct": 98,
        "feature_adoption": {"core_platform": 0.95, "advanced_analytics": 0.55, "ai_copilot": 0.42, "integrations_suite": 0.68},
        "trend_30d": "+27% MAU", "support_tickets_open": 0,
    },
    "GEKKO-016": {
        "active_seats_30d": 89, "license_utilization_pct": 99,
        "feature_adoption": {"core_platform": 0.93, "advanced_analytics": 0.71, "ai_copilot": 0.18, "integrations_suite": 0.40},
        "trend_30d": "+24% MAU", "support_tickets_open": 0,
    },
    "VANDELAY-020": {
        "active_seats_30d": 39, "license_utilization_pct": 98,
        "feature_adoption": {"core_platform": 0.88, "advanced_analytics": 0.32, "ai_copilot": 0.10, "integrations_suite": 0.25},
        "trend_30d": "+31% MAU", "support_tickets_open": 0,
    },

    # --- At-risk (retention before expansion) ---
    "TYRELL-008": {
        "active_seats_30d": 195, "license_utilization_pct": 61,
        "feature_adoption": {"core_platform": 0.58, "advanced_analytics": 0.22, "ai_copilot": 0.05, "integrations_suite": 0.18},
        "trend_30d": "-12% MAU", "support_tickets_open": 8,
    },
    "WONKA-013": {
        "active_seats_30d": 38, "license_utilization_pct": 54,
        "feature_adoption": {"core_platform": 0.51, "advanced_analytics": 0.0, "ai_copilot": 0.0, "integrations_suite": 0.08},
        "trend_30d": "-18% MAU", "support_tickets_open": 5,
    },
    "SOYLENT-019": {
        "active_seats_30d": 41, "license_utilization_pct": 48,
        "feature_adoption": {"core_platform": 0.45, "advanced_analytics": 0.0, "ai_copilot": 0.0, "integrations_suite": 0.05},
        "trend_30d": "-22% MAU", "support_tickets_open": 9,
    },

    # --- Healthy flat (renewal-timing) ---
    "CYBERDYNE-006": {
        "active_seats_30d": 240, "license_utilization_pct": 96,
        "feature_adoption": {"core_platform": 0.92, "advanced_analytics": 0.65, "ai_copilot": 0.30, "integrations_suite": 0.58},
        "trend_30d": "+2% MAU", "support_tickets_open": 1,
    },
    "PRESTIGE-011": {
        "active_seats_30d": 33, "license_utilization_pct": 94,
        "feature_adoption": {"core_platform": 0.89, "advanced_analytics": 0.40, "ai_copilot": 0.14, "integrations_suite": 0.30},
        "trend_30d": "+1% MAU", "support_tickets_open": 0,
    },
    "BLUTH-015": {
        "active_seats_30d": 23, "license_utilization_pct": 92,
        "feature_adoption": {"core_platform": 0.86, "advanced_analytics": 0.25, "ai_copilot": 0.0, "integrations_suite": 0.38},
        "trend_30d": "+3% MAU", "support_tickets_open": 1,
    },

    # --- Compliance-driven (SSO/audit upsell) ---
    "UMBRELLA-004": {
        "active_seats_30d": 132, "license_utilization_pct": 94,
        "feature_adoption": {"core_platform": 0.91, "advanced_analytics": 0.48, "ai_copilot": 0.22, "integrations_suite": 0.55},
        "trend_30d": "+7% MAU", "support_tickets_open": 1,
    },
    "WEYLAND-010": {
        "active_seats_30d": 105, "license_utilization_pct": 95,
        "feature_adoption": {"core_platform": 0.93, "advanced_analytics": 0.52, "ai_copilot": 0.25, "integrations_suite": 0.62},
        "trend_30d": "+8% MAU", "support_tickets_open": 0,
    },

    # --- Low adoption (recovery play) ---
    "NAKATOMI-017": {
        "active_seats_30d": 54, "license_utilization_pct": 60,
        "feature_adoption": {"core_platform": 0.62, "advanced_analytics": 0.10, "ai_copilot": 0.0, "integrations_suite": 0.15},
        "trend_30d": "-3% MAU", "support_tickets_open": 4,
    },
    "MONARCH-021": {
        "active_seats_30d": 130, "license_utilization_pct": 65,
        "feature_adoption": {"core_platform": 0.68, "advanced_analytics": 0.20, "ai_copilot": 0.05, "integrations_suite": 0.25},
        "trend_30d": "-5% MAU", "support_tickets_open": 6,
    },
    "BLACKMESA-023": {
        "active_seats_30d": 56, "license_utilization_pct": 70,
        "feature_adoption": {"core_platform": 0.71, "advanced_analytics": 0.15, "ai_copilot": 0.0, "integrations_suite": 0.20},
        "trend_30d": "-7% MAU", "support_tickets_open": 3,
    },
}

# Standard add-on catalog — most accounts have access to similar SKUs
DEFAULT_ADDONS = [
    {"sku": "Advanced Analytics", "list_price_per_seat": 180},
    {"sku": "AI Copilot", "list_price_per_seat": 95},
    {"sku": "Integrations Suite", "list_price_per_seat": 65},
    {"sku": "Premium Support", "list_price_flat": 25_000},
    {"sku": "Dedicated CSM", "list_price_flat": 40_000},
    {"sku": "SSO + Audit Logs", "list_price_flat": 18_000},
]

ENTERPRISE_ADDONS = DEFAULT_ADDONS + [
    {"sku": "Advanced Threat Detection", "list_price_per_seat": 220},
    {"sku": "Custom Data Residency", "list_price_flat": 60_000},
]


def _build_billing(account_id: str) -> dict:
    """Generate billing info based on account tier and current adoption."""
    acct = ACCOUNTS[account_id]
    usage = USAGE[account_id]
    adoption = usage["feature_adoption"]

    # Current SKUs = whatever has meaningful adoption (>5%)
    current_skus = ["Core Platform"]
    if adoption.get("advanced_analytics", 0) > 0.05:
        current_skus.append("Advanced Analytics")
    if adoption.get("ai_copilot", 0) > 0.05:
        current_skus.append("AI Copilot")
    if adoption.get("integrations_suite", 0) > 0.20:
        current_skus.append("Integrations Suite")

    # Build available add-ons (anything they don't already have)
    catalog = ENTERPRISE_ADDONS if acct["tier"] == "Enterprise" else DEFAULT_ADDONS
    available = [a for a in catalog if a["sku"] not in current_skus]

    # Overage rates scale with tier
    overage = {"Enterprise": 1_400, "Mid-Market": 950, "SMB": 720}[acct["tier"]]

    return {
        "current_skus": current_skus,
        "available_addons": available,
        "overage_rate_per_seat": overage,
    }


BILLING = {aid: _build_billing(aid) for aid in ACCOUNTS}


# --- Tools ------------------------------------------------------------------

@tool
def get_account_overview(account_id: str) -> dict:
    """Pull a customer's CRM record: name, tier, ARR, seats, renewal date, health score.
    Use this first when a CSM asks about an account."""
    record = ACCOUNTS.get(account_id.upper())
    if not record:
        return {"error": f"Account {account_id} not found. Available accounts: {', '.join(sorted(ACCOUNTS.keys()))}"}
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
1. Pull the account overview to understand context (tier, ARR, renewal timing, health).
2. Pull product usage to find expansion signals — over-utilization, under-adopted premium
   features, strong growth trends. Flag retention risk if health is low or trend is negative.
3. Pull billing to see current SKUs and what add-ons are available.
4. Use calculate_expansion_opportunity to quantify the opportunity in dollars.
5. Draft an outreach email tying it together.

Special cases:
- If health score is below 65 OR trend is negative, lead with retention risk before
  recommending expansion. Don't push more product on an at-risk account.
- If utilization is over 110%, prioritize a seat true-up — it's the easiest close.
- If a premium SKU has under 10% adoption, that's a recovery play, not an upsell.

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
