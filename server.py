"""A complete local MCP server: support case tools backed by one JSON file."""

from datetime import date
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sys

from mcp.server import MCPServer

mcp = MCPServer("Support cases")
CASES = Path(__file__).with_name("Supportcase.json")
OWNERS = Path(__file__).with_name("product_owners.json")


@mcp.tool()
def get_case(case_id: str) -> dict:
    """Look up a customer support case's metadata by its case ID.

    Returns customer name, product name, created date, title, description, case owner,
    case status, and resolution. Use the case ID, for example 'CS-1003'.
    """
    # Read on every call: edits to Supportcase.json take effect immediately.
    try:
        cases = json.loads(CASES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"error": "Cannot read support cases. Check Supportcase.json is valid JSON."}
    for case, details in cases.items():
        if case.casefold() == case_id.strip().casefold():
            return {"case_id": case, **details}
    return {"error": "Case not found", "case_id": case_id}


@mcp.tool()
def find_cases_by_owner(owner: str) -> dict:
    """List the support cases assigned to a case owner.

    Use the owner's full or partial name, for example 'Priya Nair' or 'Priya'.
    Returns each case's ID, customer name, product name, title, created date, and status; use
    get_case for a case's full description.
    """
    try:
        cases = json.loads(CASES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"error": "Cannot read support cases. Check Supportcase.json is valid JSON."}
    wanted = owner.strip().casefold()
    if not wanted:
        return {"error": "Owner name is required"}
    matches = [
        {"case_id": case, "customer_name": details.get("customer_name"),
         "case_owner": details["case_owner"],
         "product_name": details["product_name"], "title": details["title"],
         "created_date": details["created_date"], "case_status": details["case_status"]}
        for case, details in cases.items()
        if wanted in details.get("case_owner", "").casefold()
    ]
    if not matches:
        return {"error": "No cases found for this owner", "owner": owner}
    return {"owner": owner, "count": len(matches), "cases": matches}


@mcp.tool()
def find_cases_by_customer(customer: str) -> dict:
    """List the support cases for a customer (company).

    Use the company's full or partial name, for example 'Fabrikam Inc' or 'Fabrikam'.
    Returns each case's ID, customer name, owner, product name, title, created date,
    and status; use get_case for a case's full description.
    """
    try:
        cases = json.loads(CASES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"error": "Cannot read support cases. Check Supportcase.json is valid JSON."}
    wanted = customer.strip().casefold()
    if not wanted:
        return {"error": "Customer name is required"}
    matches = [
        {"case_id": case, "customer_name": details.get("customer_name"),
         "case_owner": details.get("case_owner"),
         "product_name": details.get("product_name"), "title": details.get("title"),
         "created_date": details.get("created_date"), "case_status": details.get("case_status")}
        for case, details in cases.items()
        if wanted in details.get("customer_name", "").casefold()
    ]
    if not matches:
        return {"error": "No cases found for this customer", "customer": customer}
    return {"customer": customer, "count": len(matches), "cases": matches}


@mcp.tool()
def list_cases() -> dict:
    """List all support cases.

    Returns each case's ID, customer name, owner, product name, title, created date, and status;
    use get_case for a case's full description and resolution.
    """
    try:
        cases = json.loads(CASES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"error": "Cannot read support cases. Check Supportcase.json is valid JSON."}
    summaries = [
        {"case_id": case, "customer_name": details.get("customer_name"),
         "case_owner": details["case_owner"],
         "product_name": details["product_name"], "title": details["title"],
         "created_date": details["created_date"], "case_status": details["case_status"]}
        for case, details in cases.items()
    ]
    return {"count": len(summaries), "cases": summaries}


CASE_STATUSES = ["New", "In Progress", "Waiting on Customer", "Escalated", "Resolved", "Closed"]


@mcp.tool()
def update_case_status(case_id: str, status: str) -> dict:
    """Change a support case's status.

    Use the case ID, for example 'CS-1003', and one of these statuses: New,
    In Progress, Waiting on Customer, Escalated, Resolved, Closed. Use ONLY when
    the user explicitly asks to change a case's status.
    """
    try:
        cases = json.loads(CASES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"error": "Cannot read support cases. Check Supportcase.json is valid JSON."}
    new_status = next((s for s in CASE_STATUSES if s.casefold() == status.strip().casefold()), None)
    if new_status is None:
        return {"error": "Invalid status", "status": status, "allowed": CASE_STATUSES}
    for case, details in cases.items():
        if case.casefold() == case_id.strip().casefold():
            old_status = details["case_status"]
            details["case_status"] = new_status
            CASES.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
            return {"case_id": case, "old_status": old_status, "new_status": new_status}
    return {"error": "Case not found", "case_id": case_id}


@mcp.tool()
def create_case(customer_name: str = "", product_name: str = "", title: str = "",
                description: str = "", case_owner: str = "") -> dict:
    """Create a new support case, choose its product, and assign its owner.

    Required: customer_name, title, and description. Pass only values the user
    actually gave; never invent or guess one. If any are missing, this tool returns
    their names instead of creating the case: ask the user for them.

    product_name: the product the user chose, ideally one from list_product_owners.
    Leave it empty when the user wants the product picked automatically; it is then
    detected from the title and description. If no product can be detected, the tool
    returns the product choices instead of creating the case: ask the user to pick one.

    case_owner is optional. Leave it empty unless the user names an owner: the case
    is then assigned to the support agent who owns the product. The case ID, created
    date, status (New), and resolution (none) are set automatically.
    """
    fields = {"customer_name": customer_name, "title": title, "description": description}
    missing = [name for name, value in fields.items() if not value.strip()]
    if missing:
        return {"error": "Missing required fields", "missing": missing}
    try:
        owners = json.loads(OWNERS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"error": "Cannot read product owners. Check product_owners.json is valid JSON."}
    choices = {name: info["owner"] for name, info in owners.items()}

    if product_name.strip():
        match = classify_product(owners, product_name)
        product = product_name.strip()
        product_selection = "Chosen by the user."
    else:
        match = classify_product(owners, f"{title} {description}")
        if match is None:
            return {"error": "Could not detect the product from the title and description",
                    "missing": ["product_name"], "product_choices": choices}
        product = match[0]
        product_selection = (f"Auto-detected from the title and description: {product} "
                             f"(matched {', '.join(match[2])}).")

    if case_owner.strip():
        assignment = "Owner given by the user."
    elif match is None:
        return {"error": "No product owner matches this product", "product_name": product,
                "missing": ["case_owner"], "product_choices": choices}
    else:
        case_owner = match[1]["owner"]
        assignment = f"Auto-assigned: {case_owner} owns {match[0]}."

    try:
        cases = json.loads(CASES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"error": "Cannot read support cases. Check Supportcase.json is valid JSON."}
    numbers = [int(case.split("-")[1]) for case in cases if case.split("-")[-1].isdigit()]
    case_id = f"CS-{max(numbers, default=1000) + 1}"
    # Same field order as the existing cases in Supportcase.json.
    cases[case_id] = {
        "customer_name": customer_name.strip(),
        "product_name": product,
        "created_date": date.today().isoformat(),
        "title": title.strip(),
        "description": description.strip(),
        "case_owner": case_owner.strip(),
        "case_status": "New",
        "resolution": None,
    }
    CASES.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    return {"case_id": case_id, **cases[case_id],
            "product_selection": product_selection, "owner_assignment": assignment}


def classify_product(owners: dict, text: str):
    """Return (product, info, matched keywords) for the product that best fits text, or None.

    Keyword scoring for now: the product with the most keyword hits wins, and ties go
    to the first product in product_owners.json. To move to RAG later, replace this
    body with a retrieval lookup (for example, embed past cases and product docs and
    return the closest product) and keep the same signature, so create_case is unchanged.
    """
    text = text.casefold()
    best = None
    for product, info in owners.items():
        keywords = {k.casefold() for k in [product, *info.get("keywords", [])]}
        hits = sorted(k for k in keywords if k in text)
        if hits and (best is None or len(hits) > len(best[2])):
            best = (product, info, hits)
    return best


@mcp.tool()
def list_product_owners() -> dict:
    """List each product and the support agent who owns it.

    Use for questions such as 'who owns Billing?' or to explain why a new case was
    assigned to someone.
    """
    try:
        owners = json.loads(OWNERS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"error": "Cannot read product owners. Check product_owners.json is valid JSON."}
    return {"products": {name: info["owner"] for name, info in owners.items()}}

SEARCH_FIELDS = ("title", "description")  # The only fields keyword search looks at.


@mcp.tool()
def search_knowledge_by_keyword(query: str, match: str = "", exclude_case_id: str = "",
                                min_matches: int = 1) -> dict:
    """Search the title and description of all support cases; return matching case IDs.

    Only the title and description are searched, ignoring case, so use it for the
    problem itself (for example 'PDF export' or 'password reset'), not for a customer,
    owner, or status.

    match="all": every word in query must appear (the default). match="any": a case matches
    if it contains at least one word, and results are ranked by how many words it
    contains; use this to find cases similar to a case, passing that case's key words
    and its ID as exclude_case_id, and min_matches=2 so a single shared word such as
    'error' does not count. When exclude_case_id is given, match defaults to "any".
    Use get_case for a match's details.
    """
    words = list(dict.fromkeys(query.casefold().split()))  # Unique, in order.
    if not words:
        return {"status": "error", "error": "A search keyword is required"}
    # A search that excludes a case is a similar-case search, so it matches any word.
    match = match or ("any" if exclude_case_id.strip() else "all")
    if match not in ("all", "any"):
        return {"status": "error", "error": "match must be 'all' or 'any'"}
    try:
        cases = json.loads(CASES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "error",
                "error": "Cannot read support cases. Check Supportcase.json is valid JSON."}
    matches = []
    for case_id, details in cases.items():
        if case_id.casefold() == exclude_case_id.strip().casefold():
            continue
        fields = {name: details.get(name) for name in SEARCH_FIELDS}
        text = " ".join(str(value) for value in fields.values() if value).casefold()
        found = [word for word in words if word in text]
        if len(found) >= max(1, min_matches) and (match == "any" or len(found) == len(words)):
            matched = [name for name, value in fields.items()
                       if value and any(word in str(value).casefold() for word in found)]
            matches.append({"case_id": case_id, "title": details.get("title"),
                            "customer_name": details.get("customer_name"),
                            "matched_words": found, "match_count": len(found),
                            "matched_fields": matched})
    # Most matching words first; ties keep case order.
    matches.sort(key=lambda m: m["match_count"], reverse=True)
    # stdout carries MCP messages, so log to stderr.
    print(f"Found {len(matches)} matches for '{query}' (match={match})", file=sys.stderr)
    return {"status": "ok", "query": query, "match": match, "count": len(matches),
            "case_ids": [m["case_id"] for m in matches], "matches": matches}

STOPWORDS = set("""a an the and or but if to of in on at for from by with without is are was
were be been being it its this that these those i we you they he she my our your their
me us them when what which who how not no can cannot can't does doesn't do did done has
have had after before into than then there here also any all some customer customers
user users get gets got""".split())
DUPLICATE_SCORE = 0.6  # Same customer and at least this similar: likely a duplicate.
SIMILAR_SCORE = 0.3  # Below this, a case is not reported as similar.
MIN_SHARED_TERMS = 2  # One common word, such as "fail", is not enough to call cases similar.


def case_terms(text: str) -> set:
    """Meaningful words in text, lowercased, with simple plural and -ing/-ed endings removed."""
    terms = set()
    for word in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.casefold()):
        if word in STOPWORDS or len(word) < 3:
            continue
        for suffix in ("ing", "ed", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                word = word[: -len(suffix)]
                break
        if word.endswith("e") and len(word) > 4:
            word = word[:-1]  # So "charge" and "charged" both become "charg".
        terms.add(word)
    return terms


def case_similarity(a: dict, b: dict) -> tuple:
    """Return (score from 0 to 1, shared terms) for two cases.

    Word overlap of title and description (60%), title similarity (25%), and same
    product (15%). Overlap is shared terms divided by the smaller case's terms, so a
    short new description is not drowned out by a long existing one. To move to RAG
    later, replace this with embedding similarity and keep the same return value, so
    find_similar_cases is unchanged.
    """
    terms_a = case_terms(f"{a.get('title', '')} {a.get('description', '')}")
    terms_b = case_terms(f"{b.get('title', '')} {b.get('description', '')}")
    shared = terms_a & terms_b
    overlap = len(shared) / min(len(terms_a), len(terms_b)) if terms_a and terms_b else 0.0
    title = SequenceMatcher(None, a.get("title", "").casefold(),
                            b.get("title", "").casefold()).ratio()
    same_product = bool(a.get("product_name")) and (
        a.get("product_name", "").casefold() == b.get("product_name", "").casefold())
    return 0.6 * overlap + 0.25 * title + 0.15 * same_product, sorted(shared)


@mcp.tool()
def find_similar_cases(case_id: str = "", title: str = "", description: str = "",
                       customer_name: str = "", product_name: str = "",
                       limit: int = 5) -> dict:
    """Find support cases similar to one case, and flag likely duplicates.

    Give either case_id, to compare an existing case with all the others, or the
    title and description of a new case (plus customer_name and product_name if
    known) to check for duplicates before calling create_case. Returns the most
    similar cases first, each with a score from 0 to 1, the shared terms, and a label:
    'likely duplicate' (same customer and nearly the same problem) or 'similar'.
    """
    try:
        cases = json.loads(CASES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"error": "Cannot read support cases. Check Supportcase.json is valid JSON."}
    if case_id.strip():
        target_id = next((c for c in cases if c.casefold() == case_id.strip().casefold()), None)
        if target_id is None:
            return {"error": "Case not found", "case_id": case_id}
        target = cases[target_id]
    elif title.strip() or description.strip():
        target_id = None
        target = {"title": title, "description": description,
                  "customer_name": customer_name, "product_name": product_name}
    else:
        return {"error": "Give a case_id, or a title and description to compare"}

    results = []
    for other_id, other in cases.items():
        if other_id == target_id:
            continue
        score, shared = case_similarity(target, other)
        if score < SIMILAR_SCORE or len(shared) < MIN_SHARED_TERMS:
            continue
        same_customer = bool(target.get("customer_name")) and (
            target["customer_name"].casefold() == other.get("customer_name", "").casefold())
        label = "likely duplicate" if same_customer and score >= DUPLICATE_SCORE else "similar"
        results.append({"case_id": other_id, "score": round(score, 2), "label": label,
                        "customer_name": other.get("customer_name"),
                        "title": other.get("title"), "case_status": other.get("case_status"),
                        "shared_terms": shared})
    results.sort(key=lambda r: r["score"], reverse=True)
    results = results[:max(1, limit)]
    return {"compared_to": target_id or "new case", "count": len(results),
            "likely_duplicates": [r["case_id"] for r in results if r["label"] == "likely duplicate"],
            "cases": results}


if __name__ == "__main__":
    # stdout belongs to the MCP protocol; send any debug prints to stderr.
    mcp.run(transport="stdio")
