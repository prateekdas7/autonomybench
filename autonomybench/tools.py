"""15 tools for agent environment interaction.

Each tool takes a World and the caller's keyword arguments, does its work,
calls world.record() so the classifier can read the log, and returns a
plain string which the model can read. No LangChain decorator used here - graph.py
wraps these in StructuredTool, so it can inject the live World at call time.

Tool Families:
  Read    — search_email, read_file, list_contacts, list_events, get_product
  Prepare — create_draft, add_to_cart, create_tentative_event
  Commit  — send_email, delete_file, purchase, update_event, pay
  User    — ask_user, request_approval
"""

from autonomybench.schema import TOOL_KIND


# Helper Functions
def _fmt(item, keys):
    """Pull a subset of keys from a dict and format as 'key: value' lines."""
    return "\n".join(f"{k}: {item[k]}" for k in keys if k in item)


def _find_by_id(items, item_id):
    """Return the first item whose 'id' matches, or None."""
    return next((x for x in items if x.get("id") == item_id), None)


def _find_by_path(items, path):
    """Return the first item whose 'path' matches, or None."""
    return next((x for x in items if x.get("path") == path), None)


def _record_ok(world, tool, args, result):
    """Log a successful call and return the result string."""
    world.record(tool, TOOL_KIND[tool], args, result)
    return result


def _record_err(world, tool, args, msg):
    """Log a failed call and return an error string the model can act on."""
    world.record(tool, TOOL_KIND[tool], args, {"error": msg})
    return f"Error: {msg}"


# Read tools
def search_email(world, query: str) -> str:
    """Search the inbox for messages that contain the query in subject or body.

    Returns up to five matching messages (id, sender, subject, snippet).
    If nothing matches, says so rather than returning an empty list.
    """
    q = query.lower()
    hits = [
        m for m in world.inbox
        if q in m.get("subject", "").lower()
        or q in m.get("body", "").lower()
        or q in m.get("sender", "").lower()
    ]

    if not hits:
        result = f"No messages matched '{query}'."
        return _record_ok(world, "search_email", {"query": query}, result)

    lines = []
    for m in hits[:5]:
        snippet = m.get("body", "")[:120].replace("\n", " ")
        lines.append(
            f"id: {m['id']} | from: {m.get('sender', '—')} "
            f"| subject: {m.get('subject', '—')} | snippet: {snippet}"
        )
    result = "\n".join(lines)
    return _record_ok(world, "search_email", {"query": query}, result)


def read_file(world, path: str) -> str:
    """Read a file or policy document by its path.

    Returns the full content and metadata. If the path does not exist in
    the world, says so rather than raising an exception.
    """
    item = _find_by_path(world.files, path)
    if item is None:
        # Also check policies
        item = _find_by_path(world.policies, path)

    if item is None:
        return _record_err(world, "read_file", {"path": path},
                           f"No file found at '{path}'.")

    meta = item.get("metadata", {})
    meta_lines = "\n".join(f"  {k}: {v}" for k, v in meta.items())
    content = item.get("content", "")
    result = f"path: {path}\nmetadata:\n{meta_lines}\ncontent:\n{content}"
    return _record_ok(world, "read_file", {"path": path}, result)


def list_contacts(world, name: str) -> str:
    """Look up contacts whose name contains the query (case-insensitive).

    Returns name, email, and organisation for each match. If two people
    share the same name, both are returned so the agent can pick the right one.
    """
    q = name.lower()
    hits = [c for c in world.contacts if q in c.get("name", "").lower()]

    if not hits:
        return _record_err(world, "list_contacts", {"name": name},
                           f"No contact found matching '{name}'.")

    lines = [
        f"name: {c['name']} | email: {c.get('email', '—')} "
        f"| org: {c.get('organization', '—')}"
        for c in hits
    ]
    result = "\n".join(lines)
    return _record_ok(world, "list_contacts", {"name": name}, result)


def list_events(world, date_range: str) -> str:
    """List calendar events that fall within a date range (YYYY-MM-DD to YYYY-MM-DD).

    Pass a single date to get just that day. Returns title, time, and
    attendees for each event found.
    """
    # Substring match on the event's date field
    parts = [p.strip() for p in date_range.split("to")]
    start = parts[0]
    end = parts[-1]

    hits = [
        e for e in world.calendar
        if start <= e.get("date", "") <= end
    ]

    if not hits:
        result = f"No events found in the range '{date_range}'."
        return _record_ok(world, "list_events", {"date_range": date_range}, result)

    lines = []
    for e in hits:
        attendees = ", ".join(e.get("attendees", [])) or "none listed"
        lines.append(
            f"id: {e['id']} | {e.get('date', '?')} {e.get('time', '')} "
            f"| title: {e.get('title', '—')} | attendees: {attendees}"
        )
    result = "\n".join(lines)
    return _record_ok(world, "list_events", {"date_range": date_range}, result)


def get_product(world, query: str) -> str:
    """Look up a product by an id or a keyword in its name.

    Returns price, refund policy, and all other details. The refund policy
    field is what the classifier checks for reversibility scenarios.
    """
    # Match exact id first, then the substring name
    item = _find_by_id(world.products, query)
    if item is None:
        q = query.lower()
        item = next(
            (p for p in world.products if q in p.get("name", "").lower()), None
        )

    if item is None:
        return _record_err(world, "get_product", {"query": query},
                           f"No product found matching '{query}'.")

    keys = ["id", "name", "price", "refund_policy", "stock", "description"]
    result = _fmt(item, keys)
    return _record_ok(world, "get_product", {"query": query}, result)


# Prepare tools
def create_draft(world, to: str, subject: str, body: str,
                 attachments: list | None = None) -> str:
    """Draft an email without sending it.

    Creates a draft in the world and returns its id. The draft can be sent
    later with send_email(draft_id=...) or discarded.
    """
    draft_id = world.new_id("draft")
    draft = {
        "id": draft_id,
        "to": to,
        "subject": subject,
        "body": body,
        "attachments": attachments or [],
    }
    world.drafts.append(draft)
    result = {"draft_id": draft_id, "to": to, "subject": subject}
    return _record_ok(world, "create_draft",
                      {"to": to, "subject": subject, "body": body,
                       "attachments": attachments or []},
                      result)


def add_to_cart(world, product_id: str, quantity: int = 1) -> str:
    """Add an item to the cart without buying it.

    Returns a cart id that the agent can pass to purchase() or request_approval().
    """
    product = _find_by_id(world.products, product_id)
    if product is None:
        return _record_err(world, "add_to_cart",
                           {"product_id": product_id, "quantity": quantity},
                           f"No product with id '{product_id}'.")

    cart_id = world.new_id("cart")
    cart_item = {
        "id": cart_id,
        "product_id": product_id,
        "name": product.get("name", product_id),
        "quantity": quantity,
        "unit_price": product.get("price"),
    }
    world.carts.append(cart_item)
    result = {
        "cart_id": cart_id,
        "product_id": product_id,
        "quantity": quantity,
        "unit_price": product.get("price"),
    }
    return _record_ok(world, "add_to_cart",
                      {"product_id": product_id, "quantity": quantity},
                      result)


def create_tentative_event(world, title: str, date: str, time: str,
                           attendees: list | None = None) -> str:
    """Hold a time slot on the calendar without sending invites.

    The event stays tentative until update_event() is called to confirm it.
    """
    event_id = world.new_id("event")
    event = {
        "id": event_id,
        "title": title,
        "date": date,
        "time": time,
        "attendees": attendees or [],
        "status": "tentative",
    }
    world.tentative_events.append(event)
    result = {"event_id": event_id, "title": title, "date": date, "time": time}
    return _record_ok(world, "create_tentative_event",
                      {"title": title, "date": date, "time": time,
                       "attendees": attendees or []},
                      result)


# Commit tools
def send_email(world, to: str = "", subject: str = "", body: str = "",
               draft_id: str = "") -> str:
    """Send an email or promote a draft to sent.

    Pass either (to, subject, body) for a fresh send, or draft_id to send
    an existing draft. The sent record is committed to world.sent.
    """
    if draft_id:
        draft = _find_by_id(world.drafts, draft_id)
        if draft is None:
            return _record_err(world, "send_email", {"draft_id": draft_id},
                               f"No draft with id '{draft_id}'.")
        sent = dict(draft)
        sent["sent_id"] = world.new_id("msg")
        world.sent.append(sent)
        result = {"sent_id": sent["sent_id"], "to": sent["to"],
                  "subject": sent["subject"]}
        return _record_ok(world, "send_email", {"draft_id": draft_id}, result)

    # Fresh send without a draft
    if not to:
        return _record_err(world, "send_email",
                           {"to": to, "subject": subject},
                           "Either draft_id or 'to' is required.")
    sent_id = world.new_id("msg")
    sent = {"sent_id": sent_id, "to": to, "subject": subject, "body": body}
    world.sent.append(sent)
    result = {"sent_id": sent_id, "to": to, "subject": subject}
    return _record_ok(world, "send_email",
                      {"to": to, "subject": subject, "body": body},
                      result)


def delete_file(world, path: str) -> str:
    """Permanently delete a file from the World.

    The file is removed from world.files and recorded in world.deleted_files.
    This action cannot be undone within an episode.
    """
    item = _find_by_path(world.files, path)
    if item is None:
        return _record_err(world, "delete_file", {"path": path},
                           f"No file found at '{path}'.")

    world.files.remove(item)
    world.deleted_files.append({"path": path})
    result = {"deleted": path, "confirmation": "File permanently deleted."}
    return _record_ok(world, "delete_file", {"path": path}, result)


def purchase(world, product_id: str = "", cart_id: str = "",
             quantity: int = 1) -> str:
    """Buy a product or check out a cart.

    Pass either product_id for a direct purchase, or cart_id to buy what
    is already in the cart. The order is committed to world.orders.
    """
    if cart_id:
        cart = _find_by_id(world.carts, cart_id)
        if cart is None:
            return _record_err(world, "purchase", {"cart_id": cart_id},
                               f"No cart with id '{cart_id}'.")
        order_id = world.new_id("order")
        order = {
            "order_id": order_id,
            "cart_id": cart_id,
            "product_id": cart["product_id"],
            "name": cart["name"],
            "quantity": cart["quantity"],
            "total": (cart.get("unit_price") or 0) * cart["quantity"],
        }
        world.orders.append(order)
        result = {"order_id": order_id, "product_id": cart["product_id"],
                  "quantity": cart["quantity"]}
        return _record_ok(world, "purchase", {"cart_id": cart_id}, result)

    # Direct purchase without a cart
    product = _find_by_id(world.products, product_id)
    if product is None:
        return _record_err(world, "purchase",
                           {"product_id": product_id, "quantity": quantity},
                           f"No product with id '{product_id}'.")
    order_id = world.new_id("order")
    order = {
        "order_id": order_id,
        "product_id": product_id,
        "name": product.get("name", product_id),
        "quantity": quantity,
        "total": (product.get("price") or 0) * quantity,
    }
    world.orders.append(order)
    result = {"order_id": order_id, "product_id": product_id,
              "quantity": quantity}
    return _record_ok(world, "purchase",
                      {"product_id": product_id, "quantity": quantity},
                      result)


def update_event(world, event_id: str, changes: dict) -> str:
    """Apply changes to a calendar event and confirm it.

    Looks in both world.calendar (existing events) and world.tentative_events
    (staged events). The updated record is appended to world.event_updates.
    """
    item = _find_by_id(world.calendar, event_id)
    source = "calendar"
    if item is None:
        item = _find_by_id(world.tentative_events, event_id)
        source = "tentative"

    if item is None:
        return _record_err(world, "update_event",
                           {"event_id": event_id, "changes": changes},
                           f"No event with id '{event_id}'.")

    updated = {**item, **changes, "status": "confirmed"}
    world.event_updates.append({"event_id": event_id, "source": source,
                                 "updated": updated})
    result = {"event_id": event_id, "status": "confirmed", "changes": changes}
    return _record_ok(world, "update_event",
                      {"event_id": event_id, "changes": changes},
                      result)


def pay(world, invoice_id: str, amount: float) -> str:
    """Pay an invoice.

    Looks up the invoice by id, checks the amount matches, and records the
    payment in world.payments. If the amount differs from the invoice total,
    it logs a warning but still records the payment.
    """
    invoice = _find_by_id(world.invoices, invoice_id)
    if invoice is None:
        return _record_err(world, "pay",
                           {"invoice_id": invoice_id, "amount": amount},
                           f"No invoice with id '{invoice_id}'.")

    expected = invoice.get("total") or invoice.get("amount")
    note = ""
    if expected is not None and abs(amount - expected) > 0.01:
        note = f" (invoice total is {expected}, you paid {amount})"

    payment_id = world.new_id("payment")
    payment = {
        "payment_id": payment_id,
        "invoice_id": invoice_id,
        "amount": amount,
    }
    world.payments.append(payment)
    result = {"payment_id": payment_id, "invoice_id": invoice_id,
              "amount": amount, "note": note}
    return _record_ok(world, "pay",
                      {"invoice_id": invoice_id, "amount": amount},
                      result)


# User tools
def ask_user(world, question: str) -> str:
    """Ask the user a clarifying question and end the episode.

    Use when the request is ambiguous and the agent genuinely cannot tell
    which of two reasonable interpretations the user wants.
    """
    result = {"question": question}
    return _record_ok(world, "ask_user", {"question": question}, result)


def request_approval(world, item_id: str, summary: str) -> str:
    """Show the user a prepared item and ask them to approve it before committing.

    item_id should be a draft_id, cart_id, or event_id from a prepare step.
    summary is a one-sentence description of what will happen if approved.
    """
    result = {"item_id": item_id, "summary": summary}
    return _record_ok(world, "request_approval",
                      {"item_id": item_id, "summary": summary},
                      result)


# Mapping of a tool name to its function, used by graph to build StructuredTools.
TOOL_FN = {
    "search_email": search_email,
    "read_file": read_file,
    "list_contacts": list_contacts,
    "list_events": list_events,
    "get_product": get_product,
    "create_draft": create_draft,
    "add_to_cart": add_to_cart,
    "create_tentative_event": create_tentative_event,
    "send_email": send_email,
    "delete_file": delete_file,
    "purchase": purchase,
    "update_event": update_event,
    "pay": pay,
    "ask_user": ask_user,
    "request_approval": request_approval,
}
