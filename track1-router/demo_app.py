from __future__ import annotations

import os
from urllib.parse import urlparse

import streamlit as st


st.set_page_config(page_title="CascadeSolo Router", page_icon="⚡", layout="wide")


def read_secret_or_env(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.environ.get(name, "")).strip()


for key in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS"):
    value = read_secret_or_env(key)
    if value:
        os.environ[key] = value

missing = [key for key in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS") if not os.environ.get(key)]
if missing:
    st.error(
        "Missing required Fireworks configuration. Set these in Streamlit secrets or local environment: "
        + ", ".join(missing)
    )
    st.stop()


from src import config  # noqa: E402
from src.cascade import route_task  # noqa: E402
from src.classify import classify  # noqa: E402
from src.fireworks_client import FireworksClient  # noqa: E402


EXAMPLES = [
    ("Factual", "Answer this factual knowledge question: What is the capital of Vietnam?"),
    ("Math", "Calculate 2 + 2 * 3."),
    ("Sentiment", "Classify the sentiment of this review: The dashboard is fast, clear, and useful."),
    ("Summary", "Summarize this text: AMD announced a developer hackathon. Teams build AI apps. Token efficiency matters."),
    ("NER", "Extract named entities from: Sarah Johnson met with representatives from Microsoft in Seattle on 2026-07-11."),
    ("Debug", "Find the bug: def add(a, b): return a - b"),
    ("Logic", "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?"),
    ("Codegen", "Write a Python function that returns the larger of two numbers."),
]

PATH_COLORS = {
    "deterministic": "#17803d",
    "fireworks": "#b35c00",
    "fireworks_retry": "#b35c00",
    "fallback": "#b42318",
}


def init_state() -> None:
    st.session_state.setdefault("prompt", EXAMPLES[1][1])
    st.session_state.setdefault("queries_run", 0)
    st.session_state.setdefault("cascade_tokens", 0)
    st.session_state.setdefault("baseline_tokens", 0)
    st.session_state.setdefault("log", [])


def reset_session() -> None:
    st.session_state.clear()
    st.rerun()


def badge(label: str, color: str) -> str:
    return (
        f"<span style='display:inline-block;padding:0.25rem 0.55rem;border-radius:0.45rem;"
        f"background:{color};color:white;font-size:0.85rem;font-weight:700'>{label}</span>"
    )


def stage_card(title: str, status: str, detail: str = "") -> None:
    color = {"done": "#17803d", "active": "#b35c00", "skip": "#667085", "fail": "#b42318"}[status]
    icon = {"done": "✅", "active": "🟠", "skip": "➖", "fail": "❌"}[status]
    st.markdown(
        f"""
        <div style="border:1px solid {color};border-radius:8px;padding:0.75rem;min-height:86px">
          <div style="font-weight:800;color:{color}">{icon} {title}</div>
          <div style="font-size:0.85rem;color:#667085;margin-top:0.35rem">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline(category: str, meta: dict[str, object]) -> None:
    path = str(meta.get("path", "fallback"))
    cols = st.columns(4)
    with cols[0]:
        stage_card("Classify", "done", category)
    with cols[1]:
        stage_card("Free solver", "done" if path == "deterministic" else "skip", "zero tokens" if path == "deterministic" else "not confident")
    with cols[2]:
        firework_status = "done" if path in {"fireworks", "fireworks_retry"} else "skip"
        if path == "fallback":
            firework_status = "fail"
        detail = f"attempts={meta.get('attempts', 1)}"
        if meta.get("retried"):
            detail += ", retried"
        stage_card("Fireworks", firework_status, detail)
    with cols[3]:
        stage_card("Validate", "fail" if path == "fallback" else "done", "accepted" if path != "fallback" else "fallback")


def run_baseline(prompt: str) -> tuple[str, int, str | None]:
    client = FireworksClient()
    result = client.complete({"prompt": prompt}, "unknown")
    return str(result.answer or ""), result.total_tokens, result.error


def truncated(text: str, length: int = 90) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= length else normalized[: length - 1] + "…"


init_state()

with st.sidebar:
    st.subheader("Model config")
    host = urlparse(config.FIREWORKS_BASE_URL).netloc or config.FIREWORKS_BASE_URL
    cheapest = config.CHEAPEST_MODEL
    st.write(f"**Fireworks host:** `{host}`")
    st.write(f"**Cheapest model:** `{cheapest}`")

    st.subheader("Session totals")
    baseline_total = int(st.session_state.baseline_tokens)
    cascade_total = int(st.session_state.cascade_tokens)
    saved = max(0, baseline_total - cascade_total)
    saved_pct = (saved / baseline_total * 100) if baseline_total else 0.0
    st.metric("Queries run", st.session_state.queries_run)
    st.metric("CascadeSolo tokens", cascade_total)
    st.metric("Naive baseline tokens", baseline_total)
    st.metric("Tokens saved", saved, delta=f"{saved_pct:.1f}% saved")

    if st.button("Reset session", use_container_width=True):
        reset_session()

st.title("CascadeSolo Router")
st.caption(
    "Answers what it can for free -- escalates only when a task genuinely needs a model. "
    "Built for AMD Developer Hackathon ACT II, Track 1."
)

for row in range(2):
    cols = st.columns(4)
    for col_index, col in enumerate(cols):
        label, example = EXAMPLES[row * 4 + col_index]
        if col.button(label, use_container_width=True):
            st.session_state.prompt = example

prompt = st.text_area("Prompt", key="prompt", height=150)
compare_baseline = st.toggle(
    "Compare with naive baseline (sends the same prompt directly to Fireworks with no routing -- costs extra tokens)",
    value=False,
)

if st.button("Run through CascadeSolo", type="primary", use_container_width=True):
    if not prompt.strip():
        st.warning("Enter a prompt first.")
        st.stop()

    task = {"prompt": prompt.strip()}
    with st.spinner("Routing through CascadeSolo..."):
        category = classify(task)
        answer, meta = route_task(task, category)

    cascade_tokens = int(meta.get("tokens", 0) or 0)
    baseline_answer = ""
    baseline_tokens: int | None = None
    baseline_error = None
    if compare_baseline:
        with st.spinner("Running naive Fireworks baseline..."):
            baseline_answer, baseline_tokens, baseline_error = run_baseline(prompt.strip())

    st.session_state.queries_run += 1
    st.session_state.cascade_tokens += cascade_tokens
    if baseline_tokens is not None:
        st.session_state.baseline_tokens += baseline_tokens

    st.session_state.log.append(
        {
            "query": truncated(prompt),
            "category": category,
            "path": meta.get("path", "fallback"),
            "cascade_tokens": cascade_tokens,
            "baseline_tokens": baseline_tokens if baseline_tokens is not None else "",
        }
    )

    st.subheader("Pipeline")
    render_pipeline(category, meta)

    st.subheader("Result")
    result_cols = st.columns([1, 1, 1])
    with result_cols[0]:
        st.markdown(badge(category, "#344054"), unsafe_allow_html=True)
    with result_cols[1]:
        path = str(meta.get("path", "fallback"))
        st.markdown(badge(path, PATH_COLORS.get(path, "#667085")), unsafe_allow_html=True)
    with result_cols[2]:
        st.metric("Tokens used", cascade_tokens)

    if category in {"code debugging", "code generation"}:
        st.code(str(answer), language="python")
    else:
        st.write(answer)

    if compare_baseline:
        st.subheader("CascadeSolo vs naive baseline")
        st.bar_chart({"tokens": {"CascadeSolo": cascade_tokens, "Naive baseline": baseline_tokens or 0}})
        if baseline_error:
            st.warning(f"Baseline error: {baseline_error}")
        with st.expander("Naive baseline answer"):
            st.write(baseline_answer)

if st.session_state.log:
    st.subheader("Session query log")
    st.dataframe(st.session_state.log, use_container_width=True, hide_index=True)

with st.expander("About"):
    st.write(
        "CascadeSolo is a hybrid router: deterministic solvers handle math, sentiment, and NER when confidence is high, "
        "while harder prompts escalate to Fireworks with compact category-specific prompts. This demonstrates the "
        "zero-token deterministic tier that improves token efficiency for AMD Developer Hackathon ACT II, Track 1."
    )
    st.link_button("GitHub repository", "https://github.com/mainamkhanh23102005/ProsoponRouter")
