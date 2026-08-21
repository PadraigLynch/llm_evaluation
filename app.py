from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="LLM Production Evaluation Dashboard",
    page_icon="🤖",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "data" / "llm_evaluation_data.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df


df = load_data()

st.title("LLM Production Evaluation Dashboard")
st.caption(
    "Fictional production data for monitoring LLM quality, groundedness, "
    "latency, cost, and pipeline failure modes."
)

# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
st.sidebar.header("Filters")

date_min = df["timestamp"].dt.date.min()
date_max = df["timestamp"].dt.date.max()

date_range = st.sidebar.date_input(
    "Date range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)

selected_models = st.sidebar.multiselect(
    "Model version",
    options=sorted(df["model_version"].unique()),
    default=sorted(df["model_version"].unique()),
)

selected_categories = st.sidebar.multiselect(
    "Query category",
    options=sorted(df["query_category"].unique()),
    default=sorted(df["query_category"].unique()),
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0]

filtered = df[
    (df["timestamp"].dt.date >= start_date)
    & (df["timestamp"].dt.date <= end_date)
    & (df["model_version"].isin(selected_models))
    & (df["query_category"].isin(selected_categories))
].copy()

if filtered.empty:
    st.warning("No requests match the selected filters.")
    st.stop()

# ---------------------------------------------------------
# KPI cards
# ---------------------------------------------------------
avg_correctness = filtered["correctness_score"].mean()
human_pass_rate = filtered["human_pass"].mean()
avg_groundedness = filtered["groundedness_score"].mean()
p95_latency = filtered["latency_seconds"].quantile(0.95)
avg_cost = filtered["cost_usd"].mean()
request_count = len(filtered)

k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric("Requests", f"{request_count:,}")
k2.metric("Correctness", f"{avg_correctness:.1%}")
k3.metric("Human Pass Rate", f"{human_pass_rate:.1%}")
k4.metric("Groundedness", f"{avg_groundedness:.1%}")
k5.metric("P95 Latency", f"{p95_latency:.2f}s")
k6.metric("Avg Cost / Request", f"${avg_cost:.4f}")

st.divider()

# ---------------------------------------------------------
# 1. Quality over time
# ---------------------------------------------------------
daily_quality = (
    filtered.assign(date=filtered["timestamp"].dt.date)
    .groupby("date", as_index=False)
    .agg(
        avg_correctness=("correctness_score", "mean"),
        avg_groundedness=("groundedness_score", "mean"),
    )
)

fig_quality = go.Figure()

fig_quality.add_trace(
    go.Scatter(
        x=daily_quality["date"],
        y=daily_quality["avg_correctness"],
        mode="lines",
        name="Correctness",
    )
)

fig_quality.add_trace(
    go.Scatter(
        x=daily_quality["date"],
        y=daily_quality["avg_groundedness"],
        mode="lines",
        name="Groundedness",
    )
)

fig_quality.update_layout(
    title="LLM Quality Over Time",
    xaxis_title="Date",
    yaxis_title="Average Score",
    hovermode="x unified",
    legend_title_text="Metric",
)

fig_quality.update_yaxes(range=[0, 1])

st.plotly_chart(fig_quality, use_container_width=True)

# ---------------------------------------------------------
# 2. Model quality comparison
# ---------------------------------------------------------
model_quality = (
    filtered.groupby("model_version", as_index=False)
    .agg(
        avg_correctness=("correctness_score", "mean"),
        human_pass_rate=("human_pass", "mean"),
    )
)

fig_models = go.Figure()

fig_models.add_trace(
    go.Bar(
        x=model_quality["model_version"],
        y=model_quality["avg_correctness"],
        name="Average Correctness",
    )
)

fig_models.add_trace(
    go.Bar(
        x=model_quality["model_version"],
        y=model_quality["human_pass_rate"],
        name="Human Pass Rate",
    )
)

fig_models.update_layout(
    title="Quality by Model Version",
    xaxis_title="Model Version",
    yaxis_title="Rate",
    barmode="group",
)

fig_models.update_yaxes(range=[0, 1])

st.plotly_chart(fig_models, use_container_width=True)

# ---------------------------------------------------------
# 3. Latency distribution
# ---------------------------------------------------------
fig_latency = go.Figure()

for model in sorted(filtered["model_version"].unique()):
    model_data = filtered.loc[
        filtered["model_version"] == model,
        "latency_seconds",
    ]

    fig_latency.add_trace(
        go.Histogram(
            x=model_data,
            name=model,
            opacity=0.65,
            nbinsx=50,
        )
    )

fig_latency.update_layout(
    title="Latency Distribution by Model Version",
    xaxis_title="Latency (seconds)",
    yaxis_title="Number of Requests",
    barmode="overlay",
)

st.plotly_chart(fig_latency, use_container_width=True)

# ---------------------------------------------------------
# 4. Failure modes
# ---------------------------------------------------------
failure_counts = (
    filtered.loc[
        filtered["failure_category"] != "none",
        "failure_category",
    ]
    .value_counts()
    .rename_axis("failure_category")
    .reset_index(name="requests")
    .sort_values("requests", ascending=True)
)

fig_failures = go.Figure()

fig_failures.add_trace(
    go.Bar(
        x=failure_counts["requests"],
        y=failure_counts["failure_category"],
        orientation="h",
        name="Requests",
    )
)

fig_failures.update_layout(
    title="Production Failure Modes",
    xaxis_title="Number of Requests",
    yaxis_title="Failure Category",
    showlegend=False,
)

st.plotly_chart(fig_failures, use_container_width=True)

# ---------------------------------------------------------
# 5. Quality versus cost
# ---------------------------------------------------------
model_cost_quality = (
    filtered.groupby("model_version", as_index=False)
    .agg(
        avg_cost=("cost_usd", "mean"),
        avg_correctness=("correctness_score", "mean"),
        request_count=("request_id", "count"),
    )
)

fig_cost_quality = go.Figure()

fig_cost_quality.add_trace(
    go.Scatter(
        x=model_cost_quality["avg_cost"],
        y=model_cost_quality["avg_correctness"],
        mode="markers+text",
        text=model_cost_quality["model_version"],
        textposition="top center",
        marker=dict(size=14),
        customdata=model_cost_quality[["request_count"]],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Average cost: $%{x:.4f}<br>"
            "Average correctness: %{y:.3f}<br>"
            "Requests: %{customdata[0]:,}"
            "<extra></extra>"
        ),
    )
)

fig_cost_quality.update_layout(
    title="Model Quality vs. Cost",
    xaxis_title="Average Cost per Request ($)",
    yaxis_title="Average Correctness",
)

fig_cost_quality.update_yaxes(range=[0, 1])

st.plotly_chart(fig_cost_quality, use_container_width=True)

# ---------------------------------------------------------
# 6. Daily P95 latency
# ---------------------------------------------------------
daily_latency = (
    filtered.assign(date=filtered["timestamp"].dt.date)
    .groupby("date")["latency_seconds"]
    .quantile(0.95)
    .reset_index(name="p95_latency")
)

fig_p95 = go.Figure()

fig_p95.add_trace(
    go.Scatter(
        x=daily_latency["date"],
        y=daily_latency["p95_latency"],
        mode="lines",
        name="P95 Latency",
    )
)

fig_p95.update_layout(
    title="Daily P95 Latency",
    xaxis_title="Date",
    yaxis_title="P95 Latency (seconds)",
    hovermode="x unified",
    showlegend=False,
)

st.plotly_chart(fig_p95, use_container_width=True)

# ---------------------------------------------------------
# Failure explorer
# ---------------------------------------------------------
st.subheader("Failure Explorer")

failures_only = filtered[filtered["failure_category"] != "none"].copy()

st.dataframe(
    failures_only[
        [
            "request_id",
            "timestamp",
            "model_version",
            "query_category",
            "retrieval_score",
            "groundedness_score",
            "correctness_score",
            "latency_seconds",
            "cost_usd",
            "failure_category",
        ]
    ].sort_values("timestamp", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "All data in this dashboard are simulated and do not represent a real company or production system."
)
