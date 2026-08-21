import numpy as np
import pandas as pd

# -----------------------------
# 1. Random seed
# -----------------------------
rng = np.random.default_rng(42)

# -----------------------------
# 2. Basic simulation settings
# -----------------------------
n_requests = 10_000

start_date = pd.Timestamp("2026-05-01")
end_date = pd.Timestamp("2026-07-29")

models = ["model_v1", "model_v2", "model_v3"]
prompt_versions = ["prompt_v1", "prompt_v2", "prompt_v3"]

query_categories = [
    "billing",
    "account",
    "technical_support",
    "product_question",
    "policy",
    "general"
]

# -----------------------------
# 3. Request IDs and timestamps
# -----------------------------
request_ids = [f"REQ-{i:05d}" for i in range(1, n_requests + 1)]

total_seconds = int((end_date - start_date).total_seconds())

timestamps = (
    start_date
    + pd.to_timedelta(
        rng.integers(0, total_seconds, size=n_requests),
        unit="s"
    )
)

timestamps = pd.Series(timestamps).sort_values().reset_index(drop=True)

# -----------------------------
# 4. Model and prompt versions
# -----------------------------
model_version = np.select(
    [
        timestamps < pd.Timestamp("2026-06-01"),
        timestamps < pd.Timestamp("2026-07-01")
    ],
    [
        "model_v1",
        "model_v2"
    ],
    default="model_v3"
)

prompt_version = np.select(
    [
        timestamps < pd.Timestamp("2026-06-01"),
        timestamps < pd.Timestamp("2026-07-01")
    ],
    [
        "prompt_v1",
        "prompt_v2"
    ],
    default="prompt_v3"
)

query_category = rng.choice(
    query_categories,
    size=n_requests,
    p=[0.18, 0.14, 0.24, 0.17, 0.12, 0.15]
)

# -----------------------------
# 5. Retrieval performance
# -----------------------------
retrieval_score = rng.beta(8, 2, size=n_requests)

retrieval_incident = (
    (timestamps >= pd.Timestamp("2026-06-18"))
    & (timestamps < pd.Timestamp("2026-06-22"))
)

retrieval_score[retrieval_incident] -= rng.uniform(
    0.15,
    0.35,
    retrieval_incident.sum()
)

retrieval_score = np.clip(retrieval_score, 0, 1)

retrieval_success = retrieval_score >= 0.60

# -----------------------------
# 6. Tool usage
# -----------------------------
tool_required = rng.random(n_requests) < 0.40

tool_accuracy = np.where(
    model_version == "model_v1",
    0.91,
    np.where(
        model_version == "model_v2",
        0.95,
        0.975
    )
)

tool_selected_correctly = (
    (~tool_required)
    | (rng.random(n_requests) < tool_accuracy)
)

# -----------------------------
# 7. Structured-output validity
# -----------------------------
structured_output_probability = np.where(
    model_version == "model_v1",
    0.970,
    np.where(
        model_version == "model_v2",
        0.985,
        0.995
    )
)

structured_output_valid = (
    rng.random(n_requests) < structured_output_probability
)

# -----------------------------
# 8. Groundedness
# -----------------------------
groundedness_score = (
    0.30
    + 0.65 * retrieval_score
    + rng.normal(0, 0.07, n_requests)
)

groundedness_score = np.clip(groundedness_score, 0, 1)

# -----------------------------
# 9. Correctness
# -----------------------------
model_quality_bonus = np.where(
    model_version == "model_v1",
    0.00,
    np.where(
        model_version == "model_v2",
        0.04,
        0.08
    )
)

correctness_score = (
    0.20
    + 0.45 * retrieval_score
    + 0.25 * groundedness_score
    + model_quality_bonus
    + 0.05 * tool_selected_correctly.astype(int)
    + rng.normal(0, 0.06, n_requests)
)

correctness_score = np.clip(correctness_score, 0, 1)

# -----------------------------
# 10. Human evaluation
# -----------------------------
human_pass_probability = (
    0.15
    + 0.85 * correctness_score
)

human_pass_probability = np.clip(
    human_pass_probability,
    0,
    1
)

human_pass = (
    rng.random(n_requests)
    < human_pass_probability
)

# -----------------------------
# 11. Latency
# -----------------------------
base_latency = np.where(
    model_version == "model_v1",
    1.4,
    np.where(
        model_version == "model_v2",
        1.8,
        2.1
    )
)

latency_seconds = (
    base_latency
    + rng.lognormal(
        mean=0.0,
        sigma=0.55,
        size=n_requests
    )
)

latency_seconds += (
    tool_required.astype(int)
    * rng.uniform(0.3, 1.5, n_requests)
)

latency_seconds = np.round(latency_seconds, 2)

# -----------------------------
# 12. Token usage
# -----------------------------
input_tokens = rng.integers(
    500,
    3500,
    size=n_requests
)

output_tokens = rng.integers(
    100,
    900,
    size=n_requests
)

# -----------------------------
# 13. Cost
# -----------------------------
input_cost_per_1k = np.where(
    model_version == "model_v1",
    0.002,
    np.where(
        model_version == "model_v2",
        0.003,
        0.004
    )
)

output_cost_per_1k = np.where(
    model_version == "model_v1",
    0.006,
    np.where(
        model_version == "model_v2",
        0.009,
        0.012
    )
)

cost_usd = (
    input_tokens / 1000 * input_cost_per_1k
    + output_tokens / 1000 * output_cost_per_1k
)

cost_usd = np.round(cost_usd, 4)

# -----------------------------
# 14. Failure categories
# -----------------------------
failure_category = np.select(
    [
        ~retrieval_success,
        tool_required & ~tool_selected_correctly,
        ~structured_output_valid,
        groundedness_score < 0.70,
        correctness_score < 0.75
    ],
    [
        "retrieval_failure",
        "tool_selection_failure",
        "invalid_structured_output",
        "grounding_failure",
        "answer_quality_failure"
    ],
    default="none"
)

# -----------------------------
# 15. Build DataFrame
# -----------------------------
df = pd.DataFrame({
    "request_id": request_ids,
    "timestamp": timestamps,
    "model_version": model_version,
    "prompt_version": prompt_version,
    "query_category": query_category,
    "retrieval_score": retrieval_score,
    "retrieval_success": retrieval_success,
    "tool_required": tool_required,
    "tool_selected_correctly": tool_selected_correctly,
    "structured_output_valid": structured_output_valid,
    "groundedness_score": groundedness_score,
    "correctness_score": correctness_score,
    "human_pass": human_pass,
    "latency_seconds": latency_seconds,
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "cost_usd": cost_usd,
    "failure_category": failure_category
})

# -----------------------------
# 16. Round display fields
# -----------------------------
df["retrieval_score"] = df["retrieval_score"].round(3)
df["groundedness_score"] = df["groundedness_score"].round(3)
df["correctness_score"] = df["correctness_score"].round(3)

# -----------------------------
# 17. Basic sanity checks
# -----------------------------
print("Dataset shape:")
print(df.shape)

print("\nData types:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

# -----------------------------
# 18. Dashboard metrics
# -----------------------------
overall_correctness = df["correctness_score"].mean()
human_pass_rate = df["human_pass"].mean()
average_groundedness = df["groundedness_score"].mean()

p50_latency = df["latency_seconds"].quantile(0.50)
p95_latency = df["latency_seconds"].quantile(0.95)
p99_latency = df["latency_seconds"].quantile(0.99)

average_cost = df["cost_usd"].mean()

print("\nDashboard metrics:")
print(f"Average correctness: {overall_correctness:.1%}")
print(f"Human pass rate: {human_pass_rate:.1%}")
print(f"Average groundedness: {average_groundedness:.1%}")
print(f"P50 latency: {p50_latency:.2f} seconds")
print(f"P95 latency: {p95_latency:.2f} seconds")
print(f"P99 latency: {p99_latency:.2f} seconds")
print(f"Average cost/request: ${average_cost:.4f}")

# -----------------------------
# 19. Model comparison
# -----------------------------
model_summary = (
    df
    .groupby("model_version")
    .agg(
        requests=("request_id", "count"),
        avg_correctness=("correctness_score", "mean"),
        human_pass_rate=("human_pass", "mean"),
        avg_groundedness=("groundedness_score", "mean"),
        avg_latency=("latency_seconds", "mean"),
        p95_latency=("latency_seconds", lambda x: x.quantile(0.95)),
        avg_cost=("cost_usd", "mean")
    )
    .reset_index()
)

print("\nModel comparison:")
print(model_summary)

# -----------------------------
# 20. Failure modes
# -----------------------------
print("\nFailure category counts:")
print(df["failure_category"].value_counts())

print("\nFailure category percentages:")
print(df["failure_category"].value_counts(normalize=True))

# -----------------------------
# 21. Save CSV
# -----------------------------
output_file = "llm_evaluation_data.csv"
df.to_csv(output_file, index=False)

print(f"\nSaved simulated data to: {output_file}")

# Optional: automatically download the CSV when running in Google Colab.
# Uncomment these two lines if desired:
#
# from google.colab import files
# files.download(output_file)
