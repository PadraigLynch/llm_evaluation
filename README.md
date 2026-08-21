# LLM Production Evaluation Dashboard

A portfolio-style Streamlit dashboard using fictional data to demonstrate monitoring and evaluation concepts for a production LLM pipeline.

## Dashboard contents

- Correctness and groundedness over time
- Human pass rate
- P95 latency
- Cost per request
- Model-version comparison
- Latency distributions
- Production failure modes
- Quality-versus-cost comparison
- Failure explorer

All data are simulated.

## Files

- `app.py` — Streamlit dashboard
- `generate_llm_evaluation_data.py` — script that creates the fictional dataset
- `data/llm_evaluation_data.csv` — generated dataset
- `requirements.txt` — Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload all files in this project, preserving the `data/` folder.
3. Commit and push to GitHub.
4. In Streamlit Community Cloud, create a new app from the GitHub repository.
5. Choose `app.py` as the entrypoint.
6. Deploy.

The dashboard reads the included CSV directly, so it does not require secrets, API keys, or external services.
