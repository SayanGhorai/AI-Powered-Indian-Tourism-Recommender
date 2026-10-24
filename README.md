# AI-Powered Indian Tourism Recommender V3

An evidence-grounded Indian tourism recommendation and provisional itinerary
planning application built with Python, Streamlit, hybrid information retrieval,
Cross-Encoder reranking, and conservative itinerary planning.

## Project overview

The system combines a verified V2 recommendation pipeline with V3 nationwide
destination selection, attraction identity review, evidence-based admission
information, conservative scheduling, PDF export, and optional AI-agent
tool selection.

The catalog contains 15,239 place records across 1,794 canonical destination
labels. The underlying review dataset contains 1,482,466 reviews.

## Core architecture

1. Detect and normalize the requested destination.
2. Retrieve candidate places using lexical TF-IDF and semantic SBERT/FAISS.
3. Fuse candidate rankings using Reciprocal Rank Fusion.
4. Apply strict city-aware retrieval and geographic constraints.
5. Rerank candidates using a Cross-Encoder.
6. Apply conservative recommendation and preference policies.
7. Classify attraction identity and operational evidence.
8. Produce provisional itinerary stops or clearly explained held entries.
9. Assess conservative scheduling without inventing travel times.
10. Present results in Streamlit and export an evidence-grounded PDF.

## Models

The application uses these pretrained models:

- `sentence-transformers/all-MiniLM-L6-v2`
- `cross-encoder/ms-marco-MiniLM-L-6-v2`

Model files may be downloaded and cached by Hugging Face during first use.
The project does not require retraining these models for normal operation.

## Environment

Verified local development environment:

- Python 3.12.2
- Streamlit 1.65.0
- Dependencies pinned in `requirements.txt`

The pinned environment was verified on Windows. Compatibility with other
operating systems and hosting environments must be tested separately.

## Local setup

Create a Python 3.12 virtual environment and install dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the application:

```powershell
.\.venv\Scripts\python.exe -m streamlit run v3_streamlit_app.py
```

## Production artifacts

The saved V2 production runtime is located at:

`production_package/runtime/Indian-Tourism-AI-Recommender-Production/`

It includes the canonical place catalog, TF-IDF artifacts, FAISS semantic
index, retrieval documents, recommendation features, and policy files.

The runtime bundle contains 17 files totaling approximately 93.04 MiB.

Do not regenerate Kaggle training artifacts when the existing verified
production bundle is available.

## Application modes

### AI recommendations

Enter a destination and interests. The V2 retrieval and reranking pipeline
provides candidate attractions. V3 then applies planning safety checks.

### Plan selected attractions

Choose a catalog destination and select attractions. Items without sufficient
identity or operational evidence may be held rather than scheduled.

### Optional AI agent

The agent can select from approved planning tools when an OpenAI API key is
configured. Without a key, the application supports deterministic offline
planning. API usage may incur charges.

Configure the optional environment variable `OPENAI_API_KEY` through your
deployment provider's secret-management facility. Never commit a real key.

## Evidence and safety limitations

- All itineraries are provisional.
- Attraction identity may require verification.
- Opening hours and admission information may be incomplete or outdated.
- Known admission subtotals do not imply complete trip costs.
- Travel times and visit durations are not invented.
- Later-stop arrival times may remain unscheduled.
- Route optimization and full-day feasibility are not guaranteed.
- An unsupported planning request does not create a fabricated itinerary.

## PDF exports

PDFs preserve the verified destination, provisional stops, held-attraction
reasons, approved evidence references where available, admission accounting,
and conservative scheduling limitations.

## Performance observations

Local Windows measurements:

- Initial Streamlit render: approximately 12.55 seconds.
- First tested V2 recommendation: approximately 12.86 seconds.
- Warm subsequent recommendation: approximately 1.26 seconds.
- Working set after inference: approximately 1.0 GiB.
- Private committed memory after inference: approximately 2.6 GiB.

These are local observations, not hosting guarantees. The tested queries
differ, and hosting memory limits must be validated before deployment.

## Deployment considerations

The application requires its production artifact bundle and Python packages.
The pretrained model weights may require additional download and cache space.

Use a hosting environment with adequate memory, CPU capacity, persistent or
reusable model caching where available, and secure environment-variable
management.

The existing Windows environment is verified; deployment to Linux or another
platform requires separate installation and smoke testing.

## Repository safety

`.gitignore` excludes local virtual environments, temporary files, backups,
and secret files. Keep required source modules, production assets, and
verified runtime artifacts in the deployment repository.

## Project status

Phases A-D and Phase E milestones E1-E7 have passed their defined local
regressions. Deployment documentation, hosting smoke testing, and final
acceptance remain subject to verification.
