# tokbench

A token-economics benchmark harness for LLM coding agents: runs the same task set
through many (model × prompt-strategy) combinations, records every token that
crosses the wire, prices it, grades the answer, and reports **which combination
buys the most quality per dollar**.

## Why this image

The bundled 2,250-cell experiment is deterministic and needs no API key, so the
container is a reproducible demo of the whole pipeline:

```bash
docker build -t tokbench .
docker run --rm tokbench                       # run the smoke experiment
docker run --rm -p 8501:8501 tokbench panel    # dashboard
```

For real providers, pass credentials at run time (never bake them into the image):

```bash
docker run --rm -e DEEPSEEK_API_KEY=... -v "$PWD/reports:/app/reports" tokbench \
  tokbench run -c examples/experiment.yaml
```

## What is in the image

- the `tokbench` CLI (Typer) and the Streamlit panel
- the 50-task benchmark set and three example experiment configs
- mock mode: deterministic, offline, `$0`, no credentials
