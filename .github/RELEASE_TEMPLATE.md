# vX.Y - <short title>

## What's new

- ...

## How to run this version

```bash
git checkout vX.Y
uv sync --extra dev
cp .env.example .env   # add ANTHROPIC_API_KEY
uv run rag ingest && uv run rag eval
```

## Results

Same evaluation dataset as every other version.

| Hit@5 | MRR | Faithfulness | Latency (p50) |
|---|---|---|---|
| | | | |

Full output: `results/vX.Y/`

## Limitations found

- ...

## Next version

What these limitations motivate: ...

**Full changelog:** https://github.com/<user>/<repo>/compare/vPREVIOUS...vX.Y
