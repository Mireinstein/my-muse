# My Muse

A personal AI assistant that answers visitors' questions about me. Go give it a try at [admiremadyira.com](https://admiremadyira.com). Also deployed at [my-muse-wine.vercel.app](https://my-muse-wine.vercel.app).

## How it works

What it knows about me comes from my resume, a short about note, and my public
GitHub repos. `build_corpus.py` pulls all three and splits them into chunks of up
to 800 characters. The resume is split by entry, so a job's bullets always stay
with its title and dates. Each chunk is embedded once with Gemini's
`gemini-embedding-001`, and the chunks and vectors are committed as
`api/index.json`.

Each question goes to a single serverless function, `api/chat.py`:

1. Llama Prompt Guard 2 scores the message, and anything it flags as an injection
   or jailbreak gets a fixed reply without reaching the answering model.
2. The question is ranked against the chunks twice, by embedding similarity and by
   BM25 keyword score, and the two rankings are merged with reciprocal rank fusion.
   A follow-up is searched together with the question before it, so "and when was
   that?" still finds the right chunk.
3. The top 8 chunks go to `gpt-oss-20b` on Groq as numbered excerpts. The model
   cites the ones it used, and those come back as sources shown under the answer.
   If the visitor wants to meet, the model calls a booking tool and the chat opens
   my Calendly scheduler.

The index is small enough to rank in plain Python, so there is no vector database
to host. Every request logs one JSON line with the guard score, retrieved and cited
chunks, tokens and latency, which Vercel keeps as searchable logs.

## Evaluation

`eval/questions.jsonl` holds 53 questions: facts, multi-fact synthesis, follow-ups,
things it shouldn't know, booking requests, a red-team set, and legitimate
questions about my security work that the guard must not block. `eval/run.py`
scores retrieval recall, judges each answer against the source material with
`qwen3.8-27b` (a different model family from the one answering), and checks the
booking and guard decisions.

Latest full run, 52 of 53 passing:

| Category | Questions | Answer correct | recall@8 |
|---|---|---|---|
| Facts | 27 | 100% | 100% |
| Multi-fact synthesis | 4 | 75% | 75% |
| Follow-ups | 3 | 100% | 100% |
| Things it shouldn't know | 4 | 100% | - |
| Booking requests | 2 | 100% | - |
| Red-team attacks | 10 | 100% | - |
| Legitimate security questions | 3 | 100% | 100% |
| **All** | **53** | **98%** | **97%** |

The one miss asks about his AI safety experience and leaves out the watermarking
project, whose chunk never uses the word "safety" and isn't retrieved.

Accuracy went from 89% to 95% when the resume was split by entry instead of
fixed-size windows, and to 98% with top-8 hybrid retrieval and the guard.

Retrieval on the 34 questions that have evidence to find:

| Retrieval | recall@8 | recall@3 | MRR |
|---|---|---|---|
| Embeddings only | 97% | 85% | 0.87 |
| BM25 only | 91% | 85% | 0.82 |
| Hybrid (default) | 97% | 82% | 0.87 |

On this corpus hybrid ties embeddings alone, so it stays as a guard against exact
terms the embeddings miss rather than as a measured win.

On the red-team set, all 10 attacks fail. The guard blocks 5 or 6 depending on the
run and the system prompt refuses the rest. None of the 3 legitimate questions
about prompt injection are blocked.

```bash
python eval/run.py               # full suite
python eval/run.py --retrieval   # dense vs BM25 vs hybrid, no model calls
```

## Update the corpus

```bash
pip install pypdf
python build_corpus.py   # needs GEMINI_API_KEY in .env
git add api/corpus.txt api/index.json && git commit -m "Refresh the corpus" && git push
```

Vercel redeploys on push.

## Tech stack

- **Frontend:** React, TypeScript, Vite, Tailwind CSS
- **Backend:** Python serverless function on Vercel
- **Model:** `gpt-oss-20b` via the Groq API
- **Embeddings:** `gemini-embedding-001` via the Gemini API
- **Guard:** Llama Prompt Guard 2 (86M) via the Groq API
