# ResearchRAG frontend

Next.js 16 UI for ResearchRAG. See the [project README](../README.md) for the full setup.

```bash
npm install
cp .env.example .env.local   # NEXT_PUBLIC_API_URL, default http://localhost:8000
npm run dev                  # http://localhost:3000
```

| Route | Page |
|---|---|
| `/` | Library: import from arXiv, upload PDFs, see ingestion status |
| `/search` | Hybrid semantic search with metadata filters |
| `/chat` | Streamed, cited answers with a sources panel |
| `/eval` | Question sets, eval runs, and retrieval and answer-quality metrics |
