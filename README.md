# GRAFF (Granular Retrieval And Factual Framework) v0.4.0

AI-powered educational chapter analysis system that transforms textbook chapters into structured, queryable knowledge databases for adaptive tutoring systems.

## What It Does

GRAFF extracts a complete knowledge representation from textbook chapters:

```
Chapter Text → Structure → Propositions → Key Takeaways
```

- **Structure**: Hierarchical sections, summary, entities, keywords
- **Propositions**: Atomic facts tagged with Bloom's Taxonomy levels
- **Key Takeaways**: Higher-order insights synthesizing across propositions

## Three-Pass Architecture

GRAFF uses a three-pass pipeline with **incremental saves** after each pass:

```
Pass 1: STRUCTURE                              → SAVE
────────────────────────────────────────────────────────
Input:  Chapter text
Output: Sections, summary, entities, keywords
        │
        ▼ (feeds into)

Pass 2: PROPOSITIONS (per-section with chunking) → SAVE per section
────────────────────────────────────────────────────────
Input:  Chapter text + structure from Pass 1
Output: All atomic facts, tagged with unit_ids
        │
        ▼ (feeds into)

Pass 3: KEY TAKEAWAYS                          → SAVE
────────────────────────────────────────────────────────
Input:  Structure + ALL propositions
Output: Synthesized insights linking proposition_ids
```

**Why incremental saves?**
- If Pass 2 fails on section 30/46, sections 1-29 are already saved
- If Pass 3 fails, all structure and propositions are preserved
- No more losing hours of analysis to a late-stage error

## Features

- **Claude Opus 4.5 Powered**: Uses Anthropic's most capable model
- **Incremental Saves**: Data persisted after each pass/section
- **Smart Chunking**: Large sections split into 4k char chunks to avoid truncation
- **JSON Salvage**: Recovers partial results from truncated LLM responses
- **Retry Logic**: Escalates max_tokens (12K→16K→20K) on truncation errors
- **Three-Pass Analysis**: Structure → Propositions → Takeaways
- **Bloom's Taxonomy Tagging**: Propositions tagged as remember/understand/apply/analyze
- **Cross-Section Synthesis**: Takeaways can bridge across multiple sections
- **Real-Time Progress**: Server-Sent Events (SSE) for live updates
- **Modern Web Interface**: Drag-and-drop upload, dark/light mode, tabbed results
- **File Format Support**: `.txt`, `.docx`, `.pdf`
- **SQLite Storage**: Persistent database with full-text search

## Quick Start

```bash
# Install dependencies
pip install -e .

# Set up Anthropic API key
export ANTHROPIC_API_KEY="your-api-key-here"

# Run the server
uvicorn src.app:app --reload --port 8000

# Open in browser
open http://localhost:8000
```

## Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  PASS 1: Structure                                              │
│  Input:  Raw chapter text                                       │
│  Output: Sections, summary, entities, keywords                  │
│  → SAVES immediately to DB                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PASS 2: Propositions (per-section with chunking)               │
│  For each section:                                              │
│    • Extract section text                                       │
│    • Split into 4k char chunks if needed                        │
│    • LLM extracts propositions with Bloom levels                │
│    • Deduplicate across chunk overlaps                          │
│    → SAVES after each section                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PASS 3: Takeaways                                              │
│  Input:  All sections + all propositions                        │
│  Output: Key takeaways linking multiple propositions            │
│  • Invalid proposition references filtered out                  │
│  → SAVES to DB                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Resilience Features

| Failure Point | What's Preserved |
|---------------|------------------|
| Pass 2 fails on section 30/46 | Sections 1-29 propositions saved |
| Pass 3 fails entirely | All structure + propositions saved |
| JSON truncated | Partial results salvaged |
| Invalid prop references | Filtered out, doesn't fail save |

## Output Structure

GRAFF produces a `ChapterAnalysis` with three connected layers:

```json
{
  "chapter_id": "ch_abc123",
  "chapter_title": "The Studio System",

  "phase1": {
    "summary": "One-paragraph overview...",
    "sections": [
      {"unit_id": "1.1", "title": "Introduction", "level": 1},
      {"unit_id": "1.2", "title": "Vertical Integration", "level": 1},
      {"unit_id": "1.2.1", "title": "Production Control", "level": 2, "parent_unit_id": "1.2"}
    ],
    "key_entities": [
      {"name": "Paramount Pictures", "type": "organization"},
      {"name": "vertical integration", "type": "concept"}
    ],
    "keywords": ["studio system", "block booking", "exhibition"]
  },

  "phase2": {
    "propositions": [
      {
        "proposition_id": "ch_abc123_1.2_p001",
        "unit_id": "1.2",
        "proposition_text": "Vertical integration refers to studio ownership of production, distribution, and exhibition.",
        "bloom_level": "remember",
        "bloom_verb": "define",
        "evidence_location": "1.2:¶002"
      }
    ],
    "key_takeaways": [
      {
        "takeaway_id": "ch_abc123_t001",
        "text": "Vertical integration allowed studios to dominate by controlling the entire supply chain.",
        "proposition_ids": ["ch_abc123_1.2_p001", "ch_abc123_1.2_p003", "ch_abc123_1.3_p002"],
        "dominant_bloom_level": "analyze"
      }
    ]
  }
}
```

## Bloom's Taxonomy Levels

### Propositions (atomic facts)
| Level | Description | Example |
|-------|-------------|---------|
| `remember` | Definitions, dates, facts | "Block booking was banned in 1948." |
| `understand` | Explanations, cause-effect | "Vertical integration eliminated dependency on third parties." |
| `apply` | Concrete examples | "Paramount owned 1,450 theaters by 1948." |
| `analyze` | Comparisons, relationships | "The Big Five owned theaters while the Little Three did not." |

### Key Takeaways (synthesis)
| Level | Description | Example |
|-------|-------------|---------|
| `analyze` | Patterns, relationships | "Theater ownership created a self-reinforcing market advantage." |
| `evaluate` | Judgments, significance | "The Paramount decision proved only partially effective." |

## Project Structure

```
GRAFF/
├── src/
│   ├── app.py                      # FastAPI server
│   ├── models.py                   # Pydantic models
│   ├── db/
│   │   ├── connection.py           # SQLite persistence layer
│   │   └── schema.sql              # Database schema
│   └── services/
│       ├── anthropic_client.py     # Claude Opus 4.5 API client
│       ├── llm_client.py           # Three-pass analysis functions
│       └── graff_orchestrator.py   # Pipeline with incremental saves
├── prompts/
│   ├── pass1_structure.txt         # Pass 1 prompt
│   ├── pass2_section.txt           # Pass 2 per-section prompt
│   └── pass3_takeaways.txt         # Pass 3 prompt
├── static/
│   ├── index.html                  # Web interface
│   └── js/app.js                   # Client-side logic
├── sample_data/                    # Sample chapters for testing
└── graff.db                        # SQLite database
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Web UI |
| `/chapters/digest` | POST | Start analysis (returns job_id) |
| `/chapters/progress/{job_id}` | GET | SSE stream of progress |
| `/chapters/list` | GET | List all chapters |
| `/chapters/{id}` | GET | Get full chapter analysis |
| `/chapters/{id}` | DELETE | Delete chapter |
| `/chapters/{id}/propositions` | GET | Get propositions (filter by bloom) |
| `/samples` | GET | List sample data files |
| `/samples/{filename}` | GET | Get sample file content |

### Example: Start Analysis

```bash
curl -X POST http://localhost:8000/chapters/digest \
  -F file=@chapter.txt \
  -F chapter_id=ch-001
```

### Example: Stream Progress

```
GET /chapters/progress/{job_id}

data: {"phase":"pass-1","message":"Extracting structure..."}
data: {"phase":"storage","message":"Structure saved (12 sections)"}
data: {"phase":"pass-2","message":"Section 5/12: Methodology"}
data: {"phase":"pass-3","message":"Synthesizing takeaways..."}
data: {"phase":"completed","message":"Done! 285 propositions, 24 takeaways"}
```

## Configuration

```bash
# Required
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional (with defaults)
export ANTHROPIC_MODEL="claude-opus-4-5-20251101"
export ANTHROPIC_MAX_TOKENS="16000"
export ANTHROPIC_TIMEOUT="300"
export ANTHROPIC_MAX_RETRIES="5"
```

## File Format Support

| Format | Library | Notes |
|--------|---------|-------|
| `.txt` | Built-in | UTF-8 or Latin-1 |
| `.docx` | python-docx | Extracts paragraph text |
| `.pdf` | PyPDF2 | Text-based PDFs only |

**Limits**: 100MB max upload, handles chapters of any length via chunking.

## Changes in v0.4.0

### LLM Switch
- **Claude Opus 4.5** replaces GPT-5.2
- Uses Anthropic SDK with retry logic for rate limits

### Resilient Pipeline
- **Incremental saves** after each pass and section
- **Section chunking** splits large sections into 4k char chunks
- **JSON salvage** recovers partial results from truncated responses
- **Retry with escalation** increases max_tokens on truncation errors
- **Invalid reference filtering** removes bad proposition IDs instead of failing

### Database
- New incremental save functions: `save_chapter_phase1`, `save_propositions`, `save_takeaways`
- Full-text search on propositions via FTS5

### Removed
- All-at-once save that could lose entire analysis on failure
- Hard truncation of large sections (now chunked instead)

## Use Cases

GRAFF powers downstream systems:

- **Adaptive Tutoring**: Query propositions by Bloom level for scaffolded learning
- **Knowledge Graphs**: Takeaways link propositions into semantic networks
- **Assessment Generation**: Generate questions from propositions at target cognitive levels
- **Content Gap Analysis**: Identify sections lacking higher-order (analyze/evaluate) content

## License

Proprietary - All rights reserved
