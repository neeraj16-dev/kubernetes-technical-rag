# 🚀 Kubernetes Technical RAG

> **Advanced Retrieval-Augmented Generation system for answering Kubernetes technical questions using hybrid retrieval, reciprocal rank fusion, cross-encoder reranking, structured citations, and grounding verification.**

This project builds a production-oriented RAG pipeline over a curated corpus of **Kubernetes technical documentation and security guidance**.

Instead of relying only on semantic vector search, the system combines **BM25 keyword retrieval + dense vector retrieval**, fuses the results using **Reciprocal Rank Fusion (RRF)**, reranks the candidates using a **Cross-Encoder**, and finally generates a citation-grounded answer using **Gemini**.

A separate verification stage checks whether the generated answer is actually supported by the retrieved sources.

---

## ✨ Features

* 🔎 **Hybrid Retrieval**

  * BM25 keyword search
  * Dense vector search
  * Reciprocal Rank Fusion (RRF)

* 🎯 **Cross-Encoder Reranking**

  * Reorders hybrid candidates based on query-document relevance
  * Final top 5 chunks are passed to the generator

* 📚 **Structured Document Ingestion**

  * 18 Kubernetes PDF documents
  * 1,667 pages
  * 8,679 structured chunks

* 🤖 **LLM Generation**

  * Gemini
  * Structured Pydantic output
  * Answers restricted to retrieved context

* 🔗 **Grounded Citations**

  * Every citation contains:

    * `chunk_id`
    * `source`
    * `page`

* 🛡️ **Grounding Verification**

  * Validates citation IDs
  * Detects phantom citations
  * Checks unsupported claims
  * Produces a hallucination score

* 📊 **Evaluation Pipeline**

  * Retrieval hit rate
  * MRR
  * Faithfulness
  * Answer relevance
  * Ground-truth comparison

* ⚡ **FastAPI Backend**

  * Modular service architecture
  * Swagger/OpenAPI documentation
  * Automatic RAG initialization on application startup

* 🎨 **Next.js Frontend**

  * Chat-based interface
  * Retrieved chunk inspection
  * Verification results
  * Performance metrics
  * Pipeline visibility

* 🐳 **Dockerized**

  * Backend container
  * Frontend container
  * Docker Compose orchestration

---

# 🏗️ Architecture

```text
                         ┌─────────────────────┐
                         │      User Query     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   FastAPI Backend   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │      Hybrid Retrieval        │
                    │                              │
                    │  ┌─────────┐   ┌─────────┐   │
                    │  │  BM25   │   │  Dense  │   │
                    │  │ Search  │   │ Search  │   │
                    │  └────┬────┘   └─────┬───┘   │
                    └───────┼──────────────┼───────┘
                            │              │
                            └──────┬───────┘
                                   ▼
                         ┌─────────────────────┐
                         │   RRF Fusion        │
                         │ Reciprocal Rank     │
                         │      Fusion         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Cross-Encoder       │
                         │    Reranking        │
                         └──────────┬──────────┘
                                    │
                              Top 5 Chunks
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Gemini Generator  │
                         │                     │
                         │ Structured Answer   │
                         │ + Citations         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Grounding           │
                         │ Verification        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Answer + Citations  │
                         │ + Verification      │
                         │ + Metrics           │
                         └─────────────────────┘
```

---

# 🔄 RAG Pipeline

The complete query pipeline is:

```text
Question
   ↓
BM25 Retrieval
   +
Dense Retrieval
   ↓
Reciprocal Rank Fusion
   ↓
Hybrid Candidate Set
   ↓
Cross-Encoder Reranking
   ↓
Top 5 Chunks
   ↓
Gemini Generation
   ↓
Structured Citations
   ↓
Grounding Verification
   ↓
Final Response
```

## 1. Document Ingestion

The corpus contains Kubernetes-related PDF documents.

Documents are loaded and processed using:

* PyMuPDF
* PyMuPDF4LLM

The extracted content preserves document structure where possible.

---

## 2. Chunking

Documents are divided into manageable retrieval units using:

```text
Chunk size:     500 characters
Chunk overlap:  100 characters
```

Each chunk receives a unique identifier.

Example:

```text
04_Workloads_p0_c119
```

The metadata contains:

```json
{
  "chunk_id": "04_Workloads_p0_c119",
  "source": "04_Workloads.pdf",
  "page": 0,
  "section": "..."
}
```

This allows retrieved information to be traced back to its original document and page.

---

# 🔎 Hybrid Retrieval

A single retrieval method has limitations.

### BM25

BM25 is effective for:

* Exact Kubernetes terminology
* Commands
* Configuration names
* Resource names
* Technical keywords

For example:

```text
hostPath
emptyDir
CrashLoopBackOff
RBAC
NetworkPolicy
```

### Dense Retrieval

Dense vector search captures semantic similarity.

For example, a query such as:

```text
Why does Kubernetes restart a container?
```

can retrieve content discussing container failures even if the exact wording differs.

The project uses:

```text
BAAI/bge-small-en-v1.5
```

for embeddings.

---

# 🔀 Reciprocal Rank Fusion

BM25 and dense retrieval produce two ranked lists.

Instead of choosing one retriever, the project combines their rankings using **Reciprocal Rank Fusion (RRF)**.

Conceptually:

```text
BM25 Results
     +
Dense Results
     ↓
    RRF
     ↓
Unified ranking
```

This allows lexical and semantic retrieval signals to complement each other.

---

# 🎯 Cross-Encoder Reranking

The hybrid retriever produces candidate chunks.

These candidates are then passed through:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The Cross-Encoder evaluates the query and candidate document together and produces a relevance score.

The candidates are reordered and the highest-ranked **5 chunks** are passed to the LLM.

```text
BM25 Top 20
       +
Dense Top 20
       ↓
     RRF
       ↓
Hybrid Candidates
       ↓
Cross-Encoder
       ↓
Final Top 5
```

---

# 🤖 Generation

The final retrieved context is passed to Gemini.

The generator uses a structured output model containing:

```text
answer
citations
```

The generation prompt explicitly instructs the model to:

* Answer only from the supplied context
* Avoid inventing Kubernetes commands
* Avoid inventing configuration options
* Say when the available context is insufficient
* Cite only retrieved chunks actually used

---

# 🔗 Citation System

Every citation returned by the generator contains:

```json
{
  "chunk_id": "04_Workloads_p0_c119",
  "source": "04_Workloads.pdf",
  "page": 0
}
```

The backend validates that every cited `chunk_id` actually exists in the retrieved context.

This prevents the model from creating citations to documents or chunks that were never retrieved.

---

# 🛡️ Grounding Verification

Generation is not treated as the final trust boundary.

After generating an answer, a separate verification stage checks the response.

The verifier produces:

```json
{
  "is_grounded": true,
  "unsupported_claims": [],
  "hallucination_score": 0.0,
  "explanation": "...",
  "phantom_citations": []
}
```

### Verification checks

#### Citation validation

Does the cited chunk actually exist?

#### Phantom citation detection

Did the model cite a chunk that wasn't retrieved?

#### Claim support

Are the claims in the answer supported by the cited context?

#### Hallucination score

```text
0.0 → grounded
1.0 → hallucinated / unsupported
```

This creates an additional guardrail between generation and the final response.

---

# 📊 Evaluation

The project includes an evaluation dataset and automated evaluation pipeline.

Current evaluation configuration uses **Top 5 reranked chunks**.

Results on the current 10-question evaluation set:

| Metric                     | Result |
| -------------------------- | -----: |
| Hit Rate                   |   100% |
| MRR                        |   1.00 |
| Faithfulness               |   100% |
| Answer Relevance           |   100% |
| Average Ground-Truth Score |    94% |

These results are based on the current evaluation dataset and are not intended to represent performance on every possible Kubernetes query.

The evaluation suite can be expanded as the corpus and question coverage grow.

---

# 📁 Project Structure

```text
k8s rag/
│
├── backend/
│   └── app/
│       ├── api/
│       │   ├── __init__.py
│       │   └── routes.py
│       │
│       ├── services/
│       │   ├── __init__.py
│       │   └── rag_service.py
│       │
│       ├── __init__.py
│       ├── main.py
│       └── schemas.py
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx
│   │   ├── page.module.css
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── Dockerfile
│   └── next.config.ts
│
├── rag/
│   ├── ingestion.py
│   ├── chunking.py
│   ├── indexing.py
│   ├── retrieval.py
│   ├── reranking.py
│   ├── generation.py
│   ├── verification.py
│   ├── evaluation.py
│   └── pipeline.py
│
├── data/
│   ├── chunks/
│   ├── extracted/
│   └── indices/
│       ├── chroma_db/
│       └── bm25_index.pkl
│
├── corpus/
│   └── Kubernetes PDF documents
│
├── tests/
│   ├── eval_dataset.json
│   ├── eval_report.json
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   ├── test_reranking.py
│   ├── test_generation.py
│   ├── test_evaluation.py
│   └── test_full_pipeline.py
│
├── docker-compose.yml
├── .dockerignore
└── README.md
```

---

# 🧩 Backend Architecture

The FastAPI backend follows a modular structure.

```text
Request
   ↓
API Route
   ↓
RAGService
   ↓
RAGPipeline
   ↓
Retrieval
   ↓
Reranking
   ↓
Generation
   ↓
Verification
   ↓
Response
```

### `main.py`

Responsible for:

* Creating the FastAPI application
* Configuring CORS
* Starting the RAG service automatically
* Managing application lifecycle

The RAG service begins initialization when the application starts instead of waiting for the first user query.

### `routes.py`

Contains API endpoints.

### `schemas.py`

Contains request and response models using Pydantic.

### `rag_service.py`

Acts as the service layer between FastAPI and the RAG pipeline.

### `pipeline.py`

Coordinates the complete RAG workflow.

---

# 🌐 API

## Health Check

```http
GET /health
```

Example:

```json
{
  "status": "healthy",
  "service": "Kubernetes Technical RAG Service"
}
```

## Query

```http
POST /api/v1/query
```

Request:

```json
{
  "query": "How does Kubernetes handle container crashes?",
  "top_n": 5,
  "verify_grounding": true
}
```

The response contains:

```text
answer
citations
verification
retrieved_chunks
metrics
```

---

# 📖 Swagger API Documentation

Once the backend is running:

```text
http://localhost:8000/api/docs
```

ReDoc:

```text
http://localhost:8000/api/redoc
```

---

# 🐳 Docker

The project includes Docker support for both the backend and frontend.

```text
                    Docker Compose
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
     ┌──────────────┐        ┌──────────────┐
     │   Backend    │        │   Frontend   │
     │   FastAPI    │        │   Next.js    │
     │   :8000      │        │   :3000      │
     └──────┬───────┘        └──────────────┘
            │
            ▼
       RAG Indexes
       BM25 + Chroma
```

## Start the application

From the project root:

```bash
docker compose build
```

Then:

```bash
docker compose up
```

Frontend:

```text
http://localhost:3000
```

Backend:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/api/docs
```

To run in detached mode:

```bash
docker compose up -d
```

To stop:

```bash
docker compose down
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

The `.env` file should **never be committed to Git**.

The frontend API URL is configured during the Docker build:

```text
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

# ⚙️ Local Development

## Backend

Activate the Python environment and install dependencies:

```bash
pip install -r backend/requirements.txt
```

Start FastAPI:

```bash
uvicorn backend.app.main:app --reload
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:3000
```

---

# 🧪 Running Tests

From the project root:

```bash
python -m pytest
```

Individual components can also be tested separately:

```bash
python -m tests.test_retrieval
python -m tests.test_reranking
python -m tests.test_generation
python -m tests.test_evaluation
python -m tests.test_full_pipeline
```

---

# 🛠️ Technology Stack

### Frontend

* Next.js
* React
* TypeScript
* CSS

### Backend

* Python
* FastAPI
* Pydantic
* Uvicorn

### RAG

* LangChain
* BM25
* ChromaDB
* Sentence Transformers
* Cross-Encoder
* Reciprocal Rank Fusion

### LLM

* Google Gemini

### Document Processing

* PyMuPDF
* PyMuPDF4LLM

### Evaluation

* Custom evaluation pipeline
* Retrieval metrics
* Answer relevance
* Faithfulness
* Grounding verification

### Deployment

* Docker
* Docker Compose

---

# 🎯 Design Goals

This project was designed around several important RAG principles:

### 1. Retrieval should not depend on one strategy

BM25 and dense retrieval solve different retrieval problems, so both are combined.

### 2. Retrieval quality should be separated from generation

The system explicitly separates:

```text
Retrieval
Reranking
Generation
Verification
```

This makes each stage measurable and replaceable.

### 3. LLM output should be traceable

Generated answers contain citations pointing back to specific chunks and pages.

### 4. Generation should not be the final trust boundary

A separate grounding verification stage checks whether generated claims are supported by retrieved information.

### 5. Evaluation should drive optimization

Chunk size, retrieval depth, reranking depth, and other pipeline decisions can be evaluated instead of being chosen purely by intuition.

---

# 🚀 Future Improvements

Possible future extensions include:

* Advanced RAG query transformation
* Multi-query retrieval
* Query decomposition
* Contextual compression
* Parent-child retrieval
* Metadata-aware filtering
* Better document/table understanding
* Image and diagram retrieval
* Larger evaluation datasets
* Retrieval latency optimization
* Streaming responses
* Authentication and rate limiting
* Production deployment

---

# 📌 Current Status

```text
Document Ingestion        ✅
Chunking                  ✅
BM25 Indexing             ✅
Dense Indexing            ✅
Hybrid Retrieval          ✅
RRF Fusion                ✅
Cross-Encoder Reranking   ✅
LLM Generation            ✅
Structured Citations      ✅
Grounding Verification    ✅
Evaluation                ✅
FastAPI Backend           ✅
Next.js Frontend          ✅
Dockerization             ✅
```

---

## 👨‍💻 Project

**Kubernetes Technical RAG**

An end-to-end RAG system focused on reliable retrieval, transparent citations, and grounded technical answers over Kubernetes documentation.
