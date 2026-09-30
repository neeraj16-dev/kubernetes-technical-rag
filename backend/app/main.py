from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.services.rag_service import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_service = None
    app.state.rag_ready = False

    async def initialize_rag():
        try:
            print("Starting RAG service initialization...")

            rag_service = await asyncio.to_thread(RAGService)

            app.state.rag_service = rag_service
            app.state.rag_ready = True

            print("RAG service initialization complete.")

        except Exception as e:
            app.state.rag_ready = False
            print(f"RAG service initialization failed: {e}")

    # Start initialization automatically when FastAPI starts
    asyncio.create_task(initialize_rag())

    yield

    app.state.rag_service = None
    app.state.rag_ready = False


app = FastAPI(
    title="Kubernetes Technical RAG Service",
    description="Production-grade hybrid search, reranked, and citation-grounded RAG API.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)