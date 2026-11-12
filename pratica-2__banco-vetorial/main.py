
import asyncio
import ipaddress
import os
import socket
import uuid
from contextlib import asynccontextmanager
from urllib.parse import urljoin, urlparse

import fitz
import httpx
import trafilatura
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel, HttpUrl
from qdrant_client import QdrantClient, models

# ============================================================
# CONFIGURAÇÕES
# ============================================================

TIMEOUT = 30
MAX_FILE_SIZE = 15 * 1024 * 1024
MAX_REDIRECTS = 5


QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

EMBEDDING_MODEL = "gemini-embedding-2"
VECTOR_SIZE = 3072

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 64

qdrant = None
embedding_model = None


# ============================================================
# INICIALIZAÇÃO
# ============================================================

def initialize_services():
    global qdrant, embedding_model

    qdrant = QdrantClient(url=QDRANT_URL)
    embedding_model = genai.Client(api_key=GOOGLE_API_KEY)

    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(initialize_services)
    yield

    if qdrant:
        qdrant.close()


app = FastAPI(
    title="Web and PDF Extractor API",
    description="Extração de conteúdo web e PDF com armazenamento vetorial no Qdrant.",
    version="2.0.0",
    lifespan=lifespan,
)


# ============================================================
# MODELOS
# ============================================================

class URLRequest(BaseModel):
    url: HttpUrl


# ============================================================
# DOWNLOAD E VALIDAÇÃO
# ============================================================

def validate_public_url(url: str):
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise HTTPException(
            400,
            "Somente URLs HTTP e HTTPS são permitidas."
        )

    if not parsed.hostname:
        raise HTTPException(400, "URL inválida.")

    try:
        addresses = socket.getaddrinfo(
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )

        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])

            if not ip.is_global:
                raise HTTPException(
                    400,
                    "URLs com endereços IP privados ou reservados não são permitidas.",
                )

    except HTTPException:
        raise

    except socket.gaierror:
        raise HTTPException(
            400,
            "Não foi possível resolver o endereço informado."
        )


async def fetch_url(url: str) -> bytes:

    async with httpx.AsyncClient(
        timeout=TIMEOUT,
        follow_redirects=False,
        trust_env=False,
        headers={"User-Agent": "WebPDFExtractor/2.0"},
    ) as client:

        for _ in range(MAX_REDIRECTS + 1):

            validate_public_url(url)

            async with client.stream("GET", url) as response:

                if response.is_redirect:
                    location = response.headers.get("location")

                    if not location:
                        raise HTTPException(
                            502,
                            "Redirecionamento inválido."
                        )

                    url = urljoin(url, location)
                    continue

                response.raise_for_status()

                content_length = response.headers.get("content-length")

                if content_length and int(content_length) > MAX_FILE_SIZE:
                    raise HTTPException(
                        413,
                        "Arquivo excede o tamanho máximo permitido."
                    )

                content = bytearray()

                async for chunk in response.aiter_bytes():

                    content.extend(chunk)

                    if len(content) > MAX_FILE_SIZE:
                        raise HTTPException(
                            413,
                            "Arquivo excede o tamanho máximo permitido."
                        )

                return bytes(content)

        raise HTTPException(
            502,
            "Número máximo de redirecionamentos excedido."
        )


# ============================================================
# CHUNKING
# ============================================================

def split_text(text: str):
    """
    Divide o texto em chunks com sobreposição.
    Tenta preservar palavras inteiras.
    """

    words = text.split()

    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:

        word_size = len(word) + 1

        if current_size + word_size > CHUNK_SIZE and current_chunk:

            chunks.append(" ".join(current_chunk))

            overlap_words = []
            overlap_size = 0

            for previous_word in reversed(current_chunk):

                size = len(previous_word) + 1

                if overlap_size + size > CHUNK_OVERLAP:
                    break

                overlap_words.insert(0, previous_word)
                overlap_size += size

            current_chunk = overlap_words
            current_size = overlap_size

        current_chunk.append(word)
        current_size += word_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ============================================================
# ARMAZENAMENTO NO QDRANT
# ============================================================

def save_documents(
    source_url: str,
    document_type: str,
    documents: list,
):
    """
    Gera embeddings e salva os documentos no Qdrant.

    documents:
        [
            {
                "text": "...",
                "page": 1
            }
        ]
    """

    points = []

    for document in documents:

        text = document["text"]
        page = document.get("page")

        chunks = split_text(text)

        if not chunks:
            continue

        # O prefixo passage: é recomendado pelo modelo E5
        result = embedding_model.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[
                types.Content(
                    parts=[types.Part.from_text(text=f"title: none | text: {chunk}")]
                )
                for chunk in chunks
            ],
        )

        embeddings = result.embeddings

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{source_url}:{document_type}:{page}:{index}"
                )
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding.values,
                    payload={
                        "source_url": source_url,
                        "document_type": document_type,
                        "page": page,
                        "chunk_index": index,
                        "text": chunk,
                    },
                )
            )

    if not points:
        raise HTTPException(
            422,
            "Nenhum conteúdo textual foi encontrado para armazenar."
        )

    # Remove os registros anteriores da mesma origem e tipo.
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.Filter(
            must=[
                models.FieldCondition(
                    key="source_url",
                    match=models.MatchValue(value=source_url),
                ),
                models.FieldCondition(
                    key="document_type",
                    match=models.MatchValue(value=document_type),
                ),
            ]
        ),
        wait=True,
    )

    # Insere os novos vetores em lotes.
    for start in range(0, len(points), BATCH_SIZE):

        batch = points[start:start + BATCH_SIZE]

        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=True,
        )

    return len(points)


# ============================================================
# ENDPOINT WEB
# ============================================================

@app.post("/web")
async def extract_web(request: URLRequest):

    url = str(request.url)

    try:

        html = await fetch_url(url)

        markdown = await asyncio.to_thread(
            trafilatura.extract,
            html,
            output_format="markdown",
            include_links=True,
            include_tables=True,
        )

        if not markdown:
            raise HTTPException(
                422,
                "Não foi possível extrair conteúdo textual da página."
            )

        documents = [
            {
                "text": markdown,
                "page": None,
            }
        ]

        total_chunks = await asyncio.to_thread(
            save_documents,
            url,
            "web",
            documents,
        )

        return {
            "message": "Conteúdo da página armazenado com sucesso.",
        }

    except HTTPException:
        raise

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            502,
            f"Erro HTTP ao acessar o site: {e.response.status_code}",
        )

    except httpx.RequestError as e:
        raise HTTPException(
            502,
            f"Erro ao acessar o site: {e}",
        )

    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            500,
            f"Erro ao processar e armazenar conteúdo: {e}",
        )


# ============================================================
# ENDPOINT PDF
# ============================================================

@app.post("/pdf")
async def extract_pdf(request: URLRequest):

    url = str(request.url)

    try:

        content = await fetch_url(url)

        if not content.startswith(b"%PDF-"):
            raise HTTPException(
                422,
                "O endereço informado não contém um PDF válido."
            )

        def extract_text():

            document = fitz.open(
                stream=content,
                filetype="pdf",
            )

            try:

                if document.is_encrypted:
                    raise HTTPException(
                        422,
                        "Não é possível extrair texto de um PDF protegido por senha."
                    )

                pages = []

                for page_number, page in enumerate(
                    document,
                    start=1,
                ):

                    text = page.get_text("text")

                    if text.strip():
                        pages.append({
                            "text": text,
                            "page": page_number,
                        })

                return pages

            finally:
                document.close()

        pages = await asyncio.to_thread(extract_text)

        if not pages:
            raise HTTPException(
                422,
                "Nenhum texto foi encontrado no PDF. Talvez seja necessário utilizar OCR."
            )

        total_chunks = await asyncio.to_thread(
            save_documents,
            url,
            "pdf",
            pages,
        )

        return {
            "message": "PDF processado e armazenado com sucesso.",
        }

    except HTTPException:
        raise

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            502,
            f"Erro HTTP ao baixar PDF: {e.response.status_code}",
        )

    except httpx.RequestError as e:
        raise HTTPException(
            502,
            f"Erro ao baixar PDF: {e}",
        )

    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            500,
            f"Erro ao processar e armazenar PDF: {e}",
        )
