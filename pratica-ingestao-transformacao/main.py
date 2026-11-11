
import asyncio
import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import fitz  # PyMuPDF
import httpx
import trafilatura
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, HttpUrl

app = FastAPI(
    title="Web and PDF Extractor API",
    description="API para extração de conteúdo textual de páginas web e PDFs.",
    version="1.0.0",
)

TIMEOUT = 30
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB
MAX_REDIRECTS = 5


class URLRequest(BaseModel):
    url: HttpUrl


def validate_public_url(url: str):
    """
    Aceita somente URLs HTTP/HTTPS com endereços IP públicos.
    Ajuda a evitar ataques SSRF.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise HTTPException(400, "Somente URLs HTTP e HTTPS são permitidas.")

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
        raise HTTPException(400, "Não foi possível resolver o endereço informado.")


async def fetch_url(url: str) -> bytes:
    """
    Baixa o conteúdo de uma URL, validando redirecionamentos
    e limitando o tamanho da resposta.
    """
    async with httpx.AsyncClient(
        timeout=TIMEOUT,
        follow_redirects=False,
        trust_env=False,
        headers={"User-Agent": "WebPDFExtractor/1.0"},
    ) as client:

        for _ in range(MAX_REDIRECTS + 1):
            validate_public_url(url)

            async with client.stream("GET", url) as response:

                if response.is_redirect:
                    location = response.headers.get("location")

                    if not location:
                        raise HTTPException(502, "Redirecionamento inválido.")

                    url = urljoin(url, location)
                    continue

                response.raise_for_status()

                content_length = response.headers.get("content-length")

                if content_length and int(content_length) > MAX_FILE_SIZE:
                    raise HTTPException(413, "Arquivo excede o tamanho máximo permitido.")

                content = bytearray()

                async for chunk in response.aiter_bytes():
                    content.extend(chunk)

                    if len(content) > MAX_FILE_SIZE:
                        raise HTTPException(413, "Arquivo excede o tamanho máximo permitido.")

                return bytes(content)

        raise HTTPException(502, "Número máximo de redirecionamentos excedido.")


@app.post("/web")
async def extract_web(request: URLRequest):
    """
    Recebe uma URL e retorna o conteúdo principal da página
    convertido para Markdown.
    """
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
                "Não foi possível extrair conteúdo textual da página.",
            )

        return Response(content=markdown, media_type="text/markdown")


    except httpx.HTTPStatusError as e:
        raise HTTPException(
            502,
            f"Erro HTTP ao acessar o site: {e.response.status_code}",
        )

    except httpx.RequestError as e:
        raise HTTPException(502, f"Erro ao acessar o site: {e}")


@app.post("/pdf")
async def extract_pdf(request: URLRequest):
    """
    Recebe uma URL de PDF e retorna o texto extraído.
    """
    url = str(request.url)

    try:
        content = await fetch_url(url)

        # Verifica a assinatura do arquivo PDF
        if not content.startswith(b"%PDF-"):
            raise HTTPException(422, "O endereço informado não contém um PDF válido.")

        def extract_text():
            document = fitz.open(stream=content, filetype="pdf")

            try:
                if document.is_encrypted:
                    raise HTTPException(
                        422,
                        "Não é possível extrair texto de um PDF protegido por senha.",
                    )

                pages = []

                for page_number, page in enumerate(document, start=1):
                    pages.append({
                        "page": page_number,
                        "text": page.get_text("text"),
                    })

                return pages

            finally:
                document.close()

        pages = await asyncio.to_thread(extract_text)

        full_text = "\n\n".join(
            f"## Página {page['page']}\n\n{page['text']}"
            for page in pages
        )

        return Response(content=full_text, media_type="text/markdown")

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            502,
            f"Erro HTTP ao baixar PDF: {e.response.status_code}",
        )

    except httpx.RequestError as e:
        raise HTTPException(502, f"Erro ao baixar PDF: {e}")
