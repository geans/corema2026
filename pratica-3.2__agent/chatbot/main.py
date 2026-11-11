
import asyncio
import os

import tools
from agno.agent import Agent
from agno.knowledge.embedder.base import Embedder
from agno.knowledge.knowledge import Knowledge
from agno.models.google import Gemini
from agno.os import AgentOS
from agno.vectordb.qdrant import Qdrant
from google import genai
from google.genai import types

QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBEDDING_MODEL = "gemini-embedding-2"
VECTOR_SIZE = 3072


class GeminiEmbedding2(Embedder):
    def __init__(self):
        super().__init__(dimensions=VECTOR_SIZE)
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def get_embedding(self, text: str) -> list:
        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=f"task: search result | query: {text}",
            config=types.EmbedContentConfig(
                output_dimensionality=VECTOR_SIZE
            ),
        )
        return response.embeddings[0].values

    def get_embedding_and_usage(self, text: str):
        return self.get_embedding(text), None

    async def async_get_embedding(self, text: str) -> list:
        return await asyncio.to_thread(self.get_embedding, text)

    async def async_get_embedding_and_usage(self, text: str):
        return await self.async_get_embedding(text), None


knowledge = Knowledge(
    vector_db=Qdrant(
        collection=COLLECTION_NAME,
        url=QDRANT_URL,
        embedder=GeminiEmbedding2(),
    )
)

agent = Agent(
    tools=[tools.estimate_solar_generation],
    name="Chatbot",
    model=Gemini(id="gemini-3.5-flash-lite"),
    knowledge=knowledge,
    search_knowledge=True,
    instructions=[
        "Você é um assistente especializado em Recursos Energéticos e Meio Ambiente do COREMA-2026.",
        "Lembre-se que o COREMA-2026 é um Congresso Regional de Energia e Meio Ambiente, realizado por estudantes de Engenharia de Energia da UFAL/CECA, sob coordenação do Prof. Will Júnior e da Profa. Amanda Peiter.",
        "Responda em português, utilizando linguagem clara, objetiva e didática.",
        "Utilize a base de conhecimento para responder.",
        "Não invente informações.",
        "Quando não encontrar informações relevantes, informe isso.",
    ],
    markdown=True,
)

agent_os = AgentOS(agents=[agent])
app = agent_os.get_app()
