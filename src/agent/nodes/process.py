import pymupdf
import uuid
from typing import Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.agent.state import AgentState

def process_node(state: AgentState) -> Dict[str, Any]:
    downloaded_papers = state.get("downloaded_papers", [])
    
    if not downloaded_papers:
        return {"current_status": "Нет скачанных статей для обработки."}
    
    # 1. Инициализация базы и модели напрямую
    client = QdrantClient(path="data/qdrant_db")
    collection_name = "research_papers"
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # 2. Создание коллекции (если ее еще нет)
    # Размер вектора 384 — это стандарт для bge-small-en-v1.5
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    
    # 3. Настройка чанкинга
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " "]
    )
    
    documents_to_insert = []
    metadata_to_insert = []
    ids_to_insert = []
    
    for paper in downloaded_papers:
        if not paper.local_path:
            continue
            
        try:
            doc = pymupdf.open(paper.local_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + "\n"
                
            chunks = text_splitter.split_text(full_text)
            
            for chunk in chunks:
                documents_to_insert.append(chunk)
                # Сохраняем сам текст прямо в метаданные (payload)
                metadata_to_insert.append({
                    "title": paper.title,
                    "url": paper.url,
                    "year": paper.year,
                    "text": chunk
                })
                ids_to_insert.append(uuid.uuid4().hex)
                
        except Exception as e:
            print(f"Ошибка при парсинге {paper.title}: {e}")
            
    # 4. Векторизация и загрузка в базу через нативный метод upsert
    if documents_to_insert:
        # Явно генерируем векторы (возвращает генератор, оборачиваем в list)
        embeddings = list(embedding_model.embed(documents_to_insert))
        
        # Формируем структуры данных для Qdrant
        points = [
            PointStruct(id=uid, vector=vector.tolist(), payload=meta)
            for uid, vector, meta in zip(ids_to_insert, embeddings, metadata_to_insert)
        ]
        
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
    return {
        "current_status": f"Распарсено и сохранено {len(documents_to_insert)} фрагментов."
    }