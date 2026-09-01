from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel

class DocumentMetadata(BaseModel):
    title: str
    authors: List[str]
    year: str
    url: str
    local_path: str | None = None

class AgentState(TypedDict):
    # Входные данные от пользователя
    original_query: str
    
    # Сгенерированные поисковые запросы для API (например, ArXiv)
    search_queries: List[str]
    
    # Найденные статьи и их метаданные
    found_papers: List[DocumentMetadata]
    
    # Статьи, которые прошли фильтрацию и были успешно скачаны
    downloaded_papers: List[DocumentMetadata]
    
    # Статус работы агента (для стриминга на фронтенд)
    current_status: str
    
    # Финальный ответ LLM
    final_answer: str