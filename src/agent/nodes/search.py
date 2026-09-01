import arxiv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Dict, Any

# Импортируем наши структуры из state.py
from src.agent.state import AgentState, DocumentMetadata

# Структура для строгого ответа LLM
class SearchQuery(BaseModel):
    queries: list[str] = Field(
        description="Список из 1-3 оптимизированных поисковых запросов для ArXiv (например: 'ti:\"causal inference\" AND cat:cs.LG')"
    )

def search_node(state: AgentState) -> Dict[str, Any]:
    original_query = state["original_query"]

    # 1. LLM генерирует оптимальные запросы для ArXiv API
    # Используем gpt-4o-mini, так как задача маршрутизации и генерации тегов простая
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Ты — академический AI-ассистент. Переведи запрос пользователя в 1-3 точных поисковых запроса для базы ArXiv. Обязательно переводи термины на английский язык. Выделяй главные концепции."),
        ("user", "{query}")
    ])
    
    # Принудительно заставляем LLM вернуть JSON, соответствующий схеме SearchQuery
    chain = prompt | llm.with_structured_output(SearchQuery)
    generated = chain.invoke({"query": original_query})
    search_queries = generated.queries

    # 2. Инициализация клиента ArXiv и выполнение поиска
    client = arxiv.Client()
    found_papers = []
    seen_urls = set() # Множество для фильтрации дубликатов
    
    for q in search_queries:
        search = arxiv.Search(
            query=q,
            max_results=3, # Ограничиваем выдачу, чтобы не скачивать сотни PDF
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        try:
            for paper in client.results(search):
                if paper.pdf_url not in seen_urls:
                    seen_urls.add(paper.pdf_url)
                    doc = DocumentMetadata(
                        title=paper.title,
                        authors=[author.name for author in paper.authors],
                        year=str(paper.published.year),
                        url=paper.pdf_url
                    )
                    found_papers.append(doc)
        except Exception as e:
            print(f"Ошибка при поиске запроса '{q}': {e}")
            continue
    
    # 3. LangGraph автоматически обновит эти ключи в глобальном State
    return {
        "search_queries": search_queries,
        "found_papers": found_papers,
        "current_status": f"Поиск завершен. Найдено {len(found_papers)} статей."
    }