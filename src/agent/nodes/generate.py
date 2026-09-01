from typing import Dict, Any
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState

def generate_node(state: AgentState) -> Dict[str, Any]:
    original_query = state["original_query"]
    
    # 1. Подключение к локальной базе и инициализация модели
    client = QdrantClient(path="data/qdrant_db")
    collection_name = "research_papers"
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    try:
        # 2. Векторизуем текстовый запрос пользователя
        query_vector = list(embedding_model.embed([original_query]))[0]
        
        # 3. Нативный векторный поиск через новый Query API
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            limit=8
        )
    except Exception as e:
        print(f"Ошибка при поиске в Qdrant (возможно, коллекция пуста): {e}")
        search_results = []

    if not search_results:
        return {
            "final_answer": "Извините, не удалось найти релевантную информацию в скачанных статьях.",
            "current_status": "Поиск завершен, данных нет."
        }

    # 4. Формирование контекстного окна
    context_blocks = []
    
    # В зависимости от версии Qdrant Client возвращает либо список, либо объект QueryResponse
    results_list = search_results.points if hasattr(search_results, 'points') else search_results
    
    for res in results_list:
        meta = res.payload
        text = meta.get("text", "")
        title = meta.get("title", "Неизвестная статья")
        
        context_blocks.append(f"[Источник: {title}]\n{text}")
        
    context_str = "\n\n---\n\n".join(context_blocks)

    # 5. Генерация ответа через LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты — эксперт-исследователь и AI-ассистент. 
        Ответь на вопрос пользователя, опираясь ТОЛЬКО на предоставленный контекст из научных статей. 
        Сравнивай методологии, если в контексте представлены разные подходы.
        Если информации в контексте недостаточно для полного ответа, прямо укажи это. 
        Обязательно ссылайся на источники в формате: (Название статьи)."""),
        ("user", "Вопрос: {query}\n\nКонтекст:\n{context}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "query": original_query, 
        "context": context_str
    })
    
    return {
        "final_answer": response.content,
        "current_status": "Ответ успешно сгенерирован."
    }