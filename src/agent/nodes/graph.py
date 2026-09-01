from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes.search import search_node
from src.agent.nodes.download import download_node
from src.agent.nodes.process import process_node
from src.agent.nodes.generate import generate_node

def build_graph():
    # 1. Инициализируем граф с нашей структурой состояния
    workflow = StateGraph(AgentState)

    # 2. Регистрируем узлы графа
    # Ключ (строка) — это внутреннее имя узла, значение — импортированная функция
    workflow.add_node("search", search_node)
    workflow.add_node("download", download_node)
    workflow.add_node("process", process_node)
    workflow.add_node("generate", generate_node)

    # 3. Выстраиваем логику переходов (Edges)
    # Точка входа: при запуске граф всегда начинает с поиска
    workflow.set_entry_point("search")
    
    # Линейный пайплайн
    workflow.add_edge("search", "download")
    workflow.add_edge("download", "process")
    workflow.add_edge("process", "generate")
    
    # Завершение работы графа
    workflow.add_edge("generate", END)

    # 4. Компилируем стейт-машину в исполняемый объект
    return workflow.compile()