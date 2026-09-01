import os
from dotenv import load_dotenv

from src.agent.nodes.graph import build_graph

def main():
    # Инициализируем переменные окружения (загружаем OPENAI_API_KEY из .env)
    load_dotenv()

    print("🤖 Research Agent RAG (Terminal Mode)")
    query = input("Введите тему для исследования (например, 'Agentic RAG architectures'):\n> ")

    if not query.strip():
        print("Запрос не может быть пустым. Завершение работы.")
        return

    # Собираем исполняемый граф
    app = build_graph()

    # Формируем стартовое состояние согласно нашему TypedDict
    initial_state = {
        "original_query": query,
        "search_queries": [],
        "found_papers": [],
        "downloaded_papers": [],
        "current_status": "Запуск пайплайна...",
        "final_answer": ""
    }

    print("\n⏳ Выполнение графа...")
    print("-" * 50)

    # Итерация по узлам в реальном времени
    try:
        for step in app.stream(initial_state):
            for node_name, state_updates in step.items():
                status = state_updates.get("current_status", "Обработка завершена.")
                print(f"[{node_name.upper()}]: {status}")
                
                # Если отработал узел генерации, красиво выводим итоговый текст
                if node_name == "generate":
                    print("-" * 50)
                    print("\n📝 ФИНАЛЬНЫЙ ОТВЕТ:\n")
                    print(state_updates.get("final_answer", "Ответ пуст."))
                    print("\n" + "=" * 50)
    except Exception as e:
        print(f"\n❌ Критическая ошибка при выполнении: {e}")

if __name__ == "__main__":
    main()