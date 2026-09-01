import urllib.request
from pathlib import Path
from typing import Dict, Any

from src.agent.state import AgentState, DocumentMetadata

def download_node(state: AgentState) -> Dict[str, Any]:
    found_papers = state.get("found_papers", [])
    downloaded_papers = []
    
    # 1. Создаем директорию для сырых данных, если ее еще нет
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    for paper in found_papers:
        try:
            # 2. Формируем безопасное имя файла из URL ArXiv
            # Пример URL: http://arxiv.org/pdf/2401.12345v1 -> Имя: 2401.12345v1.pdf
            arxiv_id = paper.url.split('/')[-1]
            if not arxiv_id.endswith('.pdf'):
                arxiv_id += '.pdf'
                
            local_path = raw_dir / arxiv_id
            
            # 3. Кэширование: скачиваем только если файла еще нет на диске
            if not local_path.exists():
                # ArXiv может блокировать дефолтные питоновские User-Agent, притворяемся браузером
                req = urllib.request.Request(
                    paper.url, 
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req) as response, open(local_path, 'wb') as out_file:
                    out_file.write(response.read())
            
            # 4. Обновляем модель данных с помощью встроенного метода Pydantic
            updated_paper = paper.model_copy(update={"local_path": str(local_path)})
            downloaded_papers.append(updated_paper)
            
        except Exception as e:
            print(f"Ошибка при скачивании '{paper.title}': {e}")
            continue
            
    return {
        "downloaded_papers": downloaded_papers,
        "current_status": f"Успешно скачано {len(downloaded_papers)} PDF-файлов."
    }