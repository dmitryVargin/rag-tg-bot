import json
import time
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task4_5.tg_bot import rag_chain, retriever


def evaluate_response(response, expected):
    response_lower = response.lower()
    expected_lower = expected.lower().strip()

    if expected_lower == "я не знаю.":
        if "не знаю" in response_lower:
            return True, 1.0
        else:
            return False, 0.0
    
    if "не знаю" in response_lower:
        return False, 0.0

    if "$%" not in response:
        return False, 0.0

    keywords = [w.strip(",.?!()").lower() for w in expected.split() if len(w) > 4]
    if not keywords:
        return True, 1.0
        
    matches = [k for k in keywords if k in response_lower]
    
    score = len(matches) / len(keywords)
    success = score > 0.3
    
    return success, round(score, 2)


def run_evaluation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    questions_path = os.path.join(base_dir, "questions.json")
    logs_path = os.path.join(base_dir, "logs.jsonl")

    with open(questions_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    results = []
    for item in test_set:
        query = item['question']
        expected = item.get('expected', "")

        print(f"Processing query: {query}")
        
        start_time = time.time()

        try:
            docs = retriever.invoke(query)
            sources = list(set([doc.metadata.get('source', 'unknown') for doc in docs]))
            chunks_found = len(docs) > 0
        except Exception as e:
            print(f"Error getting docs: {e}")
            sources = []
            chunks_found = False

        max_retries = 3
        retry_delay = 2
        response = "Error during generation"
        
        for attempt in range(max_retries):
            try:
                response = rag_chain.invoke(query)
                if response and "Error" not in response:
                    break
            except Exception as e:
                print(f"Error invoking chain (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    response = f"Error during generation: {str(e)}"
            
        latency = time.time() - start_time
        
        success, completeness = evaluate_response(response, expected)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "expected": expected,
            "latency": round(latency, 2),
            "chunks_found": chunks_found,
            "found_sources": sources,
            "response_length": len(response),
            "success": success,
            "completeness_score": completeness,
            "status": "PASS" if success else "FAIL"
        }
        results.append(entry)
        
        # Небольшая пауза между запросами для избежания Rate Limit
        time.sleep(0.5)

    with open(logs_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"Evaluation finished. Logs saved to {logs_path}")


if __name__ == "__main__":
    run_evaluation()
