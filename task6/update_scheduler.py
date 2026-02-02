import schedule
import time
from task6.incremental_update_index import update_vector_db_incremental

def run_scheduler():
    schedule.every().day.at("06:00").do(update_vector_db_incremental)
    
    print("Планировщик запущен. Ожидание задач...")
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    run_scheduler()
