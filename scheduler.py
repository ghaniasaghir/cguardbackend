from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import json
from datetime import datetime
import time

scheduler = BackgroundScheduler()

def job():
    print(f"\n🕐 Running scrape: {datetime.now()}")
    
    # Import here to avoid loading at startup
    from scraper import get_flood_data
    
    start = time.time()
    data = get_flood_data()
    elapsed = time.time() - start
    
    with open("latest_flood.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"⏱️ Completed in {elapsed:.1f} seconds")
    print(f"✅ Saved {len(data)} stations")

def start_scheduler():
    print("🚀 Starting C-GUARD Scheduler")
    
    # Run once immediately
    job()
    
    # Schedule at PMD update times
    scheduler.add_job(job, CronTrigger(hour=6, minute=5))
    scheduler.add_job(job, CronTrigger(hour=12, minute=5))
    scheduler.add_job(job, CronTrigger(hour=18, minute=5))
    
    scheduler.start()
    print("✅ Scheduler running. Next runs: 6:05, 12:05, 18:05")