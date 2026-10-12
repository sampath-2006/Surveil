import os
import time
from src.utils.logger import logger

def prune_old_files(directory: str, max_age_days: int = 7):
    """Deletes files in the specified directory older than max_age_days."""
    if not os.path.exists(directory):
        return
        
    current_time = time.time()
    max_age_seconds = max_age_days * 86400
    
    deleted_count = 0
    freed_bytes = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # Skip .gitkeep or other hidden files
            if file.startswith('.'):
                continue
                
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                try:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    freed_bytes += file_size
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to prune file {file_path}: {e}")
                    
    if deleted_count > 0:
        logger.info(f"Pruned {deleted_count} old files from {directory} (Freed {freed_bytes / (1024*1024):.2f} MB)")

def start_pruning_scheduler(max_age_days: int = 7):
    """Runs the pruner once every 24 hours."""
    logger.info(f"Starting Data Pruning Scheduler (Retention: {max_age_days} days)")
    
    directories_to_prune = [
        "data/maps",
        "data/maps/snapshots",
        "data/logs"
    ]
    
    while True:
        try:
            for directory in directories_to_prune:
                prune_old_files(directory, max_age_days)
        except Exception as e:
            logger.error(f"Error during pruning cycle: {e}")
            
        # Sleep for 24 hours
        time.sleep(86400)
