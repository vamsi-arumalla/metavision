import time, shutil, threading
import os
import logging

logger = logging.getLogger(__name__)

def janitor(ttl_hours=3):
    """Background thread - delete workspaces that have not changed in `ttl_hours`."""
    ttl = ttl_hours * 3600
    while True:
        now = time.time()
        if os.path.exists("workspaces"):
            for p in os.scandir("workspaces"):
                try:
                    if p.is_dir() and now - p.stat().st_mtime > ttl:
                        shutil.rmtree(p, ignore_errors=True)
                except FileNotFoundError:
                    pass
        else:
            logger.warning("Workspaces directory does not exist.")
        time.sleep(3600)