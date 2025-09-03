import os
import time

def write_logs(msg):
    ctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open("log.txt", "a") as f:
        f.write(f"[{ctime}] - {msg}\n")