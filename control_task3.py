import subprocess
import os
import signal
import time
import threading
from task3_hungary_resource_occupy import start, loop


start(address='10.3.40.169:8888', requirement=2)

thread_loop = threading.Thread(target=loop, daemon=True)
thread_loop.start()

s = time.time()
p_train = subprocess.Popen(['horovodrun', '-np', '2', '--host-discovery-script', './discover_hosts_3.sh', 'python', 'demo_horovod_3.py'])

while True:
    time.sleep(5)
    if p_train.poll() is not None or (time.time() - s > 180):
        break
