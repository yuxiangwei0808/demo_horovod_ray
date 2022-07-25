import subprocess
import os
import signal
import time


def end(pid):
    os.kill(pid, signal.SIGTERM)


p_start = subprocess.Popen(['python', '-c', 'import resource_occupy; resource_occupy.start()'])
time.sleep(60)
pid = p_start.pid

s = time.time()
p_train = subprocess.Popen(['horovodrun', '-np', '8', '--host-discovery-script', './discover_hosts.sh', 'python', 'demo_horovod.py'])

if p_train.poll() is not None or time.time() - s > 180:
    end(pid)