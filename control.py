import subprocess
import os
import signal

pid = 0


def begin():
    p = subprocess.Popen(['python', '-c', 'import resource_occupy; resource_occupy.start()'])
    pid = p.pid


def end(pid):
    os.kill(pid, signal.SIGTERM)