import ray
import time

ray.init(address='10.3.40.169:8888')
while True:
    time.sleep(5)
    print(ray.available_resources())