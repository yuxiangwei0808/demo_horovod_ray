import numpy as np
import ray
import numpy
import os


class BaseExecutor:
    def get_gpu_ids(self):
        return ray.get_gpu_ids()

actor = []
ip_gpu_dict = {}


def start(address=None):
    if address:
        ray.init(address=address)

    while True:
        num_workers = 1
        base_remote = ray.remote(num_gpus=num_workers)(BaseExecutor)
        base_actor = base_remote.remote()
        actor.append(base_actor)

        try:
            gpu_ids = ray.get(base_actor.get_gpu_ids.remote(), timeout=3)
            current_ip = ray._private.services.get_node_ip_address()
            actor.append(base_actor)
            print(gpu_ids)
            print(ray.available_resources())

            if current_ip in ip_gpu_dict.keys():
                gpu_list = ip_gpu_dict[current_ip]
                gpu_list.append(gpu_ids)

                dic = current_ip + ':' + str(gpu_list) + '\n'
                with open('dict.txt', 'r') as f:
                    temp_lines = f.readlines()
                    for num, line in enumerate(temp_lines):
                        if current_ip in line:
                            temp_lines[num] = dic
                with open('dict.txt', 'w') as f:
                    f.writelines(temp_lines)

                dict_sh = 'echo' + current_ip + ':' + str(len(gpu_list)) + '\n'
                with open('discover_hosts.sh', 'r') as f:
                    temp_lines = f.readlines()
                    for num, line in enumerate(temp_lines):
                        if current_ip in line:
                            temp_lines[num] = dict_sh
                with open('discover_hosts.sh', 'w') as f:
                    f.writelines(temp_lines)

            else:
                #  first write in the txt and sh, use 'a'
                ip_gpu_dict[current_ip] = [gpu_ids]
                dic = current_ip + ':' + str(gpu_ids) + '\n'
                with open('dict.txt', 'a') as f:
                    f.write(dic)

                dict_sh = 'echo' + current_ip + ':' + str(len(gpu_ids)) + '\n'
                with open('discover_hosts.sh', 'a') as f:
                    f.write(dict_sh)

        except ray.exceptions.GetTimeoutError:
            del actor[-1]


