import numpy as np
import ray
import numpy
import os


class BaseExecutor:
    def get_gpu_ids(self):
        gpu_ids = ray.get_gpu_ids()
        ip = ray._private.services.get_node_ip_address()
        for item in gpu_ids:
            current_gpu_id = item

        return ip, current_gpu_id


actor = []
ip_gpu_dict = {}


def start(address='10.3.40.169:8888'):
    if address:
        ray.init(address=address)

    ##################reserving 2 gpus at the start#############################
    '''base_remote = ray.remote(num_gpus=2)(BaseExecutor)
    base_actor = base_remote.remote()
    actor.append(base_actor)

    try:
        current_ip, gpu_ids = ray.get(base_actor.get_gpu_ids.remote(), timeout=3)

        dict_lines = []
        sh_lines = []
        ip_gpu_dict[current_ip] = [0, 1]

        for key in ip_gpu_dict:
            dict_lines.append(str(key + ':' + f'{ip_gpu_dict[key]}' + '\n'))
            sh_lines.append(str('echo ' + key + ':' + f'{len(ip_gpu_dict[key])}' + '\n'))
        with open('dict_training.txt', 'w') as f:
            f.writelines(dict_lines)
        with open('discover_hosts.sh', 'w') as f:
            f.writelines(sh_lines)

    except ray.exceptions.GetTimeoutError:
        del actor[-1]

    print('initial resources: ', ray.available_resources())
    print('initial dict: ', ip_gpu_dict)'''
    #############################################################################

    # officially start
    while True:
        num_workers = 1
        base_remote = ray.remote(num_gpus=num_workers)(BaseExecutor)
        base_actor = base_remote.remote()
        actor.append(base_actor)

        try:
            current_ip, gpu_ids = ray.get(base_actor.get_gpu_ids.remote(), timeout=3)

            dict_lines = []
            sh_lines = []
            if current_ip in ip_gpu_dict.keys():
                ip_gpu_dict[current_ip].append(int(gpu_ids))
            else:
                ip_gpu_dict[current_ip] = []
                ip_gpu_dict[current_ip].append(int(gpu_ids))

            for key in ip_gpu_dict:
                dict_lines.append(str(key + ':' + f'{ip_gpu_dict[key]}' + '\n'))
                sh_lines.append(str('echo ' + key + ':' + f'{len(ip_gpu_dict[key])}' + '\n'))
            with open('dict_training.txt', 'w') as f:
                f.writelines(dict_lines)
            with open('discover_hosts.sh', 'w') as f:
                f.writelines(sh_lines)

        except ray.exceptions.GetTimeoutError:
            del actor[-1]




start()
