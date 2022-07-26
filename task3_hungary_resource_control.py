import ray
import time
import random
from ray.util.placement_group import placement_group


class BaseExecutor:
    def get_gpu_ids(self):
        gpu_ids = ray.get_gpu_ids()
        ip = ray._private.services.get_node_ip_address()
        for item in gpu_ids:
            current_gpu_id = item

        return ip, current_gpu_id


actor = []
ip_gpu_dict = {}


def start(address, requirement):
    """start 2 actors at the beginning"""
    if address:
        ray.init(address=address)

    bd1 = {"CPU": 1, "GPU": 1}
    bd2 = {"CPU": 1, "GPU": 1}
    pg = placement_group([bd1, bd2], strategy='STRICT_SPREAD')

    for _ in range(requirement):
        base_remote = ray.remote(num_gpus=1)(BaseExecutor)
        base_actor = base_remote.options(placement_group=pg).remote()
        actor.append(base_actor)


    for idx, base_actor in enumerate(actor):
        try:
            current_ip, gpu_ids = ray.get(base_actor.get_gpu_ids.remote(), timeout=5)

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
            del actor[idx]

    """check if the resource meets the requirement"""
    while True:
        if len(actor) < requirement:
            num_workers = 1
            base_remote = ray.remote(num_gpus=num_workers)(BaseExecutor)
            base_actor = base_remote.remote()
            actor.append(base_actor)

            try:
                current_ip, gpu_ids = ray.get(base_actor.get_gpu_ids.remote(), timeout=5)

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
        else:
            break



def loop():
    while True:
        num_workers = 1
        base_remote = ray.remote(num_gpus=num_workers)(BaseExecutor)
        base_actor = base_remote.remote()
        actor.append(base_actor)

        try:
            current_ip, gpu_ids = ray.get(base_actor.get_gpu_ids.remote(), timeout=5)

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
