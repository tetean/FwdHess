"""
@author: tetean
@time: 2024/10/5 23:19
@info: 
"""
from typing import Dict, Any
from ruamel.yaml import YAML
import sys
yaml = YAML(typ='safe')
yaml.default_flow_style = False

def load_config(file_path='./config.yaml', skip=False, res_path='./result.yaml') -> Dict[str, Any]:
    """
    加载配置文件

    参数:
    file_path: 配置文件路径
    skip: Skip experiments that have already been conducted

    返回:
    配置字典
    """
    if skip:
        with open(res_path, 'r', encoding='utf-8') as file_res:
            res = yaml.load(file_res)
            if res is None:
                res = {}
            with open(file_path, 'r', encoding='utf-8') as file_conf:
                conf = yaml.load(file_conf)
                result = {k: v for k, v in conf.items() if k not in res or len(res[k]['running time']) < 3}
                if not result:
                    print("没有新的实验，程序结束。")
                    sys.exit()
                print(result)
                return result
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.load(file)


def write_res(conf={}, file_path='./result.yaml'):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            res = yaml.load(file) or {}
    except FileNotFoundError:
        res = {}

    res = merge_dicts(res, conf)

    with open(file_path, 'w', encoding='utf-8') as file:
        yaml.dump(res, file)


def merge_dicts(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            merge_dicts(dict1[key], value)
        else:
            dict1[key] = value
    return dict1
