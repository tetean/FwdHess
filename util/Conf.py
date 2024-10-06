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
    skip: 是否跳过已经完成的实验
    res_path: 结果文件路径

    返回:
    配置字典
    """

    def load_yaml(file) -> Dict[str, Any]:
        """加载 YAML 文件的辅助函数"""
        with open(file, 'r', encoding='utf-8') as f:
            return yaml.load(f) or {}

    config = load_yaml(file_path)

    if skip:
        res = load_yaml(res_path)
        if not res:
            return config

        # 过滤已经运行过的实验
        filtered_config = {
            k: v for k, v in config.items()
            if k not in res or len(res[k].get('running time', [])) < 3
        }

        if not filtered_config:
            print("没有新的实验，程序结束。")
            sys.exit()

        return filtered_config

    return config


def write_res(conf={}, file_path='./result.yaml'):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            res = yaml.load(file) or {}
    except FileNotFoundError:
        res = {}

    res = merge_dicts(res, conf)
    print(res)
    with open(file_path, 'w', encoding='utf-8') as file:
        yaml.dump(res, file)


def merge_dicts(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            merge_dicts(dict1[key], value)
        else:
            dict1[key] = value
    return dict1
