"""
@author: tetean
@time: 2024/10/5 23:19
@info: 
"""
from typing import Dict, Any
from ruamel.yaml import YAML
yaml = YAML(typ='safe')
yaml.default_flow_style = False

def load_config(file_path='./config.yaml') -> Dict[str, Any]:
    """
    加载配置文件

    参数:
    file_path: 配置文件路径

    返回:
    配置字典
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.load(file)