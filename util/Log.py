"""
@author: tetean
@time: 2024/10/5 21:19
@info: 
"""

import os, logging, time
class Log:
    def __init__(self, logger_file_path='./results'):
        if not os.path.exists(logger_file_path):
            os.makedirs(logger_file_path)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        final_log_file = os.path.join(logger_file_path, log_name)

        logger = logging.getLogger()  # 设定日志对象
        logger.setLevel(logging.INFO)  # 设定日志等级

        file_handler = logging.FileHandler(final_log_file)  # 文件输出
        console_handler = logging.StreamHandler()  # 控制台输出

        # 输出格式
        FileFormat = logging.Formatter(
            "%(asctime)s %(levelname)s - %(pathname)s[line:%(lineno)d] - : %(message)s "
        )
        ConsoleFormat = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s "
        )

        file_handler.setFormatter(FileFormat)  # 设置文件输出格式
        console_handler.setFormatter(ConsoleFormat)  # 设设置控制台输出格式
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        self.logger = logger