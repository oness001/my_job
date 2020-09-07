#!/usr/bin/env python
#_*_coding:utf-8_*_
# vim : set expandtab ts=4 sw=4 sts=4 tw=100 :

import logging
import time
import re
from logging.handlers import TimedRotatingFileHandler
from logging.handlers import RotatingFileHandler

def main(log_name):
    #日志打印格式
    log_fmt = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    formatter = logging.Formatter(log_fmt)
    #创建TimedRotatingFileHandler对象
    log_file_handler = TimedRotatingFileHandler(filename=r'C:\Users\ASUS\PycharmProjects\python3.7学习项目\loging_learning\log\%s'%log_name, when="M", interval=1, backupCount=1)

    log_file_handler.suffix = "%Y-%m-%d--%H-%M.txt"#_%H-%M
    log_file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}--\d{2}-\d{2}.txt$")#_
    log_file_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(log_file_handler)
    #循环打印日志
    log_content = "test log"
    count = 0
    while count < 30:
        logger.error(log_content)
        time.sleep(19.9)
        count = count + 1
    logger.removeHandler(log_file_handler)


if __name__ == "__main__":
    main('test_one')