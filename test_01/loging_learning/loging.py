
import logging  # 引入logging模块
import os.path
import time
# 第一步，创建一个logger
logger = logging.getLogger(__name__) #创建一个logger
logger.setLevel(level = logging.INFO)  # Log等级
# 定义输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

# 第二步，创建一个handler，用于写入日志文件
rq = time.strftime('%Y-%m-%d_%H-%M', time.localtime(time.time()))
log_path = os.getcwd() +'\loging_%s.txt'%rq
print(log_path)
fh = logging.FileHandler(filename = log_path, mode='w',encoding='utf-8')
fh.setLevel(logging.DEBUG)  # 输出到file的log等级
fh.setFormatter(formatter)
# 第四步，将logger添加到handler里面
logger.addHandler(fh)
# 日志
logger.debug('this is a logger debug message')
logger.info('this is a logger info message')
logger.warning('this is a logger warning message')
logger.error('this is a logger error message')
logger.critical('this is a logger critical message')

#
while 1:

    try:
        print(45)

    except Exception as e:

        logger.error('Failed to open file', exc_info=True)
    logger.warning('this is a logger warning message')
    logger.error('this is a logger error message')
    logger.critical('this is a logger critical message')

    time.sleep(2)


#
# import logging, sys
#
# filelog = True
# path = r'log.txt'
#
# logger = logging.getLogger('log')
# logger.setLevel(logging.DEBUG)
#
# # 调用模块时,如果错误引用，比如多次调用，每次会添加Handler，造成重复日志，这边每次都移除掉所有的handler，后面在重新添加，可以解决这类问题
# while logger.hasHandlers():
#     for i in logger.handlers:
#         logger.removeHandler(i)
#
# # file log
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# if filelog:
#     fh = logging.FileHandler(path, encoding='utf-8')
#     fh.setLevel(logging.DEBUG)
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)
#
# # console log
# formatter = logging.Formatter('%(message)s')
# ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(logging.DEBUG)
# ch.setFormatter(formatter)
# logger.addHandler(ch)
#
# if __name__ == '__main__':
#     logger.info("这是一个测试")