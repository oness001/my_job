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
# def log(func):
#    def wrapper(*arg, **kv):
#        logging.error("this is info message")
#        return func(*arg, **kv)
#    return wrapper

def log(func):
    def wrapper(*arg, **kv):

        logger.info('函数: '+str(func.__name__)+' >>执行...')

        return func(*arg, **kv)

    return wrapper




@log
def test1():
    print("test1 done")


@log
def main():
    print("main1 done")


test1()
# main()