import os
import pandas as pd
from multiprocessing import Pool #, cpu_count, Manager
from vnpy.event import EventEngine
from vnpy.app.ib_cta_backtester.engine import BacktesterEngine
from mongoengine import DateTimeField, StringField, Document, BinaryField, FloatField, BooleanField, DictField, register_connection
from hashlib import sha256
import numpy as np
import logging


logger = logging.getLogger(__name__)

class StrategyCode(Document):
    name = StringField(required=True)
    class_name = StringField(required=True)
    hash = StringField(max_length=64)
    data = BinaryField()

    meta = {
        'db_alias': 'VNPY_BACKTEST',
        "indexes": [
            {
                "fields": ("class_name",),
            },
            {
                "fields": ("name", "class_name",),
                "unique": True
            },
            {
                "fields": ("name", "class_name", "hash"),
                "unique": True,
            }
        ],
    }

class StrategiesSet(Document):
    hash = StringField(max_length=64)
    class_name = StringField()
    vt_symbol = StringField()
    interval = StringField()
    start = DateTimeField()
    end = DateTimeField()
    rate = FloatField()
    slippage = FloatField()
    size = FloatField()
    pricetick = FloatField()
    capital = FloatField()
    inverse = BooleanField()
    setting = DictField()
    result = DictField()

    meta = {
        'db_alias': 'VNPY_BACKTEST',
        "indexes": [
            {
                "name": "ss_unique",
                "fields": ("class_name", "vt_symbol", "interval", "start", "end", "rate", "slippage", "size", "pricetick", "capital", "inverse", "setting"),
                "unique": True,
            },
            {
                "fields": ("class_name",),
            },
        ],
    }

def run_backtesting(strategy_set: StrategiesSet):

    ib_backtester = BacktesterEngine(None, EventEngine())
    ib_backtester.init_engine()
    ib_backtester.event_engine.start()
    # 执行
    try:
        ss = strategy_set.to_mongo()
        ss.pop('_id')
        ss.pop('result')
        ss.pop('hash')
        ib_backtester.run_backtesting(**ss)
        strategy_statistics = ib_backtester.get_result_statistics()
        trades = ib_backtester.get_all_trades()

        if len(trades) > 0:
            pnl = []
            for o, c in zip(trades[::2], trades[1::2]):
                ov0 = o.price * -o.volume if o.direction.value == '多' else o.price * o.volume
                cv0 = c.price * -c.volume if c.direction.value == '多' else c.price * c.volume
                pnl.append(ov0 + cv0)

            s = pd.Series(pnl)
            win_rate = s[s > 0].count() * 100 / s.count() if s.count() != 0 else 0
            win_mean_num = s[s > 0].mean()
            loss_mean_num = s[s < 0].mean()
            max_num = s.max()
            min_num = s.min()
            mean_num = s.mean()
            std_num = s.std()
            max_sum = s.cumsum().max()

            strategy_statistics['win_rate'] = win_rate
            strategy_statistics['win_mean_num'] = win_mean_num
            strategy_statistics['loss_mean_num'] = loss_mean_num
            strategy_statistics['max_num'] = max_num
            strategy_statistics['min_num'] = min_num
            strategy_statistics['mean_num'] = mean_num
            strategy_statistics['std_num'] = std_num
            strategy_statistics['max_sum'] = max_sum

        else:
            strategy_statistics['win_rate'] = None
            strategy_statistics['win_mean_num'] = None
            strategy_statistics['loss_mean_num'] = None
            strategy_statistics['max_num'] = None
            strategy_statistics['min_num'] = None
            strategy_statistics['mean_num'] = None
            strategy_statistics['std_num'] = None
            strategy_statistics['max_sum'] = None

        # 转换数据类型
        for k, v in strategy_statistics.items():
            if isinstance(v, np.float64):
                strategy_statistics[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                strategy_statistics[k] = int(v)

        strategy_statistics['start_date'] = str(strategy_statistics['start_date'])
        strategy_statistics['end_date'] = str(strategy_statistics['end_date'])
        strategy_set.result = strategy_statistics

    except Exception as e:
        raise e
        # res = StrategiesSet.objects(class_name=strategy_set.clclass_name, result__ne={})
        # print(f'{s}，已经完成:{res.count()}')
        # all = StrategiesSet.objects(class_name=strategy_set.clclass_name)
        # print(f'{s}，总数:{all.count()}')
    finally:
        ib_backtester.event_engine.stop()

    return strategy_set

def result_callback(strategy_set: StrategiesSet):
    logger.info(f'保存策略结果: {strategy_set.to_json()}')
    strategy_set.save()

def log_error(e):
    logger.exception('策略执行错误', exc_info=e)

def get_strategy(s):
    logger.info(f'获取{s}策略代码以及参数')
    sc = StrategyCode.objects(class_name=s)
    sc1 = sc.first()
    ss = StrategiesSet.objects(class_name=sc1.class_name, result={})

    return sc.first(), ss

def comsumer_run(strategy_names: list, nCPU: int):
    s_path = 'strategies'
    if os.path.exists(s_path):
        raise FileExistsError('strategies文件夹已存在，请先删除')

    # 创建strategies文件夹
    logger.info('创建strategies文件夹')
    os.mkdir(s_path)
    with open(os.path.join(s_path, '__init__.py'), 'wb'):
        ...

    for s in strategy_names:
        # 数据库获取数据
        strategy_code, strategy_set = get_strategy(s)

        binary_file = strategy_code.data

        # 文件是否存在
        f_path = rf'{s_path}/{s}.py'
        if not os.path.exists(f_path):
            logger.info(f'写入策略{f_path}')
            with open(file=f_path, mode='wb') as f:
                f.write(binary_file)

        res = StrategiesSet.objects(class_name = s, result__ne={})
        print(f'{s}，已经完成: ={res.count()}')
        all = StrategiesSet.objects(class_name = s)
        print(f'{s}，总数: ={all.count()}')



        # 策略文件是否与策略集合的class_name对应
        for ss in strategy_set:
            if ss.class_name != strategy_code.class_name or ss.hash != strategy_code.hash:
                raise Exception("class_name 或 hash 不一致")

        # 放入进程池执行
        p = Pool(processes=nCPU)
        for ss in strategy_set:
            logger.info(f'参数放入进程池：{ss.to_json()}')
            p.apply_async(run_backtesting, args=(ss, ), callback=result_callback, error_callback=log_error)

        logger.info('关闭进程池')
        p.close()
        logger.info('等待进程。。。')
        p.join()
        logger.info('进程池结束')

    # 清理strategies文件夹
    logger.info('清理strategies文件夹')

    def recursive_rm(path):
        for fp in os.listdir(path):
            subpath = os.path.join(path, fp)
            if os.path.isdir(subpath):
                recursive_rm(subpath)
            else:
                os.remove(subpath)
        else:
            os.rmdir(path)
    print('回测已完成！')
    res = StrategiesSet.objects(class_name=s, result__ne={})
    all = StrategiesSet.objects(class_name=s)
    print(f'{s}，已经完成: {res.count()*100/all.count()} %')
    recursive_rm(s_path)

def save_strategy(strategy_file_path: str, strategy_name: str, all_params: dict):
    # 检查参数完备性
    for p in ("class_name", "vt_symbol", "interval", "start", "end", "rate", "slippage", "size", "pricetick", "capital", "inverse", "setting"):
        if p not in all_params:
            raise KeyError(f'all_params 缺少 {p}')

    # 读取策略文件
    with open(os.path.join(strategy_file_path, strategy_name), 'rb') as f:
        binary_file = f.read()

    # 求哈希值
    hash = sha256()
    hash.update(binary_file)
    hash_code = hash.hexdigest()

    # 判断策略文件是否有冲突
    sc = StrategyCode.objects(name=strategy_name, class_name=all_params['class_name'])
    if sc.count() != 0:
        if sc.first().hash != hash_code:
            raise Exception(f'策略{strategy_name}已存在，但hash不一致，请先删除旧策略或使用旧策略文件添加回测参数')
    else:
        strategy_code = StrategyCode(name=strategy_name, class_name=all_params['class_name'], hash=hash_code, data=binary_file)
        strategy_code.save()

    # 保存策略参数
    ss = StrategiesSet(**all_params, hash=hash_code)
    ss.save()

def del_strategy_code(class_name=None):
    """
    删除策略代码
    :param class_name: 策略模板名称
    :return: 删除条数
    """
    sc = StrategyCode.objects(class_name=class_name)
    ss = StrategiesSet.objects(class_name=class_name)
    if ss.count() > 0:
        raise Exception(f'存在策略{class_name}的策略参数，请先删除策略参数')
    return sc.delete()

def del_strategies_set(class_name=None, exclude_done=True):
    """
    删除策略代码
    :param class_name: 策略模板名称
    :param exclude_done: 是否排除已经回测完成的存在结果的策略集，默认为True，即只包含结果为空的策略集
    :return: 删除条数
    """
    ss = StrategiesSet.objects(class_name=class_name, result={}) if exclude_done else StrategiesSet.objects(class_name=class_name)
    return ss.delete()

def init():
    from vnpy.trader.setting import get_settings
    db_settting = get_settings('database.')
    register_connection(
        'VNPY_BACKTEST',
        db='VNPY_BACKTEST',
        host=db_settting['host'],
        port=db_settting['port'],
        username=db_settting['user'],
        password=db_settting['password'],
        authentication_source=db_settting['authentication_source'],
    )
    filehandler = logging.FileHandler('run_strategies.log')
    filehandler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)


if __name__ == '__main__':
    
    if 1 == True:
        import argparse
        import datetime as dt

        parser = argparse.ArgumentParser()
        parser.add_argument('--strategies', nargs='+', required=True)
        parser.add_argument('--nCPU', type=int, default=1)
        args = parser.parse_args()
        init()

        logger.info(f'使用{args.nCPU}核心，进行回测策略：{args.strategies}')
        st = dt.datetime.now()
        comsumer_run(args.strategies, args.nCPU)
        et = dt.datetime.now()
        logger.info(f'回测结束： 总用时：{et - st}')

    # 查看
    watch_data = 0
    if watch_data == True:
        init()
        res = StrategiesSet.objects(class_name='wma_tp_02', result__ne={})
        # res.count()
        print(res.count())
        all = StrategiesSet.objects(class_name='wma_tp_02')
        all.count()
        print(all.count())

    # 删除
    delete_data =0
    if delete_data == True:
        
        init()
        del_strategy_code(class_name='')
