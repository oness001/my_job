import multiprocessing
from time import sleep
from datetime import datetime, time
from logging import INFO

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine

from vnpy.gateway.onetoken import OnetokenGateway
from vnpy.app.cta_strategy import CtaStrategyApp
from vnpy.app.cta_strategy.base import EVENT_CTA_LOG


SETTINGS["log.active"] = True
SETTINGS["log.level"] = INFO
SETTINGS["log.console"] = True


exchange_setting = {
        "OT Key": "lzOQrBti-18lto3OM-IEIZwuh1-h3B55oJT",
        "OT Secret": "45Gs7Gi9-eGJN3f6z-XTUgk7Fs-VsjrtHVg",
        "交易所":  "HUOBIP",
        "账户": "oneness001",
        "会话数": 3,
        "代理地址": "127.0.0.1",
        "代理端口": 1080,
    }


def run_child():
    """
    Running in the child process.
    """
    SETTINGS["log.file"] = True

    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)

    main_engine.add_gateway(OnetokenGateway)

    cta_engine = main_engine.add_app(CtaStrategyApp)
    main_engine.write_log("主引擎创建成功")

    log_engine = main_engine.get_engine("log")
    event_engine.register(EVENT_CTA_LOG, log_engine.process_log_event)
    main_engine.write_log("注册日志事件监听")

    main_engine.connect(exchange_setting, "1TOKEN")
    main_engine.write_log("连接1TOKEN接口")

    sleep(5)

    cta_engine.init_engine()
    main_engine.write_log("CTA策略初始化完成")

    cta_engine.init_all_strategies()
    sleep(10)   # Leave enough time to complete strategy initialization
    main_engine.write_log("CTA策略全部初始化")

    cta_engine.start_all_strategies()
    main_engine.write_log("CTA策略全部启动")

    while True:
        sleep(1)


def run_parent():
    """
    Running in the parent process.
    """
    print("启动CTA策略守护父进程")

    # Chinese futures market trading period (day/night)
    DAY_START = time(0, 1)
    DAY_END = time(20, 45)

    NIGHT_START = time(20, 45)
    NIGHT_END = time(0, 0)

    child_process = None

    while True:
        current_time = datetime.now().time()
        trading = False

        # Check whether in trading period
        if (
            (current_time >= DAY_START and current_time <= DAY_END)
            or (current_time >= NIGHT_START)
            or (current_time <= NIGHT_END)
        ):
            trading = True

        # Start child process in trading period
        if trading and child_process is None:
            print("启动子进程")
            child_process = multiprocessing.Process(target=run_child)
            child_process.start()
            print("子进程启动成功")

        # 非记录时间则退出子进程
        if not trading and child_process is not None:
            print("关闭子进程")
            child_process.terminate()
            child_process.join()
            child_process = None
            print("子进程关闭成功")

        sleep(5)


if __name__ == "__main__":
    from vnpy.trader.engine import MainEngine
    from vnpy.event import EventEngine
    from vnpy.trader.constant import Direction, Exchange, Interval, Offset, Status, Product, OptionType, OrderType

    from vnpy.app.data_manager.engine import ManagerEngine


    data_ma= ManagerEngine(main_engine= MainEngine,
        event_engine= EventEngine)

    data_ma.import_data_from_csv(

        file_path=r'C:\Users\ASUS\Desktop\task\quanter\coin_dates\huobi_data\btc_xianhuo_5min\btcnew.csv',
        symbol='btcusdt',
        exchange=Exchange.HUOBI,
        interval=Interval.MINUTE,
        datetime_head= 'candle_begin_time',
        open_head='open',
        high_head='high',
        low_head='low',
        close_head ='close',
        volume_head='volume',
        open_interest_head= 'candle_begin_time',
        datetime_format= "%Y-%m-%d %H:%M:%S"

    )