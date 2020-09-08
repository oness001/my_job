import os
import traceback
import pandas as pd
import datetime as dt
from multiprocessing import Pool #, cpu_count, Manager
from vnpy.event import EventEngine
from vnpy.app.ib_cta_backtester.engine import BacktesterEngine

pd.set_option('max_rows', 99999)
pd.set_option('expand_frame_repr', False)
pd.set_option('precision', 8)



def run_backtesting(strategy_name: str, canshu: dict):


    ib_backtester = BacktesterEngine(None, EventEngine())
    ib_backtester.init_engine()
    ib_backtester.event_engine.start()
    # 执行
    try:
        if strategy_name != canshu['celue_name']:raise

        ib_backtester.run_backtesting(class_name=strategy_name, vt_symbol='HSI9999.HKFE', interval='1m',
                                      start=canshu['huice_s_t'][0], end=canshu['huice_s_t'][1],
                                      rate=0.000018, slippage=1, size=50, pricetick=1, capital=1000000,
                                      setting=canshu['setting'], inverse=False)
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

            list_res = [strategy_statistics['total_net_pnl'], strategy_statistics['max_drawdown'],
                        strategy_statistics['total_trade_count'],strategy_statistics['sharpe_ratio'],
                        strategy_statistics['total_days'],strategy_statistics['profit_days'],
                        win_rate, win_mean_num, loss_mean_num, max_num, min_num, mean_num, std_num]

            list_res.append(canshu['canshu0'])
            list_res.append(canshu['huice_s_t'][0])
            list_res.append(canshu['setting']['zhouqi'])

        else:
            ib_backtester.event_engine.stop()
            return [0,0]

    except Exception as e:
        print(e)
        print(f'参数:{canshu["canshu0"]}', '出错')
        print(traceback.format_exc())

    finally:
        ib_backtester.event_engine.stop()
        result_callback(list_res)


    return [canshu['huice_s_t'][0],canshu['setting']['zhouqi']]

def result_callback():
    # 入库
    ...

def get_strategy(s):
    ...

def run(strategy_names: list, nCPU: int):

    def tianjia(res):
        if len(res) > 0:
            print(
                f'{dt.datetime.now()}={strategy_names}添加%,时间{time_st[0]},粒度{time_zhouqi}')
        else:
            print('无交易')

    # 创建strategies文件夹
    for s in strategy_names:
        # 数据库获取数据
        results = get_strategy(s)

        binary_file = results[0]['file']

        f_path = rf'strategies/{s}.py'
        # 文件是否存在
        if not os.path.exists(f_path):
            with open(file=f_path, mode='wb') as f:
                f.write(binary_file)

        p = Pool(processes=nCPU)
        zong_can_list =results['params']
        try:
            # 遍历参数
            for j in range(0, len(zong_can_list), nCPU):
                for i in range(nCPU):
                    if j + i <= len(zong_can_list) - 1:
                        canshu0 = zong_can_list[j + i]

                        p.apply_async(run_backtesting, args=(canshu0,), callback=tianjia, )
                    else:
                        break

        except Exception as e:
            print(e)
            print(traceback.format_exc())

        p.close()
        p.join()
        print('进程池结束')




