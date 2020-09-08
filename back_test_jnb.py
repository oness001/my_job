import os
import signal
import traceback
import pandas as pd
import numpy as np
import sys, time
from multiprocessing import Pool #, cpu_count, Manager
from datetime import timedelta
import datetime as dt

from dateutil.relativedelta import relativedelta

from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.event import Event
from vnpy.app.ib_cta_backtester.engine import BacktesterEngine
from KRData.HKData import HKFuture
# ===导入模块和库
from vnpy.app.ib_cta_strategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)
from vnpy.trader.object import TradeData

pd.set_option('max_rows', 99999)
# pd.set_option('max_columns', 20)
pd.set_option('expand_frame_repr', False)
pd.set_option('precision', 8)
# from typing import List


def jisuan_corr(path, c=1, y=1, p=[5, 1]):
    '''
    将下周期的结果来验证本周期的选择
    :param path:
    :c    >> 选择的优势参数周期
    ：y  >> 选择验证周期大小
    ：p： 》》每次取几个排名的参数
    :return:
    '''

    #         da = pd.read_pickle(path)
    da = pd.read_csv(filepath_or_buffer=path, index_col=0)

    #         print(da)
    #         exit()

    # da = da.sort_values(by=['s_time'], axis=0)
    # print(da.tail(20))
    # exit()
    # da.loc[df['s_time'] != pd.NaT ,'s_time'] = pd.NaT
    # da['s_time'] =pd.to_datetime(da['s_time'])
    con = da['canshu'] == 'canshu'  # 过滤canshu字符串
    da.loc[con, ['s_time', 'end', 'canshu']] = np.nan
    da.dropna(axis=0, inplace=True)

    da['s_time'] = pd.to_datetime(da['s_time'])
    da['e_time'] = pd.to_datetime(da['e_time'])

    da['end'] = da['end'].map(lambda x: float(x))
    da['max_back'] = da['max_back'].map(lambda x: float(x))
    da['sharp_rate'] = da['sharp_rate'].map(lambda x: float(x))
    da['trade_nums'] = da['trade_nums'].map(lambda x: float(x))

    # 统计最终，不进行参数优化
    df_canshu = da.groupby(['canshu'])
    print(df_canshu)
    df_all = pd.DataFrame()
    for i, d in df_canshu:
        df_all.loc[i, '最终收益'] = d['end'].sum()
        df_all.loc[i, '平均月收益'] = d['end'].mean()
        df_all.loc[i, '平均最大回撤'] = d['max_back'].mean()
        df_all.loc[i, '平均月夏普率'] = d['sharp_rate'].mean()
        df_all.loc[i, '平均月交易次数'] = d['trade_nums'].mean()

    s_time_list = []
    # s_time_set = list(s_time_set).sort(reverse=True)
    # exit()
    # print(df_all)

    # 进行参数优化，前n名的参数优化
    df_zong = pd.DataFrame()
    dg = da.groupby(['s_time'])
    df_ = pd.DataFrame(columns=['s_time', 'canshu', 'next_zhouqi', 'now_end'])
    for i, v in dg:  # 日期分类
        s_time_list.append(i)
    s_time_list.sort(reverse=False)
    # print(s_time_list)

    for i, v in enumerate(s_time_list):
        df_zong0 = pd.DataFrame()  # 临时的分段样本
        if i + 1 < c: print(v);continue
        # print(v,'\n=============',i)
        # continue
        # 当前本月的时间  。例如：5-01
        now_time = pd.to_datetime(v)
        # 阶段回测开始。回测c个月，5-01-0==5-01
        s_t = pd.to_datetime(v) - relativedelta(months=+1 * (c - 1))
        # 阶段回测结束.回测结束的月初 5-01
        e_t = pd.to_datetime(v)  # + relativedelta(months = +1*(y))
        # 验证回测结束月初，验证几个月？ 6-01
        n_t = e_t + relativedelta(months=+1 * (y))
        # 总样本：5-01-6-01：》》》5-01-6-28
        da0 = da[da['s_time'] >= s_t]
        da0 = da0[da0['s_time'] <= n_t]
        # 回测的样本，5-01
        da_n = da0[da0['s_time'] <= now_time]

        # 验证样本，6-01
        da_y = da0[da0['s_time'] > now_time]
        # print(da_n.head())
        # print(da_n.tail())
        # print(da_y.head())
        # print(da_y.tail())
        # exit()

        for i, d in da0.groupby('canshu'):

            # 本周期回测样本
            con_ben = d['s_time'] <= now_time
            # 下周期验证样本
            con_next = d['s_time'] > now_time
            try:
                df_zong0.loc[i, 'canshu'] = i
                df_zong0.loc[i, 'end0'] = float(d.loc[con_ben, 'end'].sum())
                df_zong0.loc[i, 'end_next'] = float(d.loc[con_next, 'end'].sum())

                df_zong0.loc[i, 'ss_time'] = s_t
                df_zong0.loc[i, 'e_time'] = d.loc[con_ben, 'e_time'].values[-1]
                df_zong0.loc[i, 'next_stime'] = d.loc[con_next, 's_time'].values[-1]
                df_zong0.loc[i, 'next_etime'] = d.loc[con_next, 'e_time'].values[-1]

                df_zong0.loc[i, 'pingjia'] = (d.loc[con_ben, 'sharp_rate'].mean() * df_zong0.loc[i, 'end0']) / float(
                    d.loc[con_ben, 'max_back'].mean())
            except Exception as e:
                print(e)
                continue
        try:
            # 剔除nan数据
            df_zong0.dropna(subset=['pingjia'], inplace=True)
            # 排序
            df_zong0.sort_values(by='pingjia', ascending=True, inplace=True)
            df_zong0.reset_index(drop=True, inplace=True)
            df_zong0.dropna(subset=['next_stime'], inplace=True)

            df_zong0 = df_zong0.iloc[-1 * p[0]:-1 * p[1]].copy()  # 取前p名优秀者
            df_zong0.loc[df_zong0.index[-1], 'coor0'] = df_zong0['end_next'].corr(df_zong0['end0'])
        except Exception as e:
            print(e)
            continue
        #             print(df_zong0)
        #             raise
        df_zong = df_zong.append(df_zong0, ignore_index=True, sort=True)
        # df_zong.loc[df_zong.index[-1],'coor1'] = df_zong['end_next'].corr(df_zong['end0'])

    df_zong = df_zong[['canshu', 'pingjia', 'ss_time', 'e_time', 'next_stime', 'end0', 'end_next', 'coor0']]
    print(df_zong)
    raise
    print(df_zong['coor0'].mean())
    print((df_zong['coor0'].abs()).mean())

    print(df_zong['end_next'].corr(df_zong['end0']))

    path1 = path.split('.')[0] + str('corr.') + path.split('.')[1]
    # df_zong.to_pickle(path1)
    # print(df_zong.head())
    print(path1)
    return path1, df_zong

def kill(pid):
    try:
        a = os.kill(pid, signal.SIGKILL)
        # a = os.kill(pid, signal.9) #　与上等效
        print('已杀死pid为%s的进程,　返回值是:%s' % (pid, a))
    except Exception:
        print('没有如此进程!!!')


def canshu_list(n1=[150, 201, 10], n2=[100, 160, 10], n3=[10, 100, 5], n4=[10, 100, 5]):
    xingzhuang = [0, 0, 0, 0]
    zong_can = []
    if n1[0] < n1[1]:
        xingzhuang[0] = ([i for i in range(n1[0], n1[1], n1[2])])
    else:
        xingzhuang[0] = [-1]
    if n2[0] < n2[1]:
        xingzhuang[1] = ([i for i in range(n2[0], n2[1], n2[2])])
    else:
        xingzhuang[1] = [-1]
    if n3[0] < n3[1]:
        xingzhuang[2] = ([i for i in range(n3[0], n3[1], n3[2])])
    else:
        xingzhuang[2] = [-1]

    if n4[0] < n4[1]:
        xingzhuang[3] = [i for i in range(n4[0], n4[1], n4[2])]
    else:
        xingzhuang[3] = [-1]

    for n1 in xingzhuang[0]:
        for n2 in xingzhuang[1]:
            for n3 in xingzhuang[2]:
                for n4 in xingzhuang[3]:
                    canshu = []

                    for n in [n1, n2, n3, n4]:
                        if n > 0:
                            canshu.append(n)
                        else:
                            canshu.append(0)

                    zong_can.append(canshu)
                    continue
    # print(zong_can)
    return zong_can, xingzhuang


def s_canshu_youhua(canshu):
    #     print(canshu)
    time_kai = dt.datetime.now()
    ib_backtester = BacktesterEngine(None, EventEngine())
    ib_backtester.init_engine()
    ib_backtester.event_engine.start()

    try:
        # print(canshu['canshu0'])
        celue_name = canshu['celue_name']
        ib_backtester.run_backtesting(class_name=celue_name, vt_symbol='HSI9999.HKFE', interval='1m',
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

            list_res = [strategy_statistics['total_net_pnl'], strategy_statistics['max_drawdown'],strategy_statistics['total_trade_count'],
                        strategy_statistics['sharpe_ratio'], strategy_statistics['total_days'],strategy_statistics['profit_days'],
                        win_rate, win_mean_num, loss_mean_num,max_num,min_num,mean_num,std_num]
            list_res.append(canshu['canshu0'])
            # for i in range(0, len(canshu["canshu0"])):
            #     list_res.append(canshu["canshu0"][i])
        else:
            ib_backtester.event_engine.stop()

            return []



    except Exception as e:
        print(e)
        print(f'参数:{canshu["canshu0"]}', '出错')
        print(traceback.format_exc())

    # time_end = dt.datetime.now()
    # pid = os.getpid()

    ib_backtester.event_engine.stop()
    return [list_res]


def all_huice(func_name, time_st, zong_can, time_zhouqi, cpu_nums=3, flag_info='', symbol=''):
    # huice_df = Manager().list()
    huice_df = []
    pids = []
    canshu0 = [1, 1, 1, 1, 1]
    s_time, e_time = time_st
    path2 = os.getcwd() + f'\huice_log\{symbol}\{func_name}_{flag_info}.csv'

    #     path2_ = os.getcwd() + f'\huice_log\\{func_name}_{flag_info}.pickle'
    def tianjia(res, huice_df=huice_df, pids=pids):
        if len(res) > 0 :
            print(f'{dt.datetime.now()}={func_name}添加,{round((len(huice_df) * 100 / len(zong_can)), 3)}%,时间{time_st[0]},粒度{time_zhouqi}')
            huice_df.append(res[0])
        else:
            print('无交易')


    # 记录地址
    print('运行任务: ', path2)
    if 0 == True:
        for j in range(0, len(zong_can), cpu_nums):
            # print(j)
            for i in range(cpu_nums):
                if j + i <= len(zong_can) - 1:
                    canshu0 = zong_can[j + i]
                    canshu = {'celue_name': func_name, 'huice_s_t': [s_time, e_time],
                              'setting': {'ma_len': canshu0[0], 'bd': canshu0[1], "dk_len": canshu0[2],
                                          'acc': canshu0[3], 'zhouqi': time_zhouqi}, 'canshu0': canshu0}
                    res = s_canshu_youhua(canshu)
                    tianjia(res)

                else:
                    break

    if 1 == True:
        p = Pool(processes=cpu_nums)
        try:
            for j in range(0, len(zong_can), cpu_nums):
                for i in range(cpu_nums):
                    if j + i <= len(zong_can) - 1:
                        canshu0 = zong_can[j + i]
                        canshu = {'celue_name': func_name, 'huice_s_t': [s_time, e_time],
                                  'setting': {'ma_len': canshu0[0], 'bd': canshu0[1], "dk_len": canshu0[2],
                                              'acc': canshu0[3], 'zhouqi': time_zhouqi}, 'canshu0': canshu0}

                        p.apply_async(s_canshu_youhua, args=(canshu,), callback=tianjia, )  # , callback=tianjia
                    else:
                        break

        except Exception as e:
            print(e)
            print(traceback.format_exc())
        p.close()
        p.join()
        print('进程池joined')

    column_indexs = ['end', 'max_back', 'trade_nums', 'sharp_rate', 'all_tradedays', 'profit_days', 'win_rate',
                            'win_mean_num', 'loss_mean_num','max_num', 'min_num', 'mean_num', 'std_num', 'canshu']
    #     canshu_list = []
    #     for i in range(len(canshu0)):
    #         canshu_list.append('canshu%s' % str(i))
    #     column_indexs = column_indexs  + canshu_list
    huice_df = pd.DataFrame(huice_df, columns= column_indexs)
    huice_df['s_time'] = s_time
    huice_df['e_time'] = e_time
    huice_df = pd.DataFrame(huice_df, columns=column_indexs + ['s_time', 'e_time'])

    huice_df.sort_values(by=['end'], axis=0, ascending=False, inplace=True)
    huice_df.to_csv(path2, mode='a', index=True)

    return print(f'{func_name}-{flag_info}=参数回测结束,谢谢使用.')


if __name__ == '__main__':

    if 1 == True:
        mouth_nums = 20

        # ==收集
        func_name_canshu = []
        celue_name_canshu = []

        # 待优化策略
        cpu_nums = 4
        func_name_list = [
            # ['ma_tp_03', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
            # ['ema_tp_03', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
            # ['wma_tp_02', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
            # ['kama_tp_02', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
            ['dema_tp_03', [[30, 120, 10], [1, 5, 1], [20, 80, 10], [0, 10, 2]]],
            ['T3_tp_03', [[10, 100, 10], [1,5, 1], [20, 80, 10], [0, 10, 2]]],
        ]

        # 注册参数及对应策略
        for f in func_name_list:
            name = f[0]
            zongcan, a = canshu_list(n1=f[1][0], n2=f[1][1], n3=f[1][2], n4=f[1][3])
            func_name_canshu.append([name, zongcan])

        start_t = dt.datetime(2019, 6, 1)
        end_t = dt.datetime(2020, 6, 28)

        # 策略的主循环
        for t in range(0, mouth_nums + 1):
            s_time = start_t + relativedelta(months=+t)
            if s_time >= end_t: break
            e_time = s_time + timedelta(days=27)
            time_st = [s_time, e_time]
            time_zhouqi = [1, 3]

            if 1 == True:
                # ===多策略优化
                symbol = 'MEBD03_pre_res'
                flag_info1 = f"{start_t.strftime('%Y-%m-%d')}_"  # 回测标记信息
                flag_info0 = '8_18'  # 回测标记信息
                for zq in time_zhouqi:
                    flag_info = flag_info1 + str(zq) + 'T_' + flag_info0
                    for name in func_name_canshu:
                        func_name1 = name[0]
                        zong_can = name[1]
                        print(f'策略{func_name1}' f'\n{zq}', '总数:', len(zong_can))

                        all_huice(func_name1, time_st, zong_can[:], time_zhouqi=zq, cpu_nums=cpu_nums, flag_info=flag_info,symbol=symbol)



    if 0 == True:
        path_ = os.getcwd() + r'\huice_log' + '\8_10_14_10_ma_tp_01_hsi9999.csv'
        print(path_)
        jisuan_corr(path=path_, y=1, p=[2, 1])


