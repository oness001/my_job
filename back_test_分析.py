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

pd.set_option('max_rows', 99999)
# pd.set_option('max_columns', 20)
pd.set_option('expand_frame_repr', False)
pd.set_option('precision', 8)
from typing import List
from vnpy.trader.object import TradeData


def yinzi_cal(i,d,df_zong0,s_t,e_t,train_res=[1,pd.Series()]):
    '''

    :param d: 计算周期因子
    :param s_t: 开始周期时间
    :param now_time: 当前时间
    :param train_res: 【是否要传入训练结果，样本训练结果】
    :return:
    '''
    try:
        if float(d['end'].sum()) != 0 or d['end'].sum() != np.NAN:
            pass
        else:
            return pd.DataFrame(),df_zong0
        df_zong0.loc[i, 'canshu'] = i
        df_zong0.loc[i, '本周期收益'] = float(d['end'].sum())
        df_zong0.loc[i, '平均月收益'] = float(d['end'].mean())
        df_zong0.loc[i, '最大值'] = (d['end'].cumsum()).max()
        df_zong0.loc[i, '收益std'] = (d['end'].std())
        df_zong0.loc[i, '偏度'] = (d['end'].skew())
        df_zong0.loc[i, '峰度'] = (d['end'].kurt())

        df_zong0.loc[i, '月最大回撤'] = d['max_back'].min()
        df_zong0.loc[i, '平均最大回撤'] = d['max_back'].mean()
        df_zong0.loc[i, '回撤std'] = (d['max_back'].std() * -1)
        df_zong0.loc[i, '平均月夏普率'] = d['sharp_rate'].mean()
        df_zong0.loc[i, '平均月夏普率std'] = d['sharp_rate'].std()
        df_zong0.loc[i, '平均月交易次数'] = d['trade_nums'].mean()
        df_zong0.loc[i, '平均月交易次数std'] = d['trade_nums'].std()

        # 因子组合
        df_zong0.loc[i, '夏普稳健因子'] = d['sharp_rate'].mean() / d['sharp_rate'].std() if d[
                                                                                          'sharp_rate'].std() != 0 else 0
        df_zong0.loc[i, '风险因子01'] = (d['max_back'].mean() / (d['max_back'].std() * -1)) if d[
                                                                                               'max_back'].std() != 0 else 0
        df_zong0.loc[i, '盈利因子01'] = d['end'].sum() * d['end'].mean() / (d['end'].std()) if d[
                                                                                               'end'].std() != 0 else 0
        df_zong0.loc[i, '分布因子01'] = (d['end'].skew() + 0.3) * (d['end'].kurt() - 1.5)
        df_zong0.loc[i, '效率因子01'] = (d['trade_nums'].std() / d['trade_nums'].mean()) * d['end'].sum() / abs(
            d['max_back'].mean()) if d['max_back'].mean() != 0 else 0


        # 参数回测的开始结束
        df_zong0.loc[i, 'ss_time'] = s_t
        # 最近一个月的开始结束
        df_zong0.loc[i, 'c_stime'] = e_t


        if train_res[0] == True:
            d1 = train_res[1]
            df_zong0.loc[i, '预测周期收益'] = float(d1['end'].sum())  # c=1, y=1,
            df_zong0.loc[i, '预期偏差'] =float(d1['end'].sum()) - float(d.iloc[-1]['end'].sum())
            return df_zong0, df_zong0
        else:
            # 平稳测试计算
            df_zong0.loc[i, '预测周期收益'] = float(0)  # c=1, y=1,
            df_zong0.loc[i, '预期偏差'] = float(0)
            return df_zong0 ,df_zong0
    except Exception as e:
        exc_type, exc_value, exc_traceback_obj = sys.exc_info()
        traceback.print_tb(exc_traceback_obj)
        print('单参数统计出错', d.tail())
        return df_zong0,df_zong0


def jisuan_corr(path, c=3, y=1, p=[5, 1],zongtest =0,jiqixuexi= 1):
    '''
    将下周期的结果来验证本周期的选择
    :param path:
    :c    >> 选择的优势参数周期
    ：y  >> 选择验证周期大小
    ：p： 》》每次取几个排名的参数
    :return:
    '''

    da = pd.read_csv(filepath_or_buffer=path, index_col=0)

    # da = da.sort_values(by=['s_time'], axis=0)
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
    # da['trade_nums'] = da['trade_nums'].map(lambda x: float(x))
    if zongtest ==True:
        # 统计最终，所有参数在回测时间段里的表现均值
        df_canshu = da.groupby(['canshu'])
        df_all = pd.DataFrame()
        for i, d in df_canshu:
            d = d.iloc[:].copy()
            dn = d.iloc[-1].copy()
            df_all.loc[i, '回测最终收益'] = d['end'].sum()
            df_all.loc[i, '平均月收益'] = d['end'].mean()
            df_all.loc[i, '最大值'] =( d['end'].cumsum()).max()
            df_all.loc[i, '收益std'] =( d['end'].std() )
            df_all.loc[i, '偏度'] =( d['end'].skew()  )
            df_all.loc[i, '峰度'] =( d['end'].kurt()  )

            df_all.loc[i, '平均最大回撤'] = d['max_back'].mean()
            df_all.loc[i, '回撤std'] =( d['max_back'].std()*-1 )
            df_all.loc[i, '平均月夏普率'] = d['sharp_rate'].mean()
            df_all.loc[i, '平均月夏普率std'] = d['sharp_rate'].std()


            df_all.loc[i, '平均月交易次数'] = d['trade_nums'].mean()
            df_all.loc[i, '平均月交易次数std'] = d['trade_nums'].std()

            df_all.loc[i, '本周期收益'] = d['end'].sum()
            df_all.loc[i, '未来周期收益'] = dn['end'].sum()

            df_all.loc[i, '夏普稳健因子'] = d['sharp_rate'].mean() /d['sharp_rate'].std() if d['sharp_rate'].std() !=0 else 0
            df_all.loc[i, '效率因子01'] = (d['end'].sum()/abs(d['max_back'].mean()))*(d['trade_nums'].std()/d['trade_nums'].mean()) if d['max_back'].mean() !=0 else 0
            df_all.loc[i, '风险因子01'] = d['end'].sum()/abs(d['max_back'].mean()) if d['max_back'].mean() !=0 else 0
            df_all.loc[i, '盈利因子01'] = (d['end'].sum()*d['end'].mean()/d['end'].std()) if d['end'].std() !=0 else 0
            df_all.loc[i, '分布因子01'] = (d['end'].skew()+0.3)*(d['end'].kurt()) #正偏态高峰
            df_all.loc[i, '综合因子02'] = -1*d['end'].mean()/(d['end'].std()*d['max_back'].mean()*d['max_back'].std()*d['trade_nums'].mean()) if d['max_back'].mean() !=0 else 0#正偏态高峰


        df_all0 =df_all[df_all['平均月收益'] > df_all['平均月收益'].std()]
        df_all0 =df_all0[df_all0['分布因子01'] > 0 ]
        # df_all0 =df_all0[df_all0['回测最终收益'] > 0] #df_all0['回测最终收益'].mean()+df_all0['回测最终收益'].std()] #
        df_all0 =df_all0[df_all0['效率因子01'] > 0.7*df_all0['效率因子01'].std() ]
        df_all0 =df_all0[df_all0['盈利因子01'] > df_all0['盈利因子01'].mean() - 0.5*df_all0['盈利因子01'].std()]
        df_all0 =df_all0[df_all0['夏普稳健因子'] > df_all0['夏普稳健因子'].mean()]



        df_all0.sort_values(by=['平均月收益'],ascending=True ,inplace=True)
        print(df_all0)
        return df_all0

    if jiqixuexi == True:
        # 统计最终，所有参数在回测时间段里的表现均值
        df_canshu = da.groupby(['canshu'])
        df_all = pd.DataFrame()
        df_all1 = pd.DataFrame()

        for i, d0 in df_canshu:

            # 训练样本
            # d = d.iloc[:].copy()
            d1 = d0.iloc[:-2].copy()
            s_time = d1['s_time'].tolist()[0]
            e_time =d1['s_time'].tolist()[-1]
            # 训练样本结果
            dn = d0.iloc[-2].copy()
            # print(dn.tail())

            res, df_all = yinzi_cal(i, d1, df_all, s_t=s_time, e_t=e_time,train_res=[1, dn])
            if res.empty:
                print('出错了！')
            else:
                pass
                # print('ok!')
            # 训练样本
            d2 = d0.iloc[1:-1].copy()
            # 训练样本结果
            dn2 = d0.iloc[-1].copy()
            # print(dn2.tail())
            # exit()
            s_time = d2['s_time'].tolist()[0]
            e_time = d2['s_time'].tolist()[-1]
            res, df_all = yinzi_cal(i, d2, df_all1, s_t=s_time, e_t=e_time,train_res=[1, dn2])

            # , '盈利因子01', '风险因子01', '分布因子01', '效率因子01', '夏普稳健因子',
        yinzi = ['本周期收益','最大值','收益std', '偏度','平均月收益','分布因子01',
            '平均最大回撤','平均月夏普率', '回撤std','平均月夏普率std','平均月交易次数','平均月交易次数std']

        print(df_all[['本周期收益','预测周期收益','c_stime']].head())
        print(df_all[['本周期收益','预测周期收益','c_stime']].tail())
        print(df_all1[['本周期收益','预测周期收益','c_stime']].head())
        print(df_all1[['本周期收益','预测周期收益','c_stime']].tail())

        model,prelist = polynomial_regression(X_train=df_all[yinzi], y_train=df_all['预测周期收益'],
                                              X_test=df_all1[yinzi], y_test=df_all1['预测周期收益'],
                                                degree=3, include_bias=False,normalize=True)
        df_n_p = pd.DataFrame({'未来实际收益':prelist[0],'预测收益':prelist[1]})

        print(df_n_p)
        # exit()


        return df_all,df_n_p

    s_time_list = []


    # 进行参数优化，前n名的参数优化
    da.sort_values(by=['s_time'],ascending=True,inplace=True)
    # print(da.s_time)
    df_zong = pd.DataFrame()
    dg = da.groupby(['s_time'])
    for i, v in dg:  # 日期分类
        s_time_list.append(i)
    s_time_list.sort(reverse=False)

    for i, v in enumerate(s_time_list):
        if i <= c: print(v);continue
        # 当前周期的时间  。例如：5-01
        now_time = pd.to_datetime(v)
        # 阶段回测开始。回测c个月，5-01-c==3-01 c=2
        s_t = pd.to_datetime(v) - relativedelta(months=1 * (c))
        # 验证回测结束月初，验证几个月？ 5-01 +y = 6-01 y=1
        n_t = now_time + relativedelta(months=1 * (y))
        # 总样本：5-01-6-01：》》》5-01-6-28
        da0 = da[da['s_time'] >= s_t]
        da0 = da0[da0['s_time'] <= n_t]

        # 回测训练总样本
        df_zong0 =pd.DataFrame()
        da1= da0[da0['s_time'] <= now_time]
        for i, d0 in da1.groupby('canshu'):
            # 训练样本
            d = d0.iloc[:-1].copy()
            #训练样本结果
            d1 = d0.iloc[-1].copy()

            res,df_zong0 = yinzi_cal(i,d,df_zong0, s_t=s_t, e_t=now_time, train_res=[1, d1])

            if res.empty:
                continue
        # 预测样本计算参数
        try:
            df_zong1 = pd.DataFrame()
            for i, d in da0.groupby('canshu'):
                d = da0.iloc[1:-1].copy()#测试
                dn =  da0.iloc[-1].copy()#测试结果

                # 预测样本
                d = d
                #预测样本的结果
                # 预测样本计算因子
                res,df_zong1 = yinzi_cal(i,d, df_zong1,s_t=s_t + relativedelta(months=1 ), e_t= now_time, train_res=[1, dn])
                if res.empty:
                    continue
                else:
                    df_zong1 = df_zong1.append(res,ignore_index=True)

            yinzi = ['本周期收益', '平均月收益', '最大值', '收益std', '偏度', '峰度',
                     '月最大回撤','平均最大回撤', '回撤std', '平均月夏普率', '平均月夏普率std', '平均月交易次数', '平均月交易次数std',
                     '夏普稳健因子', '风险因子01', '盈利因子01','分布因子01', '效率因子01']

            model, df_zong1 = polynomial_regression(X_train=df_zong0[yinzi], y_train=df_zong0['预测周期收益']
                                              , X_test=df_zong1[yinzi],y_test=df_zong1['预测周期收益'],
                                              degree=2, include_bias=False, normalize=True)

        except Exception as e:
            print(e)
            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            traceback.print_tb(exc_traceback_obj)



        try:

            # 剔除nan数据
            # df_zong0.dropna(subset=['风险因子01','盈利因子01','综合因子01'], inplace=True)
            df_zong1.dropna(subset=['c_stime'], inplace=True)
            # 排序
            # df_zong0 = df_zong0[df_zong0['平均月收益'] > 0]  #
            # df_zong0 = df_zong0[df_zong0['分布因子01'] > 0]
            # df_zong0 = df_zong0[df_zong0['夏普稳健因子'] > df_zong0['夏普稳健因子'].std()]
            # df_zong0 = df_zong0[df_zong0['盈利因子01'] > df_zong0['盈利因子01'].mean()-0.5*df_zong0['盈利因子01'].std()]
            # df_zong0 = df_zong0[df_zong0['效率因子01'] >  0.7*df_zong0['效率因子01'].std()]
            df_zong1.sort_values(by=['predict_res'], inplace=True)
            df_zong1 = df_zong1.iloc[-1 * p[0]:-1 * p[1]].copy()  # 取前p名优秀者
            print(df_zong1)
            df_zong0.loc[df_zong0.index[-1], '单周期相关性'] = df_zong0['本周期收益'].corr(df_zong0['预测周期收益'])

        except Exception as e:
            print(df_zong0.tail(5))
            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            traceback.print_tb(exc_traceback_obj)
            continue
        #             print(df_zong0)
        #             raise
        df_zong = df_zong.append(df_zong0, ignore_index=True, sort=True,)
        # df_zong.loc[df_zong.index[-1],'coor1'] = df_zong['end_next'].corr(df_zong['end0'])

    print(df_zong[['c_stime','n_stime','ss_time','canshu','最终收益','本周期收益','未来周期收益','预期偏差','盈利因子01','效率因子01','夏普稳健因子','风险因子01']])



    path1 = path.split('.')[0] + str('corr.') + path.split('.')[1]

    print(path1)
    return df_zong

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
        list_res = [strategy_statistics['total_net_pnl'], strategy_statistics['max_drawdown'],
                    strategy_statistics['total_trade_count'], strategy_statistics['sharpe_ratio']]

        # print(f'参数:{canshu["canshu0"]}：{list_res}')
        list_res.append(canshu['canshu0'])
        for i in range(0, len(canshu["canshu0"])):
            list_res.append(canshu["canshu0"][i])

    except Exception as e:
        print(e)
        print(f'参数:{canshu["canshu0"]}', '出错')
        print(traceback.format_exc())

    time_end = dt.datetime.now()

    ib_backtester.event_engine.stop()
    pid = os.getpid()
    return [list_res]


def all_huice(func_name, time_st, zong_can, time_zhouqi, cpu_nums=3, flag_info='', symbol=''):
    # huice_df = Manager().list()
    huice_df = []
    pids = []
    canshu0 = [1, 1, 1, 1, 1]
    s_time, e_time = time_st
    path2 = os.getcwd() + f'\huice_log\\{func_name}_{flag_info}.csv'

    #     path2_ = os.getcwd() + f'\huice_log\\{func_name}_{flag_info}.pickle'

    def tianjia(res, huice_df=huice_df):

        print(f'{dt.datetime.now()}={func_name}添加,{round((len(huice_df) * 100 / len(zong_can)),3)}%,时间{time_st[0]},粒度{time_zhouqi}')
        huice_df.append(res[0])

    #         if len(huice_df) == len(zong_can:
    #             print(f'{func_name}-所有参数回测结束,谢谢使用.')
    #             print(f'{func_name}-正在保存.')

    # for pid in pids:
    #     kill(pid)
    # print()
    # column_indexs = ['end', 'ma_back', 'trand_nums', 'sharprate', 'canshu']
    # canshu_list = []
    # for i in range(len(canshu0)):
    #     canshu_list.append('canshu%s' % str(i))
    # column_indexs = column_indexs + canshu_list
    # huice_df = pd.DataFrame(huice_df, columns=column_indexs)
    # huice_df['s_time'] = s_time
    # huice_df['e_time'] = e_time
    # huice_df = pd.DataFrame(huice_df, columns=column_indexs + ['s_time', 'e_time'])
    #
    # huice_df.sort_values(by=['end'], axis=0, ascending=False, inplace=True)
    # print(huice_df)
    # huice_df.to_csv(path2, mode='a', index=True)
    #
    # huice_df.to_pickle(path2_)
    # return

    # 记录地址

    print('运行任务: ', path2)
    if 0 == True:
        for j in range(0, len(zong_can), cpu_nums):
            #             print(j)
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
                        # print(canshu)
                        # exit()

                        p.apply_async(s_canshu_youhua, args=(canshu,), callback=tianjia, )  # , callback=tianjia
                    else:
                        break


        except Exception as e:
            print(e)
            print(traceback.format_exc())
        p.close()
        p.join()
        print('进程池joined')
    column_indexs = ['end', 'max_back', 'trade_nums', 'sharp_rate', 'canshu']
    canshu_list = []
    for i in range(len(canshu0)):
        canshu_list.append('canshu%s' % str(i))
    column_indexs = column_indexs + canshu_list
    huice_df = pd.DataFrame(huice_df, columns=column_indexs)
    huice_df['s_time'] = s_time
    huice_df['e_time'] = e_time
    huice_df = pd.DataFrame(huice_df, columns=column_indexs + ['s_time', 'e_time'])

    huice_df.sort_values(by=['end'], axis=0, ascending=False, inplace=True)
    huice_df.to_csv(path2, mode='a', index=True)

    #     huice_df.to_pickle(path2_)

    return print(f'{func_name}-{flag_info}=参数回测结束,谢谢使用.')


if __name__ == '__main__':
    from 统计分析 import *
    from 机器学习函数 import *

    if 0 == True:
        mouth_nums = 20

        # ==收集
        func_name_canshu = []
        celue_name_canshu = []

        # 待优化策略

        func_name_list = [
            ['ma_tp_01', [[30, 150, 10], [2, 6, 1], [20, 90, 10], [0, 10, 2]]],
            ['ema_tp_01', [[30, 150, 10], [2, 6, 1], [20, 90, 10], [0, 10, 2]]],
            ['wma_tp_01', [[30, 150, 10], [2, 6, 1], [20, 90, 10], [0, 10, 2]]],
            ['kama_tp_01', [[30, 150, 10], [2, 6, 1], [20, 90, 10], [0, 10, 2]]],
            ['dema_tp_01', [[30, 150, 10], [2, 6, 1], [20, 90, 10], [0, 10, 2]]],
            ['T3_tp_01', [[20, 80, 10], [2, 6, 1], [20, 90, 10], [0, 10, 2]]],
        ]

        # 注册参数及对应策略
        for f in func_name_list:
            name = f[0]
            zongcan, a = canshu_list(n1=f[1][0], n2=f[1][1], n3=f[1][2], n4=f[1][3])
            func_name_canshu.append([name, zongcan])

        start_t = dt.datetime(2019, 3, 1)
        end_t = dt.datetime(2020, 6, 28)

        # 策略的主循环
        for t in range(0, mouth_nums + 1):
            s_time = start_t + relativedelta(months=+t)
            if s_time >= end_t: break
            e_time = s_time + timedelta(days=27)
            time_st = [s_time, e_time]
            time_zhouqi = [1, 3, 5, 10]

            if 1 == True:
                # ===多策略优化
                symbol = 'hsi9999'
                flag_info1 = 'BASE_01='  # 回测标记信息
                flag_info0 = '8_10'  # 回测标记信息
                for zq in time_zhouqi:
                    flag_info = flag_info1 + str(zq) + 'T=' + flag_info0
                    for name in func_name_canshu:
                        func_name1 = name[0]
                        zong_can = name[1]
                        print(f'策略{func_name1}' '\n', '总数:', len(zong_can))

                        all_huice(func_name1, time_st, zong_can, time_zhouqi=zq, cpu_nums=5, flag_info=flag_info,
                                  symbol=symbol)

    if 1 == True:

        path_ = os.getcwd() + r'\huice_log' + '\ma_tp_01_BASE_01=3T.csv'
        print(path_)
        df_all ,pre = jisuan_corr(path=r'F:\回测文件\dema_tp_01_BASE_01=1T=8_10=3T=8_10.csv', c=6, y=1,  p=[5, 1],zongtest=0,jiqixuexi=1)
        # print(pre[1].index)
        # print(pd.DataFrame(pre[1]))
        # print(pd.DataFrame(pre[1])['未来周期收益'])

        # print(pre[1][-10:])

        # exit()
        # path = os.getcwd() + r'\huice_log' + '\ma_tp_01_BASE_01=3T-res.csv'
        # df_all.to_csv(path,mode='a')
        pre2 = pre[pre['未来实际收益'] != 0].copy()
        plot_fenbutu(pre2['未来实际收益'],pre2['预测收益'])



    if 0==True:
        ib_backtester = BacktesterEngine(None, EventEngine())
        ib_backtester.init_engine()
        ib_backtester.event_engine.start()

        try:
            canshu = {'celue_name': 'ma_tp_01', 'huice_s_t': [dt.datetime(2019, 3, 1, 0, 0), dt.datetime(2019, 3, 28, 0, 0)], 'setting': {'ma_len': 30, 'bd': 2, 'dk_len': 20, 'acc': 0, 'zhouqi': 1}, 'canshu0': [30, 2, 20, 0]}

            # print(canshu['canshu0'])
            celue_name = canshu['celue_name']
            ib_backtester.run_backtesting(class_name=celue_name, vt_symbol='HSI9999.HKFE', interval='1m',
                                          start=canshu['huice_s_t'][0], end=canshu['huice_s_t'][1],
                                          rate=0.000018, slippage=1, size=50, pricetick=1, capital=1000000,
                                          setting=canshu['setting'], inverse=False)
            strategy_statistics = ib_backtester.get_result_statistics()
            per_trades: List[TradeData] = ib_backtester.get_all_trades()
            list_res = [strategy_statistics['total_net_pnl'], strategy_statistics['max_drawdown'],
                        strategy_statistics['total_trade_count'], strategy_statistics['sharpe_ratio']]

            print(per_trades)
            exit()
            list_res.append(canshu['canshu0'])
            for i in range(0, len(canshu["canshu0"])):
                list_res.append(canshu["canshu0"][i])

        except Exception as e:
            print(e)
            print(f'参数:{canshu["canshu0"]}', '出错')
            print(traceback.format_exc())

