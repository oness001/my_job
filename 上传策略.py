import json
import datetime as dt
import os
import os
import signal
import traceback
import pandas as pd
import numpy as np
import sys, time
import datetime as dt





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



func_name_list = [
            ['ma_tp_04', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
            ['ema_tp_04', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
            ['wma_tp_04', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
            ['kama_tp_04', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
            ['dema_tp_04', [[30, 120, 10], [1, 5, 1], [20, 80, 10], [0, 10, 2]]],
            ['T3_tp_04', [[10, 100, 10], [1,5, 1], [20, 80, 10], [0, 10, 2]]],
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
    # ===多策略优化
    symbol = 'MEBD03_pre_res'
    flag_info1 = f"{start_t.strftime('%Y-%m-%d')}_"  # 回测标记信息
    flag_info0 = '8_18'  # 回测标记信息
    for zq in time_zhouqi:
        flag_info = flag_info1 + str(zq) + 'T_' + flag_info0
        for name in func_name_canshu:
            func_name = name[0]
            zong_can = name[1]

            for canshu0 in zong_can:


                canshu = {'celue_name': func_name, 'huice_s_t': [s_time, e_time],
                      'setting': {'ma_len': canshu0[0], 'bd': canshu0[1], "dk_len": canshu0[2],
                                  'acc': canshu0[3], 'zhouqi': time_zhouqi}, 'canshu0': canshu0}






            ib_backtester.run_backtesting(class_name=canshu['celue_name'], vt_symbol='HSI9999.HKFE', interval='1m',
                                      start=canshu['huice_s_t'][0], end=canshu['huice_s_t'][1],
                                      rate=0.000018, slippage=1, size=50, pricetick=1, capital=1000000,
                                      setting=canshu['setting'], inverse=False)
        canshu_data=[]