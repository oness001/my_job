from run_strategies import save_strategy, init
import datetime as dt
from datetime import timedelta
from mongoengine.errors import NotUniqueError

from dateutil.relativedelta import relativedelta
init()


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
    ['ma_tp_03', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
    ['ema_tp_03', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
    ['dema_tp_03', [[30, 120, 10], [1, 5, 1], [20, 80, 10], [0, 10, 2]]],
    ['T3_tp_03', [[10, 100, 10], [1,5, 1], [20, 80, 10], [0, 10, 2]]],
    ['wma_tp_03', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
    ['kama_tp_03', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],

    ['ma_tp_01', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
    ['ema_tp_01', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
    ['dema_tp_01', [[30, 120, 10], [1, 5, 1], [20, 80, 10], [0, 10, 2]]],
    ['T3_tp_01', [[10, 100, 10], [1,5, 1], [20, 80, 10], [0, 10, 2]]],
    ['wma_tp_01', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
    ['kama_tp_01', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],

    ['ma_tp_02', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
    ['ema_tp_02', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
    ['dema_tp_02', [[30, 120, 10], [1, 5, 1], [20, 80, 10], [0, 10, 2]]],
    ['T3_tp_02', [[10, 100, 10], [1,5, 1], [20, 80, 10], [0, 10, 2]]],
    ['wma_tp_02', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
    ['kama_tp_02', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],

    ['ma_tp_04', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
    ['ema_tp_04', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
    ['dema_tp_04', [[30, 120, 10], [1, 5, 1], [20, 80, 10], [0, 10, 2]]],
    ['T3_tp_04', [[10, 100, 10], [1,5, 1], [20, 80, 10], [0, 10, 2]]],
    ['wma_tp_04', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
    ['kama_tp_04', [[30, 120, 10], [1, 6, 1], [20, 80, 10], [0, 10, 2]]],
]


start_t = dt.datetime(2020, 7, 1)
end_t = dt.datetime(2020, 8, 27)
time_zhouqi = [1,3]

func_name_canshu = []
# 注册参数及对应策略
for f in func_name_list:
    name = f[0]
    zongcan, a = canshu_list(n1=f[1][0], n2=f[1][1], n3=f[1][2], n4=f[1][3])
    func_name_canshu.append([name, zongcan])

# 生成参数的主循环
t =0
while True:
    s_time = start_t + relativedelta(months=+t)
    e_time = s_time + timedelta(days=27)
    if s_time >= end_t: break
    t +=1
    # print(s_time)
    # continue
    for name in func_name_canshu:   #策略名
        for zq in time_zhouqi:      #周期
            for canshu0 in name[1]: #对应所有参数遍历
                ps = {"class_name": str(name[0]), "vt_symbol": "HSI9999.HKFE", "interval": "1m",
                      "start": s_time, "end": e_time, "rate": 0.000018, "slippage": 1,
                      "size": 50, "pricetick": 1, "capital": 1000000, "inverse": False,
                      "setting": {'ma_len': canshu0[0], 'bd': canshu0[1],"dk_len": canshu0[2],
                                  'acc': canshu0[3], 'zhouqi': zq}
                      }
                
                try:
                    save_strategy(r"F:\new_0811\strategies", f"{name[0]}.py", ps)
                    print(f'添加参数成功：{s_time}-{name[0]}{ps["setting"]}')
                except NotUniqueError:
                    print(f'{s_time}-{name[0]}{ps["setting"]}已存在!')


