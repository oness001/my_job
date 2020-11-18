from numba import jit
import numpy as np
from numba.types import List
import time
import datetime as dt
from datetime import timedelta
import pandas as pd

pd_display_rows = 10
pd_display_cols = 100
pd_display_width = 1000
pd.set_option('display.max_rows', pd_display_rows)
pd.set_option('display.max_columns', pd_display_cols)
pd.set_option('display.width', pd_display_width)
pd.set_option('display.max_colwidth', pd_display_width)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('expand_frame_repr', False)
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 20000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

@jit(nopython=True)
def cal_signal(df0,a=10, sxf=1,slip=1):

    for i in range(a+10,df0.shape[0]):

        trade_time_con = ((df0[i,12]>30) and (df0[i,11]==9)) and((df0[i,12]<20) and (df0[i,11]==16)) or ((df0[i,11]>9)and(df0[i,11]<16))
        # 信号生成
        if trade_time_con :
            # 信号条件
            long_condition = df0[i][4] > np.max(df0[i - a:i,2]) and df0[i-1][7] != 1                            #and False
            close_long = df0[i][4] < (np.max(df0[i - a:i,2])+np.min(df0[i - a:i,3])) and df0[i-1][7] == 1       #and False

            short_condition = df0[i][4] < np.min(df0[i - a:i,3]) and df0[i-1][7] != -1                          and False
            close_short = df0[i][4] > (np.max(df0[i - a:i,2])+np.min(df0[i - a:i,3])) and df0[i-1][7] == -1     and False
            # 开多
            if long_condition:
                df0[i][6] = 1
            # 平多
            elif close_long:
                df0[i][6] = 0
            #开空
            if short_condition:
                df0[i][6] = -1
            # 平空
            elif close_short:
                df0[i][6] = 0
        else:
            # 有持仓，平仓！
            if df0[i - 1][7] != 0:
                df0[i][6] = 0


        #  收益统计
        # 上一根信号=1 仓位 != 1 ,记录持仓价【8】，当日盈亏【9】-手续费和滑点
        if df0[i-1][6]==1 and df0[i-1][7] != 1:
            df0[i][7] = 1
            df0[i][8] = df0[i][1] +sxf+slip
            df0[i][9] = (df0[i][4]-df0[i][1])*df0[i][7] - sxf-slip
        # 上一根信号=1 仓位 != 1,记录持仓价【8】，当日盈亏【9】-手续费和滑点
        elif df0[i-1][6]==-1 and df0[i-1][7] != -1:
            df0[i][7] = -1
            df0[i][8] = df0[i][1] - sxf-slip
            df0[i][9] = (df0[i][4]-df0[i][1])*df0[i][7] - sxf-slip

        # 上一根信号=0 仓位 != 0,记录平仓价【8】，当日盈亏【9】-手续费和滑点
        elif df0[i - 1][6] ==0 and abs(df0[i-1][7]) > 0:
            df0[i][7] = 0
            if df0[i-1][7] >0:
                df0[i][8] = df0[i][1] -(sxf+slip)
                df0[i][10] = 666.66 if df0[i][1] > df0[i-1][8] else -666.66
            elif df0[i-1][7] <0:
                df0[i][8] = df0[i][1]+sxf+slip
                df0[i][10] = 666.66 if df0[i][1] < df0[i-1][8] else -666.66

            df0[i][9] = (df0[i][1]-df0[i-1][4])*df0[i-1][7] - sxf-slip

        # 仓位不变,记录开/平仓价【8】，当日盈亏【9】，变化点数乘以仓位
        else:
            df0[i][7] = df0[i-1][7]
            df0[i][8] = df0[i-1][8]
            df0[i][9] = (df0[i][4]-df0[i-1][4])*df0[i][7]

    df0[:,9][np.isnan(df0[:,9])] = 0
    # 返回交易次数和开仓手数
    return df0

def signal_test(df_input,canshu=200):

    df = df_input.copy()
    cols = df.keys()
    # 转化成np.array
    df0 = np.array(df.values[:,:])

    res = cal_signal(df0,a=canshu)
    zjline = res[:, 9].cumsum()
    trades = [i for i in res[:, 10] if abs(i) >0]
    end_zj = zjline[-1]
    max_zj = max(zjline)
    min_zj = min(zjline)
    std_zj = np.std(zjline)
    mean_zj = np.mean(zjline)
    counts = len(trades)
    sl = len([i for i in trades if i >0])/counts
    yk_rate = sl/(1-sl) if sl !=1 else 1
    res0 = {'最后收益': end_zj, '最大收益': max_zj, '最小收益': min_zj, '收益std': std_zj, '平均收益': mean_zj, '开仓次数': counts, '胜率': sl, '盈亏比': yk_rate}
    return res0

if __name__ == '__main__':
    df_time_list = [['2017-01-01 09:15:00', '2019-12-26 16:25:00']]
    s_time, e_time = df_time_list[0]
    datapath = r'F:\task\恒生股指期货\hsi_data_1min\HSI2011-2019_12.csv'
    print('数据地址：', datapath)
    df = pd.read_csv(filepath_or_buffer=datapath)
    print(df.tail())
    df['candle_begin_time'] = df['datetime']
    df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'])
    df = df[df['candle_begin_time'] >= pd.to_datetime(s_time)]
    df = df[df['candle_begin_time'] <= pd.to_datetime(e_time)]
    df = df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume']]
    df.reset_index(inplace=True, drop=True)
    df.ffill(inplace=True)
    # print(df.tail())
    df['candle_begin_time'] = (df['candle_begin_time'] - np.datetime64(0, 's')) / timedelta(seconds=1)
    df['signal'] = np.nan
    df['pos'] = np.nan
    df['opne_price'] = np.nan
    df['per_lr'] = np.nan
    df['sl'] = np.nan
    df['huors'] = pd.to_datetime(df["candle_begin_time"], unit='s').apply(lambda x: float(x.to_pydatetime().hour))
    df['minutes'] = pd.to_datetime(df["candle_begin_time"], unit='s').apply(lambda x: float(x.to_pydatetime().minute))
    resdf = pd.DataFrame()
    for cs in range(200,300,2):
        a = time.clock()
        res = signal_test(df_input=df,canshu=cs)
        res['参数'] = np.array(cs)
        resdf=resdf.append(res,ignore_index=True)
        print('run：',time.clock() -a ,' s')
    resdf = resdf[['参数','最后收益', '最大收益', '最小收益', '收益std', '平均收益', '开仓次数', '胜率', '盈亏比']]
    resdf.sort_values(by='最后收益',inplace=True)
    print(resdf)