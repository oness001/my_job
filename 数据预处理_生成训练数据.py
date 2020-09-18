import os
import signal
import traceback
import pandas as pd
import numpy as np
import sys, time
import pickle
import datetime as dt

from dateutil.relativedelta import relativedelta

from vnpy.event import EventEngine

from vnpy.app.ib_cta_backtester.engine import BacktesterEngine


pd.set_option('max_rows', 99999)
# pd.set_option('max_columns', 20)
pd.set_option('expand_frame_repr', False)
pd.set_option('precision', 8)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from typing import List
from vnpy.trader.object import TradeData


# 特征因子计算
def  cal_yinzi0(celuename,train_da0, s_t, e_t, res_t, train_res):
    '''
    :param d: 计算周期因子
    :param s_t: 开始周期时间
    :param now_time: 当前时间
    :param train_res: 【是否要传入训练结果，样本训练结果】
    :return:
    ['end', 'max_back', 'trade_nums', 'sharp_rate', 'all_tradedays',
       'profit_days', 'win_rate', 'win_mean_num', 'loss_mean_num', 'max_num',
       'min_num', 'mean_num', 'std_num',

    '''
    df_zong0 = pd.DataFrame()  # 保存参数的统计因子
    # 计算每个参数的统计因子
    train_da0 = train_da0.copy()
    # 遍历所有参数，计算本期的统计量
    for i, d in train_da0.groupby('canshu'):
        try:
            d.fillna(0, inplace=True)
            # 训练开始
            df_zong0.loc[i, 'train_s'] = s_t
            # 训练结束
            df_zong0.loc[i, 'train_e'] = e_t
            #  预测时间
            df_zong0.loc[i, 'res_t'] = res_t

            df_zong0.loc[i, 'canshu'] = i
            df_zong0.loc[i, 'celuename'] = celuename
            if float(d['end'].sum()) == 0 or d['end'].sum() == np.NAN:
                continue
            df_zong0.loc[i, 'para1'] = float(d['para1'].mean())
            df_zong0.loc[i, 'para2'] = float(d['para2'].mean())
            df_zong0.loc[i, 'para3'] = float(d['para3'].mean())
            df_zong0.loc[i, 'para4'] = float(d['para4'].mean())

            df_zong0.loc[i, '本周期总收益'] = float(d['end'].sum())
            df_zong0.loc[i, '最近周期收益'] = float(d.iloc[-1]['end'].sum())
            df_zong0.loc[i, '最大回撤'] = d['max_back'].min()
            df_zong0.loc[i, '最大值'] = (d['end'].cumsum()).max()
            df_zong0.loc[i, '收益std'] = (d['end'].std())
            df_zong0.loc[i, '偏度'] = (d['end'].skew())
            df_zong0.loc[i, '峰度'] = (d['end'].kurt())
            df_zong0.loc[i, '平均月收益'] = d['end'].mean()
            df_zong0.loc[i, '平均月最大收益'] = d['max_sum'].mean()
            df_zong0.loc[i, '平均月最大回撤'] = d['max_back'].mean()
            df_zong0.loc[i, '平均月夏普率'] = d['sharp_rate'].mean()
            df_zong0.loc[i, '平均月交易次数'] = d['trade_nums'].mean()
            df_zong0.loc[i, '月均交易天数'] = d['total_days'].mean()
            df_zong0.loc[i, '月均盈利天数'] = d['profit_days'].mean()
            df_zong0.loc[i, '月均开单收益std'] = d['std_num'].mean()
            df_zong0.loc[i, '月均开单最大收益'] = d['max_num'].mean()
            df_zong0.loc[i, '月均亏单平均亏损'] = d['loss_mean_num'].mean()

            df_zong0.loc[i, '月均胜单平均盈利'] = d['win_mean_num'].mean()
            df_zong0.loc[i, '月均胜单平均盈利偏度'] = d['win_mean_num'].skew()
            df_zong0.loc[i, '月均胜单平均盈利std'] = d['win_mean_num'].std()

            df_zong0.loc[i, '月均交易胜率'] = d['win_rate'].mean()
            df_zong0.loc[i, '月均交易胜率偏度'] = d['win_rate'].skew()
            df_zong0.loc[i, '月均交易胜率std'] = d['win_rate'].std()

            df_zong0.loc[i, '月均开单平均收益'] = d['mean_num'].mean()
            df_zong0.loc[i, '月均开单平均收益偏度'] = d['mean_num'].skew()
            df_zong0.loc[i, '月均开单平均收益std'] = d['mean_num'].std()

            df_zong0.loc[i, '回撤std'] = (d['max_back'].std() * -1)

            df_zong0.loc[i, '盈撤比'] = (d['max_sum'].mean() / (-1 * d['max_back'].mean())) if (d['max_back'].mean()) != 0 else 0
            df_zong0.loc[i, '盈利因子01'] = d['max_sum'].sum() * d['end'].mean()  / (d['end'].std()) if (d['end'].std() != 0) else 0

            if train_res.empty:
                df_zong0.loc[i, '预测周期真实收益'] = float(0)  # c=1, y=1,
            else:
                # 训练结果
                d1 = train_res
                df_zong0.loc[i, '预测周期真实收益'] = float(d1.loc[d1['canshu'] == i, 'end'].sum())

        except Exception as e:
            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            traceback.print_tb(exc_traceback_obj)
            print('单参数统计出错', d.tail())
        finally:
            df_zong0.fillna(0, inplace=True)
            # print(df_zong0.tail())

    return df_zong0

# 计算所有数据的特征因子
def cal_all_yinzi(celuename,zong_t, hc_zq, gd_zq, data):

    '''
    生成目标月份的统计数据。
    :param data: 原始数据
    :param index_name:统计列表名字
    :return: 以时间为序列的统计特征字典{时间：数据，时间：数据。。。}
      因子1，因子2。。。
    dt1
    dt2
    。
    。
    。
    '''
    to_nums = ['para1', 'para2', 'para3', 'para4', 'win_rate', 'win_mean_num', 'loss_mean_num', 'max_num', 'min_num',
               'mean_num', 'std_num', 'max_sum', 'end', 'max_back', 'trade_nums',
               'sharp_rate', 'total_days', 'profit_days', 'total_slippage']
    yinzi_dict = {}
    try:
        i = 0
        # 月份循环
        while True:
            pass
            # 回测的时间
            hc_end = zong_t[-1]
            now_t0 = zong_t[0] + relativedelta(months=int(i + hc_zq))
            train_s0 = zong_t[0] + relativedelta(months=int(i))
            train_e0 = now_t0 - relativedelta(months=int(gd_zq))
            # 当前月为，训练结果月，大于回测结束时间，训练结果为空，一旦训练数据的结束月大于回测结束时间，跳出。
            if now_t0 > hc_end:
                train_res0 = pd.DataFrame()
                if train_e0 > zong_t[-1]:
                    break
            else:train_res0 = data[data['s_time'] == now_t0].copy()

            # 本次训练数据
            train_da0 = data[data['s_time'] >= train_s0]
            train_da0 = train_da0[train_da0['s_time'] <= train_e0]

            train_da0.sort_values(by=['s_time'], ascending=True, inplace=True)

            print(f'{celuename}训练数据月份：',now_t0.strftime(format="%Y-%m-%d"))

            # 每次计算一次
            dt_yinzi = cal_yinzi0(celuename,train_da0, train_s0, train_e0, res_t=now_t0, train_res=train_res0)

            yinzi_dict[now_t0.strftime(format="%Y-%m-%d")] = dt_yinzi

            if dt_yinzi.empty == False:
                print(f'{now_t0}完成。')
                # print(dt_yinzi.sample(5))
            else:
                print(f'*****'*3)
            print(f',最后:')
            print(dt_yinzi.tail())
            i = i + gd_zq

    except Exception as e:
        print(traceback.format_exc())
        print(e)

    print(f'所有的统计数据生成！\n特征列名字：{yinzi_dict[zong_t[-1].strftime(format="%Y-%m-%d")].columns}')
    # exit()

    return yinzi_dict


# 生成训练集
def generate_train_data(da,zong_t = [], hc_zq=6,gd_zq=1):
    '''

    以实际结果为训练结果，训练结果的月份为标记月份，

    最总生成，以结果月为序列的训练集。
    最后一个月为预测数据集，月份比传入最后回测日期大一个月，最后一个月的结果为零，就是训练集的结果为零。

    将下周期的结果来验证本周期的选择

    :return:
    '''

    zong_t = zong_t
    # 滚动周期
    gd_zq = gd_zq
    # 每次回测周期
    hc_zq = hc_zq

    #  注意，必须先进行字段处理干净，才能选择日期，否则报错。
    da['s_time'] = pd.to_datetime(da['s_time'])
    da = da[pd.to_datetime(da['s_time']) >= zong_t[0]]
    da = da[ pd.to_datetime(da['s_time']) <= zong_t[-1]]

    #   简单处理， 目标特征设置为float格式
    to_nums = ['para1', 'para2', 'para3', 'para4','win_rate', 'win_mean_num', 'loss_mean_num', 'max_num', 'min_num',
                'mean_num', 'std_num', 'max_sum', 'end', 'max_back', 'trade_nums',
                'sharp_rate', 'total_days', 'profit_days', 'total_slippage']

    da[to_nums] = da[to_nums].applymap(float)
    da['canshu'] = da['canshu'].apply(lambda x:str(x))
    celuename = da.iloc[-1]['celue_name']
    da['canshu_'] = da['canshu']
    da.set_index('canshu_', drop=True, inplace=True)
    #  上小下大
    da.sort_values(by=['s_time'], ascending=True, inplace=True)
    da.fillna(0,inplace=True)
    #   计算da数据的因子(统计指标）==》生成所有月份的训练数据集
    print(da.head(2))
    print(da.tail(2))
    # exit()
    # 计算该策略的统计量特征
    df_yinzi = cal_all_yinzi(celuename,zong_t,hc_zq,gd_zq,da)

    # print(df_yinzi.sample(10))
    # exit()
    #   查看一下。
    for i,v in df_yinzi.items():
        print(i)
        print(v.tail(3))
        print('=='*10)
    #  保存成pickle
    # path = pkl_path0 + str(celuename) + 'train_data.pickle'
    # with open(path,mode='wb') as f:
    #     pickle.dump(df_yinzi, f)
    # print(f'训练数据集，保存完毕！\n{path}')
    print(f'训练数据集，生成完毕！')

    return df_yinzi


# 特征因子计算
def cal_yinzi2(celuename,train_da0, s_t, e_t, res_t, train_res,h_data=None):

    '''
    :param d: 计算周期因子
    :param s_t: 开始周期时间
    :param now_time: 当前时间
    :param train_res: 【是否要传入训练结果，样本训练结果】
    :return:
    ['end', 'max_back', 'trade_nums', 'sharp_rate', 'all_tradedays',
       'profit_days', 'win_rate', 'win_mean_num', 'loss_mean_num', 'max_num',
       'min_num', 'mean_num', 'std_num',

    '''
    yz = ['canshu','win_rate', 'win_mean_num', 'loss_mean_num', 'max_num', 'min_num',
               'mean_num', 'std_num', 'max_sum', 'end', 'max_back', 'trade_nums',
               'sharp_rate', 'total_days', 'profit_days', 'total_slippage']
    df_zong0 = pd.DataFrame()  # 保存参数的统计因子
    # 计算每个参数的统计因子
    # train_da0.sort_values(by='s_time',ascending=True,inplace=True)
    # 遍历所有参数，计算本期的统计量
    ix = 0
    for i, d in (train_da0.groupby('s_time')):
        # continue
        ix +=1
        print(celuename,i)
        h_data0= h_data[h_data['datetime'] > i]
        h_data0= h_data0[h_data0['datetime'] <= i+datetime.timedelta(days=27)]
        print(h_data0.tail())

        desc = h_data0['close'].describe()

        if i == s_t:
            new_columns = {}
            for name in yz:
                if name == 'canshu':
                    new_columns[name] = name
                else:
                    new_columns[name] = name+ '_' + f'{ix}'

            d = d.rename(columns=new_columns)
            df_zong0 = df_zong0.append(d)
            # print(df_zong0.shape)
        else:
            new_columns = {}
            for name in yz:
                if name =='canshu':
                    new_columns[name] = name
                else:
                    new_columns[name] = name+ '_'+f'{ix}'

            d = d.rename(columns=new_columns)

            df_zong0 = df_zong0.merge(d[new_columns.values()],how='outer',on ='canshu',left_index=True)
        new_columns[name] = name + '_' + f'{ix}'

        df_zong0['市场_mean_m' + f'{ix}'] = desc['mean']/desc['min']
        df_zong0['市场_std_mean' + f'{ix}'] = desc['std'] /  desc['mean']
        df_zong0['市场_std_mm' + f'{ix}'] = desc['std']/(desc['max']- desc['min'])
        df_zong0['市场_mm_m' + f'{ix}'] = (desc['max']- desc['min'])/desc['mean']
        vmm= h_data0['volume'].max() -h_data0['volume'].min()
        mean_vol = (h_data0['volume'].diff()).mean()
        df_zong0['市场_vol_mm' + f'{ix}'] =vmm
        df_zong0['市场_vol_std_mm' + f'{ix}'] = h_data0['close'].std()/vmm
        df_zong0['市场_vol_mm_m' + f'{ix}'] =vmm/mean_vol
        df_zong0['市场_vol_skew' + f'{ix}'] =h_data0['volume'].skew()
        df_zong0['市场_vol_kurt' + f'{ix}'] =h_data0['volume'].kurt()
        df_zong0['市场_skew_' + f'{ix}'] = h_data0['close'].skew()
        df_zong0['市场_kurt_' + f'{ix}'] = h_data0['close'].kurt()
    df_zong0['celuename'] =df_zong0['canshu'] +'_'+ celuename
    df_zong0['e_time'] =e_t
    df_zong0['res_time'] =res_t


    if train_res.empty:
        df_zong0['预测周期真实收益'] = 0
        print(res_t)
        print(fr'{celuename}完成！', df_zong0.tail())

    else:
        print(train_res.tail())
        # exit()
        train_res =train_res.rename(columns={f'end':'预测周期真实收益'})
        df_zong0 = df_zong0.merge(train_res[['canshu','预测周期真实收益']],how='outer',on ='canshu',left_index=True)

    cols = ['celuename', 'canshu','celue_name', 's_time','e_time', 'res_time', '预测周期真实收益' ,'para1', 'para2', 'para3', 'para4']
    col = cols + list(set(df_zong0.keys())-set(cols))
    df_zong0 = df_zong0[col]
    print(fr'{celuename}完成！',df_zong0.tail())


    return df_zong0

# 计算所有数据的特征因子
def cal_all_yinzi2(celuename, zong_t, hc_zq, gd_zq, data,h_data):
    '''
    生成目标月份的统计数据。
    :param data: 原始数据
    :param index_name:统计列表名字
    :return: 以时间为序列的统计特征字典{时间：数据，时间：数据。。。}
      因子1，因子2。。。
    dt1
    dt2
    。
    '''
    to_nums = ['para1', 'para2', 'para3', 'para4', 'win_rate', 'win_mean_num', 'loss_mean_num', 'max_num', 'min_num',
               'mean_num', 'std_num', 'max_sum', 'end', 'max_back', 'trade_nums',
               'sharp_rate', 'total_days', 'profit_days', 'total_slippage']
    yinzi_dict = {}
    try:
        i = 0
        # 月份循环
        while True:
            pass
            # 回测的时间
            hc_end = zong_t[-1]

            now_t0 = zong_t[0] + relativedelta(months=int(i + hc_zq))
            train_s0 = zong_t[0] + relativedelta(months=int(i))
            train_e0 = now_t0 - relativedelta(months=int(gd_zq))
            # 当前月为，训练结果月，大于回测结束时间，训练结果为空，一旦训练数据的结束月大于回测结束时间，跳出。
            if now_t0 > hc_end:
                train_res0 = pd.DataFrame()
                print(data[data['s_time'] == now_t0])

                if train_e0 > zong_t[-1]:
                    break
            else:
                train_res0 = data[data['s_time'] == now_t0].copy()

            # 本次训练数据
            train_da0 = data[data['s_time'] >= train_s0]
            train_da0 = train_da0[train_da0['s_time'] <= train_e0]

            train_da0.sort_values(by=['s_time'], ascending=True, inplace=True)

            print(f'{celuename}训练数据结果的月份：', now_t0.strftime(format="%Y-%m-%d"))

            # 每次计算一次
            h_data0 = h_data[h_data['datetime']>train_s0]
            h_data0 = h_data0[h_data0['datetime']<train_e0+datetime.timedelta(days=28)]
            dt_yinzi = cal_yinzi2(celuename, train_da0, train_s0, train_e0, res_t=now_t0, train_res=train_res0,h_data=h_data0)

            yinzi_dict[now_t0.strftime(format="%Y-%m-%d")] = dt_yinzi

            if now_t0 > hc_end:
                break
            print(f',最后:')
            print(dt_yinzi.tail())
            i = i + gd_zq

    except Exception as e:
        print(traceback.format_exc())
        print(e)

    print(f'所有的统计数据生成！\n特征列名字：{yinzi_dict[zong_t[-1].strftime(format="%Y-%m-%d")].columns}')
    # exit()

    return yinzi_dict

# 生成训练集
def generate_train_data2(da,zong_t = [], hc_zq=6,gd_zq=1,h_data=[]):
    '''

    以实际结果为训练结果，训练结果的月份为标记月份，

    最总生成，以结果月为序列的训练集。
    最后一个月为预测数据集，月份比传入最后回测日期大一个月，最后一个月的结果为零，就是训练集的结果为零。

    将下周期的结果来验证本周期的选择

    :return:
    '''


    zong_t = zong_t
    # 滚动周期
    gd_zq = gd_zq
    # 每次回测周期
    hc_zq = hc_zq

    #  注意，必须先进行字段处理干净，才能选择日期，否则报错。
    da['s_time'] = pd.to_datetime(da['s_time'])
    da = da[pd.to_datetime(da['s_time']) >= zong_t[0]]
    da = da[ pd.to_datetime(da['s_time']) <= zong_t[-1]]

    #   简单处理， 目标特征设置为float格式
    to_nums = ['para1', 'para2', 'para3', 'para4','win_rate', 'win_mean_num', 'loss_mean_num', 'max_num', 'min_num',
                'mean_num', 'std_num', 'max_sum', 'end', 'max_back', 'trade_nums',
                'sharp_rate', 'total_days', 'profit_days', 'total_slippage']

    da[to_nums] = da[to_nums].applymap(float)
    da['canshu'] = da['canshu'].apply(lambda x:str(x))
    celuename = da.iloc[-1]['celue_name']
    da['canshu_'] = da['canshu']
    da.set_index('canshu_', drop=True, inplace=True)
    #  上小下大
    da.sort_values(by=['s_time'], ascending=True, inplace=True)
    da.fillna(0,inplace=True)
    #   计算da数据的因子(统计指标）==》生成所有月份的训练数据集
    print('所有的数据始终')
    print(da.head(2))
    print(da.tail(2))
    # exit()
    # 计算该策略的统计量特征
    df_yinzi = cal_all_yinzi2(celuename,zong_t,hc_zq,gd_zq,da,h_data)

    # print(df_yinzi.sample(10))
    # exit()
    #   查看一下。
    for i,v in df_yinzi.items():
        print(i)
        print(v.tail(3))
        print('=='*10)
    #  保存成pickle
    # path = pkl_path0 + str(celuename) + 'train_data.pickle'
    # with open(path,mode='wb') as f:
    #     pickle.dump(df_yinzi, f)
    # print(f'训练数据集，保存完毕！\n{path}')
    print(f'训练数据集，生成完毕！')

    return df_yinzi

# 训练前的数据处理
def data_precess(train_data,yinzi = [], pre_style = 'max_min'):
    import sklearn.preprocessing as spp

    train_data.fillna(0, inplace=True)
    train_data = train_data[train_data['平均月收益'] != 0].copy()
    train_data0 = pd.DataFrame()

    # 数据处理
    if pre_style == 'max_min':
        train_data0 = spp.MinMaxScaler().fit_transform(train_data[yinzi])

    elif pre_style == 'max_abs':
        train_data0 = spp.MaxAbsScaler().fit_transform(train_data[yinzi])

    elif pre_style == 'standar':
        train_data0 = spp.StandardScaler().fit_transform(train_data[yinzi])

    elif pre_style == 'normal':
        train_data0 = spp.Normalizer().fit_transform(train_data[yinzi])


    train_data0 = pd.DataFrame(train_data0, columns=yinzi, index=train_data.index)
    train_data0.loc[:, '预测周期真实收益'] = pd.Series(train_data['预测周期真实收益'])

    return train_data0


def data_fliter_fb(data):

    # 过滤算法

    data = data[data['平均月收益'] > data['平均月收益'].mean()]
    # data = data[data['效率因子01'] > 0]


    return pd.DataFrame(data)



# 打包原回测文件生成训练数据。
def pack_to_traindatas(data_path, train_path, zong_t,time_lidu,hc_zq=3,watch_data=False):
    '''
   1.读取回测文件。
   2.读取训练数据地址，保存一个空文件。
   3.按策略名字生成训练数据数据  》数据格式：{时间：数据，时间：数据。。。}
   4.最终整理所有数据：》{策略名字：{时间：数据，时间：数据。。。}，
                        策略名字：{时间：数据，时间：数据。。。}，
                        策略名字：{时间：数据，时间：数据。。。}}

    '''
    if watch_data:
        with open(data_path, 'rb') as f0:
            train_data = pickle.load(f0)
            for k, v in train_data.items():
                if k.split('_')[-1] == f'{time_lidu}':
                    print(k)
                    print(v.sample(10))
        return print('查看回测数据完毕。')

    #  1.读取回测文件
    train_datas = {}
    with open(data_path, 'rb') as f0:
        resdf = pickle.load(f0)

    #  2.1读取训练数据地址，保存一个空文件。
    time0 = time.strftime("%m-%d", time.localtime())
    train_path = train_path.split('.')[0] + f'time0_{time_lidu}_{hc_zq}_new_{time0}' + '.pickle'
    if os.path.exists(train_path):  # 读取保存地址
        print(f'地址存在{train_path}。')

    #  2.2读取训练数据地址，保存一个空文件。
    else:

        print(f'文件地址可用，新建一个空文件:{train_path}')
        with open(train_path, 'wb') as f_new:
            pickle.dump(train_datas, f_new)
            print(f'空文件保存在：{train_path}')

    #  3.按策略名字生成训练数据数据
    for k in resdf.keys():  # 遍历里面的策略名
        da = resdf[k]
        #   生成数据
        if len(da.keys()) > 0:
            print(f'{k}开始生成训练数据')
            # 数据格式：{时间：数据，时间：数据。。。}
            trainda = generate_train_data(da, zong_t=zong_t, hc_zq=hc_zq)
            train_datas[k] = trainda
            print(f'训练数据已经添加，当前：{k}')
            with open(train_path, 'wb') as f1:
                print('已存在》key：', train_datas.keys())
                pickle.dump(train_datas, f1)
                print(f'训练数据存放完毕：\n{k}保存在：{train_path}')
    with open(train_path, 'wb') as f1:
        print('生成完毕：已存在key：')
        for i in train_datas.keys():
            print(i)
        print(f'\n保存在：{train_path}')

def updata_to_local_traindatas(data_path,train_path,clnames=[],dates=[],hc_zq=6):
    with open(data_path,'rb') as f:
        hc_res = pickle.load(f)

    with open(train_path,'rb') as f1:
        local_train_res = pickle.load(f1)

    for clm in clnames:
        # 拿到指定策略的本地回测数据
        hc_local_df = hc_res[clm]
        # 检查一下更新的月份是否存在
        for y in dates:
            if y in local_train_res[clm].keys():
                print(f'{clm}训练数据的月份{y}，存在！')
                continue
            # 不存在，需要更新！


def pack_to_traindatas2(data_path, train_path, zong_t,time_lidu,hc_zq=3,watch_data=False):
    '''
    模式二，直接进行历史读取
   1.读取回测文件。
   2.读取训练数据地址，保存一个空文件。
   3.按策略名字生成，训练数据数据  》数据格式：{时间：数据，时间：数据。。。}
   4.最终整理所有数据：》{策略名字：{时间：数据，时间：数据。。。}，
                        策略名字：{时间：数据，时间：数据。。。}，
                        策略名字：{时间：数据，时间：数据。。。}}

    '''
    if watch_data:
        with open(data_path, 'rb') as f0:
            train_data = pickle.load(f0)
            for k, v in train_data.items():
                if k.split('_')[-1] == f'{time_lidu}':
                    print(k)
                    print(v.sample(10))
        return print('查看回测数据完毕。')

    #  1.读取回测文件
    train_datas = {}
    with open(data_path, 'rb') as f0:
        resdf = pickle.load(f0)

    #  2.1读取训练数据地址，保存一个空文件。
    time0 = time.strftime("%m-%d", time.localtime())
    train_path = train_path.split('.')[0] + f'time0_{time_lidu}_{hc_zq}_new_{time0}' + '.pickle'
    if os.path.exists(train_path):  # 读取保存地址
        print(f'地址存在{train_path}。')

    #  2.2读取训练数据地址，保存一个空文件。
    else:
        print(f'文件地址可用，新建一个空文件:{train_path}')
        with open(train_path, 'wb') as f_new:
            pickle.dump(train_datas, f_new)
            print(f'空文件保存在：{train_path}')
    # 3.0 获取历史信息。
    hf = HKFuture()  # 首次使用先配置..KRData
    h_data = hf.get_main_contract_bars('HSI', start=zong_t[0], end=(zong_t[-1] + datetime.timedelta(days=27)))
    #  3.按策略名字生成训练数据数据
    for k in resdf.keys():  # 遍历里面的策略名
        da = resdf[k] #da是df数据，所有月份和参数。
        #   生成数据
        if len(da.keys()) > 0:
            print(f'{k}开始生成训练数据')
            # trainda 数据格式：{时间：数据，时间：数据。}
            trainda = generate_train_data2(da, zong_t=zong_t, hc_zq=hc_zq,h_data = h_data)
            train_datas[k] = trainda
            print(f'训练数据已经添加，当前：{k}')
            print('已存在》key：', train_datas.keys())
            #
            # with open(train_path, 'wb') as f1:
            #     print('已存在》key：', train_datas.keys())
            #     pickle.dump(train_datas, f1)
            #     print(f'训练数据存放完毕：\n{k}保存在：{train_path}')

    with open(train_path, 'wb') as f1:
        pickle.dump(train_datas, f1)
        print('生成完毕：已存在key：')
        for i in train_datas.keys():
            print(i)
        print(f'\n保存在：{train_path}')



#结果分析
def analyze_res(res_paths,info = ''):
    cols =[]
    df_zong = pd.DataFrame()
    for r in res_paths:

        df_pre = pd.DataFrame()
        with open(r,mode='rb') as f:
            pre_res = pickle.load(f)

        cl_name = ''
        for i,v in pre_res.items():
            df0 = pd.DataFrame()
            print(f'滚动观察：{i}')
            if v['预测数据'].empty:
                print(f'{i} =》没有数据，不用操作！')
                df0.at[0,'s_Time'] = i

                df_pre = df_pre.append(df0, ignore_index=True, )
                continue

            print(f'{i}:有数据。')

            v= v['预测数据']
            cl_name = v.iloc[-1]['celuename']
            df0['预测周期真实收益'] = v['预测周期真实收益']
            df0['s_Time'] = i
            #参考col
            cols =['end_预测值',
                   '本周期总收益','最近周期收益','最大回撤', '平均月最大回撤','平均月收益','平均月夏普率','平均月交易次数']
            for index_ in cols :
                if index_ == 'end_预测值':
                    wei = len(str(int(v.iloc[-1][index_]))) -5
                    if wei > 0:
                        base = pow(10,wei)
                        df0[index_] =  v[index_]/base
                        continue
                df0[index_] = v[index_]

            df0=df0[['s_Time','预测周期真实收益']+cols]
            # print(df0)
            # time.sleep(0.1)

            df0.sort_values(by='end_预测值',inplace=True)
            df0 = df0.iloc[-5:].copy()
            df_pre = df_pre.append(df0,ignore_index=True,)

        df_pre['策略'] = cl_name
        df_pre.fillna(0,inplace=True)
        df_zong =df_zong.append(df_pre,ignore_index=True)

    df_zong.sort_values(by = 's_Time',inplace=True)
    df_zong = df_zong[df_zong['end_预测值'] > 0].copy()
    print(df_zong[[ '策略','s_Time','预测周期真实收益']+cols])
    corr_list= []
    df_corr = pd.DataFrame(index=cols)
    for i in cols:
        for j in cols:

            # corr_list.append(np.corrcoef(df_zong[i],df_zong[j])[0])
            df_corr.loc[i,j] = min(np.corrcoef(df_zong[i],df_zong[j])[0])

    print('相关矩阵：')

    print(df_corr)
    exit()

    dong_scatter(data = df_zong[['s_Time','end_预测值','预测周期真实收益']],info = info)
    # echart_plot_3d(data = df_zong[['s_Time','end_预测值','预测周期真实收益']])
    # plot_fenbutu(df_zong['end_预测值'], df_zong['预测周期真实收益'])
    # plot_fenbutu02(df_zong['s_Time'], df_zong['预测周期真实收益'],df_zong['end_预测值'])

    print(df_zong)

def clm_tezheng(train_data):

    clm = set(sorted(train_data['celuename'].to_list()))
    clm_dict= {}
    for i, c in enumerate(clm):
        clm_dict[c]={'clm%s'%i:1}
    for ix, c in enumerate(clm_dict.keys()):
        for i, c0 in enumerate(clm):
            if 'clm%s' % i in clm_dict[c].keys():continue
            else: clm_dict[c]['clm%s' % i] = 0

    for i in clm:
        con = train_data['celuename'] == i
        # 同名策略名统一命名。
        for m,v in clm_dict[i].items():
            train_data.loc[con,m]=v

    # print(clm_dict)
    return train_data


if __name__ == '__main__':
    from 统计分析 import *
    from 机器学习函数 import *
    from 算法组合 import *
    from DB_in_out_strategies.查看_下载_strategies_res import show_local_data

    sys.path.append(r"F:\task\恒生股指期货\KRData_master\KRData")
    from KRData.HKData import HKMarket, HKFuture

    zong_t = [dt.datetime(2019,1, 1), dt.datetime(2020, 8, 1)]


    #
    # hf = HKFuture()  # 首次使用先配置..KRData
    # data = hf.get_main_contract_bars('HSI', start=zong_t[0], end=[-1])
    # print(data.head())
    # print(data.tail())
    # exit()

    # 单策略运行
    zong_t = [dt.datetime(2019,1, 1), dt.datetime(2020, 8, 1)]
    # dapath = r'F:\new_0811\DB_in_out_strategies\res_df_in_dict_to03_09-09.pickle'

    dapath = r'F:\new_0811\huice_log\train_res\MEBD_03_yy训练数据time0_1_3_new_09-14.pickle'

    # show_local_data(dapath, show=True)
    # exit()

    # 生成训练数据
    if 1 == True:
        path_ = os.getcwd() + r'\huice_log' + '\MEBD03\dema_tp_03_2019-06-01_3T_8_18.csv'
        data_path = r'F:\new_0811\DB_in_out_strategies\res_df_in_dict_to04_09-09.pickle'
        train_path = r'F:\new_0811\huice_log\train_res\MEBD_04_yy训练数据.pkl'
        res_path0 = r'F:\new_0811\huice_log\MEBD03_pre_res'
        time_lidu = 1
        zong_t = [dt.datetime(2019, 1, 1), dt.datetime(2020, 8, 1)]
        pack_to_traindatas2(data_path, train_path, zong_t,time_lidu=1,hc_zq=3, watch_data=False)


