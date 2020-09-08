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


from typing import List
from vnpy.trader.object import TradeData


# 特征因子计算
def cal_yinzi0(celuename,train_da0, s_t, e_t, res_t, train_res):
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
    for i, d in train_da0.groupby('canshu'):
        # print(i,'计算本周期因子。')
        try:
            d.fillna(0, inplace=True)
            # 训练开始
            df_zong0.loc[i, 'train_s'] = s_t
            # 训练结束
            df_zong0.loc[i, 'train_e'] = e_t
            #  预测时间
            df_zong0.loc[i, 'res_t'] = res_t

            if float(d['end'].sum()) == 0 or d['end'].sum() == np.NAN:
                continue
            df_zong0.loc[i, 'canshu'] = i
            df_zong0.loc[i, 'celuename'] = celuename

            df_zong0.loc[i, '本周期总收益'] = float(d['end'].sum())
            df_zong0.loc[i, '最近周期收益'] = float(d.iloc[-1]['end'].sum())

            df_zong0.loc[i, '月最大回撤'] = d['max_back'].min()
            df_zong0.loc[i, '平均月收益'] = d['end'].mean()
            df_zong0.loc[i, '平均最大回撤'] = d['max_back'].mean()
            df_zong0.loc[i, '平均月夏普率'] = d['sharp_rate'].mean()
            df_zong0.loc[i, '平均月交易次数'] = d['trade_nums'].mean()
            df_zong0.loc[i, '月均交易天数'] = d['all_tradedays'].mean()
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

            df_zong0.loc[i, '最大值'] = (d['end'].cumsum()).max()
            df_zong0.loc[i, '收益std'] = (d['end'].std())
            df_zong0.loc[i, '偏度'] = (d['end'].skew())
            df_zong0.loc[i, '峰度'] = (d['end'].kurt())

            df_zong0.loc[i, '回撤std'] = (d['max_back'].std() * -1)
            df_zong0.loc[i, '平均月夏普率std'] = d['sharp_rate'].std()

            df_zong0.loc[i, '盈撤比'] = (d['end'].mean() / (-1 * d['max_back'].mean())) if (d[
                                                                                             'max_back'].mean()) != 0 else 0
            df_zong0.loc[i, '盈利因子01'] = d['end'].sum() * d['end'].mean() * d['std_num'].mean() / (d['end'].std()) if d[
                                                                                                                         'end'].std() != 0 else 0

            if train_res.empty:
                df_zong0.loc[i, '预测周期真实收益'] = float(0)  # c=1, y=1,
            else:
                # 训练结果
                d1 = train_res
                df_zong0.loc[i, '预测周期真实收益'] = float(d1.loc[d1['canshu'] == i, 'end'].sum())
                # df_zong0.loc[i, '预期偏差'] = float(d1.loc[d1['canshu'] == i,'end'].sum()) - float(d.iloc[-1]['end'].sum())

        except Exception as e:
            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            traceback.print_tb(exc_traceback_obj)
            print('单参数统计出错', d.tail())
        finally:
            df_zong0.fillna(0, inplace=True)

    return df_zong0


# 计算所有数据的特征因子
def cal_all_yinzi(celuename,zong_t, hc_zq, gd_zq, data):
    '''
    生成目标月份的统计数据。
    :param data: 原始数据
    :param index_name:统计列表名字
    :return: 以时间为序列的统计特征字典或者df
      因子1，因子2。。。
    dt1
    dt2
    。
    。
    。
    '''
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
            else:
                train_res0 = data[data['s_time'] == now_t0].copy()

            # 本次训练数据
            train_da0 = data[data['s_time'] >= train_s0]
            train_da0 = train_da0[train_da0['s_time'] <= train_e0]

            dt_yinzi = cal_yinzi0(celuename,train_da0, train_s0, train_e0, res_t=now_t0, train_res=train_res0)

            yinzi_dict[now_t0.strftime(format="%Y-%m-%d")] = dt_yinzi

            if dt_yinzi.empty == False:
                print(f'{now_t0}完成')
                # print(dt_yinzi.sample(5))
            else:
                print(f'*****'*3)
            print(f',最后:')
            print(dt_yinzi.tail())
            i = i + gd_zq

    except Exception as e:
        print(e)

    print(f'所有的统计数据生成！\n特征列名字：{yinzi_dict[zong_t[-1].strftime(format="%Y-%m-%d")].columns}')

    return yinzi_dict

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

# 算法组合
def cal_zuhe3(train_data=None,model_list={}, yinzi=[], Train=True ,last =False):
    if not train_data.empty :
        # 数据过滤
        train_data = data_fliter_fb(train_data)
        # 数据处理：==》处理完的数据
        # print(train_data)
        train_data0 = data_precess(train_data, yinzi=yinzi, pre_style='max_min')

    rf_fl_num = 0
    if last==False:
        # 随机森林分类:输入train_data0，输出train_data0
        if 1 == True:
            if Train:
                fl_con1 = train_data0['预测周期真实收益'] > train_data0['预测周期真实收益'].mean()
                fl_con11 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con1, '预测周期真实收益'].mean()
                fl_con0 = train_data0['预测周期真实收益'] < train_data0['预测周期真实收益'].mean()

                # print(train_data0[['预测周期真实收益']])
                # exit()
                train_data0.loc[fl_con1, 'rf_分类结果'] = 1
                train_data0.loc[fl_con11, 'rf_分类结果'] = 2
                train_data0.loc[fl_con0, 'rf_分类结果'] = -1

                if len(train_data0['rf_分类结果'].tolist()) == 0 \
                        or len(train_data0[train_data0['rf_分类结果'] == 2]['rf_分类结果'].tolist()) == 0 :

                    print('\n======\nrf：训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, 'rf：分类问题'

                # print(train_data0[['预测周期真实收益','rf_分类结果']])
                train_data0.dropna(axis=0,inplace=True)
                model1, X_train_pre, mean_score = randomforest_classify0(X_train=train_data0, yinzi=yinzi,
                                                                         y_train=train_data0['rf_分类结果'],
                                                                         n_estimators=50, criterion='gini', max_depth=4,
                                                                         max_features='log2', min_samples_split=10,
                                                                         random_state=0)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] > rf_fl_num].copy()
                train_data0.rename(columns={'预测值': 'rf_预测值'}, inplace=True)

                model_list['model1'] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list['model1'][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, 'rf_预测值'] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0['rf_预测值'] > rf_fl_num].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, 'rf：预测分类出错'
        # 极限森林分类:输入train_data0，输出train_data0
        if 0 == True:
            if Train:
                fl_con_1 = train_data0['预测周期真实收益'] > 0
                fl_con_0 = train_data0['预测周期真实收益'] < 0
                fl_con_ = train_data0['预测周期真实收益'] == 0

                train_data0.loc[fl_con_1 , 'exf_分类结果'] = 1
                train_data0.loc[fl_con_0 , 'exf_分类结果'] = -1
                train_data0.loc[fl_con_, 'exf_分类结果'] = 0
                train_data0.dropna(axis=0,inplace=Train)
                if len(train_data0['exf_分类结果'].tolist()) == 0 \
                        or len(train_data0[train_data0['exf_分类结果'] == 1]['exf_分类结果'].tolist()) == 0:
                    print('\n======\nexf：训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, 'exf：分类问题'
                # print(train_data0)
                # exit()
                model1, X_train_pre, mean_score = extraforest_classify0(X_train = train_data0, yinzi = yinzi, y_train=train_data0['exf_分类结果'],n_estimators=30,criterion='gini', max_depth=2,min_samples_split = 2, random_state = 0)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] > 1].copy()
                train_data0.rename(columns ={'预测值':'exf_预测值'},inplace=True)
                model_list['model1'] = [model1,mean_score]
            else:
                #拿到预测模型
                model1 = model_list['model1'][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, 'exf_预测值'] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0['exf_预测值'] > 0].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，exf,分类出错')
                    return {}, 'exf：预测分类出错'
        # svc分类:输入train_data0，输出train_data0
        if 1 == True:
            if Train:
                fl_con = train_data0['预测周期真实收益'] > 0#train_data0['预测周期真实收益'].mean()
                fl_con_ = train_data0['预测周期真实收益'] <= 0#train_data0['预测周期真实收益'].mean()

                train_data0.loc[fl_con, 'svc_分类结果'] = 1
                train_data0.loc[fl_con_, 'svc_分类结果'] = 0
                train_data0.fillna(0, inplace=True)

                if len(train_data0['svc_分类结果'].tolist()) == 0 \
                    or len(train_data0[train_data0['svc_分类结果'] == 0]['svc_分类结果'].tolist()) == 0 \
                    or len(train_data0[train_data0['svc_分类结果'] == 1]['svc_分类结果'].tolist()) == 0:
                    print('\n======\nsvc： 训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, 'svc：分类问题'
                # 二次训练
                model2, X_train_pre ,mean_score = svc_classify0(X_train=train_data0,yinzi =yinzi, y_train=train_data0['svc_分类结果'],
                                                                kernel='poly', degree=5, c=3)#kernel='poly', degree=3, c=0.5
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']

                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] == 1].copy()
                train_data0.rename(columns ={'预测值':'svc_预测值'},inplace=True)

                model_list['model2'] = [model2,mean_score]
            else:
                model2 = model_list['model2'][0]
                cat_pre = model2.predict(train_data0[yinzi])
                train_data0.loc[:, 'svc_预测值'] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0['svc_预测值'] == 1].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，svc：分类出错')
                    return {}, 'svc：分类问题'
        # 回归:输入train_data0，输出train_data0
        if 1==True:
            if train_data0[yinzi].empty:
                return model_list, '回归，无预测数据'
            if Train:
                model3, X_train_pre ,mean_score = polynomial_regression0(X_train=train_data0,yinzi=yinzi, y_train=train_data0['预测周期真实收益'],
                                                     degree=3, include_bias=False, normalize=False)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                X_train_pre.rename(columns={'预测值': 'rg_预测值'}, inplace=True)
                train_data0 = X_train_pre

                model_list['model3'] = [model3,mean_score]
            else:
                model3 = model_list['model3'][0]
                reg_pre = model3.predict(train_data0[yinzi])
                train_data0.loc[:, 'rg_预测值'] = pd.Series(reg_pre, index=train_data0.index)
                print('\n======\n预测结果\n回归的结果，线性corr:', np.corrcoef(train_data0['rg_预测值'], train_data0['预测周期真实收益'])[0], '\n')
                print("R2得分:", metrics.r2_score(train_data0['预测周期真实收益'], train_data0['rg_预测值']))

    # 最后一个月只有预测
    if last == True:

        # ===拿到预测模型
        model1 = model_list['model1'][0]
        cat_pre = model1.predict(train_data0[yinzi])
        train_data0.loc[:, 'rf_预测值'] = pd.Series(cat_pre, index=train_data0.index)
        train_data0 = train_data0[train_data0['rf_预测值'] > rf_fl_num].copy()
        if train_data0[yinzi].empty:
            print('预测数据，rf:分类出错')
            print(train_data0.sample(5))
            return {}, 'rf:分类出错'

        # ===拿到预测模型
        model2 = model_list['model2'][0]
        cat_pre = model2.predict(train_data0[yinzi])
        train_data0.loc[:, 'svc_预测值'] = pd.Series(cat_pre, index = train_data0.index)
        train_data0 = train_data0[train_data0['svc_预测值'] >0].copy()
        if train_data0[yinzi].empty:
            print('预测数据，svc:分类出错')
            print(train_data0.sample(5))
            return {}, 'svc:分类出错'

        # ===拿到预测模型
        if train_data0[yinzi].empty:
            print( '回归，无预测数据!')
            return model_list, '回归，无预测数据!'
        else:
            model3 = model_list['model3'][0]
            reg_pre = model3.predict(train_data0[yinzi])
            train_data0.loc[:, 'rg_预测值'] = pd.Series(reg_pre, index=train_data0.index)

        train_data0['canshu'] = train_data0.index
        cols = [ i for i in list(train_data0.columns ) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols],train_data,how='inner',on=['canshu','预测周期真实收益'],left_index=True)
        return model_list, train_data0

    if Train ==False:
        train_data0['canshu'] = train_data0.index
        cols = [i for i in list(train_data0.columns) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols], train_data, how='inner', on=['canshu', '预测周期真实收益'], left_index=True)

    return model_list,train_data0


# 生成训练集
def generate_train_data(path,pkl_path,zong_t = [], hc_zq=6,gd_zq=1):
    '''
    传入文件数据的地址和，需要统计回测的时间，以及每次滚动回测回溯周期，

    以实际结果为训练结果，训练结果的月份为标记月份，

    最总生成，以结果月为序列的训练集。
    最后一个月为预测数据集，月份比传入最后回测日期大一个月，最后一个月的结果为零，就是训练集的结果为零。

    将下周期的结果来验证本周期的选择

    :return:
    '''
    n=''
    for n0 in str(path.split('\\')[-1]).split('_')[:3]:
        n += n0+'_'
    celuename = n+str(path.split('\\')[-1]).split('_')[-3]
    print(celuename)
    da = pd.read_csv(filepath_or_buffer=path, index_col=0)
    zong_t = zong_t
    # 滚动周期
    gd_zq = gd_zq
    # 每次回测周期
    hc_zq = hc_zq
    # 总周期数据进行处理
    con = da['canshu'] == 'canshu'  # 过滤canshu字符串
    da.loc[con, ['s_time', 'end', 'canshu']] = np.nan
    da.dropna(axis=0, inplace=True)

    #  注意，必须先进行字段处理干净，才能选择日期，否则报错。
    da['e_time'] = pd.to_datetime(da['e_time'])
    da['s_time'] = pd.to_datetime(da['s_time'])
    da = da[pd.to_datetime(da['s_time'])>=zong_t[0]]
    da = da[ pd.to_datetime(da['s_time'])<=zong_t[-1]]

    #   简单处理， 目标特征设置为float格式
    to_nums = ['end', 'max_back', 'trade_nums', 'sharp_rate', 'all_tradedays',
               'profit_days', 'win_rate', 'win_mean_num', 'loss_mean_num', 'max_num',
               'min_num', 'mean_num', 'std_num']
    da[to_nums] = da[to_nums].applymap(float)
    #  上小下大
    da.sort_values(by=['s_time'], ascending=True, inplace=True)

    #   计算da数据的因子(统计指标）==》生成所有月份的训练数据集
    df_yinzi = cal_all_yinzi(celuename,zong_t,hc_zq,gd_zq,da)
    #   查看一下。
    for i,v in df_yinzi.items():
        print(i)
        print(v.tail(3))
        print('=='*10)
    #  保存成pickle
    with open(pkl_path,mode='wb') as f:
        pickle.dump(df_yinzi, f)
    print(f'训练数据集，保存完毕！\n{pkl_path}')
    return pkl_path

# 学习，预测
def train_and_predict(path_pkl,res_path ,zong_t = [], hc_zq=6, gd_zq=1):

    # 导入训练集
    with open(path_pkl, mode='rb') as f:
        df_yinzi = pickle.load(f)

    i = 0
    df_zong = {}  # 收集每一阶段的优势参数
    while True:

        # 当前的回测的时间,训练结果月
        now_t0 = zong_t[0] + relativedelta(months=int(i + hc_zq))
        now_ = now_t0.strftime(format="%Y-%m-%d")
        # 训练结束时间
        train_t0 = now_t0 - relativedelta(months=int(gd_zq))
        # 预测结束时间，预测结果月
        pre_t0 = now_t0 + relativedelta(months=int(gd_zq))
        pre_ = pre_t0.strftime(format="%Y-%m-%d")
        print('----------')

        print(train_t0,now_,pre_)

        # 训练数据结束月=回测结束，意味着，没有训练结果了
        if train_t0 == zong_t[-1]:
            break

        df_zong[now_t0] = {}
        # 训练数据
        train_da0 =  pd.DataFrame(df_yinzi[now_].copy())
        # 预测数据
        pre_da0 =  pd.DataFrame(df_yinzi[pre_].copy())
        # 机器学习预测
        try:
            print(now_t0, '开始训练。。。')
            yinzi0 = [  '本周期总收益', '最近周期收益','月最大回撤', '平均月收益','平均最大回撤', '平均月夏普率', '平均月交易次数',
                        '月均交易天数', '月均盈利天数', '月均开单收益std','月均开单最大收益', '月均亏单平均亏损', '月均胜单平均盈利',
                        '月均胜单平均盈利偏度', '月均胜单平均盈利std','月均交易胜率', '月均交易胜率偏度', '月均交易胜率std',
                        '月均开单平均收益', '月均开单平均收益偏度','月均开单平均收益std', '最大值', '收益std', '偏度', '峰度',
                        '回撤std', '平均月夏普率std', '盈撤比','盈利因子01']

            res0 = ['预测周期真实收益', '预期偏差', 'rg_预测值']

            # 训练
            model, df_pre0 = cal_zuhe3(train_data=train_da0, model_list={}, yinzi=yinzi0,Train=True)
            if isinstance(df_pre0, str):
                print(f'{df_pre0}，下一循环。')
                df_zong[now_t0]['预测数据'] = pd.DataFrame()
                df_zong[now_t0]['预测model'] = model
                i= i+gd_zq
                continue
            else:
                df_zong[now_t0]['预测model'] = model


            res_index = 'rg_预测值'
            # 预测# 最后一个月，只给出预测数据，
            if now_t0 >= zong_t[-1]:
                model_list, pre_data = cal_zuhe3(train_data=pre_da0, model_list=model, yinzi=yinzi0, Train=False,last=True)
                if isinstance(pre_data, str):
                    print(f'{pre_data}，最后一个月，无预测数据。')
                    df_zong[now_t0]['预测数据'] = pd.DataFrame()
                else:
                    pre_data.dropna(axis=0, subset=[res_index], inplace=True)
                    pre_data.fillna(0, inplace=True)
                    pre_data.sort_values(by=[res_index], ascending=True, inplace=True)
                    df_zong[now_t0]['预测数据'] = pd.DataFrame(pre_data)
                    break
            #  预测
            else:
                # 预测
                model_list, pre_data = cal_zuhe3(train_data=pre_da0, model_list=model,yinzi=yinzi0, Train=False)
                if isinstance(pre_data, str):
                    print(f'{pre_data}，下一循环。')
                    df_zong[now_t0]['预测数据'] = pd.DataFrame()
                    i = i + gd_zq
                    continue

            # print(pre_data)

            pre_data.dropna(axis=0, subset=[res_index], inplace=True)
            pre_data.fillna(0, inplace=True)
            pre_data.sort_values(by=[res_index], ascending=True, inplace=True)

            res = ['celuename','rf_预测值', 'svc_预测值', 'rg_预测值', '预测周期真实收益',  '本周期总收益', '平均月收益', '平均月交易次数', '月均交易胜率', '平均最大回撤',
                   '平均月夏普率', '月最大回撤',
                   '月均交易天数', '月均盈利天数', '月均开单收益std', '月均开单最大收益', '月均亏单平均亏损',
                   '月均胜单平均盈利', '月均胜单平均盈利偏度', '月均胜单平均盈利std', '月均交易胜率偏度',
                   '月均交易胜率std', '月均开单平均收益', '月均开单平均收益偏度', '月均开单平均收益std', '最大值', '收益std',
                   '偏度', '峰度', '回撤std', '平均月夏普率std', '盈撤比', '盈利因子01', 'canshu', 'train_s',
                   'train_e', 'res_t']
            if pre_data.empty:pass
            else:print(pre_data[res].tail())

            df_zong[now_t0]['预测数据'] = pd.DataFrame(pre_data)
            i =i+gd_zq
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            i =i+1
            continue

    print(df_zong.keys())
    with open(res_path,mode='wb') as f:
        pickle.dump(df_zong,f)
        print('逐月训练结果，已保存！')

    return res_path

#结果分析
def analyze_res(res_paths):

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
            cols =['rg_预测值', '本周期总收益','最近周期收益', '月最大回撤',
                           '平均月收益','平均最大回撤','平均月夏普率','平均月交易次数']
            for index_ in cols :
                if index_ == 'rg_预测值':
                    wei = len(str(int(v.iloc[-1][index_]))) -5
                    if wei > 0:
                        base = pow(10,wei)
                        df0[index_] =  v[index_]/base
                        continue
                df0[index_] = v[index_]

            df0=df0[['s_Time','rg_预测值','预测周期真实收益','本周期总收益',
                     '最近周期收益', '月最大回撤','平均月收益','平均最大回撤','平均月夏普率','平均月交易次数']]
            # print(df0)
            # time.sleep(0.1)

            df0.sort_values(by='rg_预测值',inplace=True)
            df0 = df0.iloc[-5:].copy()
            df_pre = df_pre.append(df0,ignore_index=True,)

        df_pre['策略'] = cl_name
        df_pre.fillna(0,inplace=True)
        df_zong =df_zong.append(df_pre,ignore_index=True)

    df_zong.sort_values(by = 's_Time',inplace=True)
    df_zong = df_zong[df_zong['rg_预测值']>0].copy()
    print(df_zong[[ '策略','s_Time', '预测周期真实收益','rg_预测值',  '平均最大回撤', '平均月交易次数', '平均月夏普率', '平均月收益', '最近周期收益',
       '月最大回撤', '本周期总收益']])

    # exit()
    dong_scatter(data = df_zong[['s_Time','rg_预测值','预测周期真实收益']])
    # echart_plot_3d(data = df_zong[['s_Time','rg_预测值','预测周期真实收益']])
    # plot_fenbutu(df_zong['rg_预测值'], df_zong['预测周期真实收益'])
    # plot_fenbutu02(df_zong['s_Time'], df_zong['预测周期真实收益'],df_zong['rg_预测值'])

    print(df_zong)


if __name__ == '__main__':
    from 统计分析 import *
    from 机器学习函数 import *
    # 单策略运行
    if 0 == True:

        path_ = os.getcwd() + r'\huice_log' + '\MEBD03\dema_tp_03_2019-06-01_3T_8_18.csv'
        data_path = r'F:\new0811\huice_log\MEBD03\dema_tp_03_2019-06-01_3T_8_18.csv'
        train_path = r'F:\new0811\huice_log\MEBD03\dema_tp_03_2019-06-01_3T_8_18_特征.pkl'
        res_path = r'F:\new0811\huice_log\MEBD03\dema_tp_03_2019-06-01_3T_8_18_预测结果.pkl'

        zong_t =[dt.datetime(2019, 6, 1), dt.datetime(2020, 6, 1)]
        #   生成数据
        train_path = generate_train_data(path=data_path,zong_t=zong_t)

        # res_path = train_and_predict(train_path,zong_t)
        analyze_res(res_path=res_path)

        # plot_fenbutu(df_zong['预测值'],df_zong['预测周期真实收益'])

    # 多策略运行
    if 1 == True:

            path0 = os.getcwd() + r'\huice_log\MEBD03'
            path1 = [
                     r'\dema_tp_03_2019-06-01_3T_8_18',
                    r'\dema_tp_03_2019-06-01_1T_8_18',
                     r'\T3_tp_03_2019-06-01_1T_8_18',
                     r'\T3_tp_03_2019-06-01_3T_8_18']

            data_paths = [path0 +p1 + r'.csv' for p1 in path1]
            train_paths = [path0 +p1 +  r'_特征.pkl' for p1 in path1]
            res_paths = [path0 +p1 +  r'_预测结果.pkl' for p1 in path1]

            zong_t = [dt.datetime(2019, 6, 1), dt.datetime(2020, 6, 1)]
            #   集中生成特征数据
            if 0 == True:
                for p,k in zip(data_paths,train_paths):

                    train_path = generate_train_data(path=p,pkl_path=k,zong_t=zong_t)

            #   集中生成计算模型结果，的数据
            if 0 ==True:
                for t ,r in zip(train_paths,res_paths):

                    res_path = train_and_predict(t,r,zong_t)

            #   集中统计结果，的数据
            if 1 == True:
                analyze_res(res_paths=res_paths)
            # plot_fenbutu(df_zong['预测值'],df_zong['预测周期真实收益'])
