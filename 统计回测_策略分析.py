import os
import signal
import traceback
import pandas as pd
import numpy as np
import sys, time

import datetime as dt

from dateutil.relativedelta import relativedelta

from vnpy.event import EventEngine

from vnpy.app.ib_cta_backtester.engine import BacktesterEngine


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
            df_zong0.loc[i, '预测周期真实收益'] = float(d1['end'].sum())  # c=1, y=1,
            df_zong0.loc[i, '预期偏差'] =float(d1['end'].sum()) - float(d.iloc[-1]['end'].sum())
            return df_zong0, df_zong0
        else:
            df_zong0.loc[i, '预测周期真实收益'] = float(0)  # c=1, y=1,
            df_zong0.loc[i, '预期偏差'] = float(0)
            return df_zong0 ,df_zong0
    except Exception as e:
        exc_type, exc_value, exc_traceback_obj = sys.exc_info()
        traceback.print_tb(exc_traceback_obj)
        print('单参数统计出错', d.tail())
        return df_zong0,df_zong0

def data_precess(train_data,yinzi = [], pre_style = 'max_min'):
    import sklearn.preprocessing as spp
    # 初步过滤筛选
    # yinzi = ['平均月收益', '收益std', '偏度', '夏普稳健因子', '风险因子01', '盈利因子01', '分布因子01', '效率因子01']
    # yinzi = ['本周期收益', '平均月收益', '最大值', '收益std', '偏度', '峰度',
    #          '月最大回撤', '平均最大回撤', '回撤std', '平均月夏普率', '平均月夏普率std', '平均月交易次数', '平均月交易次数std',
    #          '夏普稳健因子', '风险因子01', '盈利因子01', '分布因子01', '效率因子01']
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

    data = data[data['平均月收益'] > 0]
    data = data[data['效率因子01'] > 0]


    return pd.DataFrame(data)

def cal_zuhe1(train_data,pre_data,model_list = [], yinzi =[],Train = True):

    if Train :

        train_data = data_fliter_fb(train_data)

        train_data0 = data_precess(train_data,yinzi=yinzi ,pre_style = 'max_min')
        print('训练开始：数据特征：',len(train_data.columns))
        # 分类定类
        # 训练标签设置
        train_data0.loc[train_data0['预测周期真实收益'] > train_data0['预测周期真实收益'].std(),'未来收益正负'] = 1
        train_data0.loc[train_data0['预测周期真实收益'] <= train_data0['预测周期真实收益'].std(),'未来收益正负'] = 0
        if len(train_data0['未来收益正负'].tolist()) == 0 \
            or len(train_data0[train_data0['未来收益正负']==0]['未来收益正负'].tolist()) ==0\
            or len(train_data0[train_data0['未来收益正负'] ==1]['未来收益正负'].tolist()) ==0 :
            print('\n======\n训练结果分类，出问题。')
            return [],'分类问题'
        pipeline_model1,X_train_pre = polynomial_Logistic_classify0(X_train=train_data0[yinzi], y_train=train_data0['未来收益正负'],
                                             degree = 3 , penalty='l1', solver='liblinear')
        X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
        train_data1 =  X_train_pre[X_train_pre['预测值'] >0].copy()

        # 回归
        if train_data1[yinzi].empty:
            return [], '回归问题'
        pipeline_model2,X_train_pre2 = polynomial_regression0(X_train=train_data1[yinzi], y_train=train_data1['预测周期真实收益'],
                                                                degree=3, include_bias=False, normalize=False)

        X_train_pre2['预测周期真实收益'] = train_data1['预测周期真实收益']
        print('回归的结果准确度',np.corrcoef(X_train_pre['预测值'],X_train_pre['预测周期真实收益']),'\n')

        return [pipeline_model1,pipeline_model2], X_train_pre2

    if not Train:
        pipeline_model1, pipeline_model2 = model_list
        # 过滤数据
        pre_data = data_fliter_fb(pre_data)

        # 预测数据
        print('预测开始：数据特征：',len(pre_data.columns)) #,
        pre_data1 = data_precess(pre_data, yinzi=yinzi, pre_style='max_min')
        # 分类
        pre_data1 = pd.DataFrame(pre_data1)
        cat_pre = pipeline_model1.predict(pre_data1[yinzi])
        pre_data1.loc[:, '预测值'] = pd.Series(cat_pre, index=pre_data1.index)
        pre_data1 = pre_data1[pre_data1['预测值'] > 0].copy()
        # 回归
        pre_data1 = pd.DataFrame(pre_data1)
        reg_pre = pipeline_model2.predict(pre_data1[yinzi])
        pre_data.loc[:, '预测值'] = pd.Series(reg_pre, index=pre_data1.index)
        # print(pre_data.tail(200))
        # exit()


        return  [pipeline_model1,pipeline_model2], pre_data


def cal_corr_1(path, zongtest =0,jiqixuexi= 1):
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
    to_nums = ['end','max_back','sharp_rate','trade_nums']
    da[to_nums] = da[to_nums].applymap(float)

    # da['end'] = da['end'].map(lambda x: float(x))
    # da['max_back'] = da['max_back'].map(lambda x: float(x))
    # da['sharp_rate'] = da['sharp_rate'].map(lambda x: float(x))
    # da['trade_nums'] = da['trade_nums'].map(lambda x: float(x))
    pass

    # 分布一阶段测试
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
    # sklearn一阶段测试
    if jiqixuexi == True:
        # 统计最终，所有参数在回测时间段里的表现均值
        df_canshu = da.groupby(['canshu'])
        df_all = pd.DataFrame()
        df_all1 = pd.DataFrame()

        for i, d0 in df_canshu:
            # 训练样本
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
            res, df_all1 = yinzi_cal(i, d2, df_all1, s_t=s_time, e_t=e_time,train_res=[1, dn2])
            if res.empty:
                print('出错了！')
        # yinzi = ['平均月收益','收益std', '偏度','夏普稳健因子', '风险因子01', '盈利因子01', '分布因子01', '效率因子01']
        # yinzi = ['本周期收益', '平均月收益', '最大值', '收益std', '偏度', '峰度',
        #          '月最大回撤', '平均最大回撤', '回撤std', '平均月夏普率', '平均月夏普率std', '平均月交易次数', '平均月交易次数std',
        #          '夏普稳健因子', '风险因子01', '盈利因子01', '分布因子01', '效率因子01']
        df_all.fillna(0,inplace=True)
        df_all = df_all[df_all['收益std'] != 0].copy()
        df_all1 = df_all1[df_all1['收益std'] != 0].copy()
        # model,prelist = pol_Bayes_regression(X_train= df_all[yinzi], y_train=df_all['预测周期真实收益'],
        #                                       X_test= df_all1[yinzi], y_test=df_all1['预测周期真实收益'],
        #                                         degree= 2, include_bias=False,normalize=True)
        yinzi0 = ['本周期收益', '平均月收益', '最大值', '收益std', '偏度', '峰度',
                 '月最大回撤', '平均最大回撤', '回撤std', '平均月夏普率', '平均月夏普率std', '平均月交易次数', '平均月交易次数std',
                 '夏普稳健因子', '风险因子01', '盈利因子01', '分布因子01', '效率因子01']
        model_list,df_pre0 = cal_zuhe1(train_data=df_all,pre_data=df_all1,model_list=[], yinzi=yinzi0,)
        model_list ,pre_data = cal_zuhe1(train_data=df_all,pre_data=df_all1,model_list=model_list, yinzi=yinzi0,Train=False)
        pre_data.dropna(axis=0, inplace=True)
        y_test, pre_y = pre_data['预测周期真实收益'], pre_data['预测值']
        print("平均绝对值误差:", metrics.mean_absolute_error(y_test, pre_y))
        print("平均平方误差:", metrics.mean_squared_error(y_test, pre_y))
        print("中位绝对值误差:", metrics.median_absolute_error(y_test, pre_y))
        print("R2得分:", metrics.r2_score(y_test, pre_y))

        # exit()

        return pre_data


def cal_corr_n(path, c=3, y=1, p=[5, 1]):
    '''
    将下周期的结果来验证本周期的选择
    :param path:
    :c    >> 选择的优势参数周期
    ：y  >> 选择验证周期大小
    ：p： 》》每次取几个排名的参数
    :return:
    '''

    da = pd.read_csv(filepath_or_buffer=path, index_col=0)

    con = da['canshu'] == 'canshu'  # 过滤canshu字符串
    da.loc[con, ['s_time', 'end', 'canshu']] = np.nan
    da.dropna(axis=0, inplace=True)

    da['s_time'] = pd.to_datetime(da['s_time'])
    da['e_time'] = pd.to_datetime(da['e_time'])

    to_nums = ['end','max_back','sharp_rate','trade_nums']
    da[to_nums] = da[to_nums].applymap(float)
    # da['end'] = da['end'].map(lambda x: float(x))
    # da['max_back'] = da['max_back'].map(lambda x: float(x))
    # da['sharp_rate'] = da['sharp_rate'].map(lambda x: float(x))
    # da['trade_nums'] = da['trade_nums'].map(lambda x: float(x))
    # da['trade_nums'] = da['trade_nums'].map(lambda x: float(x))


    s_time_list = []
    # 进行参数优化，前n名的参数优化
    da.sort_values(by=['s_time'], ascending=True, inplace=True)
    df_zong = pd.DataFrame()   # 收集每一阶段的优势参数
    # 日期分类
    dg = da.groupby(['s_time'])
    for i, v in dg:
        s_time_list.append(i)
    s_time_list.sort(reverse=False)


    # 模拟循环
    for i, v in enumerate(s_time_list):
        # 需要获取到c个月之前数据实际是，c+1月。
        if i <= c: print(v);continue

        # 当前周期的时间  。例如：5-01
        now_time = pd.to_datetime(v)
        # 阶段回测开始。回测c个月，5-01-c==3-01 c=1
        s_t = pd.to_datetime(v) - relativedelta(months=1 * (c))
        # 验证回测结束月初，验证几个月？ 5-01 +y = 6-01 y=1
        n_t = now_time + relativedelta(months=1 * (y))
        if n_t > s_time_list[-1]:
            print('结束：' ,v)
            continue

        # 总样本：5-01-6-01：》》》5-01-6-28
        da0 = da[da['s_time'] >= s_t]
        da0 = da0[da0['s_time'] <= n_t]

        df_zong0 = pd.DataFrame()
        df_zong1 = pd.DataFrame()

        # 训练总样本计算因子
        try:
            print(now_time,'训练数据,==计算因子')
            # 拿到训练样本和结果
            da1 = da0[da0['s_time'] <= now_time]
            for i, d0 in da1.groupby('canshu'):
                # 训练样本
                d = d0.iloc[:-1].copy()
                # 训练样本结果
                dn = d0.iloc[-1].copy()
                # 特征因子计算
                res, df_zong0 = yinzi_cal(i, d, df_zong0, s_t=s_t, e_t=now_time, train_res=[1, dn])

                if res.empty:
                    continue
        except Exception as e:
            print(e,'\n训练出错')
            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            traceback.print_tb(exc_traceback_obj)

        # 预测样本计算因子
        try:
            print(now_time,'预测数据，==计算因子')

            for i, d0 in da0.groupby('canshu'):
                d = d0.iloc[1:-1].copy()  # 测试样本，预测
                dn = d0.iloc[-1].copy()  # 测试结果

                # 预测样本计算因子
                res, df_zong1 = yinzi_cal(i, d, df_zong1, s_t = s_t + relativedelta(months=1), e_t=now_time,
                                          train_res=[1, dn])
                # print(df_zong1.columns)
                # time.sleep(1)
                # continue
                if res.empty:
                    continue
        except Exception as e:
            print(e,'预测时出错')
            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            traceback.print_tb(exc_traceback_obj)

        # 机器学习预测
        try:
            print(now_time,'开始学习！')

            df_zong0.fillna(0, inplace=True)
            df_zong0 = df_zong0[df_zong0['收益std'] != 0].copy()
            df_zong1 = df_zong1[df_zong1['收益std'] != 0].copy()

            yinzi0 = ['本周期收益', '平均月收益', '最大值', '收益std', '偏度', '峰度',
                      '月最大回撤', '平均最大回撤', '回撤std', '平均月夏普率', '平均月夏普率std', '平均月交易次数', '平均月交易次数std',
                      '夏普稳健因子', '风险因子01', '盈利因子01', '分布因子01', '效率因子01']

            model_list, df_pre0 = cal_zuhe1(train_data=df_zong0, pre_data=df_zong1, model_list=[], yinzi=yinzi0, Train=True)
            if len(model_list)==0:
                print(f'{df_pre0}，下一循环。')
                df_zong = df_zong.append(pd.DataFrame(), ignore_index=True, sort=True)

                continue
            model_list, pre_data = cal_zuhe1(train_data=df_zong0, pre_data=df_zong1, model_list=model_list, yinzi=yinzi0,Train=False)

            pre_data.dropna(axis = 0, subset = ["预测值"],inplace = True)
            pre_data.fillna(0,inplace=True)
            print("R2得分:", metrics.r2_score(pre_data['预测周期真实收益'], pre_data['预测值']))

            pre_data.sort_values(by=['预测值'], ascending=True, inplace=True)
            pre_data = pre_data.iloc[-1*p[0] : -1*p[1]].copy()  # 取前p名优秀者

            print(pre_data)
            pre_data.loc[pre_data.index[-1], '单周期相关性'] = pre_data['预测值'].corr(pre_data['预测周期真实收益'])
            df_zong = df_zong.append(pre_data, ignore_index=True, sort=True)
            # print(df_zong[['本周期收益','预测周期真实收益','平均月收益','预测值']].tail())
            # break
        except Exception as e:
            print(e)
            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            traceback.print_tb(exc_traceback_obj)
            continue


    print(df_zong[['本周期收益','预测周期真实收益','平均月收益','预测值']])

    return df_zong




if __name__ == '__main__':
    from 统计分析 import *
    from 机器学习函数 import *



    if 1 == True:

        path_ = os.getcwd() + r'\huice_log' + '\ma_tp_01_BASE_01=3T.csv'
        print(path_)
        # df_zong = cal_corr_1(path=r'F:\回测文件\dema_tp_01_BASE_01=1T=8_10=3T=8_10.csv', zongtest=0,jiqixuexi=1)
        #
        df_zong = cal_corr_n(path=r'F:\回测文件\dema_tp_01_BASE_01=1T=8_10=3T=8_10.csv', c=5, y=1, p=[10, 1])

        plot_fenbutu(df_zong['预测周期真实收益'],df_zong['预测值'])




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

