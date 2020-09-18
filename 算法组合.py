import os
import traceback
import pandas as pd
import numpy as np
import sys, time
import pickle
import datetime as dt
from dateutil.relativedelta import relativedelta
from 机器学习函数 import *
pd.set_option('max_rows', 99999)
# pd.set_option('max_columns', 20)
pd.set_option('expand_frame_repr', False)
pd.set_option('precision', 8)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

yinzi0 = [  '本周期总收益',
       '最近周期收益', '最大回撤', '最大值', '收益std', '偏度', '峰度', '平均月收益', '平均月最大收益',
       '平均月最大回撤', '平均月夏普率', '平均月交易次数', '月均交易天数', '月均盈利天数', '月均开单收益std',
       '月均开单最大收益', '月均亏单平均亏损', '月均胜单平均盈利', '月均胜单平均盈利偏度', '月均胜单平均盈利std',
       '月均交易胜率', '月均交易胜率偏度', '月均交易胜率std', '月均开单平均收益', '月均开单平均收益偏度',
       '月均开单平均收益std', '回撤std', '盈撤比', '盈利因子01']


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def k_pac(train_data,yinzi,N=10, kernel='linear',n_jobs=1):
    from sklearn.decomposition import KernelPCA
    if N > len(yinzi):N = len(yinzi)-2
    print(f'开始进行：PAC降维：{N}')
    train_data0 = data_precess(train_data,yinzi, pre_style = 'standar')
    transformer = KernelPCA(n_components=N, kernel=kernel,n_jobs=n_jobs )
    print(train_data0[yinzi].tail(2))
    # print(train_data0[yinzi].dtypes)
    X_transformed = transformer.fit_transform(train_data0[yinzi])
    yinzis = [str(i+1) for i in range(0,X_transformed.shape[1])]
    X_transformed = pd.DataFrame(X_transformed,index=train_data.index,columns=yinzis)
    X_transformed['预测周期真实收益']=train_data['预测周期真实收益']
    X_transformed['canshu']=train_data['canshu']
    X_transformed['celue_name'] = train_data['celue_name']
    X_transformed.set_index(train_data.index,drop=True,inplace=True)
    print(f'数据降维后：',X_transformed.shape)
    print(f'数据降维后，除特征的列名：',[x for x in X_transformed.keys() if x not in yinzis])

    return X_transformed,train_data,yinzis

def data_fliter_fb(data,yinzi0):
    yinzi0 = ['本周期总收益',
              '最近周期收益', '最大回撤', '最大值', '收益std', '偏度', '峰度', '平均月收益', '平均月最大收益',
              '平均月最大回撤', '平均月夏普率', '平均月交易次数', '月均交易天数', '月均盈利天数', '月均开单收益std',
              '月均开单最大收益', '月均亏单平均亏损', '月均胜单平均盈利', '月均胜单平均盈利偏度', '月均胜单平均盈利std',
              '月均交易胜率', '月均交易胜率偏度', '月均交易胜率std', '月均开单平均收益', '月均开单平均收益偏度',
              '月均开单平均收益std', '回撤std', '盈撤比', '盈利因子01']
    # 过滤算法
    data = data[data[ '平均月交易次数'] > data[ '平均月交易次数'].mean()]
    data = data[data['本周期总收益'] > data['本周期总收益'].mean()]
    #

    return pd.DataFrame(data)

def data_precess(train_data,yinzi = [], pre_style = 'max_min'):
    '''

    :param train_data:
    :param yinzi:
    :param pre_style: [max_min,max_abs,normal,standar]
    :return:
    '''
    import sklearn.preprocessing as spp
    # train_data[yinzi].dropna(axis=1,inplace=True)
    # print(train_data.tail(20))
    train_data[yinzi].fillna(0, inplace=True)
    # train_data = train_data[train_data['平均月收益'] != 0].copy()
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
    # train_data0.loc[:, '预测周期真实收益'] = pd.Series(train_data['预测周期真实收益'])

    return train_data0

def corr_jw(train_data=None,n=10, yinzi=[],res_index = '预测周期真实收益'):

    train_data[yinzi].fillna(0,inplace=True)
    train_data[res_index].fillna(0,inplace=True)

    n_f = len(yinzi)
    train_res = normalization(train_data[res_index]) #归一化
    # print(train_res.tail(10))
    df_corr = pd.DataFrame(index=yinzi)

    for i in yinzi:
        # print(i)
        # print(min(np.corrcoef(train_data[i], train_res)[0]))
        col0 = normalization(train_data[i])
        df_corr.loc[i, 'corr_nums'] = (min(np.corrcoef(col0, train_res)[0]))
        df_corr.loc[i, 'name_columns'] = i

    df_corr.sort_values(by='corr_nums',ascending=True,inplace=True)
    # print(df_corr)
    # exit()
    yinzi_cols = df_corr.iloc[:n]['name_columns'].tolist()+df_corr.iloc[-n:]['name_columns'].tolist()


    return train_data ,yinzi_cols

def build_yinzi(train_data, yinzi=[],res_index = '预测周期真实收益'):
    yinzi0 = yinzi.copy()
    new_yinzi = []
    li = []
    for i,c in enumerate(list(yinzi)):
        # print(len(yinzi))
        # print(c)
        li.append(c)
        # yinzi.remove(c)
        # print(list(set(yinzi)-set(li)))
        # print(len(list(set(yinzi)-set(li))))
        for c2 in list(set(yinzi)-set(li)):

            if c == c2:continue

            train_data[c+str('/')+c2] =  (train_data[c]/(train_data[c2]+0.1))#归一化
            # train_data[c+str('+')+c2] =  (train_data[c]+train_data[c2]) #归一化
            train_data[c+str('0.5')+c2] =  (train_data[c]**2+train_data[c2]**2)**0.5 #归一化

            new_yinzi.append(c+str('/')+c2)
            # new_yinzi.append(c+str('+')+c2)
            new_yinzi.append(c+str('0.5')+c2)


    new_yinzi = new_yinzi + yinzi0
    train_data[new_yinzi].fillna(0,inplace=True)
    # print(train_data.tail())

    return train_data,new_yinzi


if 1==True:
    #算法组合。
    pass

# 算法组合
def cal_zuhe3(train_data=None,model_list={}, yinzi=[], Train=True ,last =False,n_jobs=2):
    if not train_data.empty :
        # 数据过滤
        train_data = data_fliter_fb(train_data)
        # 数据处理：==》处理完的数据
        train_data0 = data_precess(train_data, yinzi=yinzi, pre_style='max_min')
        # print('\n\n====',train_data0)

    fl_num = 0
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
                train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': '1_预测值'}, inplace=True)

                model_list['model1'] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list['model1'][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, '1_预测值'] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0['1_预测值'] > fl_num].copy()

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
                train_data0.rename(columns ={'预测值':'2_预测值'},inplace=True)

                model_list['model2'] = [model2,mean_score]
            else:
                model2 = model_list['model2'][0]
                cat_pre = model2.predict(train_data0[yinzi])
                train_data0.loc[:, '2_预测值'] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0['2_预测值'] == 1].copy()

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
                X_train_pre.rename(columns={'预测值': 'end_预测值'}, inplace=True)
                train_data0 = X_train_pre

                model_list['model3'] = [model3,mean_score]
            else:
                model3 = model_list['model3'][0]
                reg_pre = model3.predict(train_data0[yinzi])
                train_data0.loc[:, 'end_预测值'] = pd.Series(reg_pre, index=train_data0.index)
                print('\n======\n预测结果\n回归的结果，线性corr:', np.corrcoef(train_data0['end_预测值'], train_data0['预测周期真实收益'])[0], '\n')
                print("R2得分:", metrics.r2_score(train_data0['预测周期真实收益'], train_data0['end_预测值']))

    # 最后一个月只有预测
    if last == True:

        # ===拿到预测模型
        model1 = model_list['model1'][0]
        cat_pre = model1.predict(train_data0[yinzi])
        train_data0.loc[:, '1_预测值'] = pd.Series(cat_pre, index=train_data0.index)
        train_data0 = train_data0[train_data0['1_预测值'] > fl_num].copy()
        if train_data0[yinzi].empty:
            print('预测数据，rf:分类出错')
            print(train_data0.sample(5))
            return {}, 'rf:分类出错'

        # ===拿到预测模型
        model2 = model_list['model2'][0]
        cat_pre = model2.predict(train_data0[yinzi])
        train_data0.loc[:, '2_预测值'] = pd.Series(cat_pre, index = train_data0.index)
        train_data0 = train_data0[train_data0['2_预测值'] >0].copy()
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
            train_data0.loc[:, 'end_预测值'] = pd.Series(reg_pre, index=train_data0.index)

        train_data0['canshu'] = train_data0.index
        cols = [ i for i in list(train_data0.columns ) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols],train_data,how='inner',on=['canshu','预测周期真实收益'],left_index=True)
        return model_list, train_data0

    if Train ==False:
        print('预测：')
        # print(train_data0.tail())
        # print(train_data['canshu'])
        train_data0['canshu'] = train_data0.index

        # exit()
        cols = [i for i in list(train_data0.columns) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols], train_data, how='inner', on=['canshu', '预测周期真实收益'], left_index=True)

    return model_list,train_data0

def cal_zuhe1(train_data=None,model_list={}, yinzi=[], Train=True ,last =False,n_jobs=2):
    '''
    组合1，在前两次的训练里面，使用了随机遍历的超参数拿最优进行，模型选择。

    '''
    if not train_data.empty :
        # 数据过滤
        train_data = data_fliter_fb(train_data)
        # 数据处理：==》处理完的数据
        train_data0 = data_precess(train_data, yinzi=yinzi, pre_style='max_min')
        # print('\n\n====',train_data0)

    fl_num = 0
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
                model1, X_train_pre, mean_score = randomforest_classify_grid(X_train=train_data0, yinzi=yinzi,
                                                                         y_train=train_data0['rf_分类结果'])
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': '1_预测值'}, inplace=True)

                model_list['model1'] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list['model1'][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, '1_预测值'] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0['1_预测值'] > fl_num].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, 'rf：预测分类出错'

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

                model2, X_train_pre ,mean_score = svc_classify_grid(X_train=train_data0,yinzi =yinzi, y_train=train_data0['svc_分类结果'])
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']

                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] == 1].copy()
                train_data0.rename(columns ={'预测值':'2_预测值'},inplace=True)

                model_list['model2'] = [model2,mean_score]
            else:
                model2 = model_list['model2'][0]
                cat_pre = model2.predict(train_data0[yinzi])
                train_data0.loc[:, '2_预测值'] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0['2_预测值'] == 1].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，svc：分类出错')
                    return {}, 'svc：分类问题'

        # 回归:输入train_data0，输出train_data0
        if 1==True:
            if train_data0[yinzi].empty:
                return model_list, '回归，无预测数据'
            if Train:
                model3, X_train_pre ,mean_score = lars_regression(X_train=train_data0,yinzi=yinzi, y_train=train_data0['预测周期真实收益'],
                                                     degree=3)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                X_train_pre.rename(columns={'预测值': 'end_预测值'}, inplace=True)
                train_data0 = X_train_pre

                model_list['model3'] = [model3,mean_score]
            else:
                model3 = model_list['model3'][0]
                reg_pre = model3.predict(train_data0[yinzi])
                train_data0.loc[:, 'end_预测值'] = pd.Series(reg_pre, index=train_data0.index)
                print('\n======\n预测结果\n回归的结果，线性corr:', np.corrcoef(train_data0['end_预测值'], train_data0['预测周期真实收益'])[0], '\n')
                print("R2得分:", metrics.r2_score(train_data0['预测周期真实收益'], train_data0['end_预测值']))

    # 最后一个月只有预测
    if last == True:

        # ===拿到预测模型
        model1 = model_list['model1'][0]
        cat_pre = model1.predict(train_data0[yinzi])
        train_data0.loc[:, '1_预测值'] = pd.Series(cat_pre, index=train_data0.index)
        train_data0 = train_data0[train_data0['1_预测值'] > fl_num].copy()
        if train_data0[yinzi].empty:
            print('预测数据，rf:分类出错')
            print(train_data0.sample(5))
            return {}, 'rf:分类出错'

        # ===拿到预测模型
        model2 = model_list['model2'][0]
        cat_pre = model2.predict(train_data0[yinzi])
        train_data0.loc[:, '2_预测值'] = pd.Series(cat_pre, index = train_data0.index)
        train_data0 = train_data0[train_data0['2_预测值'] >0].copy()
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
            train_data0.loc[:, 'end_预测值'] = pd.Series(reg_pre, index=train_data0.index)

        train_data0['canshu'] = train_data0.index
        cols = [ i for i in list(train_data0.columns ) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols],train_data,how='inner',on=['canshu','预测周期真实收益'],left_index=True)
        return model_list, train_data0

    if Train ==False:
        print('预测：')
        # print(train_data0.tail())
        # print(train_data['canshu'])
        train_data0['canshu'] = train_data0.index

        # exit()
        cols = [i for i in list(train_data0.columns) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols], train_data, how='inner', on=['canshu', '预测周期真实收益'], left_index=True)

    return model_list,train_data0

def cal_zuhe2(train_data=None,model_list={}, yinzi=[], Train=True ,last =False,n_jobs=2):
    '''
    第一次分类，随机森林
    第二次分类，svc分类，["rbf",'sigmoid','linear','poly']
    分类都采取，随机grid方式过滤。
    最后进行回归。
    '''
    if not train_data.empty :
        # 数据过滤
        # train_data = data_fliter_fb(train_data)
        # 数据处理：==》处理完的数据
        train_data0 = data_precess(train_data, yinzi=yinzi, pre_style='max_min')
        # print('\n\n====',train_data0)

    fl_num = 0
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
                model1, X_train_pre, mean_score = randomforest_classify_grid(X_train=train_data0, yinzi=yinzi,
                                                                         y_train=train_data0['rf_分类结果'],n_iter=10)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': '1_预测值'}, inplace=True)

                model_list['model1'] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list['model1'][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, '1_预测值'] = pd.Series(cat_pre, index = train_data0.index)
                train_data0 = train_data0[train_data0['1_预测值'] > fl_num].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, 'rf：预测分类出错'

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

                model2, X_train_pre ,mean_score = svc_classify_grid(X_train=train_data0,yinzi =yinzi,
                                                                    y_train=train_data0['svc_分类结果'],n_iter=20,kernel=["rbf",'sigmoid','linear','poly']  )
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']

                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] == 1].copy()
                train_data0.rename(columns ={'预测值':'2_预测值'},inplace=True)

                model_list['model2'] = [model2,mean_score]
            else:
                model2 = model_list['model2'][0]
                cat_pre = model2.predict(train_data0[yinzi])
                train_data0.loc[:, '2_预测值'] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0['2_预测值'] == 1].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，svc：分类出错')
                    return {}, 'svc：分类问题'

        # 回归:输入train_data0，输出train_data0
        if 1==True:
            if train_data0[yinzi].empty:
                return model_list, '回归，无预测数据'
            if Train:
                model3, X_train_pre ,mean_score = lars_regression(X_train=train_data0,yinzi=yinzi, y_train=train_data0['预测周期真实收益'],
                                                     degree=3)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                X_train_pre.rename(columns={'预测值': 'end_预测值'}, inplace=True)
                train_data0 = X_train_pre

                model_list['model3'] = [model3,mean_score]
            else:
                model3 = model_list['model3'][0]
                reg_pre = model3.predict(train_data0[yinzi])
                train_data0.loc[:, 'end_预测值'] = pd.Series(reg_pre, index=train_data0.index)
                print('\n======\n预测结果\n回归的结果，线性corr:', np.corrcoef(train_data0['end_预测值'], train_data0['预测周期真实收益'])[0], '\n')
                print("R2得分:", metrics.r2_score(train_data0['预测周期真实收益'], train_data0['end_预测值']))

    # 最后一个月只有预测
    if last == True:

        # ===拿到预测模型
        model1 = model_list['model1'][0]
        cat_pre = model1.predict(train_data0[yinzi])
        train_data0.loc[:, '1_预测值'] = pd.Series(cat_pre, index=train_data0.index)
        train_data0 = train_data0[train_data0['1_预测值'] > fl_num].copy()
        if train_data0[yinzi].empty:
            print('预测数据，rf:分类出错')
            print(train_data0.sample(5))
            return {}, 'rf:分类出错'

        # ===拿到预测模型
        model2 = model_list['model2'][0]
        cat_pre = model2.predict(train_data0[yinzi])
        train_data0.loc[:, '2_预测值'] = pd.Series(cat_pre, index = train_data0.index)
        train_data0 = train_data0[train_data0['2_预测值'] >0].copy()
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
            train_data0.loc[:, 'end_预测值'] = pd.Series(reg_pre, index=train_data0.index)

        train_data0['canshu'] = train_data0.index
        cols = [ i for i in list(train_data0.columns ) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols],train_data,how='inner',on=['canshu','预测周期真实收益'],left_index=True)
        return model_list, train_data0

    if Train ==False:
        print('预测：')
        # print(train_data0.tail())
        # print(train_data['canshu'])
        train_data0['canshu'] = train_data0.index

        # exit()
        cols = [i for i in list(train_data0.columns) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols], train_data, how='inner', on=['canshu', '预测周期真实收益'], left_index=True)

    return model_list,train_data0

def cal_zuhe4(train_data=None,model_list={}, yinzi=[], Train=True ,last =False,n_jobs=2):
    '''
    第一次分类，adaboost算法
    第二次分类，svc分类，["rbf",'sigmoid','linear','poly']
    分类都采取，随机grid方式过滤。
    最后进行回归。
    '''
    print('欢迎使用组合4')
    if not train_data.empty :
        # 数据过滤
        # train_data = data_fliter_fb(train_data)
        # 数据处理：==》处理完的数据
        train_data0 = data_precess(train_data, yinzi=yinzi, pre_style='max_min')
        # print('\n\n====',train_data0)

    fl_num = 0
    if last==False:
        # ada_dtree:输入train_data0，输出train_data0
        if 1 == True:
            if Train:
                fl_con1 = train_data0['预测周期真实收益'] > train_data0['预测周期真实收益'].mean()
                fl_con11 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con1, '预测周期真实收益'].mean()
                fl_con0 = train_data0['预测周期真实收益'] < train_data0['预测周期真实收益'].mean()

                # print(train_data0[['预测周期真实收益']])
                # exit()
                train_data0.loc[fl_con1, 'adt_分类结果'] = 1
                train_data0.loc[fl_con11, 'adt_分类结果'] = 2
                train_data0.loc[fl_con0, 'adt_分类结果'] = -1

                if len(train_data0['adt_分类结果'].tolist()) == 0 \
                        or len(train_data0[train_data0['adt_分类结果'] == 2]['adt_分类结果'].tolist()) == 0 :

                    print('\n======\nrf：训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, 'rf：分类问题'

                # print(train_data0[['预测周期真实收益','rf_分类结果']])
                train_data0.dropna(axis=0,inplace=True)
                model1, X_train_pre, mean_score = adaboosting_dtree_grid(X_train=train_data0, yinzi=yinzi,
                                                                         y_train=train_data0['adt_分类结果'],n_iter=10,n_jobs=n_jobs)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': '1_预测值'}, inplace=True)

                model_list['model1'] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list['model1'][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, '1_预测值'] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0['1_预测值'] > fl_num].copy()
                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, '1_预测值：预测分类出错'

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

                model2, X_train_pre ,mean_score = svc_classify_grid(X_train=train_data0,yinzi =yinzi,
                                                                    y_train=train_data0['svc_分类结果'],
                                                                    n_iter=20,kernel=["rbf",'sigmoid','linear','poly'],n_jobs=n_jobs)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']

                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] == 1].copy()
                train_data0.rename(columns ={'预测值':'2_预测值'},inplace=True)

                model_list['model2'] = [model2,mean_score]
            else:
                model2 = model_list['model2'][0]
                cat_pre = model2.predict(train_data0[yinzi])
                train_data0.loc[:, '2_预测值'] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0['2_预测值'] == 1].copy()
                if train_data0[yinzi].empty:
                    print('预测数据，2_预测值：分类出错')
                    return {}, '2_预测值：分类问题'

        # 回归:输入train_data0，输出train_data0
        if 1==True:
            if train_data0[yinzi].empty:
                return model_list, '回归，无预测数据'
            if Train:
                model3, X_train_pre ,mean_score = lars_regression(X_train=train_data0,yinzi=yinzi, y_train=train_data0['预测周期真实收益'],
                                                     degree=3)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                X_train_pre.rename(columns={'预测值': 'end_预测值'}, inplace=True)
                train_data0 = X_train_pre

                model_list['model3'] = [model3,mean_score]
            else:
                model3 = model_list['model3'][0]
                reg_pre = model3.predict(train_data0[yinzi])
                train_data0.loc[:, 'end_预测值'] = pd.Series(reg_pre, index=train_data0.index)
                print('\n======\n预测结果\n回归的结果，线性corr:', np.corrcoef(train_data0['end_预测值'], train_data0['预测周期真实收益'])[0], '\n')
                print("R2得分:", metrics.r2_score(train_data0['预测周期真实收益'], train_data0['end_预测值']))

    # 最后一个月只有预测
    if last == True:

        # ===拿到预测模型
        model1 = model_list['model1'][0]
        cat_pre = model1.predict(train_data0[yinzi])
        train_data0.loc[:, '1_预测值'] = pd.Series(cat_pre, index=train_data0.index)
        train_data0 = train_data0[train_data0['1_预测值'] > fl_num].copy()
        if train_data0[yinzi].empty:
            print('预测数据，rf:分类出错')
            print(train_data0.sample(5))
            return {}, '1_预测值:分类出错'

        # ===拿到预测模型
        model2 = model_list['model2'][0]
        cat_pre = model2.predict(train_data0[yinzi])
        train_data0.loc[:, '2_预测值'] = pd.Series(cat_pre, index = train_data0.index)
        train_data0 = train_data0[train_data0['2_预测值'] >0].copy()
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
            train_data0.loc[:, 'end_预测值'] = pd.Series(reg_pre, index=train_data0.index)

        train_data0['canshu'] = train_data0.index
        cols = [ i for i in list(train_data0.columns ) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols],train_data,how='inner',on=['canshu','预测周期真实收益'],left_index=True)
        return model_list, train_data0

    if Train ==False:
        print('预测：')
        # print(train_data0.tail())
        # print(train_data['canshu'])
        train_data0['canshu'] = train_data0.index

        # exit()
        cols = [i for i in list(train_data0.columns) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols], train_data, how='inner', on=['canshu', '预测周期真实收益'], left_index=True)

    return model_list,train_data0

def cal_zuhe5(train_data=None,model_list={}, yinzi=[], Train=True ,last =False,n_jobs=2):
    '''
    数据归一化
    第一次分类，adaboost算法
    第二次分类，random——Forrest分类
    第3次分类，svc分类，["rbf",'sigmoid','linear','poly']
    分类都采取，随机grid方式过滤。
    最后进行回归。
    '''
    print(f'欢迎使用{sys._getframe().f_code.co_name}：fb-ada-rf-svc-poly')

    if Train:
        print(f'训练，本次数据量：{train_data[yinzi].shape}')
        # 数据过滤
        train_data = data_fliter_fb(train_data)
    else:print(f'预测，本次数据量：{train_data[yinzi].shape}')

    if not train_data.empty :

        # 数据处理：==》处理完的数据
        train_data0 = data_precess(train_data, yinzi=yinzi, pre_style='normal')
        # print('\n\n====',train_data0)

    pre_res_names = []
    fl_num = 0
    # ======
    if last==False:

        # ada_dtree:输入train_data0，输出train_data0
        if 1 == True:
            pre_res_name = '1_预测值'
            model_name = 'model_1'
            pre_res_names.append(pre_res_name)
            if Train:
                fl_con1 = train_data0['预测周期真实收益'] > train_data0['预测周期真实收益'].mean()
                fl_con11 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con1, '预测周期真实收益'].mean()
                fl_con0 = train_data0['预测周期真实收益'] < train_data0['预测周期真实收益'].mean()

                train_data0.loc[fl_con1, 'adt_分类结果'] = 1
                train_data0.loc[fl_con11, 'adt_分类结果'] = 2
                train_data0.loc[fl_con0, 'adt_分类结果'] = -1

                if len(train_data0['adt_分类结果'].tolist()) == 0 \
                        or len(train_data0[train_data0['adt_分类结果'] == 2]['adt_分类结果'].tolist()) == 0:
                    print('\n======\nrf：训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, 'rf：分类问题'

                # print(train_data0[['预测周期真实收益','rf_分类结果']])
                train_data0.dropna(axis=0, inplace=True)
                print(f'{model_name}数据量：{train_data0[yinzi].shape}')

                model1, X_train_pre, mean_score = adaboosting_dtree_grid(X_train=train_data0, yinzi=yinzi,
                                                                         y_train=train_data0['adt_分类结果'], n_iter=10,
                                                                         n_jobs=n_jobs)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list[model_name][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0[pre_res_name] > fl_num].copy()
                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, f'{pre_res_name}：预测分类出错'

        # 随机森林分类:输入train_data0，输出train_data0
        if 1 == True:
            pre_res_name = '2_预测值'
            model_name = 'model_2'

            pre_res_names.append(pre_res_name)
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
                        or len(train_data0[train_data0['rf_分类结果'] == 2]['rf_分类结果'].tolist()) == 0:
                    print('\n======\nrf：训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, 'rf：分类问题'

                train_data0.dropna(axis=0, inplace=True)
                print(f'{model_name}数据量：{train_data0[yinzi].shape}')
                model1, X_train_pre, mean_score = randomforest_classify_grid(X_train=train_data0, yinzi=yinzi,
                                                                             y_train=train_data0['rf_分类结果'],
                                                                             n_iter=10)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list[model_name][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, 'pre_res_name'] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0['pre_res_name'] > fl_num].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, 'rf：预测分类出错'

        # svc分类:输入train_data0，输出train_data0
        if 0 == True:
            pre_res_name = '3_预测值'
            model_name = 'model_3'

            pre_res_names.append(pre_res_name)

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
                    return model_list, f'{pre_res_name}：分类问题'
                # zai次训练
                print(f'{model_name}数据量：{train_data0[yinzi].shape}')
                model3, X_train_pre ,mean_score = svc_classify_grid(X_train=train_data0,yinzi =yinzi,
                                                                    y_train=train_data0['svc_分类结果'],
                                                                    n_iter=10,kernel=["rbf",'sigmoid','linear'],n_jobs=n_jobs)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']

                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] == 1].copy()
                train_data0.rename(columns ={'预测值':pre_res_name},inplace=True)

                model_list[model_name] = [model3, mean_score]
            else:
                model2 = model_list[model_name][0]
                cat_pre = model2.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0[pre_res_name] == 1].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，svc：分类出错')
                    return {}, f'{pre_res_names}：分类问题'

        # 回归:输入train_data0，输出train_data0
        if 1==True:
            pre_res_name = 'end_预测值'
            model_name = 'model_4'
            pre_res_names.append(pre_res_name)
            if train_data0[yinzi].empty:
                return model_list, '回归，无预测数据'
            if Train:
                print(f'{model_name}数据量：{train_data0[yinzi].shape}')

                model4, X_train_pre ,mean_score = polynomial_regression_grid(X_train=train_data0,yinzi=yinzi, y_train=train_data0['预测周期真实收益'], n_iter=3, n_jobs=n_jobs)

                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                X_train_pre.rename(columns={'预测值': pre_res_name}, inplace=True)
                train_data0 = X_train_pre

                model_list[model_name] = [model4, mean_score]
            else:
                model4 = model_list[model_name][0]
                reg_pre = model4.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(reg_pre, index=train_data0.index)
                print('\n======\n预测结果\n回归的结果，线性corr:', np.corrcoef(train_data0[pre_res_name], train_data0['预测周期真实收益'])[0], '\n')
                print("R2得分:", metrics.r2_score(train_data0['预测周期真实收益'], train_data0[pre_res_name]))

    # 最后一个月只有预测
    if last == True:
        if 'model_1' in model_list.keys():
            # ===拿到预测模型
            model1 = model_list['model_1'][0]
            cat_pre = model1.predict(train_data0[yinzi])
            train_data0.loc[:, '1_预测值'] = pd.Series(cat_pre, index=train_data0.index)
            train_data0 = train_data0[train_data0['1_预测值'] > fl_num].copy()
            if train_data0[yinzi].empty:
                print(f'预测数据，1:分类出错')
            return {}, '1:分类出错'

        if 'model_2' in model_list.keys():
            # ===拿到预测模型
            model2 = model_list['model_2'][0]
            cat_pre = model2.predict(train_data0[yinzi])
            train_data0.loc[:, '2_预测值'] = pd.Series(cat_pre, index = train_data0.index)
            train_data0 = train_data0[train_data0['2_预测值'] >0].copy()
            if train_data0[yinzi].empty:
                print('预测数据，2:分类出错')
                print(train_data0.sample(5))
                return {}, '2:分类出错'

        if 'model_3' in model_list.keys():
            # ===拿到预测模型
            model3 = model_list['model_3'][0]
            cat_pre = model3.predict(train_data0[yinzi])
            train_data0.loc[:, '3_预测值'] = pd.Series(cat_pre, index=train_data0.index)
            train_data0 = train_data0[train_data0['3_预测值'] > 0].copy()
            if train_data0[yinzi].empty:
                print('预测数据，3:分类出错')
                print(train_data0.sample(5))
                return {}, '3:分类出错'


        # ===拿到预测模型
        if train_data0[yinzi].empty:
            print( '回归，无预测数据!')
            return model_list, '回归，无预测数据!'
        else:
            model4 = model_list['model_4'][0]
            reg_pre = model4.predict(train_data0[yinzi])
            train_data0.loc[:, 'end_预测值'] = pd.Series(reg_pre, index=train_data0.index)

        train_data0['canshu'] = train_data0.index
        cols = [ i for i in list(train_data0.columns ) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols],train_data,how='inner',on=['canshu','预测周期真实收益'],left_index=True)
        return model_list, train_data0

    if Train ==False:
        print('预测：')
        # print(train_data0.tail())
        # print(train_data['canshu'])
        train_data0['canshu'] = train_data0.index

        # exit()
        cols = [i for i in list(train_data0.columns) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols], train_data, how='inner', on=['canshu', '预测周期真实收益'], left_index=True)

    return model_list,train_data0

def cal_zuhe6(train_data=None,model_list={}, yinzi=[], Train=True ,last =False,n_jobs=2):
    '''

    '''
    print(f'欢迎使用{sys._getframe().f_code.co_name}：fb-ada-rf-svc-poly')
    if Train:
        print(f'训练，本次数据量：{train_data[yinzi].shape}')
    else:print(f'预测，本次数据量：{train_data[yinzi].shape}')

    if not train_data.empty :
        # 数据过滤
        # train_data = data_fliter_fb(train_data)
        # 数据处理：==》处理完的数据
        train_data0 = data_precess(train_data, yinzi=yinzi, pre_style='normal')
        # print('\n\n====',train_data0)

    pre_res_names = []
    fl_num = 0

    if last==False:

        # ada_dtree:输入train_data0，输出train_data0
        if 1 == True:
            pre_res_name = '1_预测值'
            model_name = 'model_1'
            pre_res_names.append(pre_res_name)
            if Train:
                fl_con1 = train_data0['预测周期真实收益'] > train_data0['预测周期真实收益'].mean()
                fl_con11 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con1, '预测周期真实收益'].mean()
                fl_con0 = train_data0['预测周期真实收益'] < train_data0['预测周期真实收益'].mean()

                # print(train_data0[['预测周期真实收益']])
                # exit()
                train_data0.loc[fl_con1, 'adt_分类结果'] = 1
                train_data0.loc[fl_con11, 'adt_分类结果'] = 2
                train_data0.loc[fl_con0, 'adt_分类结果'] = -1

                if len(train_data0['adt_分类结果'].tolist()) == 0 \
                        or len(train_data0[train_data0['adt_分类结果'] == 2]['adt_分类结果'].tolist()) == 0:
                    print('\n======\nrf：训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, 'rf：分类问题'

                # print(train_data0[['预测周期真实收益','rf_分类结果']])
                train_data0.dropna(axis=0, inplace=True)
                print(f'数据量：{train_data[yinzi].shape}')

                model1, X_train_pre, mean_score = adaboosting_dtree_grid(X_train=train_data0, yinzi=yinzi,
                                                                         y_train=train_data0['adt_分类结果'], n_iter=10,
                                                                         n_jobs=n_jobs)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list[model_name][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0[pre_res_name] > fl_num].copy()
                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, f'{pre_res_name}：预测分类出错'

        # 随机森林分类:输入train_data0，输出train_data0
        if 1 == True:
            pre_res_name = '2_预测值'
            model_name = 'model_2'

            pre_res_names.append(pre_res_name)

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
                        or len(train_data0[train_data0['rf_分类结果'] == 2]['rf_分类结果'].tolist()) == 0:
                    print('\n======\nrf：训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, 'rf：分类问题'

                train_data0.dropna(axis=0, inplace=True)
                model1, X_train_pre, mean_score = randomforest_classify_grid(X_train=train_data0, yinzi=yinzi,
                                                                             y_train=train_data0['rf_分类结果'],
                                                                             n_iter=10)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list[model_name][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, 'pre_res_name'] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0['pre_res_name'] > fl_num].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, 'rf：预测分类出错'

        # svc分类:输入train_data0，输出train_data0
        if 0 == True:
            pre_res_name = '3_预测值'
            model_name = 'model_3'

            pre_res_names.append(pre_res_name)

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
                    return model_list, f'{pre_res_name}：分类问题'
                # zai次训练

                model3, X_train_pre ,mean_score = svc_classify_grid(X_train=train_data0,yinzi =yinzi,
                                                                    y_train=train_data0['svc_分类结果'],
                                                                    n_iter=10,kernel=["rbf",'sigmoid','linear'],n_jobs=n_jobs)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']

                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] == 1].copy()
                train_data0.rename(columns ={'预测值':pre_res_name},inplace=True)

                model_list[model_name] = [model3, mean_score]
            else:
                model2 = model_list[model_name][0]
                cat_pre = model2.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0[pre_res_name] == 1].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，svc：分类出错')
                    return {}, f'{pre_res_names}：分类问题'

        # ada_rg回归:输入train_data0，输出train_data0
        if 1==True:
            pre_res_name = 'end_预测值'
            model_name = 'model_4'
            pre_res_names.append(pre_res_name)
            if train_data0[yinzi].empty:
                return model_list, '回归，无预测数据'
            if Train:
                model4, X_train_pre ,mean_score = ada_dt_regression_grid(X_train=train_data0,yinzi=yinzi, y_train=train_data0['预测周期真实收益'],
                                                    n_jobs=n_jobs)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                X_train_pre.rename(columns={'预测值': pre_res_name}, inplace=True)
                train_data0 = X_train_pre

                model_list[model_name] = [model4, mean_score]
            else:
                model4 = model_list[model_name][0]
                reg_pre = model4.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(reg_pre, index=train_data0.index)
                print('\n======\n预测结果\n回归的结果，线性corr:', np.corrcoef(train_data0[pre_res_name], train_data0['预测周期真实收益'])[0], '\n')
                print("R2得分:", metrics.r2_score(train_data0['预测周期真实收益'], train_data0[pre_res_name]))

    # 最后一个月只有预测
    if last == True:
        if 'model_1' in model_list.keys():
            # ===拿到预测模型
            model1 = model_list['model_1'][0]
            cat_pre = model1.predict(train_data0[yinzi])
            train_data0.loc[:, '1_预测值'] = pd.Series(cat_pre, index=train_data0.index)
            train_data0 = train_data0[train_data0['1_预测值'] > fl_num].copy()
            if train_data0[yinzi].empty:
                print(f'预测数据，1:分类出错')
            return {}, '1:分类出错'

        if 'model_2' in model_list.keys():
            # ===拿到预测模型
            model2 = model_list['model_2'][0]
            cat_pre = model2.predict(train_data0[yinzi])
            train_data0.loc[:, '2_预测值'] = pd.Series(cat_pre, index = train_data0.index)
            train_data0 = train_data0[train_data0['2_预测值'] >0].copy()
            if train_data0[yinzi].empty:
                print('预测数据，2:分类出错')
                print(train_data0.sample(5))
                return {}, '2:分类出错'

        if 'model_3' in model_list.keys():
            # ===拿到预测模型
            model3 = model_list['model_3'][0]
            cat_pre = model3.predict(train_data0[yinzi])
            train_data0.loc[:, '3_预测值'] = pd.Series(cat_pre, index=train_data0.index)
            train_data0 = train_data0[train_data0['3_预测值'] > 0].copy()
            if train_data0[yinzi].empty:
                print('预测数据，3:分类出错')
                print(train_data0.sample(5))
                return {}, '3:分类出错'


        # ===拿到预测模型
        if train_data0[yinzi].empty:
            print( '回归，无预测数据!')
            return model_list, '回归，无预测数据!'
        else:
            model4 = model_list['model_4'][0]
            reg_pre = model4.predict(train_data0[yinzi])
            train_data0.loc[:, 'end_预测值'] = pd.Series(reg_pre, index=train_data0.index)

        train_data0['canshu'] = train_data0.index
        cols = [ i for i in list(train_data0.columns ) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols],train_data,how='inner',on=['canshu','预测周期真实收益'],left_index=True)
        return model_list, train_data0

    if Train ==False:
        print('预测：')
        # print(train_data0.tail())
        # print(train_data['canshu'])
        train_data0['canshu'] = train_data0.index

        # exit()
        cols = [i for i in list(train_data0.columns) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols], train_data, how='inner', on=['canshu', '预测周期真实收益'], left_index=True)

    return model_list,train_data0


def cal_zuhe_gl_moban0(train_data=None, model_list={}, yinzi=[], Train=True, last=False, n_jobs=2):
    '''
    数据归一化
    第一次分类，adaboost算法
    第二次分类，random——Forrest分类
    第3次分类，svc分类，["rbf",'sigmoid','linear','poly']
    分类都采取，随机grid方式过滤。
    最后进行回归。
    '''

    def data_fliter_fb(data, yinzi0):
        yinzi0 = ['本周期总收益',
                  '最近周期收益', '最大回撤', '最大值', '收益std', '偏度', '峰度', '平均月收益', '平均月最大收益',
                  '平均月最大回撤', '平均月夏普率', '平均月交易次数', '月均交易天数', '月均盈利天数', '月均开单收益std',
                  '月均开单最大收益', '月均亏单平均亏损', '月均胜单平均盈利', '月均胜单平均盈利偏度', '月均胜单平均盈利std',
                  '月均交易胜率', '月均交易胜率偏度', '月均交易胜率std', '月均开单平均收益', '月均开单平均收益偏度',
                  '月均开单平均收益std', '回撤std', '盈撤比', '盈利因子01']
        # 过滤算法
        data = data[data['平均月收益'] > data['平均月收益'].mean()]
        data = data[data['平均月交易次数'] > 2]
        return pd.DataFrame(data)


    print(f'欢迎使用{sys._getframe().f_code.co_name}：fb-ada-rf-svc-poly')

    if Train:
        # 数据过滤
        train_data = data_fliter_fb(train_data)
        print(f'过滤之后，本次训练数据量：{train_data[yinzi].shape}')
    else:
        print(f'预测，本次数据量：{train_data[yinzi].shape}')

    if not train_data.empty:
        # 数据处理：==》处理完的数据
        train_data0 = data_precess(train_data, yinzi=yinzi, pre_style='normal')
        print('\n训练数据标准化处理。')

    pre_res_names = []
    fl_num = 0
    if last == False:

        # ada_dtree:输入train_data0，输出train_data0
        if 1 == True:
            pre_res_name = '1_预测值'
            model_name = 'model_1'
            pre_res_names.append(pre_res_name)
            if Train:
                fl_con1 = train_data0['预测周期真实收益'] > train_data0['预测周期真实收益'].mean()
                fl_con11 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con1, '预测周期真实收益'].mean()
                fl_con0 = train_data0['预测周期真实收益'] < train_data0['预测周期真实收益'].mean()

                train_data0.loc[fl_con1, 'adt_分类结果'] = 1
                train_data0.loc[fl_con11, 'adt_分类结果'] = 2
                train_data0.loc[fl_con0, 'adt_分类结果'] = -1

                if len(train_data0['adt_分类结果'].tolist()) == 0 \
                        or len(train_data0[train_data0['adt_分类结果'] == 2]['adt_分类结果'].tolist()) == 0:
                    print('\n======\nrf：训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, 'rf：分类问题'

                # print(train_data0[['预测周期真实收益','rf_分类结果']])
                train_data0.dropna(axis=0, inplace=True)
                print(f'{model_name}数据量：{train_data0[yinzi].shape}')

                model1, X_train_pre, mean_score = adaboosting_dtree_grid(X_train=train_data0, yinzi=yinzi,
                                                                         y_train=train_data0['adt_分类结果'], n_iter=10,
                                                                         n_jobs=n_jobs)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list[model_name][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0[pre_res_name] > fl_num].copy()
                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, f'{pre_res_name}：预测分类出错'

        # 随机森林分类:输入train_data0，输出train_data0
        if 1 == True:
            pre_res_name = '2_预测值'
            model_name = 'model_2'

            pre_res_names.append(pre_res_name)
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
                        or len(train_data0[train_data0['rf_分类结果'] == 2]['rf_分类结果'].tolist()) == 0:
                    print('\n======\nrf：训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, 'rf：分类问题'

                train_data0.dropna(axis=0, inplace=True)
                print(f'{model_name}数据量：{train_data0[yinzi].shape}')
                model1, X_train_pre, mean_score = randomforest_classify_grid(X_train=train_data0, yinzi=yinzi,
                                                                             y_train=train_data0['rf_分类结果'],
                                                                             n_iter=10)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list[model_name][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, 'pre_res_name'] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0['pre_res_name'] > fl_num].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, 'rf：预测分类出错'

        # svc分类:输入train_data0，输出train_data0
        if 0 == True:
            pre_res_name = '3_预测值'
            model_name = 'model_3'

            pre_res_names.append(pre_res_name)

            if Train:
                fl_con = train_data0['预测周期真实收益'] > 0  # train_data0['预测周期真实收益'].mean()
                fl_con_ = train_data0['预测周期真实收益'] <= 0  # train_data0['预测周期真实收益'].mean()

                train_data0.loc[fl_con, 'svc_分类结果'] = 1
                train_data0.loc[fl_con_, 'svc_分类结果'] = 0
                train_data0.fillna(0, inplace=True)

                if len(train_data0['svc_分类结果'].tolist()) == 0 \
                        or len(train_data0[train_data0['svc_分类结果'] == 0]['svc_分类结果'].tolist()) == 0 \
                        or len(train_data0[train_data0['svc_分类结果'] == 1]['svc_分类结果'].tolist()) == 0:
                    print('\n======\nsvc： 训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, f'{pre_res_name}：分类问题'
                # zai次训练
                print(f'{model_name}数据量：{train_data0[yinzi].shape}')
                model3, X_train_pre, mean_score = svc_classify_grid(X_train=train_data0, yinzi=yinzi,
                                                                    y_train=train_data0['svc_分类结果'],
                                                                    n_iter=10, kernel=["rbf", 'sigmoid', 'linear'],
                                                                    n_jobs=n_jobs)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']

                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] == 1].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model3, mean_score]
            else:
                model2 = model_list[model_name][0]
                cat_pre = model2.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0[pre_res_name] == 1].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，svc：分类出错')
                    return {}, f'{pre_res_names}：分类问题'

        # 回归:输入train_data0，输出train_data0
        if 1 == True:
            pre_res_name = 'end_预测值'
            model_name = 'model_4'
            pre_res_names.append(pre_res_name)
            if train_data0[yinzi].empty:
                return model_list, '回归，无预测数据'
            if Train:
                print(f'{model_name}数据量：{train_data0[yinzi].shape}')

                model4, X_train_pre, mean_score = polynomial_regression_grid(X_train=train_data0, yinzi=yinzi,
                                                                             y_train=train_data0['预测周期真实收益'], n_iter=3,
                                                                             n_jobs=n_jobs)

                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                X_train_pre.rename(columns={'预测值': pre_res_name}, inplace=True)
                train_data0 = X_train_pre

                model_list[model_name] = [model4, mean_score]
            else:
                model4 = model_list[model_name][0]
                reg_pre = model4.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(reg_pre, index=train_data0.index)
                print('\n======\n预测结果\n回归的结果，线性corr:',
                      np.corrcoef(train_data0[pre_res_name], train_data0['预测周期真实收益'])[0], '\n')
                print("R2得分:", metrics.r2_score(train_data0['预测周期真实收益'], train_data0[pre_res_name]))

    # 最后一个月只有预测
    if last == True:
        if 'model_1' in model_list.keys():
            # ===拿到预测模型
            model1 = model_list['model_1'][0]
            cat_pre = model1.predict(train_data0[yinzi])
            train_data0.loc[:, '1_预测值'] = pd.Series(cat_pre, index=train_data0.index)
            train_data0 = train_data0[train_data0['1_预测值'] > fl_num].copy()
            if train_data0[yinzi].empty:
                print(f'预测数据，1:分类出错')
            return {}, '1:分类出错'

        if 'model_2' in model_list.keys():
            # ===拿到预测模型
            model2 = model_list['model_2'][0]
            cat_pre = model2.predict(train_data0[yinzi])
            train_data0.loc[:, '2_预测值'] = pd.Series(cat_pre, index=train_data0.index)
            train_data0 = train_data0[train_data0['2_预测值'] > 0].copy()
            if train_data0[yinzi].empty:
                print('预测数据，2:分类出错')
                print(train_data0.sample(5))
                return {}, '2:分类出错'

        if 'model_3' in model_list.keys():
            # ===拿到预测模型
            model3 = model_list['model_3'][0]
            cat_pre = model3.predict(train_data0[yinzi])
            train_data0.loc[:, '3_预测值'] = pd.Series(cat_pre, index=train_data0.index)
            train_data0 = train_data0[train_data0['3_预测值'] > 0].copy()
            if train_data0[yinzi].empty:
                print('预测数据，3:分类出错')
                print(train_data0.sample(5))
                return {}, '3:分类出错'

        # ===拿到预测模型
        if train_data0[yinzi].empty:
            print('回归，无预测数据!')
            return model_list, '回归，无预测数据!'
        else:
            model4 = model_list['model_4'][0]
            reg_pre = model4.predict(train_data0[yinzi])
            train_data0.loc[:, 'end_预测值'] = pd.Series(reg_pre, index=train_data0.index)

        train_data0['canshu'] = train_data0.index
        cols = [i for i in list(train_data0.columns) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols], train_data, how='inner', on=['canshu', '预测周期真实收益'], left_index=True)
        return model_list, train_data0

    if Train == False:
        print('预测：')
        # print(train_data0.tail())
        # print(train_data['canshu'])
        train_data0['canshu'] = train_data0.index

        # exit()
        cols = [i for i in list(train_data0.columns) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols], train_data, how='inner', on=['canshu', '预测周期真实收益'], left_index=True)

    return model_list, train_data0

def cal_zuhe_gl_01(train_data=None, model_list={}, yinzi=[], Train=True, last=False, n_jobs=2):
    '''
    数据归一化
    第一次分类，adaboost算法
    第二次分类，random——Forrest分类
    第3次分类，svc分类，["rbf",'sigmoid','linear','poly']
    分类都采取，随机grid方式过滤。
    最后进行回归。
    '''

    def data_fliter_fb(data, yinzi0):
        yinzi0 = ['本周期总收益',
                  '最近周期收益', '最大回撤', '最大值', '收益std', '偏度', '峰度', '平均月收益', '平均月最大收益',
                  '平均月最大回撤', '平均月夏普率', '平均月交易次数', '月均交易天数', '月均盈利天数', '月均开单收益std',
                  '月均开单最大收益', '月均亏单平均亏损', '月均胜单平均盈利', '月均胜单平均盈利偏度', '月均胜单平均盈利std',
                  '月均交易胜率', '月均交易胜率偏度', '月均交易胜率std', '月均开单平均收益', '月均开单平均收益偏度',
                  '月均开单平均收益std', '回撤std', '盈撤比', '盈利因子01']
        # 过滤算法
        data = data[data['平均月收益'] > data['平均月收益'].mean()]
        data = data[data['平均月交易次数'] > 2]
        return pd.DataFrame(data)

    print(f'欢迎使用{sys._getframe().f_code.co_name}：fb-ada-rf-svc-poly')

    if Train:
        # 数据过滤
        train_data = data_fliter_fb(train_data,yinzi)
        print(f'过滤之后，本次训练数据量：{train_data[yinzi].shape}')
    else:
        print(f'预测，本次数据量：{train_data[yinzi].shape}')

    if not train_data.empty:
        # 数据处理：==》处理完的数据
        train_data0 = data_precess(train_data, yinzi=yinzi, pre_style='normal')
        print('\n训练数据标准化处理。')

    pre_res_names = []
    fl_num = 0
    if last == False:

        # ada_dtree:输入train_data0，输出train_data0
        if 1 == True:
            pre_res_name = '1_预测值'
            model_name = 'model_1'
            pre_res_names.append(pre_res_name)
            if Train:
                fl_con1 = train_data0['预测周期真实收益'] > train_data0['预测周期真实收益'].mean()
                fl_con11 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con1, '预测周期真实收益'].mean()
                fl_con0 = train_data0['预测周期真实收益'] < train_data0['预测周期真实收益'].mean()

                train_data0.loc[fl_con1, 'adt_分类结果'] = 1
                train_data0.loc[fl_con11, 'adt_分类结果'] = 2
                train_data0.loc[fl_con0, 'adt_分类结果'] = -1

                if len(train_data0['adt_分类结果'].tolist()) == 0 \
                        or len(train_data0[train_data0['adt_分类结果'] == 2]['adt_分类结果'].tolist()) == 0:
                    print('\n======\nrf：训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, 'rf：分类问题'

                # print(train_data0[['预测周期真实收益','rf_分类结果']])
                train_data0.dropna(axis=0, inplace=True)
                print(f'{model_name}数据量：{train_data0[yinzi].shape}')

                model1, X_train_pre, mean_score = adaboosting_dtree_grid(X_train=train_data0, yinzi=yinzi,
                                                                         y_train=train_data0['adt_分类结果'], n_iter=10,
                                                                         n_jobs=n_jobs)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list[model_name][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0[pre_res_name] > fl_num].copy()
                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, f'{pre_res_name}：预测分类出错'

        #svc_bagging
        if 1 == True:
                pre_res_name = '1_预测值'
                model_name = 'model_1'
                pre_res_names.append(pre_res_name)
                if Train:
                    fl_con1 = train_data0['预测周期真实收益'] > train_data0['预测周期真实收益'].mean()
                    # fl_con11 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con1, '预测周期真实收益'].mean()
                    fl_con0 = train_data0['预测周期真实收益'] < train_data0['预测周期真实收益'].mean()

                    train_data0.loc[fl_con1, '1分类结果'] = 1
                    # train_data0.loc[fl_con11, '1分类结果'] = 2
                    train_data0.loc[fl_con0, '1分类结果'] = -1

                    if len(train_data0['1分类结果'].tolist()) == 0 \
                            or len(train_data0[train_data0['1分类结果'] == 1]['1分类结果'].tolist()) == 0:
                        print('\n======\n1：训练结果分类，出问题。')
                        print(train_data0['预测周期真实收益'].sample(5))
                        return model_list, '1：分类问题'

                    train_data0.dropna(axis=0, inplace=True)
                    print(f'{model_name}数据量：{train_data0[yinzi].shape}')

                    model1, X_train_pre, mean_score = bag_svc(X_train=train_data0, yinzi=yinzi,
                                                                             y_train=train_data0['1分类结果'], n_iter=20,
                                                                             n_jobs=n_jobs)
                    X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                    # 目标分类
                    train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                    train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                    model_list[model_name] = [model1, mean_score]
                else:
                    # 拿到预测模型
                    model1 = model_list[model_name][0]
                    cat_pre = model1.predict(train_data0[yinzi])
                    train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                    train_data0 = train_data0[train_data0[pre_res_name] > fl_num].copy()
                    if train_data0[yinzi].empty:
                        print('预测数据，分类出错')
                        return {}, f'{pre_res_name}：预测分类出错'

        # 随机森林分类:输入train_data0，输出train_data0
        if 0 == True:
            pre_res_name = '2_预测值'
            model_name = 'model_2'

            pre_res_names.append(pre_res_name)
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
                        or len(train_data0[train_data0['rf_分类结果'] >= 1]['rf_分类结果'].tolist()) == 0:
                    print('\n======\nrf：训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, 'rf：分类问题'

                train_data0.dropna(axis=0, inplace=True)
                print(f'{model_name}随机森林分类,数据量大小：{train_data0[yinzi].shape}')
                model1, X_train_pre, mean_score = randomforest_classify_grid(X_train=train_data0, yinzi=yinzi,
                                                                             y_train=train_data0['rf_分类结果'],
                                                                             n_iter=10)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list[model_name][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, 'pre_res_name'] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0[pre_res_name] > fl_num].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, 'rf：预测分类出错'

        # svc分类:输入train_data0，输出train_data0
        if 0 == True:
            pre_res_name = '3_预测值'
            model_name = 'model_3'

            pre_res_names.append(pre_res_name)

            if Train:
                fl_con = train_data0['预测周期真实收益'] > train_data0['预测周期真实收益'].mean()
                fl_con_ = train_data0['预测周期真实收益'] <  train_data0['预测周期真实收益'].mean()

                train_data0.loc[fl_con, 'svc_分类结果'] = 1
                train_data0.loc[fl_con_, 'svc_分类结果'] = -1
                train_data0.fillna(0, inplace=True)

                if len(train_data0['svc_分类结果'].tolist()) == 0 \
                        or len(train_data0[train_data0['svc_分类结果'] != 1]['svc_分类结果'].tolist()) == 0 \
                        or len(train_data0[train_data0['svc_分类结果'] == 1]['svc_分类结果'].tolist()) == 0:
                    print('\n======\nsvc： 训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, f'{pre_res_name}：分类问题'
                print(f'{model_name}：svc：数据量大小：{train_data0[yinzi].shape}')
                model3, X_train_pre, mean_score = svc_classify_grid(X_train=train_data0, yinzi=yinzi,
                                                                    y_train=train_data0['svc_分类结果'],
                                                                    n_iter=10, kernel=["rbf"],
                                                                    n_jobs=n_jobs)

                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']

                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] == 1].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model3, mean_score]
            else:
                model2 = model_list[model_name][0]
                cat_pre = model2.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0[pre_res_name] == 1].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，svc：分类出错')
                    return {}, f'{pre_res_names}：分类问题'

        # 回归:输入train_data0，输出train_data0
        if 1 == True:
            pre_res_name = 'end_预测值'
            model_name = 'model_4'
            pre_res_names.append(pre_res_name)
            if train_data0[yinzi].empty:
                return model_list, '回归，无预测数据'
            if Train:
                print(f'{model_name}数据量：{train_data0[yinzi].shape}')

                model4, X_train_pre, mean_score = polynomial_regression0(X_train=train_data0, yinzi=yinzi,
                                                                             y_train=train_data0['预测周期真实收益'], degree=2)

                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                X_train_pre.rename(columns={'预测值': pre_res_name}, inplace=True)
                train_data0 = X_train_pre

                model_list[model_name] = [model4, mean_score]
            else:
                model4 = model_list[model_name][0]
                reg_pre = model4.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(reg_pre, index=train_data0.index)
                print('\n======\n预测结果\n回归的结果，线性corr:',
                      np.corrcoef(train_data0[pre_res_name], train_data0['预测周期真实收益'])[0], '\n')
                print("R2得分:", metrics.r2_score(train_data0['预测周期真实收益'], train_data0[pre_res_name]))

    # 最后一个月只有预测
    if last == True:
        if 'model_1' in model_list.keys():
            # ===拿到预测模型
            model1 = model_list['model_1'][0]
            cat_pre = model1.predict(train_data0[yinzi])
            train_data0.loc[:, '1_预测值'] = pd.Series(cat_pre, index=train_data0.index)
            train_data0 = train_data0[train_data0['1_预测值'] > fl_num].copy()
            if train_data0[yinzi].empty:
                print(f'预测数据，1:分类出错')
            return {}, '1:分类出错'

        if 'model_2' in model_list.keys():
            # ===拿到预测模型
            model2 = model_list['model_2'][0]
            cat_pre = model2.predict(train_data0[yinzi])
            train_data0.loc[:, '2_预测值'] = pd.Series(cat_pre, index=train_data0.index)
            train_data0 = train_data0[train_data0['2_预测值'] > 0].copy()
            if train_data0[yinzi].empty:
                print('预测数据，2:分类出错')
                print(train_data0.sample(5))
                return {}, '2:分类出错'

        if 'model_3' in model_list.keys():
            # ===拿到预测模型
            model3 = model_list['model_3'][0]
            cat_pre = model3.predict(train_data0[yinzi])
            train_data0.loc[:, '3_预测值'] = pd.Series(cat_pre, index=train_data0.index)
            train_data0 = train_data0[train_data0['3_预测值'] > 0].copy()
            if train_data0[yinzi].empty:
                print('预测数据，3:分类出错')
                print(train_data0.sample(5))
                return {}, '3:分类出错'

        # ===拿到预测模型
        if train_data0[yinzi].empty:
            print('回归，无预测数据!')
            return model_list, '回归，无预测数据!'
        else:
            model4 = model_list['model_4'][0]
            reg_pre = model4.predict(train_data0[yinzi])
            train_data0.loc[:, 'end_预测值'] = pd.Series(reg_pre, index=train_data0.index)

        train_data0['canshu'] = train_data0.index
        cols = [i for i in list(train_data0.columns) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols], train_data, how='inner', on=['canshu', '预测周期真实收益'], left_index=True)
        return model_list, train_data0

    if Train == False:
        print('预测：')
        # print(train_data0.tail())
        # print(train_data['canshu'])
        train_data0['canshu'] = train_data0.index

        # exit()
        cols = [i for i in list(train_data0.columns) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols], train_data, how='inner', on=['canshu', '预测周期真实收益'], left_index=True)

    return model_list, train_data0

def cal_zuhe_gl_02(train_data=None, model_list={}, yinzi=[], Train=True, last=False, n_jobs=2):
    '''
    数据归一化
    第一次分类，adaboost算法
    第二次分类，random——Forrest分类
    第3次分类，svc分类，["rbf",'sigmoid','linear','poly']
    分类都采取，随机grid方式过滤。
    最后进行回归。
    '''

    def data_fliter_fb(data, yinzi=[]):

        # 过滤算法
        data = data[data['total_days_3'] > 3]
        data = data[data['trade_nums_3'] > 10]
        data = data[data['trade_nums_2'] > 10]
        data = data[data['trade_nums_3'] < 200]
        data = data[data['trade_nums_2'] < 200]
        data.fillna(0, inplace=True)

        return pd.DataFrame(data)

    print(f'欢迎使用{sys._getframe().f_code.co_name}：fb-svc_bag-svc-poly')
    print(f'本次传入数据量：{train_data[yinzi].shape}')


    # print(yinzi)
    # print(train_data.tail())
    # exit()
    if Train:
        # 数据过滤
        train_data = data_fliter_fb(train_data, yinzi)
        train_data0, train_data_, yinzi = k_pac(train_data, yinzi, N=40, kernel='linear', n_jobs=1)
        print(f'过滤之后，本次,训练：{train_data0[yinzi].shape}')
    else:
        train_data = data_fliter_fb(train_data, yinzi)

        train_data0, train_data_, yinzi = k_pac(train_data, yinzi, N=40, kernel='linear', n_jobs=1)
        # train_data0 = train_data
        print(f'预测，本次数据量：{train_data0[yinzi].shape}')

    # 数据处理：==》False
    if not train_data0.empty and False:
        train_data0 = data_precess(train_data0, yinzi=yinzi, pre_style='normal')
        print('\n训练数据标准化处理。')

    pre_res_names = []
    fl_num = 0
    if last == False:
        # ada_dtree:输入train_data0，输出train_data0
        if 0 == True:
            pre_res_name = '1_预测值'
            model_name = 'model_1'
            pre_res_names.append(pre_res_name)
            if Train:
                fl_con1 = train_data0['预测周期真实收益'] > train_data0['预测周期真实收益'].mean()
                fl_con11 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con1, '预测周期真实收益'].mean()
                fl_con0 = train_data0['预测周期真实收益'] < train_data0['预测周期真实收益'].mean()

                train_data0.loc[fl_con1, 'adt_分类结果'] = 1
                train_data0.loc[fl_con11, 'adt_分类结果'] = 2
                train_data0.loc[fl_con0, 'adt_分类结果'] = -1

                if len(train_data0['adt_分类结果'].tolist()) == 0 \
                        or len(train_data0[train_data0['adt_分类结果'] == 2]['adt_分类结果'].tolist()) == 0:
                    print('\n======\nrf：训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, 'rf：分类问题'

                # print(train_data0[['预测周期真实收益','rf_分类结果']])
                train_data0.dropna(axis=0, inplace=True)
                print(f'{model_name}数据量：{train_data0[yinzi].shape}')

                model1, X_train_pre, mean_score = adaboosting_dtree_grid(X_train=train_data0, yinzi=yinzi,
                                                                         y_train=train_data0['adt_分类结果'], n_iter=10,
                                                                         n_jobs=n_jobs)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list[model_name][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0[pre_res_name] > fl_num].copy()
                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, f'{pre_res_name}：预测分类出错'

        # 随机森林分类:输入train_data0，输出train_data0
        if 1 == True:
            pre_res_name = '1_预测值'
            model_name = 'model_1'

            pre_res_names.append(pre_res_name)
            if Train:
                fl_con1 = train_data0['预测周期真实收益'] > 0 #train_data0['预测周期真实收益'].mean()
                fl_con11 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con1, '预测周期真实收益'].mean()
                fl_con0 = train_data0['预测周期真实收益'] < 0 #train_data0['预测周期真实收益'].mean()

                # print(train_data0[['预测周期真实收益']])
                # exit()
                train_data0.loc[fl_con1, 'rf_分类结果'] = 1
                train_data0.loc[fl_con11, 'rf_分类结果'] = 2
                train_data0.loc[fl_con0, 'rf_分类结果'] = -1

                if len(train_data0['rf_分类结果'].tolist()) == 0 \
                        or len(train_data0[train_data0['rf_分类结果'] >= 1]['rf_分类结果'].tolist()) == 0:
                    print('\n======\nrf：训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, 'rf：分类问题'

                train_data0.dropna(axis=0, inplace=True)
                print(f'{model_name}随机森林分类,数据量大小：{train_data0[yinzi].shape}')
                model1, X_train_pre, mean_score = randomforest_classify_grid(X_train=train_data0, yinzi=yinzi,
                                                                             y_train=train_data0['rf_分类结果'],
                                                                             n_iter=10)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list[model_name][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0[pre_res_name] > fl_num].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, 'rf：预测分类出错'

        #svc_bagging
        if 0 == True:
                pre_res_name = '2_预测值'
                model_name = 'model_2'
                pre_res_names.append(pre_res_name)
                if Train:
                    fl_con1 = train_data0['预测周期真实收益'] > 0 #train_data0['预测周期真实收益'].mean()
                    # fl_con11 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con1, '预测周期真实收益'].mean()
                    fl_con0 = train_data0['预测周期真实收益'] < 0 #train_data0['预测周期真实收益'].mean()

                    train_data0.loc[fl_con1, '1分类结果'] = 1
                    # train_data0.loc[fl_con11, '1分类结果'] = 2
                    train_data0.loc[fl_con0, '1分类结果'] = -1

                    if len(train_data0['1分类结果'].tolist()) == 0 \
                            or len(train_data0[train_data0['1分类结果'] == 1]['1分类结果'].tolist()) == 0:
                        print('\n======\n1：训练结果分类，出问题。')
                        print(train_data0['预测周期真实收益'].sample(5))
                        return model_list, '1：分类问题'

                    train_data0.dropna(axis=0, inplace=True)
                    print(f'{model_name}数据量：{train_data0[yinzi].shape}')

                    model1, X_train_pre, mean_score = bag_svc(X_train=train_data0, yinzi=yinzi,
                                                                             y_train=train_data0['1分类结果'], n_iter=60,
                                                                             n_jobs=n_jobs)

                    X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                    # 目标分类
                    train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                    train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                    model_list[model_name] = [model1, mean_score]
                else:
                    # 拿到预测模型
                    model1 = model_list[model_name][0]
                    cat_pre = model1.predict(train_data0[yinzi])
                    train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                    train_data0 = train_data0[train_data0[pre_res_name] > fl_num].copy()
                    if train_data0[yinzi].empty:
                        print('预测数据，分类出错')
                        return {}, f'{pre_res_name}：预测分类出错'

        # svc分类:输入train_data0，输出train_data0
        if 1 == True:
            pre_res_name = '2_预测值'
            model_name = 'model_2'

            pre_res_names.append(pre_res_name)

            if Train:
                fl_con = train_data0['预测周期真实收益'] > train_data0['预测周期真实收益'].mean()
                fl_con1 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con, '预测周期真实收益'].mean()
                fl_con_ = train_data0['预测周期真实收益'] < train_data0['预测周期真实收益'].mean()
                fl_con_1 = train_data0['预测周期真实收益'] < train_data0.loc[fl_con_, '预测周期真实收益'].mean()

                train_data0.loc[fl_con, 'svc_分类结果'] = 1
                train_data0.loc[fl_con1, 'svc_分类结果'] = 2
                train_data0.loc[fl_con_, 'svc_分类结果'] = -1
                train_data0.loc[fl_con_, 'svc_分类结果'] = -2

                train_data0.fillna(0, inplace=True)

                if len(train_data0['svc_分类结果'].tolist()) == 0 :
                    print('\n======\nsvc： 训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, f'{pre_res_name}：分类问题'
                print(f'{model_name}：svc：数据量大小：{train_data0[yinzi].shape}')
                model3, X_train_pre, mean_score = svc_classify_grid(X_train=train_data0, yinzi=yinzi,
                                                                    y_train=train_data0['svc_分类结果'],
                                                                    n_iter=10, kernel=["rbf",'poly'],
                                                                    n_jobs=n_jobs)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']

                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] == 1].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model3, mean_score]
            else:
                model2 = model_list[model_name][0]
                cat_pre = model2.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0[pre_res_name] != -2].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，svc：分类出错')
                    return {}, f'{pre_res_names}：分类问题'

        # 回归:输入train_data0，输出train_data0
        if 1== True:
            pre_res_name = 'end_预测值'
            model_name = 'model_end'
            pre_res_names.append(pre_res_name)
            if train_data0[yinzi].empty:
                return model_list, '回归，无预测数据'
            if Train:
                print(f'{model_name}数据量：{train_data0[yinzi].shape}')

                model4, X_train_pre, mean_score = poly_ridge_regression(X_train=train_data0, yinzi=yinzi,
                                                                             y_train=train_data0['预测周期真实收益'], degree=4)

                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                X_train_pre.rename(columns={'预测值': pre_res_name}, inplace=True)
                train_data0 = X_train_pre

                model_list[model_name] = [model4, mean_score]
            else:
                model4 = model_list[model_name][0]
                reg_pre = model4.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(reg_pre, index=train_data0.index)
                print('\n======\n预测结果\n回归的结果，线性corr:',
                      np.corrcoef(train_data0[pre_res_name], train_data0['预测周期真实收益'])[0], '\n')
                print("R2得分:", metrics.r2_score(train_data0['预测周期真实收益'], train_data0[pre_res_name]))

            # 随机森林分类:输入train_data0，输出train_data0
        #分类预测输出：
        if 0 == True:
            pre_res_name = 'end_预测值'
            model_name = 'model_end'

            pre_res_names.append(pre_res_name)
            if Train:
                fl_con1 = train_data0['预测周期真实收益'] > train_data0['预测周期真实收益'].mean()
                fl_con11 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con1, '预测周期真实收益'].mean()

                fl_con0 = train_data0['预测周期真实收益'] < train_data0['预测周期真实收益'].mean()
                fl_con01 = train_data0['预测周期真实收益'] < train_data0.loc[fl_con0, '预测周期真实收益'].mean()


                # print(train_data0[['预测周期真实收益']])
                # exit()
                train_data0.loc[fl_con1, 'rf_分类结果'] = 1
                train_data0.loc[fl_con11, 'rf_分类结果'] = 2
                train_data0.loc[fl_con0, 'rf_分类结果'] = -1
                train_data0.loc[fl_con01, 'rf_分类结果'] = -2



                train_data0.dropna(axis=0, inplace=True)
                print(f'{model_name}随机森林分类,数据量大小：{train_data0[yinzi].shape}')
                model1, X_train_pre, mean_score = randomforest_classify_grid(X_train=train_data0, yinzi=yinzi,
                                                                             y_train=train_data0['rf_分类结果'],
                                                                             n_iter=10)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                # train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list[model_name][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                # train_data0 = train_data0[train_data0[pre_res_name] > fl_num].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, 'rf：预测分类出错'


    # 最后一个月只有预测
    if last == True:
        if 'model_1' in model_list.keys():
            # ===拿到预测模型
            model1 = model_list['model_1'][0]
            pre = model1.predict(train_data0[yinzi])
            train_data0.loc[:, '1_预测值'] = pd.Series(pre, index=train_data0.index)
            train_data0 = train_data0[train_data0['1_预测值'] > 0].copy()
            # print(train_data0)

            if train_data0.empty:
                print(f'预测数据，1:分类出错')
                return {}, '1:分类出错'

        if 'model_2' in model_list.keys():
            # ===拿到预测模型
            model2 = model_list['model_2'][0]
            cat_pre = model2.predict(train_data0[yinzi])
            print(cat_pre)
            train_data0.loc[:, '2_预测值'] = pd.Series(cat_pre, index=train_data0.index)

            train_data0 = train_data0[train_data0['2_预测值'] >= -2].copy()
            if train_data0[yinzi].empty:
                print('预测数据，2:分类出错,预测数据为空')
                print(pd.Series(cat_pre, index=train_data0.index).tail(50))
                return {}, '2:分类出错'

        if 'model_3' in model_list.keys():
            # ===拿到预测模型
            model3 = model_list['model_3'][0]
            cat_pre = model3.predict(train_data0[yinzi])
            train_data0.loc[:, '3_预测值'] = pd.Series(cat_pre, index=train_data0.index)
            train_data0 = train_data0[train_data0['3_预测值'] > 0].copy()
            if train_data0[yinzi].empty:
                print('预测数据，3:分类出错')
                print(train_data0.sample(5))
                return {}, '3:分类出错'

        # ===拿到预测模型
        if train_data0[yinzi].empty:
            print('回归，无预测数据!')
            return model_list, '回归，无预测数据!'
        else:
            model4 = model_list['model_end'][0]
            reg_pre = model4.predict(train_data0[yinzi])
            train_data0.loc[:, 'end_预测值'] = pd.Series(reg_pre, index=train_data0.index)

        train_data0['canshu'] = train_data0.index
        cols = [i for i in list(train_data0.columns) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols], train_data, how='inner', on=['canshu', '预测周期真实收益'], left_index=True)
        return model_list, train_data0

    if Train == False:
        print('预测：')
        # print(train_data0.tail())
        # print(train_data['canshu'])
        train_data0['canshu'] = train_data0.index

        # exit()
        cols = [i for i in list(train_data0.columns) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols], train_data, how='inner', on=['canshu', '预测周期真实收益'], left_index=True)

    return model_list, train_data0

def cal_zuhe_gl_03(train_data=None, model_list={}, yinzi=[], Train=True, last=False, n_jobs=2):
    def data_fliter_fb(data):
        yinzi0 = ['本周期总收益',
                  '最近周期收益', '最大回撤', '最大值', '收益std', '偏度', '峰度', '平均月收益', '平均月最大收益',
                  '平均月最大回撤', '平均月夏普率', '平均月交易次数', '月均交易天数', '月均盈利天数', '月均开单收益std',
                  '月均开单最大收益', '月均亏单平均亏损', '月均胜单平均盈利', '月均胜单平均盈利偏度', '月均胜单平均盈利std',
                  '月均交易胜率', '月均交易胜率偏度', '月均交易胜率std', '月均开单平均收益', '月均开单平均收益偏度',
                  '月均开单平均收益std', '回撤std', '盈撤比', '盈利因子01']
        # 过滤算法
        data = data[data['平均月夏普率'] > data['平均月夏普率'].mean()]
        # data = data[data['最近周期收益'] > data['最近周期收益'].mean()]
        # data = data[data['平均月交易次数'] > 1]
        data.fillna(0, inplace=True)
        data = data[data['平均月收益'] != 0].copy()

        return pd.DataFrame(data)

    print(f'欢迎使用{sys._getframe().f_code.co_name}')
    print(f'本次传入数据量：{train_data[yinzi].shape}')
    # 因子重构
    train_data = data_fliter_fb(train_data) #先处理，
    train_da0, new_yinzi = build_yinzi(train_data=train_data, yinzi=yinzi, res_index='预测周期真实收益')
    train_da0, yinzi0 = corr_jw(train_data=train_da0, n=20, yinzi=new_yinzi, res_index='预测周期真实收益')

    if Train:
        # 数据过滤
        print(f'训练：====>\n')
        train_data0, train_data_, yinzi = k_pac(train_data, yinzi, N=10, kernel='linear', n_jobs=1)
        print(f'过滤之后，本次训练数据量：{train_data0[yinzi].shape}')
    else:
        print(f'预测：====>\n')


        train_data0, train_data_, yinzi = k_pac(train_data, yinzi, N=10, kernel='linear', n_jobs=1)
        print(f'预测，本次数据量：{train_data0[yinzi].shape}')

    # 数据处理：==》False
    if not train_data0.empty and False:
        train_data0 = data_precess(train_data0, yinzi=yinzi, pre_style='normal')
        print('\n训练数据标准化处理。')

    pre_res_names = []
    fl_num = 0
    if last == False:
        # ada_dtree:输入train_data0，输出train_data0
        if 0 == True:
            pre_res_name = '1_预测值'
            model_name = 'model_1'
            pre_res_names.append(pre_res_name)
            if Train:
                fl_con1 = train_data0['预测周期真实收益'] > train_data0['预测周期真实收益'].mean()
                fl_con11 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con1, '预测周期真实收益'].mean()
                fl_con0 = train_data0['预测周期真实收益'] < train_data0['预测周期真实收益'].mean()

                train_data0.loc[fl_con1, 'adt_分类结果'] = 1
                train_data0.loc[fl_con11, 'adt_分类结果'] = 2
                train_data0.loc[fl_con0, 'adt_分类结果'] = -1

                if len(train_data0['adt_分类结果'].tolist()) == 0 \
                        or len(train_data0[train_data0['adt_分类结果'] == 2]['adt_分类结果'].tolist()) == 0:
                    print('\n======\nrf：训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, 'rf：分类问题'

                # print(train_data0[['预测周期真实收益','rf_分类结果']])
                train_data0.dropna(axis=0, inplace=True)
                print(f'{model_name}数据量：{train_data0[yinzi].shape}')

                model1, X_train_pre, mean_score = adaboosting_dtree_grid(X_train=train_data0, yinzi=yinzi,
                                                                         y_train=train_data0['adt_分类结果'], n_iter=10,
                                                                         n_jobs=n_jobs)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list[model_name][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0[pre_res_name] > fl_num].copy()
                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, f'{pre_res_name}：预测分类出错'

        # 随机森林分类:输入train_data0，输出train_data0
        if 1 == True:
            pre_res_name = '1_预测值'
            model_name = 'model_1'

            pre_res_names.append(pre_res_name)
            if Train:
                fl_con1 = train_data0['预测周期真实收益'] > 0 #train_data0['预测周期真实收益'].mean()
                fl_con11 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con1, '预测周期真实收益'].mean()
                fl_con0 = train_data0['预测周期真实收益'] < 0 #train_data0['预测周期真实收益'].mean()

                # print(train_data0[['预测周期真实收益']])
                # exit()
                train_data0.loc[fl_con1, 'rf_分类结果'] = 1
                train_data0.loc[fl_con11, 'rf_分类结果'] = 2
                train_data0.loc[fl_con0, 'rf_分类结果'] = -1

                if len(train_data0['rf_分类结果'].tolist()) == 0 \
                        or len(train_data0[train_data0['rf_分类结果'] >= 1]['rf_分类结果'].tolist()) == 0:
                    print('\n======\nrf：训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, 'rf：分类问题'

                train_data0.dropna(axis=0, inplace=True)
                print(f'{model_name}随机森林分类,数据量大小：{train_data0[yinzi].shape}')
                model1, X_train_pre, mean_score = randomforest_classify_grid(X_train=train_data0, yinzi=yinzi,
                                                                             y_train=train_data0['rf_分类结果'],
                                                                             n_iter=10)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list[model_name][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0[pre_res_name] > fl_num].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, 'rf：预测分类出错'

        #svc_bagging
        if 0 == True:
                pre_res_name = '2_预测值'
                model_name = 'model_2'
                pre_res_names.append(pre_res_name)
                if Train:
                    fl_con1 = train_data0['预测周期真实收益'] > 0 #train_data0['预测周期真实收益'].mean()
                    # fl_con11 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con1, '预测周期真实收益'].mean()
                    fl_con0 = train_data0['预测周期真实收益'] < 0 #train_data0['预测周期真实收益'].mean()

                    train_data0.loc[fl_con1, '1分类结果'] = 1
                    # train_data0.loc[fl_con11, '1分类结果'] = 2
                    train_data0.loc[fl_con0, '1分类结果'] = -1

                    if len(train_data0['1分类结果'].tolist()) == 0 \
                            or len(train_data0[train_data0['1分类结果'] == 1]['1分类结果'].tolist()) == 0:
                        print('\n======\n1：训练结果分类，出问题。')
                        print(train_data0['预测周期真实收益'].sample(5))
                        return model_list, '1：分类问题'

                    train_data0.dropna(axis=0, inplace=True)
                    print(f'{model_name}数据量：{train_data0[yinzi].shape}')

                    model1, X_train_pre, mean_score = bag_svc(X_train=train_data0, yinzi=yinzi,
                                                                             y_train=train_data0['1分类结果'], n_iter=60,
                                                                             n_jobs=n_jobs)

                    X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                    # 目标分类
                    train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                    train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                    model_list[model_name] = [model1, mean_score]
                else:
                    # 拿到预测模型
                    model1 = model_list[model_name][0]
                    cat_pre = model1.predict(train_data0[yinzi])
                    train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                    train_data0 = train_data0[train_data0[pre_res_name] > fl_num].copy()
                    if train_data0[yinzi].empty:
                        print('预测数据，分类出错')
                        return {}, f'{pre_res_name}：预测分类出错'

        # svc分类:输入train_data0，输出train_data0
        if 1 == True:
            pre_res_name = '2_预测值'
            model_name = 'model_2'

            pre_res_names.append(pre_res_name)

            if Train:
                fl_con = train_data0['预测周期真实收益'] > train_data0['预测周期真实收益'].mean()
                fl_con1 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con, '预测周期真实收益'].mean()
                fl_con_ = train_data0['预测周期真实收益'] < train_data0['预测周期真实收益'].mean()
                fl_con_1 = train_data0['预测周期真实收益'] < train_data0.loc[fl_con_, '预测周期真实收益'].mean()

                train_data0.loc[fl_con, 'svc_分类结果'] = 1
                train_data0.loc[fl_con1, 'svc_分类结果'] = 2
                train_data0.loc[fl_con_, 'svc_分类结果'] = -1
                train_data0.loc[fl_con_, 'svc_分类结果'] = -2

                train_data0.fillna(0, inplace=True)

                if len(train_data0['svc_分类结果'].tolist()) == 0 :
                    print('\n======\nsvc： 训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, f'{pre_res_name}：分类问题'
                print(f'{model_name}：svc：数据量大小：{train_data0[yinzi].shape}')
                model3, X_train_pre, mean_score = svc_classify_grid(X_train=train_data0, yinzi=yinzi,
                                                                    y_train=train_data0['svc_分类结果'],
                                                                    n_iter=10, kernel=["rbf",'poly'],
                                                                    n_jobs=n_jobs)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']

                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] == 1].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model3, mean_score]
            else:
                model2 = model_list[model_name][0]
                cat_pre = model2.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0[pre_res_name] != -2].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，svc：分类出错')
                    return {}, f'{pre_res_names}：分类问题'

        # 回归:输入train_data0，输出train_data0
        if 1== True:
            pre_res_name = 'end_预测值'
            model_name = 'model_end'
            pre_res_names.append(pre_res_name)
            if train_data0[yinzi].empty:
                return model_list, '回归，无预测数据'
            if Train:
                print(f'{model_name}数据量：{train_data0[yinzi].shape}')

                model4, X_train_pre, mean_score = poly_ridge_regression(X_train=train_data0, yinzi=yinzi,
                                                                             y_train=train_data0['预测周期真实收益'], degree=4)

                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                X_train_pre.rename(columns={'预测值': pre_res_name}, inplace=True)
                train_data0 = X_train_pre

                model_list[model_name] = [model4, mean_score]
            else:
                model4 = model_list[model_name][0]
                reg_pre = model4.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(reg_pre, index=train_data0.index)
                print('\n======\n预测结果\n回归的结果，线性corr:',
                      np.corrcoef(train_data0[pre_res_name], train_data0['预测周期真实收益'])[0], '\n')
                print("R2得分:", metrics.r2_score(train_data0['预测周期真实收益'], train_data0[pre_res_name]))

            # 随机森林分类:输入train_data0，输出train_data0
        #分类预测输出：
        if 0 == True:
            pre_res_name = 'end_预测值'
            model_name = 'model_end'

            pre_res_names.append(pre_res_name)
            if Train:
                fl_con1 = train_data0['预测周期真实收益'] > train_data0['预测周期真实收益'].mean()
                fl_con11 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con1, '预测周期真实收益'].mean()

                fl_con0 = train_data0['预测周期真实收益'] < train_data0['预测周期真实收益'].mean()
                fl_con01 = train_data0['预测周期真实收益'] < train_data0.loc[fl_con0, '预测周期真实收益'].mean()


                # print(train_data0[['预测周期真实收益']])
                # exit()
                train_data0.loc[fl_con1, 'rf_分类结果'] = 1
                train_data0.loc[fl_con11, 'rf_分类结果'] = 2
                train_data0.loc[fl_con0, 'rf_分类结果'] = -1
                train_data0.loc[fl_con01, 'rf_分类结果'] = -2



                train_data0.dropna(axis=0, inplace=True)
                print(f'{model_name}随机森林分类,数据量大小：{train_data0[yinzi].shape}')
                model1, X_train_pre, mean_score = randomforest_classify_grid(X_train=train_data0, yinzi=yinzi,
                                                                             y_train=train_data0['rf_分类结果'],
                                                                             n_iter=10)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                # train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list[model_name][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                # train_data0 = train_data0[train_data0[pre_res_name] > fl_num].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, 'rf：预测分类出错'


    # 最后一个月只有预测
    if last == True:
        if 'model_1' in model_list.keys():
            # ===拿到预测模型
            model1 = model_list['model_1'][0]
            pre = model1.predict(train_data0[yinzi])
            train_data0.loc[:, '1_预测值'] = pd.Series(pre, index=train_data0.index)
            train_data0 = train_data0[train_data0['1_预测值'] > 0].copy()
            # print(train_data0)

            if train_data0.empty:
                print(f'预测数据，1:分类出错')
                return {}, '1:分类出错'

        if 'model_2' in model_list.keys():
            # ===拿到预测模型
            model2 = model_list['model_2'][0]
            cat_pre = model2.predict(train_data0[yinzi])
            print(cat_pre)
            train_data0.loc[:, '2_预测值'] = pd.Series(cat_pre, index=train_data0.index)

            train_data0 = train_data0[train_data0['2_预测值'] >= -2].copy()
            if train_data0[yinzi].empty:
                print('预测数据，2:分类出错,预测数据为空')
                print(pd.Series(cat_pre, index=train_data0.index).tail(50))
                return {}, '2:分类出错'

        if 'model_3' in model_list.keys():
            # ===拿到预测模型
            model3 = model_list['model_3'][0]
            cat_pre = model3.predict(train_data0[yinzi])
            train_data0.loc[:, '3_预测值'] = pd.Series(cat_pre, index=train_data0.index)
            train_data0 = train_data0[train_data0['3_预测值'] > 0].copy()
            if train_data0[yinzi].empty:
                print('预测数据，3:分类出错')
                print(train_data0.sample(5))
                return {}, '3:分类出错'

        # ===拿到预测模型
        if train_data0[yinzi].empty:
            print('回归，无预测数据!')
            return model_list, '回归，无预测数据!'
        else:
            model4 = model_list['model_end'][0]
            reg_pre = model4.predict(train_data0[yinzi])
            train_data0.loc[:, 'end_预测值'] = pd.Series(reg_pre, index=train_data0.index)

        train_data0['canshu'] = train_data0.index
        cols = [i for i in list(train_data0.columns) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols], train_data, how='inner', on=['canshu', '预测周期真实收益'], left_index=True)
        return model_list, train_data0

    if Train == False:
        print('预测：')
        # print(train_data0.tail())
        # print(train_data['canshu'])
        train_data0['canshu'] = train_data0.index

        # exit()
        cols = [i for i in list(train_data0.columns) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols], train_data, how='inner', on=['canshu', '预测周期真实收益'], left_index=True)

    return model_list, train_data0

def cal_zuhe_gl_04_2(train_data=None, model_list={}, yinzi=[], Train=True, last=False, n_jobs=2):
    '''
    数据归一化
    第一次分类，adaboost算法
    第二次分类，random——Forrest分类
    第3次分类，svc分类，["rbf",'sigmoid','linear','poly']
    分类都采取，随机grid方式过滤。
    最后进行回归。
    '''

    def data_fliter_fb(data, yinzi=[]):

        # 过滤算法
        data = data[data['total_days_3'] > 3]
        data = data[data['trade_nums_3'] > 10]
        data = data[data['trade_nums_2'] > 10]
        data = data[data['trade_nums_3'] < 200]
        data = data[data['trade_nums_2'] < 200]
        data.fillna(0, inplace=True)

        return pd.DataFrame(data)

    print(f'\n======\n欢迎使用{sys._getframe().f_code.co_name}：fb-svc_bag-svc-poly')
    print(f'本次传入数据量：{train_data[yinzi].shape}')


    # print(yinzi)
    # print(train_data.tail())
    # exit()
    if Train:
        # 数据过滤
        train_data = data_fliter_fb(train_data, yinzi)
        train_data0, train_data_, yinzi = k_pac(train_data, yinzi, N=40, kernel='linear', n_jobs=1)
        print(f'训练，本次数据量：{train_data0[yinzi].shape}')
    else:
        train_data = data_fliter_fb(train_data, yinzi)

        train_data0, train_data_, yinzi = k_pac(train_data, yinzi, N=40, kernel='linear', n_jobs=1)
        # train_data0 = train_data
        print(f'预测，本次数据量：{train_data0[yinzi].shape}')

    # 数据处理：==》False
    if not train_data0.empty and False:
        train_data0 = data_precess(train_data0, yinzi=yinzi, pre_style='normal')
        print('\n训练数据标准化处理。')

    pre_res_names = []
    fl_num = 0
    if last == False:
        # ada_dtree:输入train_data0，输出train_data0
        if 0 == True:
            pre_res_name = '1_预测值'
            model_name = 'model_1'
            pre_res_names.append(pre_res_name)
            if Train:
                fl_con1 = train_data0['预测周期真实收益'] > train_data0['预测周期真实收益'].mean()
                fl_con11 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con1, '预测周期真实收益'].mean()
                fl_con0 = train_data0['预测周期真实收益'] < train_data0['预测周期真实收益'].mean()

                train_data0.loc[fl_con1, 'adt_分类结果'] = 1
                train_data0.loc[fl_con11, 'adt_分类结果'] = 2
                train_data0.loc[fl_con0, 'adt_分类结果'] = -1

                if len(train_data0['adt_分类结果'].tolist()) == 0 \
                        or len(train_data0[train_data0['adt_分类结果'] == 2]['adt_分类结果'].tolist()) == 0:
                    print('\n======\nrf：训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, 'rf：分类问题'

                # print(train_data0[['预测周期真实收益','rf_分类结果']])
                train_data0.dropna(axis=0, inplace=True)
                print(f'{model_name}数据量：{train_data0[yinzi].shape}')

                model1, X_train_pre, mean_score = adaboosting_dtree_grid(X_train=train_data0, yinzi=yinzi,
                                                                         y_train=train_data0['adt_分类结果'], n_iter=10,
                                                                         n_jobs=n_jobs)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model1, mean_score]
            else:
                # 拿到预测模型
                model1 = model_list[model_name][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0[pre_res_name] > fl_num].copy()
                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, f'{pre_res_name}：预测分类出错'

        # 随机森林分类:输入train_data0，输出train_data0
        if 0 == True:
            pre_res_name = '1_预测值'
            model_name = 'model_1'

            pre_res_names.append(pre_res_name)
            if Train:
                fl_con1 = train_data0['预测周期真实收益'] > 0 #train_data0['预测周期真实收益'].mean()
                fl_con11 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con1, '预测周期真实收益'].mean()
                fl_con0 = train_data0['预测周期真实收益'] < 0 #train_data0['预测周期真实收益'].mean()

                # print(train_data0[['预测周期真实收益']])
                # exit()
                train_data0.loc[fl_con1, 'rf_分类结果'] = 1
                train_data0.loc[fl_con11, 'rf_分类结果'] = 2
                train_data0.loc[fl_con0, 'rf_分类结果'] = -1

                if len(train_data0['rf_分类结果'].tolist()) == 0 \
                        or len(train_data0[train_data0['rf_分类结果'] >= 1]['rf_分类结果'].tolist()) == 0:
                    print('\n======\nrf：训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, 'rf：分类问题'

                train_data0.dropna(axis=0, inplace=True)
                print(f'{model_name}随机森林分类,数据量大小：{train_data0[yinzi].shape}')
                model1, X_train_pre, mean_score = randomforest_classify_grid(X_train=train_data0, yinzi=yinzi,
                                                                             y_train=train_data0['rf_分类结果'],
                                                                             n_iter=10)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model1, mean_score]
            else:
                # 拿到预测模型
                print(f'{model_name}预测：')
                model1 = model_list[model_name][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                print(train_data0[pre_res_name].sample(100))
                train_data0 = train_data0[train_data0[pre_res_name] > fl_num].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, 'rf：预测分类出错'

        #svc_bagging
        if 0 == True:
                pre_res_name = '2_预测值'
                model_name = 'model_2'
                pre_res_names.append(pre_res_name)
                if Train:
                    fl_con1 = train_data0['预测周期真实收益'] > 0 #train_data0['预测周期真实收益'].mean()
                    # fl_con11 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con1, '预测周期真实收益'].mean()
                    fl_con0 = train_data0['预测周期真实收益'] < 0 #train_data0['预测周期真实收益'].mean()

                    train_data0.loc[fl_con1, '1分类结果'] = 1
                    # train_data0.loc[fl_con11, '1分类结果'] = 2
                    train_data0.loc[fl_con0, '1分类结果'] = -1

                    if len(train_data0['1分类结果'].tolist()) == 0 \
                            or len(train_data0[train_data0['1分类结果'] == 1]['1分类结果'].tolist()) == 0:
                        print('\n======\n1：训练结果分类，出问题。')
                        print(train_data0['预测周期真实收益'].sample(5))
                        return model_list, '1：分类问题'

                    train_data0.dropna(axis=0, inplace=True)
                    print(f'{model_name}数据量：{train_data0[yinzi].shape}')

                    model1, X_train_pre, mean_score = bag_svc(X_train=train_data0, yinzi=yinzi,
                                                                             y_train=train_data0['1分类结果'], n_iter=60,
                                                                             n_jobs=n_jobs)

                    X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                    # 目标分类
                    train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                    train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                    model_list[model_name] = [model1, mean_score]
                else:
                    # 拿到预测模型
                    model1 = model_list[model_name][0]
                    cat_pre = model1.predict(train_data0[yinzi])
                    train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                    train_data0 = train_data0[train_data0[pre_res_name] > fl_num].copy()
                    if train_data0[yinzi].empty:
                        print('预测数据，分类出错')
                        return {}, f'{pre_res_name}：预测分类出错'

        # svc分类:输入train_data0，输出train_data0
        if 0 == True:
            pre_res_name = '2_预测值'
            model_name = 'model_2'

            pre_res_names.append(pre_res_name)

            if Train:
                fl_con = train_data0['预测周期真实收益'] > train_data0['预测周期真实收益'].mean()
                fl_con1 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con, '预测周期真实收益'].mean()
                fl_con_ = train_data0['预测周期真实收益'] < train_data0['预测周期真实收益'].mean()
                fl_con_1 = train_data0['预测周期真实收益'] < train_data0.loc[fl_con_, '预测周期真实收益'].mean()

                train_data0.loc[fl_con, 'svc_分类结果'] = 1
                train_data0.loc[fl_con1, 'svc_分类结果'] = 2
                train_data0.loc[fl_con_, 'svc_分类结果'] = -1
                train_data0.loc[fl_con_, 'svc_分类结果'] = -2

                train_data0.fillna(0, inplace=True)

                if len(train_data0['svc_分类结果'].tolist()) == 0 :
                    print('\n======\nsvc： 训练结果分类，出问题。')
                    print(train_data0['预测周期真实收益'].sample(5))
                    return model_list, f'{pre_res_name}：分类问题'
                print(f'{model_name}：svc：数据量大小：{train_data0[yinzi].shape}')
                model3, X_train_pre, mean_score = svc_classify_grid(X_train=train_data0, yinzi=yinzi,
                                                                    y_train=train_data0['svc_分类结果'],
                                                                    n_iter=10, kernel=["rbf",'poly'],
                                                                    n_jobs=n_jobs)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']

                # 目标分类
                train_data0 = X_train_pre[X_train_pre['预测值'] == 1].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)

                model_list[model_name] = [model3, mean_score]
            else:
                model2 = model_list[model_name][0]
                cat_pre = model2.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                train_data0 = train_data0[train_data0[pre_res_name] != -2].copy()

                if train_data0[yinzi].empty:
                    print('预测数据，svc：分类出错')
                    return {}, f'{pre_res_names}：分类问题'

        # 回归:输入train_data0，输出train_data0
        if 0== True:
            pre_res_name = 'end_预测值'
            model_name = 'model_end'
            pre_res_names.append(pre_res_name)
            if train_data0[yinzi].empty:
                return model_list, '回归，无预测数据'
            if Train:
                print(f'{model_name}数据量：{train_data0[yinzi].shape}')

                model4, X_train_pre, mean_score = poly_ridge_regression(X_train=train_data0, yinzi=yinzi,
                                                                             y_train=train_data0['预测周期真实收益'], degree=4)

                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                X_train_pre.rename(columns={'预测值': pre_res_name}, inplace=True)
                train_data0 = X_train_pre

                model_list[model_name] = [model4, mean_score]
            else:
                model4 = model_list[model_name][0]
                reg_pre = model4.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(reg_pre, index=train_data0.index)
                print('\n======\n预测结果\n回归的结果，线性corr:',
                      np.corrcoef(train_data0[pre_res_name], train_data0['预测周期真实收益'])[0], '\n')
                print("R2得分:", metrics.r2_score(train_data0['预测周期真实收益'], train_data0[pre_res_name]))

            # 随机森林分类:输入train_data0，输出train_data0
        #分类预测输出：
        if 1 == True:
            pre_res_name = 'end_预测值'
            model_name = 'model_end'

            pre_res_names.append(pre_res_name)
            if Train:
                fl_con1 = train_data0['预测周期真实收益'] > train_data0['预测周期真实收益'].mean()
                fl_con11 = train_data0['预测周期真实收益'] > train_data0.loc[fl_con1, '预测周期真实收益'].mean()

                fl_con0 = train_data0['预测周期真实收益'] < train_data0['预测周期真实收益'].mean()
                fl_con01 = train_data0['预测周期真实收益'] < train_data0.loc[fl_con0, '预测周期真实收益'].mean()


                # print(train_data0[['预测周期真实收益']])
                # exit()
                train_data0.loc[fl_con1, 'rf_分类结果'] = 1
                train_data0.loc[fl_con11, 'rf_分类结果'] = 2
                train_data0.loc[fl_con0, 'rf_分类结果'] = -1
                train_data0.loc[fl_con01, 'rf_分类结果'] = -2



                train_data0.dropna(axis=0, inplace=True)
                print(f'{model_name}随机森林分类,数据量大小：{train_data0[yinzi].shape}')
                model1, X_train_pre, mean_score = randomforest_classify_grid(X_train=train_data0, yinzi=yinzi,
                                                                             y_train=train_data0['rf_分类结果'],
                                                                             n_iter=10)
                X_train_pre['预测周期真实收益'] = train_data0['预测周期真实收益']
                # 目标分类
                # train_data0 = X_train_pre[X_train_pre['预测值'] > fl_num].copy()
                train_data0.rename(columns={'预测值': pre_res_name}, inplace=True)
                # pre = model1.predict(train_data0[yinzi])
                # print(set(pre))
                model_list[model_name] = [model1, mean_score]
            else:
                print(f'{model_name}预测：')

                # 拿到预测模型
                model1 = model_list[model_name][0]
                cat_pre = model1.predict(train_data0[yinzi])
                train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
                # train_data0 = train_data0[train_data0[pre_res_name] > fl_num].copy()
                print(train_data0[pre_res_name].sample(10))

                if train_data0[yinzi].empty:
                    print('预测数据，分类出错')
                    return {}, 'rf：预测分类出错'


    # 最后一个月只有预测
    if last == True:
        if 'model_1' in model_list.keys():
            print(f'model_1，最后预测：')

            # ===拿到预测模型
            model1 = model_list['model_1'][0]
            pre = model1.predict(train_data0[yinzi])
            train_data0.loc[:, '1_预测值'] = pd.Series(pre, index=train_data0.index)
            print(train_data0['1_预测值'].sample(10))
            print('选大于零的。')
            train_data0 = train_data0[train_data0['1_预测值'] > 0].copy()

            # print(train_data0)

            if train_data0.empty:
                print(f'预测数据，1:分类出错')
                return {}, '1:分类出错'

        if 'model_2' in model_list.keys():
            # ===拿到预测模型
            model2 = model_list['model_2'][0]
            cat_pre = model2.predict(train_data0[yinzi])
            print(cat_pre)
            train_data0.loc[:, '2_预测值'] = pd.Series(cat_pre, index=train_data0.index)

            train_data0 = train_data0[train_data0['2_预测值'] >= -2].copy()
            if train_data0[yinzi].empty:
                print('预测数据，2:分类出错,预测数据为空')
                print(pd.Series(cat_pre, index=train_data0.index).tail(50))
                return {}, '2:分类出错'

        if 'model_3' in model_list.keys():
            # ===拿到预测模型
            model3 = model_list['model_3'][0]
            cat_pre = model3.predict(train_data0[yinzi])
            train_data0.loc[:, '3_预测值'] = pd.Series(cat_pre, index=train_data0.index)
            train_data0 = train_data0[train_data0['3_预测值'] > 0].copy()
            if train_data0[yinzi].empty:
                print('预测数据，3:分类出错')
                print(train_data0.sample(5))
                return {}, '3:分类出错'

        # ===拿到预测模型
        if train_data0[yinzi].empty:
            print('回归，无预测数据!')
            return model_list, '回归，无预测数据!'
        else:
            # print(f"最后！\n{model_list['model_end']}")
            modelend = model_list['model_end'][0]
            reg_pre = modelend.predict(train_data0[yinzi])

            train_data0.loc[:, 'end_预测值'] = pd.Series(reg_pre, index=train_data0.index)
            print(train_data0['end_预测值'].sample(10))
        train_data0['canshu'] = train_data0.index
        cols = [i for i in list(train_data0.columns) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols], train_data, how='inner', on=['canshu', '预测周期真实收益'], left_index=True)
        return model_list, train_data0

    if Train == False:
        print('预测：')
        # print(train_data0.tail())
        # print(train_data['canshu'])
        train_data0['canshu'] = train_data0.index

        # exit()
        cols = [i for i in list(train_data0.columns) if i not in yinzi]
        train_data0 = pd.merge(train_data0[cols], train_data, how='inner', on=['canshu', '预测周期真实收益'], left_index=True)

    return model_list, train_data0

def zhunquedu(zhenshi ,yuce):
    from sklearn.metrics import accuracy_score
    zqd = accuracy_score(zhenshi,yuce)
    return zqd

def zuhe_911_01(train_data=None,res_to_df ={} , yinzi=[], Train=True, last=False, n_jobs=2,**canshu):
    '''
    数据归一化
    第一次分类，adaboost算法
    第二次分类，random——Forrest分类
    第3次分类，svc分类，["rbf",'sigmoid','linear','poly']
    分类都采取，随机grid方式过滤。
    最后进行回归。
    '''

    gaishu = f' 使用两层选择：' \
             f'\n第一层：使用几个分类器进行分类！' \
             f'\n第二层：在预测结束之后，可以得到一个：结果，进行保存：时间：df结果= 策略，参数，分类器名字，训练得分，训练的预测结果，预测结果的结果 ，预测周期真实值' \
             f'\n对其保存！'

    fl_list = ['randomforest_classify_grid',
               # 'adaboosting_dtree_grid',
               # 'svc_classify_grid',
               ]
    def data_fliter_fb(data):

        # 过滤算法
        data = data[data['total_days_3'] > 4]
        data = data[data['trade_nums_3'] > 10]
        data = data[data['trade_nums_2'] > 10]
        data = data[data['trade_nums_3'] < 200]
        data = data[data['trade_nums_2'] < 200]
        data.fillna(0, inplace=True)

        return pd.DataFrame(data)

    print(f'\n======\n欢迎使用{sys._getframe().f_code.co_name}')
    print(f'本次传入数据量：{train_data[yinzi].shape}')

    train_data = data_fliter_fb(train_data)
    if Train:
        # 数据过滤
        print(f'本次训练，本次数据量：{train_data[yinzi].shape}')
        train_data0, train_data_, yinzi = k_pac(train_data, yinzi, N=40, kernel='linear', n_jobs=1)

        train_data0['预测周期真实分类'] =pd.qcut(train_data0['预测周期真实收益'],q = 8,labels=[x for x in range(1,9,1)],precision=1)

    else:
        print(f'预测，本次数据量：{train_data[yinzi].shape}')
        train_data0, train_data_, yinzi = k_pac(train_data, yinzi, N=40, kernel='linear', n_jobs=1)
        # 最后一个月只有预测
        if last == False:
            train_data0['预测周期真实分类'] = pd.qcut(train_data0['预测周期真实收益'], q=8, labels=[x for x in range(1, 9, 1)], precision=1)
        else:
            train_data0['预测周期真实分类'] = 0
    # # 数据处理：==》False
    # if not train_data0.empty and False:
    #     train_data0 = data_precess(train_data0, yinzi=yinzi, pre_style='normal')
    #     print('\n训练数据标准化处理。')

    pre_res_names = []
    train_data0 = train_data0
    for i in fl_list:
        pre_res_name = 'data_%s'%i
        model_name = 'model_%s'%i
        pre_res_names.append(pre_res_name)
        #训练 》》model
        if Train and last == False:
            print(f'训练！：{model_name}数据量：{train_data0.shape}，因子量:{len(yinzi)}')
            model1, X_train_pre, scores = eval(i)(X_train=train_data0, yinzi=yinzi, y_train=train_data0['预测周期真实分类'],
                                                                     n_iter=10,n_jobs=n_jobs)

            train_data0.rename(columns={'预测周期真实分类':'测试集_真实分类'},inplace =True)
            train_data0.rename(columns={'预测周期真实收益':'测试集_真实收益'},inplace =True)

            train_data0['测试集_预测值'] = X_train_pre['预测值']
            train_data0['best_score'] = scores[0]
            train_data0['sj_acc_score'] = scores[1]
            train_data0['sj_ba_acc_score'] = scores[2]
            train_data0['model_name'] = model_name

            train_data0 =train_data0[['canshu','celue_name','model_name','测试集_真实收益','测试集_真实分类','测试集_预测值',
                                                     'best_score', 'sj_acc_score', 'sj_ba_acc_score']]
            res_to_df[pre_res_name] = train_data0
            # print(train_data0.keys())
            # print(train_data0.keys())
            res_to_df[model_name] = model1
        # 预测》》预测结果
        elif Train==False and last == False:
            print(f'预测！：{model_name}数据量：{train_data0.shape}，因子量:{len(yinzi)}')

            # 拿到预测模型
            model1 = res_to_df[model_name]
            cat_pre = model1.predict(train_data0[yinzi])
            train_data0['预测值'] = pd.Series(cat_pre, index=train_data0.index)
            train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
            if len(set(train_data0['预测周期真实分类'].values.tolist())) == 1:
                train_data0['预测准确度']  = zhunquedu(train_data0['预测周期真实分类'], train_data0['预测值'])
            else:
                train_data0['预测准确度']  = 0


            train_data0['model_name'] = model_name
            train_data0 = train_data0[[ 'canshu', 'celue_name',  'model_name','预测周期真实收益','预测周期真实分类','预测值']]
            data_t = res_to_df[pre_res_name]
            res_to_df[pre_res_name] = pd.merge(data_t, train_data0, on=[ 'model_name','celue_name', 'canshu'], how='inner')
            # print(train_data0.keys())

        elif last == True:
            print('最后一个月！预测！')
            print(f'训练！：{model_name}数据量：{train_data0.shape}，因子量:{len(yinzi)}')

            # 拿到预测模型
            model1 = res_to_df[model_name]
            cat_pre = model1.predict(train_data0[yinzi])
            train_data0['预测值'] = pd.Series(cat_pre, index=train_data0.index)
            train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
            if train_data0['预测周期真实分类'].mean() != 0:
                train_data0['预测准确度']  = zhunquedu(train_data0['预测周期真实分类'], train_data0['预测值'])
            else:
                train_data0['预测准确度']  = 0

            train_data0['model_name'] = model_name
            train_data0 = train_data0[['canshu', 'celue_name', 'model_name', '预测周期真实收益', '预测周期真实分类','预测值']]
            data_t = res_to_df[pre_res_name]
            res_to_df[pre_res_name] = pd.merge(data_t, train_data0, on=[ 'model_name','celue_name', 'canshu'], how='inner')
            print('预测结束：',train_data0.keys())


    return res_to_df, fl_list

def zuhe_911_02(train_data=None,res_to_df ={} ,yinzi=[], Train=True, last=False, n_jobs=2,**canshu):
    '''
    数据归一化
    第一次分类，adaboost算法
    第二次分类，random——Forrest分类
    第3次分类，svc分类，["rbf",'sigmoid','linear','poly']
    分类都采取，随机grid方式过滤。
    最后进行回归。
    '''
    pac_k = canshu['pac_k']
    q = canshu['q']
    ['para1', 'para2', 'para3', 'para4', 'clm0', 'clm1', 'clm2', 'clm3', 'clm4', 'clm5']
    gaishu = f' 使用两层选择：' \
             f'\n第一层：使用几个分类器进行分类！' \
             f'\n第二层：在预测结束之后，可以得到一个：结果，进行保存：时间：df结果= 策略，参数，分类器名字，训练得分，训练的预测结果，预测结果的结果 ，预测周期真实值' \
             f'\n对其保存！'

    fl_list = ['randomforest_classify_grid',
               # 'svc_classify_grid',
               'adaboosting_dtree_grid',
               ]
    def data_fliter_fb(data):

        # 过滤算法
        data = data[data['total_days_3'] > 4]
        data = data[data['trade_nums_3'] > 10]
        data = data[data['trade_nums_2'] > 10]
        data = data[data['trade_nums_3'] < 200]
        data = data[data['trade_nums_2'] < 200]
        data.fillna(0, inplace=True)

        return pd.DataFrame(data)

    print(f'\n======\n欢迎使用{sys._getframe().f_code.co_name}')
    print(f'本次传入数据量：{train_data[yinzi].shape}')

    train_data = data_fliter_fb(train_data)
    yinzi_k = [x for x in yinzi if (str(x).startswith('para') or str(x).startswith('clm')) == False]
    yinzi_y = [x for x in yinzi if (str(x).startswith('para') or str(x).startswith('clm'))]
    yinzi_sheng = [x for x in list(train_data.keys()) if x not in (yinzi_k+yinzi_y)]
    # print(yinzi_k,'\n',yinzi_y)
    # exit()

    if Train:
        # 数据过滤
        print(f'本次训练，本次数据量：{train_data[yinzi].shape}')

        train_data0, train_data_, yinzi_k = k_pac(train_data, yinzi_k, N=pac_k, kernel='linear', n_jobs=1)
        con1 = train_data0['预测周期真实收益'] <= 0
        con2 = train_data0['预测周期真实收益'] > 0
        train_data0.loc[con1, '预测周期真实分类'] = pd.qcut(train_data0[con1]['预测周期真实收益'], q=int(q / 2),
                                                 labels=[x for x in range(1, int(q / 2) + 1, 1)], precision=1)
        train_data0.loc[con2, '预测周期真实分类'] = pd.qcut(train_data0[con2]['预测周期真实收益'], q=int(q / 2),
                                                 labels=[x for x in range(int(q / 2) + 1, q + 1, 1)], precision=1)


    else:
        print(f'预测，本次数据量：{train_data[yinzi].shape}')
        train_data0, train_data_, yinzi_k = k_pac(train_data, yinzi_k, N=pac_k, kernel='linear', n_jobs=1)
        # 最后一个月只有预测
        if last == False:
            con1 = train_data0['预测周期真实收益'] <= 0
            con2 = train_data0['预测周期真实收益'] > 0
            train_data0.loc[con1, '预测周期真实分类'] = pd.qcut(train_data0[con1]['预测周期真实收益'], q=int(q / 2),
                                                        labels=[x for x in range(1, int(q / 2) + 1, 1)], precision=1)
            train_data0.loc[con2, '预测周期真实分类'] = pd.qcut(train_data0[con2]['预测周期真实收益'], q=int(q / 2),
                                                        labels=[x for x in range(int(q / 2) + 1, q + 1, 1)], precision=1)
        else:
            train_data0['预测周期真实分类'] = 0
    # # 数据处理：==》False
    # if not train_data0.empty and False:
    #     train_data0 = data_precess(train_data0, yinzi=yinzi, pre_style='normal')
    #     print('\n训练数据标准化处理。')

    pre_res_names = []
    yinzi = yinzi_k + yinzi_y
    for i in train_data.keys():
        if i in (yinzi_sheng+yinzi_y):
            train_data0[i] = train_data[i]
    train_data1 = train_data0.copy()

    for i in fl_list:
        train_data0 = train_data1.copy()
        pre_res_name = 'data_%s'%i
        model_name = 'model_%s'%i
        pre_res_names.append(pre_res_name)
        #训练 》》model
        if Train and last == False:
            print(f'训练：{model_name}数据量：{train_data0.shape}，因子量:{len(yinzi)}')
            # print(train_data0[yinzi].tail());exit()
            model1, X_train_pre, scores = eval(i)(X_train=train_data0, yinzi=yinzi, y_train=train_data0['预测周期真实分类'],n_iter=6,n_jobs=n_jobs)

            train_data0.rename(columns={'预测周期真实分类':'训练集_真实分类'},inplace =True)
            train_data0.rename(columns={'预测周期真实收益':'训练集_真实收益'},inplace =True)

            train_data0['训练集_best_score'] = scores[0]
            train_data0['训练集_sj_acc_score'] = scores[1]
            train_data0['训练集_sj_ba_acc_score'] = scores[2]
            train_data0['model_name'] = model_name
            train_data0['训练集_预测值'] = X_train_pre['预测值']
            train_data0 = train_data0[['canshu','celue_name','model_name','训练集_真实收益','训练集_真实分类',
                                                     '训练集_best_score', '训练集_sj_acc_score', '训练集_sj_ba_acc_score','训练集_预测值']]
            res_to_df[pre_res_name] = train_data0
            # print(train_data0.keys())
            # print(train_data0.keys())
            res_to_df[model_name] = model1
        # 预测》》预测结果
        elif Train==False and last == False:
            print(f'预测模型：{model_name}数据量：{train_data0.shape}，因子量:{len(yinzi)}')
            # 拿到预测模型
            model1 = res_to_df[model_name]
            cat_pre = model1.predict(train_data0[yinzi])
            train_data0['预测集_预测值'] = pd.Series(cat_pre, index=train_data0.index)
            train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
            train_data0['model_name'] = model_name
            train_data0 = train_data0[[ 'canshu', 'celue_name',  'model_name','预测周期真实收益','预测周期真实分类','预测集_预测值']]
            data_t = res_to_df[pre_res_name]
            res_to_df[pre_res_name] = pd.merge(data_t, train_data0, on=[ 'model_name','celue_name', 'canshu'], how='inner')
            # print(train_data0.keys())

        elif last == True:
            print('最后一个月！预测！')
            print(f'预测模型：{model_name} 数据量：{train_data0.shape}，因子量:{len(yinzi)}')

            # 拿到预测模型
            model1 = res_to_df[model_name]
            cat_pre = model1.predict(train_data0[yinzi])
            train_data0['预测集_预测值'] = pd.Series(cat_pre, index=train_data0.index)
            train_data0.loc[:, pre_res_name] = pd.Series(cat_pre, index=train_data0.index)
            train_data0['model_name'] = model_name
            train_data0 = train_data0[['canshu', 'celue_name', 'model_name', '预测周期真实收益', '预测周期真实分类','预测集_预测值']]
            data_t = res_to_df[pre_res_name]
            res_to_df[pre_res_name] = pd.merge(data_t, train_data0, on=[ 'model_name','celue_name', 'canshu'], how='inner')
    if Train == False:
        # 选出俩个的相同的
        per_date = pd.DataFrame()
        for n in range(1,q+1,1):
            da_list=[]
            for k ,da in res_to_df.items():
                if str(k).startswith('data'):
                    da['new'] = da['canshu'] + da['celue_name']
                    per_date = da[da['训练集_预测值']==n]
                    da_list.append(set(per_date['new'].values.tolist()))
            new_set = set()
            for i in da_list:
                new_set &=i
            per_date['new'].apply(lambda x:True if x in new_set else False )
            per_date = per_date[per_date['new']==True]
        res_to_df['data_hb'] = per_date
        print(per_date)
    return res_to_df, fl_list

def second_fl01(train_data=None,res_to_df ={} ,yinzi=[],resinx='', Train=True,  n_jobs=2,**canshu):
    '''
    数据归一化
    第一次分类，adaboost算法
    第二次分类，random——Forrest分类
    第3次分类，svc分类，["rbf",'sigmoid','linear','poly']
    分类都采取，随机grid方式过滤。
    最后进行回归。
    '''
    pac_k = canshu['pac_k']
    q = canshu['q']
    gaishu = f' 使用两层选择：' \
             f'\n第一层：使用几个分类器进行分类！' \
             f'\n第二层：在预测结束之后，可以得到一个：结果，进行保存：时间：df结果= 策略，参数，分类器名字，训练得分，训练的预测结果，预测结果的结果 ，预测周期真实值' \
             f'\n对其保存！'

    fl_list = [
               # 'adaboosting_dtree_grid',
               # 'svc_classify_grid',
                'randomforest_classify_grid',
               ]

    def data_fliter_fb(data):

        # 过滤算法
        # data = data[data['total_days_3'] > 4]
        data = data[data['trade_nums_3'] > 10]
        # data = data[data['trade_nums_2'] > 10]
        data = data[data['trade_nums_3'] < 200]
        # data = data[data['trade_nums_2'] < 200]
        # data.fillna(0, inplace=True)
        pass

        return pd.DataFrame(data)

    print(f'\n======\n欢迎使用{sys._getframe().f_code.co_name}')
    print(f'本次传入数据量：{train_data[yinzi].shape}')

    train_data = data_fliter_fb(train_data)
    train_data[yinzi].fillna(0, inplace=True)
    train_data[resinx].fillna(0, inplace=True)

    yinzi_k = [x for x in yinzi if not str(x).startswith('para')]
    # yinzi_k = [x for x in yinzi_k if not str(x).startswith('clm')]

    yinzi_y = [x for x in yinzi if x not in yinzi_k]

    print('计算因子特征：',len(yinzi_k))
    print('参数和策略：',len(yinzi_y))
    train_data1 = data_precess(train_data,yinzi_k,'max_min')
    for col in train_data1.keys():
        train_data[col] = train_data1[col]
        # print(col)
        # exit()
    train_data0=train_data
    print(train_data0.tail(10))
    # exit()
    # yinzi_k = list(train_data0.keys())
    if Train:
        # 数据过滤
        print(f'本次训练，本次数据量：{train_data[yinzi].shape}')
        # print(train_data.tail(100))
        # train_data0, train_data_, yinzi_k = k_pac(train_data, yinzi_k, N=40, kernel='linear', n_jobs=1)
        # print(yinzi_k)
        # exit()
        train_data0.loc[:, '预测周期真实分类2'] = pd.qcut(train_data[resinx], q=int(q ),
                                                 labels=[x for x in range(1, int(q) + 1, 1)], precision=1)
        # train_data0.loc[:, '预测周期真实分类'] = pd.qcut(train_data0[train_data0[resinx] > 0][resinx], q=int(q / 2),
        #                                          labels=[x for x in range(int(q / 2) + 1, q + 1, 1)], precision=1)

    else:
        print(f'预测，本次数据量：{train_data[yinzi].shape}')
        # train_data0, train_data_, yinzi_k = k_pac(train_data, yinzi_k, N=pac_k, kernel='linear', n_jobs=1)

    # # 数据处理：==》False
    # if not train_data0.empty and False:
    #     train_data0 = data_precess(train_data0, yinzi=yinzi, pre_style='normal')
    #     print('\n训练数据标准化处理。')

    pre_res_names = []
    train_data0 = train_data0
    yinzi = yinzi_k + yinzi_y

    for i in fl_list:
        pre_res_name = 'data_%s'%i
        model_name = 'model_%s'%i
        pre_res_names.append(pre_res_name)
        #训练 》》model
        if Train:
            print(f'训练！：{model_name}数据量：{train_data0.shape}，因子量:{len(yinzi)}')
            # print(train_data0['预测周期真实分类'].sample(100))
            # print(train_data0[yinzi].sample(100))
            model1, X_train_pre, scores = eval(i)(X_train=train_data0, yinzi=yinzi, y_train=train_data0['预测周期真实分类2'],
                                                                     n_iter=10,n_jobs=n_jobs)

            train_data0.rename(columns={'预测周期真实分类2': '测试集_真实分类'},inplace =True)
            train_data0.rename(columns={resinx:'训练_真实收益'},inplace =True)

            train_data0['训练_预测值'] = X_train_pre['预测值']
            train_data0['训练_best_score'] = scores[0]
            train_data0['训练_sj_acc_score'] = scores[1]
            train_data0['训练_sj_ba_acc_score'] = scores[2]
            train_data0['训练_model_name'] = model_name

            res_to_df[pre_res_name] = train_data0
            # print(train_data0.keys())
            # print(train_data0.keys())
            res_to_df[model_name] = model1
        # 预测》》预测结果
        elif Train==False :
            print(f'预测=：{model_name}数据量：{train_data0.shape}，因子量:{len(yinzi)}')

            # 拿到预测模型
            model1 = res_to_df[model_name]
            cat_pre = model1.predict(train_data0[yinzi])
            train_data0['最终预测值'] = pd.Series(cat_pre, index=train_data0.index)
            train_data0['训练_model_name'] = model_name

            data_t = res_to_df[pre_res_name]

            res_to_df[pre_res_name] = pd.merge(data_t, train_data0[['训练_model_name','celue_name', 'canshu','最终预测值','s_Time']], on=['训练_model_name','celue_name', 'canshu'], how='inner')
            # print(res_to_df[pre_res_name].tail(100))

    return res_to_df, fl_list


if __name__ == '__main__':
    from 统计分析 import *

