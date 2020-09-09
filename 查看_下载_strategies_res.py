from mongoengine import DateTimeField, StringField, Document, BinaryField, FloatField, BooleanField, DictField, register_connection
import pandas as pd
import pickle
import os
import time
import datetime
pd.set_option('max_rows', 99999)
# pd.set_option('max_columns', 20)
pd.set_option('expand_frame_repr', False)
pd.set_option('precision', 8)

class StrategyCode(Document):
    name = StringField(required=True)
    class_name = StringField(required=True)
    hash = StringField(max_length=64)
    data = BinaryField()

    meta = {
        'db_alias': 'VNPY_BACKTEST',
        "indexes": [
            {
                "fields": ("class_name",),
            },
            {
                "fields": ("name", "class_name",),
                "unique": True
            },
            {
                "fields": ("name", "class_name", "hash"),
                "unique": True,
            }
        ],
    }

class StrategiesSet(Document):
    hash = StringField(max_length=64)
    class_name = StringField()
    vt_symbol = StringField()
    interval = StringField()
    start = DateTimeField()
    end = DateTimeField()
    rate = FloatField()
    slippage = FloatField()
    size = FloatField()
    pricetick = FloatField()
    capital = FloatField()
    inverse = BooleanField()
    setting = DictField()
    result = DictField()

    meta = {
        'db_alias': 'VNPY_BACKTEST',
        "indexes": [
            {
                "name": "ss_unique",
                "fields": ("class_name", "vt_symbol", "interval", "start", "end", "rate", "slippage", "size", "pricetick", "capital", "inverse", "setting"),
                "unique": True,
            },
            {
                "fields": ("class_name",),
            },
        ],
    }

def init():
    from vnpy.trader.setting import get_settings
    db_settting = get_settings('database.')
    register_connection(
        'VNPY_BACKTEST',
        db='VNPY_BACKTEST',
        host=db_settting['host'],
        port=db_settting['port'],
        username=db_settting['user'],
        password=db_settting['password'],
        authentication_source=db_settting['authentication_source'],
    )

# all_data
def DBdata_to_dfs(res_path, moxing=['04']):
    '''
    1.读取本地临时文件。（实际不需要
    2.更新moxing中对应的db数据
    3.今日日期结尾进行保存。
    '''

    init()
    all = StrategyCode.objects(class_name__exists=True)
    all0 = [a.class_name for a in all]
    result_pickle = {}

    # with open(res_path, 'rb') as f0:
    #     local_res = pickle.load(f0)
    #     print('本地keys：', local_res.keys())

    try:
        for a0 in all0:
            res = StrategiesSet.objects(class_name=a0, result__ne={})
            all = StrategiesSet.objects(class_name=a0)
            print(f'策略：{a0}， 已完成：{res.count()}， 总共：{all.count()}, 完成率：{float(res.count()) * 100 / float(all.count())}%')
            # continue
            # 开始，进行数据加工
            if res.count() == all.count():
                if a0.split('_')[-1] not in moxing: continue
                print(a0)
                # 本地判断
                # if a0 in [l.strip('_' + l.split('_')[-1]) for l in local_res.keys()]:
                #     print(f'策略：{a0},已存在本地。')
                #     continue

                all = StrategiesSet.objects(class_name=a0)
                result_dict = {}
                i = 0
                cols = []

                for index_, s0 in enumerate(all):
                    result_dict0 = {}
                    result_dict0['celue_name'] = s0.class_name + str('_') + str(s0.setting['zhouqi'])
                    result_dict0['s_time'] = s0.start
                    result_dict0['para1'] = s0.setting['ma_len']
                    result_dict0['para2'] = s0.setting['bd']
                    result_dict0['para3'] = s0.setting['dk_len']
                    result_dict0['para4'] = s0.setting['acc']
                    result_dict0['canshu'] = [s0.setting['ma_len'], s0.setting['bd'], s0.setting['dk_len'],
                                              s0.setting['acc']]
                    # 赋值结果
                    if s0.result['win_rate'] != None:
                        result_dict0['win_rate'] = s0.result['win_rate']
                        result_dict0['win_mean_num'] = s0.result['win_mean_num']
                        result_dict0['loss_mean_num'] = s0.result['loss_mean_num']
                        result_dict0['max_num'] = s0.result['max_num']
                        result_dict0['min_num'] = s0.result['min_num']
                        result_dict0['mean_num'] = s0.result['mean_num']
                        result_dict0['std_num'] = s0.result['std_num']
                        result_dict0['max_sum'] = s0.result['max_sum']
                        result_dict0['end'] = s0.result['total_net_pnl']
                        result_dict0['max_back'] = s0.result['max_drawdown']
                        result_dict0['trade_nums'] = s0.result['total_trade_count']
                        result_dict0['sharp_rate'] = s0.result['sharpe_ratio']
                        result_dict0['total_days'] = s0.result['total_days']
                        result_dict0['profit_days'] = s0.result['profit_days']
                        result_dict0['total_slippage'] = s0.result['total_slippage']
                        result_dict0['win_rate'] = s0.result['win_rate']
                        result_dict0['win_mean_num'] = s0.result['win_mean_num']
                        result_dict0['win_mean_num'] = s0.result['loss_mean_num']
                        result_dict0['max_num'] = s0.result['max_num']
                        result_dict0['min_num'] = s0.result['min_num']
                        result_dict0['mean_num'] = s0.result['mean_num']
                        result_dict0['std_num'] = s0.result['std_num']
                        cols = result_dict0.keys()

                    result_dict[index_] = result_dict0

                df_res0 = pd.DataFrame(result_dict).T
                # print(df_res0)
                df_res0 = df_res0[list(cols)]
                for name in [c for c in list(set(df_res0['celue_name'].tolist()))]:
                    result_pickle[str(name)] = df_res0.loc[df_res0['celue_name'] == name, :].copy()
                    print(f'{name}，已生成！')
                # result_pickle[str(a0)] = df_res0
            else:
                print(f'{a0}，没有完成！')

        if len(result_pickle.keys()) == 0:
            print('无数据，结束。 ')
            exit()

        print('当前的key：', result_pickle.keys())
        res_in = input('确定保存：y/n')
        if res_in == 'y':
            time0 = time.strftime("%m-%d", time.localtime())

            respath = res_path.split('.')[0] + moxing[0]+'_'+time0 + '.pickle'
            with open(respath,'wb') as f:
                pickle.dump(result_pickle, f)
                print('dict 已保存，keys：', result_pickle.keys())
                print(f'数据存放地址：》{respath}')

        else:
            print('没有保存。')


    except Exception as e:
        print(e)

    return result_pickle

def show_DB_data():
    init()
    all = StrategyCode.objects(class_name__exists=True)
    all0 = [a.class_name for a in all]
    done = []
    undone = []
    for a0 in all0:
        res = StrategiesSet.objects(class_name=a0, result__ne={})
        all = StrategiesSet.objects(class_name=a0)
        print(f'策略：{a0}， 已完成：{res.count()}， 总共：{all.count()}, 完成率：{float(res.count())*100/float(all.count())}%' )
        if float(res.count())/float(all.count())==1:
            done.append(a0)
        else:
            undone.append(a0)

    print(f'\n已完成：{done}\n未完成：{undone}')
    return [all0]

def save_to_local(to_pickle_data,to_pickle_flag):

    path0 = os.getcwd()+'\\'
    for i in list(set([k.split('_')[-2] for k in to_pickle_data.keys()])):

        cl_to_pickle_data = {}

        for k in to_pickle_data.keys():
            if k.split('_')[-2] == i:
                cl_to_pickle_data[k] = to_pickle_data[k]


        to_path = path0 + str(to_pickle_flag) + f'_{i}.pickle'
        if os.path.exists(to_path):
            to_path = path0 + str(to_pickle_flag) + f'_{i}_new.pickle'
            print(f'文件地址{to_path}已经存在!')
            new = input('是否建立新文件？y/n：')
            if new == 'y':
                with open(to_path, 'wb') as f:
                    pickle.dump(cl_to_pickle_data, f)
                    print(f'{cl_to_pickle_data.keys()}\n文件已保存到：{to_path}')
            # print(cl_to_pickle_data.keys())
            continue
        else:
            to_path = path0 + str(to_pickle_flag) + f'_{i}.pickle'
            # print(to_path,'\n', cl_to_pickle_data.keys())
            # continue

            with open(to_path, 'wb') as f:
                pickle.dump(cl_to_pickle_data, f)
                print(f'{cl_to_pickle_data.keys()}\n文件已保存到：{to_path}')


    return

def show_local_data(local_data_path,show =False):

    if os.path.exists(local_data_path):
        print('该地址文件存在：')
        with open(local_data_path,'rb') as f:
            res_pic = pickle.load(f)
            print(type(res_pic))
            print(f'{local_data_path}\n本地的内容：',res_pic.keys())
            if show == True:
                for k,v in res_pic.items():
                    print(k)
                    print(v.keys())
                    print(v.sample(50))

                    exit()
                    v.sort_values(by='s_time',inplace=True)
                    print(v.head(1))
                    print(v.tail(1))
                    print(set(sorted(v['s_time'].tolist())))

                    print(k,v.shape)
                    print('\n===========')
        return res_pic
    else:
        print('本地没有该文件：保存。。。')
        return None

# 拿到最新数据保存到本地，时间条件更新。
def newDBdata_to_local(local_res_path, moxing=['04'], new_time=datetime.datetime(2020, 7, 1)):
    local_res_path0 = local_res_path.split('\\')[-1].split('.')[0]
    time0 = time.strftime("%m-%d", time.localtime())
    init()
    all = StrategyCode.objects(class_name__exists=True)
    all0 = [a.class_name for a in all]
    result_pickle = {}

    # 打开本地文件
    with open(local_res_path, 'rb') as f0:
        local_res = pickle.load(f0)
        print('本地keys：', local_res.keys())
        # 遍历所有策略名
        for a0 in all0:
            res = StrategiesSet.objects(class_name=a0, result__ne={})
            all = StrategiesSet.objects(class_name=a0)
            print(f'策略：{a0}， 已完成：{res.count()}， 总共：{all.count()}, 完成率：{float(res.count()) * 100 / float(all.count())}%')
            # 开始，进行数据加工
            if res.count() == all.count():

                if a0.split('_')[-1] not in moxing: continue
                print(a0)
                # 大于newtime的数据，==目标跟新数据。
                all = StrategiesSet.objects(class_name=a0, end__gt=new_time)
                result_dict = {}
                i = 0
                cols = []
                for index_, s0 in enumerate(all):
                    result_dict0 = {}
                    result_dict0['celue_name'] = s0.class_name + str('_') + str(s0.setting['zhouqi'])
                    result_dict0['s_time'] = s0.start
                    result_dict0['para1'] = s0.setting['ma_len']
                    result_dict0['para2'] = s0.setting['bd']
                    result_dict0['para3'] = s0.setting['dk_len']
                    result_dict0['para4'] = s0.setting['acc']
                    result_dict0['canshu'] = [s0.setting['ma_len'], s0.setting['bd'], s0.setting['dk_len'],
                                              s0.setting['acc']]
                    # 赋值结果
                    if s0.result['win_rate'] != None:
                        result_dict0['win_rate'] = s0.result['win_rate']
                        result_dict0['win_mean_num'] = s0.result['win_mean_num']
                        result_dict0['loss_mean_num'] = s0.result['loss_mean_num']
                        result_dict0['max_num'] = s0.result['max_num']
                        result_dict0['min_num'] = s0.result['min_num']
                        result_dict0['mean_num'] = s0.result['mean_num']
                        result_dict0['std_num'] = s0.result['std_num']
                        result_dict0['max_sum'] = s0.result['max_sum']
                        result_dict0['end'] = s0.result['total_net_pnl']
                        result_dict0['max_back'] = s0.result['max_drawdown']
                        result_dict0['trade_nums'] = s0.result['total_trade_count']
                        result_dict0['sharp_rate'] = s0.result['sharpe_ratio']
                        result_dict0['total_days'] = s0.result['total_days']
                        result_dict0['profit_days'] = s0.result['profit_days']
                        result_dict0['total_slippage'] = s0.result['total_slippage']
                        result_dict0['win_rate'] = s0.result['win_rate']
                        result_dict0['win_mean_num'] = s0.result['win_mean_num']
                        result_dict0['win_mean_num'] = s0.result['loss_mean_num']
                        result_dict0['max_num'] = s0.result['max_num']
                        result_dict0['min_num'] = s0.result['min_num']
                        result_dict0['mean_num'] = s0.result['mean_num']
                        result_dict0['std_num'] = s0.result['std_num']
                        cols = result_dict0.keys()

                    result_dict[index_] = result_dict0

                df_res0 = pd.DataFrame(result_dict).T
                df_res0 = df_res0[list(cols)]
                # print(df_res0.tail())
                # exit()
                for name in [c for c in list(set(df_res0['celue_name'].tolist()))]:
                    todata = df_res0.loc[df_res0['celue_name'] == name].copy()
                    todata.sort_values(by='s_time', ascending=True, inplace=True)
                    result_pickle[str(name)] = todata
                    print(f'{name}，已生成！')
                # result_pickle[str(a0)] = df_res0
            else:
                print(f'{a0}，没有完成！')

        if len(result_pickle.keys()) == 0:
            print('无数据，结束。 ')
            exit()

        print('当前的key：', result_pickle.keys())
        res_in = input('确定保存：y/n')
        if res_in == 'y':
            res_path = local_res_path0 + time0
            with open(res_path, 'wb') as f:
                pickle.dump(result_pickle, f)
                print('dict 已保存，keys：', result_pickle.keys())
                print(f'数据存放地址：》{res_path}')

        else:
            print('没有保存。')



    return result_pickle

# 最新数据对应更新到本地
def newdata_updata_dfs(new_data, local_res_path):
    '''
    new_data数据是{策略名：最新数据}
    :param new_data:
    :param local_res_path:
    :return:
    '''

    local_res_path0 = local_res_path.splie('\\')[-1].splite('.')[0]

    # 读取目标策略集的策略
    with open(local_res_path, 'rb') as f0:
        local_res = pickle.load(f0)
        print('本地keys：', local_res.keys())
    # 对本地数据的策略每个进行判断
    for k, v in local_res.items():
        # 最新数据的开始月份 》》本地数据结束月份
        if new_data[k].iloc[0]['s_time'] > v.iloc[-1]['s_time']:
            print(f'{k},数据判断存在。')
        else:
            print(f'{k}策略，存在于本地数据！')
            continue

    time0 = time.strftime("%m-%d", time.localtime())
    with open(local_res_path0 + time0, 'wb') as f:
        pickle.dump(local_res, f)
        print('dict 已保存，keys：', local_res.keys())
        print(f'数据存放地址：》{local_res_path0 + time0}')

    return local_res_path0 + time0



# 更新数据保存到本地
def updata_to_local(local_path,updatas_names=[],new_time=''):
    '''添加本地对应策略回测记录:
    1.连接数据库。拿到最新数据信息。
    2.将最新数据保存到本地
    updatas = {clm,:data=df,
                clm,:data=df,
                    }
    '''
    time0 = time.strftime("%m-%d", time.localtime())
    init()
    all = StrategyCode.objects(class_name__exists=True)
    all0 = [a.class_name for a in all]
    result_pickle = {}
    for a0 in updatas_names:
        res = StrategiesSet.objects(class_name=a0, result__ne={})
        all = StrategiesSet.objects(class_name=a0)
        print(f'策略：{a0}， 已完成：{res.count()}， 总共：{all.count()}, 完成率：{float(res.count()) * 100 / float(all.count())}%')
        # 开始，进行数据加工
        if res.count() == all.count():
            print(a0,'可以更新！')
            # 大于newtime的数据，==目标跟新数据。
            all = StrategiesSet.objects(class_name=a0, end__gt=new_time)
            result_dict = {}
            cols = []
            for index_, s0 in enumerate(all):
                result_dict0 = {}
                result_dict0['celue_name'] = s0.class_name + str('_') + str(s0.setting['zhouqi'])
                result_dict0['s_time'] = s0.start
                result_dict0['para1'] = s0.setting['ma_len']
                result_dict0['para2'] = s0.setting['bd']
                result_dict0['para3'] = s0.setting['dk_len']
                result_dict0['para4'] = s0.setting['acc']
                result_dict0['canshu'] = [s0.setting['ma_len'], s0.setting['bd'], s0.setting['dk_len'],
                                          s0.setting['acc']]
                # 赋值结果
                if s0.result['win_rate'] != None:
                    result_dict0['win_rate'] = s0.result['win_rate']
                    result_dict0['win_mean_num'] = s0.result['win_mean_num']
                    result_dict0['loss_mean_num'] = s0.result['loss_mean_num']
                    result_dict0['max_num'] = s0.result['max_num']
                    result_dict0['min_num'] = s0.result['min_num']
                    result_dict0['mean_num'] = s0.result['mean_num']
                    result_dict0['std_num'] = s0.result['std_num']
                    result_dict0['max_sum'] = s0.result['max_sum']
                    result_dict0['end'] = s0.result['total_net_pnl']
                    result_dict0['max_back'] = s0.result['max_drawdown']
                    result_dict0['trade_nums'] = s0.result['total_trade_count']
                    result_dict0['sharp_rate'] = s0.result['sharpe_ratio']
                    result_dict0['total_days'] = s0.result['total_days']
                    result_dict0['profit_days'] = s0.result['profit_days']
                    result_dict0['total_slippage'] = s0.result['total_slippage']
                    result_dict0['win_rate'] = s0.result['win_rate']
                    result_dict0['win_mean_num'] = s0.result['win_mean_num']
                    result_dict0['win_mean_num'] = s0.result['loss_mean_num']
                    result_dict0['max_num'] = s0.result['max_num']
                    result_dict0['min_num'] = s0.result['min_num']
                    result_dict0['mean_num'] = s0.result['mean_num']
                    result_dict0['std_num'] = s0.result['std_num']
                    cols = result_dict0.keys()

                result_dict[index_] = result_dict0
            df_res0 = pd.DataFrame(result_dict).T
            # 生成最终的df数据
            df_res0 = df_res0[list(cols)]
            # print(df_res0.tail())
            # exit()
            # 此包含多个时间周期的策略，要分离
            for name in [c for c in list(set(df_res0['celue_name'].tolist()))]:
                todata = df_res0.loc[df_res0['celue_name'] == name].copy()
                todata.sort_values(by='s_time', ascending=True, inplace=True)
                result_pickle[str(name)] = todata
                print(f'{name}，已生成！')
                # result_pickle[str(a0)] = df_res0
        else:
            print(f'{a0}，没有回测完成！')


    if len(result_pickle.keys()) == 0:
        print('无数据更新，为空，退出。 ')
        exit()

    print('当前的已经更新策略名字：\n', result_pickle.keys())

    with open(local_path,'rb') as lf:
        local_df = pickle.load(lf)

    for clm in result_pickle.keys():
        if clm in local_df.keys():
            new_data  = pd.DataFrame(local_df[clm]).append(result_pickle[clm],ignore_index=True)
            new_data.drop_duplicates(subset=['s_time','canshu'],keep='first',inplace=True)
            local_df[clm] = new_data
        else:
            local_df[clm] = result_pickle[clm]
    with open(local_path,'wb') as tof:
        pickle.dump(local_df,tof)






if __name__ == '__main__':

    res_pickle_path0 = r'F:\new_0811\DB_in_out_strategies\res_df_in_dict_to0309-07.pickle'
    #查看DBdata
    # show_DB_data()

    # 更新
    # newDBdata_to_local(local_res_path=res_path , moxing=['03'], new_time=datetime.datetime(2020, 6, 1))





    #重新保存新文件模型。
    # #转换DB文件成为df，保存到res，，
    res_path = os.getcwd() + r'\res_df_in_dict_to.pickle'
    to_pickle_data = DBdata_to_dfs(res_path,moxing=['04'])
    # res_pickle_path = r'F:\new_0811\DB_in_out_strategies\res_df_in_dict_to0309-07.pickle'
    # with open(res_pickle_path,'rb') as f:
    #     to_pickle_data = pickle.load(f)
    #     print(to_pickle_data.keys())
    #     print(to_pickle_data['T3_tp_02_3'].iloc[-1]['s_time'])
    # save_to_local(to_pickle_data, to_pickle_flag='MEBD策略模型')
    # show_local_data(r'F:\new_0811\DB_in_out_strategies\MEBD策略模型_01.pickle',show=True)









