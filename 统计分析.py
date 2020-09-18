import datetime
import time
import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import scipy.stats as stats

pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 斯皮尔曼秩相关系数和p检验
stats.spearmanr(a=[],b=[])

def plot_fenbutu(a, b):


    fig = plt.figure(figsize = (10,6))
    ax1 = fig.add_subplot(2,1,1)  # 创建子图1
    ax1.scatter(a,b)
    plt.grid()
    # 绘制数据分布图

    # 线性拟合，可以返回斜率，截距，r 值，p 值，标准误差
    k_value, b_value, r_value, p_value, std_err = stats.linregress(a, b)
    ax1.plot(a, a*k_value+b_value, color='green', label='other')

    ax2 = fig.add_subplot(2,1,2)  # 创建子图2
    a.hist(bins=30, alpha=0.5, ax=ax2)
    a.plot(kind='kde', secondary_y=True, ax=ax2,c='g')
    b.hist(bins=30,alpha = 0.5,ax = ax2)
    b.plot(kind = 'kde', secondary_y=True,ax = ax2,c='r')
    plt.grid()
    plt.show()
    # 绘制直方图
    # 呈现较明显的正太性


def plot_fenbutu02(a,b,c):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)  # 创建子图1
    ax1.scatter(a, b,c='r')
    ax1.scatter(a, c,c='g')


    plt.grid()
    plt.show()
    # 绘制数据分布图

    # # 线性拟合，可以返回斜率，截距，r 值，p 值，标准误差
    # k_value, b_value, r_value, p_value, std_err = stats.linregress(a, b)
    # ax1.plot(a, a * k_value + b_value, color='green', label='other')

def plot_3d(data):
    fig = plt.figure()
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率
    ax = fig.add_subplot(111, projection='3d')


    flag =['*','+','^','o','*','+','^','o','*','+','^','o','*','+','^','o']
    color_list = ['r','b','y','g','pink','black','#3d0101','#005001','#353541','#300541',]
    for i, z0 in enumerate(set(data['s_Time'].tolist())):
        print(i,z0)

        ax.scatter(xs=0, ys=data[data['s_Time']==z0]['预测周期真实收益'], zs=data[data['s_Time']==z0]['预测值'], c=color_list[i], s=30, alpha=1, label='', marker=flag[i])

    ax.set_xticklabels(list(set(data['s_Time'].tolist())), fontsize=10)
    ax.set_yticklabels([" ", " ", "predict_price", " ", " "], fontsize=10)
    ax.set_zlabel('True_price', fontsize=16)

    plt.tight_layout(rect=(0, 0, 1, 1))
    # plt.savefig('student_score.pdf')
    plt.show()

import random


def echart_plot_3d(data):
    from pyecharts.charts import Bar3D
    from pyecharts import options as opts


    print(data.tail())

    df = data[['s_Time', '预测周期真实收益' ,'预测值' ]].values.tolist()



    (Bar3D(init_opts=opts.InitOpts(width="1600px", height="800px"))
    .add(
        "",
        df,
        xaxis3d_opts=opts.Axis3DOpts( type_="category"),
        yaxis3d_opts=opts.Axis3DOpts( type_="value"),
        zaxis3d_opts=opts.Axis3DOpts( type_="value"),
    )
    .set_global_opts(
        visualmap_opts=opts.VisualMapOpts(max_=100000),
        title_opts=opts.TitleOpts(title="predict"),
    )
    .render("predict.html")
)

def dong_scatter(data,info='',path0=''):
    from pyecharts import options as opts
    from pyecharts.commons.utils import JsCode
    from pyecharts.charts import Scatter, Timeline
    # print(data.columns)
    title = data.iloc[-1]['策略']

    df = data
    data['测试集_真实分类'] = data['测试集_真实分类'].apply(lambda x: int(x))
    print(data['测试集_真实分类'].values)
    min_pre = min(data['测试集_真实分类'].values.tolist())
    max_pre = max(data['测试集_真实分类'].values.tolist())
    df['预测周期真实收益']=df['预测周期真实收益'].apply(lambda x: int(x))
    df['预测值']=df['预测值'].apply(lambda x: int(x))

    df['s_Time'] = pd.to_datetime(df['s_Time'])#.apply(lambda x:x.strftime(format="%Y-%m-%d"))
    df.sort_values(by=['s_Time'], ascending=True, inplace=True)
    tl = Timeline()
    timelist = list(set(df['s_Time'].values.tolist()))
    list.sort(timelist)
    df_date = [time.strftime('%Y-%m-%d',time.localtime(i/1000000000) ) for i in timelist]
    print(df_date)

    for k,i in enumerate(df_date):
        # print(k,i)
        xdata = df.loc[df['s_Time'] == i, '预测值'].values.tolist()
        ydata = df.loc[df['s_Time']==i,['预测周期真实收益','预测值']].values.tolist()
        # print(ydata)
        Scatter0 = (
            Scatter()
            .add_xaxis(xdata)
            .add_yaxis('预测周期真实收益',ydata,label_opts = opts.LabelOpts(is_show=False))
            .set_series_opts()

            .set_global_opts(
                xaxis_opts=opts.AxisOpts(name = '预测值：',type_="value",axistick_opts=opts.AxisTickOpts(is_show=True)),
                yaxis_opts=opts.AxisOpts(name = '真实值：',type_="value",axistick_opts=opts.AxisTickOpts(is_show=True)),
                title_opts =opts.TitleOpts(title=f"{title}==:{i}月份的数据"),

                tooltip_opts = opts.TooltipOpts(formatter=JsCode("function (params) { return '真实：'+params.value[1] +' == 预测：'+ params.value[2];}")),
                visualmap_opts=opts.VisualMapOpts(min_=min_pre,max_= max_pre),
                ))
        tl.add(Scatter0, "{}月".format(i))
    tl.render(path0+f"{info}.html")
    print(path0+f"{info}.html")

def corr_plot(data:list):
    from pyecharts import options as opts
    from pyecharts.charts import HeatMap
    from pyecharts.faker import Faker

    corr = (
        HeatMap()
        .add_xaxis('series0')
        .add_yaxis(
            "series1",
            data,
            label_opts=opts.LabelOpts(is_show=True, position="inside"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="HeatMap-Label 显示"),
            visualmap_opts=opts.VisualMapOpts(),
        )
        .render("相关性图.html")
    )