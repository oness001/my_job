import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
import traceback
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import metrics

def polynomial_regression(X_train=[],y_train=[],X_test=[],y_test=[], degree=1, include_bias=False, normalize =False):
    # 构建多项式特征
    polynomial_features = PolynomialFeatures(degree=degree,include_bias=include_bias)
    # 线性回归模型的数据正则化，（普通最小二乘线性回归。）
    linear_regression = LinearRegression(normalize=True)
    # 添加队列
    pipeline_model = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])

    pipeline_model.fit(X_train, y_train)
    train_score = pipeline_model.score(X_train, y_train)
    print(f'测试得分：{train_score}')
    # 预测数据
    pre_y = pipeline_model.predict(X_test)
    if len(y_test) != 0:
        cv_score = pipeline_model.score(X_test, y_test)
        # 评估当前模型
        print("平均绝对值误差:", metrics.mean_absolute_error(y_test, pre_y))
        print("平均平方误差:", metrics.mean_squared_error(y_test, pre_y))
        print("中位绝对值误差:", metrics.median_absolute_error(y_test, pre_y))
        print("R2得分:", metrics.r2_score(y_test, pre_y))
        print(f'预测得分：{cv_score}')
        return pipeline_model,[y_test,pre_y]

    X_test['predict_res'] = pd.DataFrame(pre_y)
    return pipeline_model ,X_test

def pol_ridge_regression(X_train=[],y_train=[],X_test=[],y_test=[], degree=1, include_bias=False,alpha=0.5):
    # 构建多项式特征
    polynomial_features = PolynomialFeatures(degree=degree,include_bias=include_bias)
    # 线性回归模型的数据正则化，（普通最小二乘线性回归。）

    ridge_regression = Ridge(alpha=alpha)
    # 添加队列
    pipeline_model = Pipeline([("polynomial_features", polynomial_features),
                         ("ridge_regression", ridge_regression)])

    pipeline_model.fit(X_train, y_train)
    train_score = pipeline_model.score(X_train, y_train)
    print(f'测试得分：{train_score}')
    # 预测数据
    pre_y = pipeline_model.predict(X_test)
    if len(y_test) != 0:
        cv_score = pipeline_model.score(X_test, y_test)
        # 评估当前模型
        print("平均绝对值误差:", metrics.mean_absolute_error(y_test, pre_y))
        print("平均平方误差:", metrics.mean_squared_error(y_test, pre_y))
        print("中位绝对值误差:", metrics.median_absolute_error(y_test, pre_y))
        print("R2得分:", metrics.r2_score(y_test, pre_y))
        print(f'预测得分：{cv_score}')
        return pipeline_model,[y_test,pre_y]

    X_test['predict_res'] = pd.DataFrame(pre_y)
    return pipeline_model ,X_test

def pol_Bayes_regression(X_train=[],yinzi =[],y_train=[],X_test=[],y_test=[], degree=1, include_bias=False,normalize=True):
    # 构建多项式特征
    polynomial_features = PolynomialFeatures(degree=degree,include_bias=include_bias)
    Bayesian_Ridge = BayesianRidge(normalize=normalize)

    # linear_regression = ridge_regression(alpha=alpha)
    # 添加队列
    pipeline_model = Pipeline([("polynomial_features", polynomial_features),
                         ("Bayesian_Ridge", Bayesian_Ridge)])

    pipeline_model.fit(X_train[yinzi], y_train)
    train_score = pipeline_model.score(X_train[yinzi], y_train)
    print(f'测试得分：{train_score}')
    # 预测数据
    pre_y = pipeline_model.predict(X_test)
    if len(y_test) != 0:
        cv_score = pipeline_model.score(X_test, y_test)
        # 评估当前模型
        print("平均绝对值误差:", metrics.mean_absolute_error(y_test, pre_y))
        print("平均平方误差:", metrics.mean_squared_error(y_test, pre_y))
        print("中位绝对值误差:", metrics.median_absolute_error(y_test, pre_y))
        print("R2得分:", metrics.r2_score(y_test, pre_y))
        print(f'预测得分：{cv_score}')
        return pipeline_model,[y_test,pre_y]

    X_test['predict_res'] = pd.DataFrame(pre_y)
    return pipeline_model ,X_test

def polynomial_Logistic_classify(X_train=pd.DataFrame(),y_train=pd.DataFrame(),X_test=pd.DataFrame(),y_test=pd.DataFrame(), degree=1,**kwarg):
    # 构建多项式特征
    polynomial_features = PolynomialFeatures(degree=degree,include_bias=False)
    Logistic_Regression = LogisticRegression(**kwarg)
    # 添加队列
    pipeline_model = Pipeline([("polynomial_features", polynomial_features),
                         ("Logistic_Regression", Logistic_Regression)])

    pipeline_model.fit(X_train, y_train)
    train_score = pipeline_model.score(X_train, y_train)
    print(f'测试得分：{train_score}')
    pre_y = pipeline_model.predict(X_test)
    if len(y_test) != 0:
        pre_y = pipeline_model.predict(X_test)
        cv_score = pipeline_model.score(X_test, y_test)
        # 评估当前模型
        print("平均绝对值误差:", metrics.mean_absolute_error(y_test, pre_y))
        print("平均平方误差:", metrics.mean_squared_error(y_test, pre_y))
        print("中位绝对值误差:", metrics.median_absolute_error(y_test, pre_y))
        print("R2得分:", metrics.r2_score(y_test, pre_y))
        print(f'预测得分：{cv_score}')
        return pipeline_model,[y_test,pre_y]
    X_test['predict_res'] = pd.DataFrame(pre_y)

    return pipeline_model ,X_test

def pol_kn_classify(X_train=pd.DataFrame(),y_train=pd.DataFrame(),X_test=pd.DataFrame(),y_test=pd.DataFrame(), degree=1,**kwarg):
    '''
    n_neighbors=5, weights='distance', p = 2, algorithm='auto', leaf_size=30

    '''
    # 构建多项式特征
    polynomial_features = PolynomialFeatures(degree=degree,include_bias=False)
    KNeighbors_Classifier = KNeighborsClassifier(**kwarg)
    # 添加队列
    pipeline_model = Pipeline([("polynomial_features", polynomial_features),
                         ("KNeighbors_Classifier", KNeighbors_Classifier)])

    pipeline_model.fit(X_train, y_train)
    train_score = pipeline_model.score(X_train, y_train)
    print(f'测试得分：{train_score}')
    pre_y = pipeline_model.predict(X_test)
    if len(y_test) != 0:
        pre_y = pipeline_model.predict(X_test)
        cv_score = pipeline_model.score(X_test, y_test)
        # 评估当前模型
        print("平均绝对值误差:", metrics.mean_absolute_error(y_test, pre_y))
        print("平均平方误差:", metrics.mean_squared_error(y_test, pre_y))
        print("中位绝对值误差:", metrics.median_absolute_error(y_test, pre_y))
        print("R2得分:", metrics.r2_score(y_test, pre_y))
        print(f'预测得分：{cv_score}')
        return pipeline_model,[y_test,pre_y]
    X_test['predict_res'] = pd.DataFrame(pre_y)

    return pipeline_model ,X_test

def polynomial_Logistic_classify0(X_train=pd.DataFrame(),y_train=pd.DataFrame(), degree=1,**kwarg):


    # 构建多项式特征
    polynomial_features = PolynomialFeatures(degree=degree,include_bias=False)
    Logistic_Regression = LogisticRegression(**kwarg)
    # 添加队列
    pipeline_model = Pipeline([("polynomial_features", polynomial_features),
                         ("Logistic_Regression", Logistic_Regression)])

    pipeline_model.fit(X_train, y_train)
    train_score = pipeline_model.score(X_train, y_train)
    print(f'二分类，测试得分：{train_score}')

    pre = pipeline_model.predict(X_train)
    X_train = pd.DataFrame(X_train)

    X_train.loc[:,'预测值'] = pd.Series(pre,index=X_train.index)


    return pipeline_model,X_train

def polynomial_Logistic_classify1(X_train=[],yinzi=[],y_train=[], degree=1,**kwarg):


    # 构建多项式特征
    polynomial_features = PolynomialFeatures(degree=degree,include_bias=False)
    Logistic_Regression = LogisticRegression(**kwarg)
    # 添加队列
    pipeline_model = Pipeline([("polynomial_features", polynomial_features),
                         ("Logistic_Regression", Logistic_Regression)])

    pipeline_model.fit(X_train[yinzi], y_train)
    train_score = pipeline_model.score(X_train[yinzi], y_train)
    print(f'二分类，测试得分：{train_score}')

    pre = pipeline_model.predict(X_train[yinzi])

    X_train['预测值'] = pre


    return pipeline_model,X_train ,train_score

def polynomial_regression0(X_train=[],yinzi=[],y_train=[], degree=1, include_bias=False, normalize =False):
    # 构建多项式特征
    polynomial_features = PolynomialFeatures(degree=degree,include_bias=include_bias)
    # 线性回归模型的数据正则化，（普通最小二乘线性回归。）
    linear_regression = LinearRegression(normalize=True)
    # 添加队列
    pipeline_model = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline_model.fit(X_train[yinzi], y_train)
    train_score = pipeline_model.score(X_train[yinzi], y_train)
    print(f'最小二乘法，回归测试得分：{train_score}')
    # 预测数据
    pre = pipeline_model.predict(X_train[yinzi])
    X_train = pd.DataFrame(X_train)
    X_train['预测值'] = pre
    # X_train.loc[X_train.index,'预测值'] = pre #pd.Series(pre,index=X_train.index)

    return pipeline_model,X_train ,train_score

def svc_classify0(X_train=pd.DataFrame(),yinzi = [],y_train=pd.DataFrame(),kernel = 'poly',degree=2,c=0.8):
    from sklearn.svm import SVC

    '''

    :param X_train:
    :param y_train:
    :param degree:
    :param kwarg:
    kernel = 'poly',degree=2,c=0.8,
                kernel = 'poly',核函数采用多项式形式。
                degree=2, 多项式对应的多项式的最高阶数
                c=0.8,误差处理，软间隔的处理。
                   
    :return:
    '''

    # 构建多项式特征
    svc = SVC(kernel = kernel,degree=degree,C=c)
    svc.fit(X_train[yinzi], y_train)
    train_score = svc.score(X_train[yinzi], y_train)
    print(f'svc，测试得分：{train_score}')

    pre = svc.predict(X_train[yinzi])
    X_train = pd.DataFrame(X_train)
    X_train.loc[:,'预测值'] = pd.Series(pre,index=X_train.index)

    return svc,X_train,train_score

def randomforest_classify0(X_train=pd.DataFrame(),yinzi=[],y_train=pd.DataFrame(),**kwarg):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_validate

    '''

    :param X_train:
    :param y_train:
    :param degree:
    :param kwarg:
    n_estimators=50,criterion='gini', max_depth=3,max_features='log2', min_samples_split=10,random_state=0
                n_estimators=50, 决策树的个数
                criterion='gini', max_depth=3,每棵分裂深度
                                 max_features='log2', 特征个数
                                 min_samples_split=10, 分裂最小的样本数
                                 random_state=0
    :return:
    '''

    rfcl = RandomForestClassifier(**kwarg)
    scores = cross_validate(rfcl, X_train[yinzi], y_train, cv=10)
    rfcl.fit(X_train[yinzi], y_train)
    print(f"random_forest，测试得分：{scores['test_score'].mean()}")

    pre = rfcl.predict(X_train[yinzi])
    X_train = pd.DataFrame(X_train)
    X_train.loc[:,'预测值'] = pd.Series(pre,index=X_train.index)


    return rfcl,X_train,scores['test_score'].mean()

def poly_ridge_regression(X_train=[], yinzi=[], y_train=[], degree=1, **kwarg):
    from sklearn.model_selection import cross_validate
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import  Ridge

    # 构建多项式特征
    polynomial_features = PolynomialFeatures(degree=degree)
    # 线性回归模型的数据正则化，（普通最小二乘线性回归。）

    ridge_regression = Ridge(alpha=0.5)
    # 添加队列
    pipeline_model = Pipeline([("polynomial_features", polynomial_features),
                               ("ridge_regression", ridge_regression)])
    scores = cross_validate(pipeline_model, X_train[yinzi], y_train, cv=5)

    pipeline_model = pipeline_model.fit(X_train[yinzi], y_train)
    print(f'ridge，平均回归测试得分：{scores["test_score"].mean()}')
    # 预测数据
    pre = pipeline_model.predict(X_train[yinzi])

    X_train['预测值'] = pre
    return pipeline_model, X_train, scores["test_score"].mean()

def lars_regression(X_train=[],yinzi=[],y_train=[], degree=1):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LassoLars
    from sklearn.model_selection import cross_validate


    # 构建多项式特征
    polynomial_features = PolynomialFeatures(degree=degree)
    # （lars回归。）
    LassoLars_model = LassoLars(alpha=0.95,normalize=True)
    # 添加队列
    pipeline_model = Pipeline([("polynomial_features", polynomial_features),
                         ("LassoLars_model", LassoLars_model)])
    if X_train.shape[0]>5:
        scores = cross_validate(pipeline_model, X_train[yinzi], y_train, cv=5)
    else:
        scores = cross_validate(pipeline_model, X_train[yinzi], y_train, cv=2)
    pipeline_model = pipeline_model.fit(X_train[yinzi], y_train)
    print(f'lars，回归测试得分：{scores["test_score"].mean()}')
    # 预测数据
    pre = pipeline_model.predict(X_train[yinzi])
    X_train = pd.DataFrame(X_train)
    X_train['预测值'] = pre
    # X_train.loc[X_train.index,'预测值'] = pre #pd.Series(pre,index=X_train.index)

    return pipeline_model,X_train ,scores["test_score"].mean()

def extraforest_classify0(X_train=pd.DataFrame(),yinzi=[],y_train=pd.DataFrame(),**kwarg):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import cross_validate

    '''

    :param X_train:
    :param y_train:
    :param degree:
    :param kwarg:
                    n_estimators=50, 决策树的个数
                    criterion='gini', max_depth=3,每棵分裂深度
                                 max_features='log2', 特征个数
                                 min_samples_split=10, 分裂最小的样本数
                                 random_state=0
    :return:
    '''
    try:
        rfcl = ExtraTreesClassifier(**kwarg)
        scores = cross_validate(rfcl, X_train[yinzi], y_train, cv=10)
        rfcl.fit(X_train[yinzi], y_train)
        print(f"exf，测试得分：{scores['test_score'].mean()}")

        pre = rfcl.predict(X_train[yinzi])
        X_train = pd.DataFrame(X_train)
        X_train.loc[:,'预测值'] = pd.Series(pre,index=X_train.index)
    except Exception as e:
        print('ExtraTreesClassifier:出错。')
        print(traceback.format_exc())

        print(e)


    return rfcl,X_train,scores['test_score'].mean()

def polynomial_regression_grid(X_train=[],yinzi=[],y_train=[],n_iter=3, n_jobs=2):

    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import RandomizedSearchCV

    # 构建多项式特征
    pip_list = []

    for i in range(2,5,1):
        polynomial_features = PolynomialFeatures(degree=i)
        for j in [False,True]:
            linear_regression = LinearRegression(normalize=j)
            pip0 = [("polynomial_features", polynomial_features),
             ("linear_regression", linear_regression)]
            pip_list.append(pip0)

    param_grid = {'steps':pip_list}

    poly_rg_grid = RandomizedSearchCV(Pipeline(), param_grid, n_iter=n_iter, cv=5, n_jobs=n_jobs)
    poly_rg_grid = poly_rg_grid.fit(X_train[yinzi], y_train)
    poly_rg_best = poly_rg_grid.best_estimator_

    print(f'polygrid，回归测试得分：{poly_rg_grid.best_score_}')
    # 预测数据
    pre = poly_rg_best.predict(X_train[yinzi])
    X_train = pd.DataFrame(X_train)
    X_train['预测值'] = pre
    return poly_rg_best, X_train, poly_rg_grid.best_score_

def ada_dt_regression_grid(X_train=[],yinzi=[],y_train=[], n_jobs=2):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.model_selection import RandomizedSearchCV

    param_grid =  {'base_estimator':[DecisionTreeRegressor(max_depth=i) for i in range(3,6,1) ],
               "n_estimators": [ i for i in range(75,141,20) ]}
    if X_train.shape[0]>5:
        grid = RandomizedSearchCV(AdaBoostRegressor(), param_grid,n_iter=3, cv=5,n_jobs=n_jobs)
    else:grid = RandomizedSearchCV(AdaBoostRegressor(),param_grid,n_iter=3, cv=2,n_jobs=n_jobs)

    grid = grid.fit(X_train[yinzi], y_train)
    regr_best = grid.best_estimator_

    print(f'ada_grid，回归测试得分：{grid.best_score_}')
    # 预测数据
    pre = regr_best.predict(X_train[yinzi])
    X_train = pd.DataFrame(X_train)
    X_train['预测值'] = pre

    return regr_best,X_train ,grid.best_score_

def bag_svc(X_train=pd.DataFrame(),yinzi=[],y_train=pd.DataFrame(),n_iter=5,n_jobs=2):
    from sklearn.svm import SVC
    from sklearn.ensemble import BaggingClassifier

    clfb = BaggingClassifier(base_estimator=SVC(),n_estimators=n_iter,max_samples=0.6,max_features=0.4,bootstrap=True,
                        bootstrap_features=True, random_state=0)
    clfb = clfb.fit(X_train[yinzi], y_train)
    score0 = clfb.score(X_train[yinzi], y_train)
    print(f"bag_svc，得分：{score0}")
    pre = clfb.predict(X_train[yinzi])
    X_train = pd.DataFrame(X_train)
    X_train.loc[:,'预测值'] = pd.Series(pre,index=X_train.index)
    return clfb,X_train,score0

def bag_svr(X_train=pd.DataFrame(), yinzi=[], y_train=pd.DataFrame(), n_iter=10, n_jobs=2):
    from sklearn.svm import SVR
    from sklearn.ensemble import BaggingRegressor

    bag_rg = BaggingRegressor(base_estimator=SVR(), n_estimators=n_iter, max_samples=0.5, max_features=0.5,
                             bootstrap=True,bootstrap_features=True, random_state=0,n_jobs =n_jobs)
    bag_rg = bag_rg.fit(X_train[yinzi], y_train)
    score0=bag_rg.score(X_train[yinzi], y_train)
    print(f"bag_rg，得分：{score0}")
    pre = bag_rg.predict(X_train[yinzi])
    X_train = pd.DataFrame(X_train)
    X_train.loc[:,'预测值'] = pd.Series(pre,index=X_train.index)
    return bag_rg,X_train,score0

def k_pac(X_train=pd.DataFrame(),yinzi=[],N=10,kernel=5,n_jobs=2):
    from sklearn.decomposition import KernelPCA
    transformer = KernelPCA(n_components=10, kernel='rbf', n_jobs=1)
    X_transformed = transformer.fit_transform(X_train[yinzi])



#==最新
def randomforest_classify_grid(X_train=pd.DataFrame(),yinzi=[],y_train=pd.DataFrame(),n_iter=5,n_jobs=2):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    res_col = str(y_train.name)

    param_grid={'n_estimators' : [120,140,150,200], 'criterion' : ['gini'], 'max_depth' : [5,6,7,8],
    'max_features' : ['log2','sqrt'] ,'min_samples_split' : [3]}

    grid = RandomizedSearchCV(RandomForestClassifier(), param_grid,n_iter=n_iter, cv=5,n_jobs=n_jobs)
    grid.fit(X_train[yinzi], y_train)
    best_grid = grid.best_estimator_

    pre = best_grid.predict(X_train[yinzi])
    X_train = pd.DataFrame(X_train)
    X_train.loc[:,'预测值'] = pd.Series(pre,index=X_train.index)

    new = X_train[['预测值', res_col]].sample(int((X_train.shape[0]) * 0.618))

    sj_acc_score = accuracy_score(new[res_col], new['预测值'])
    sj_ba_score = balanced_accuracy_score(new[res_col], new['预测值'])

    print(f"random_forest，随机遍历最优得分：{[grid.best_score_,sj_acc_score,sj_ba_score]}")
    return best_grid,X_train,[grid.best_score_,sj_acc_score,sj_ba_score]

def adaboosting_dtree_grid(X_train=pd.DataFrame(),yinzi=[],y_train=pd.DataFrame(),n_iter=5,n_jobs=2):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    res_col = str(y_train.name)

    b_estimator_list = []
    for n in range(3, 7, 1):
        b_estimator_list.append(DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=n))
    if len(b_estimator_list) == 0:
        raise
    param_grid = {'base_estimator': b_estimator_list,
                  'n_estimators': [80, 100,120,150,180]}
    adadtcl_grid = RandomizedSearchCV(AdaBoostClassifier(), param_grid, n_iter=n_iter, cv=6, n_jobs=n_jobs)
    adadtcl_grid.fit(X_train[yinzi], y_train)
    adadtcl_best = adadtcl_grid.best_estimator_

    pre = adadtcl_best.predict(X_train[yinzi])
    X_train = pd.DataFrame(X_train)
    X_train.loc[:,'预测值'] = pd.Series(pre,index=X_train.index)

    new  = X_train[['预测值',res_col]].sample(int((X_train.shape[0])*0.618))
    sj_acc_score = accuracy_score(new[res_col], new['预测值'])
    sj_ba_score = balanced_accuracy_score(new[res_col], new['预测值'])

    print(f"adaboost_d_tree，随机遍历最优得分：{[adadtcl_grid.best_score_,sj_acc_score,sj_ba_score]}")
    return adadtcl_best,X_train,[adadtcl_grid.best_score_,sj_acc_score,sj_ba_score]

def svc_classify_grid(X_train=pd.DataFrame(), yinzi=[], y_train=pd.DataFrame(),n_iter=5,n_jobs=2):
    from sklearn.svm import SVC
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    res_col = str(y_train.name)
    # print(y_train.name)
    kernel = ["rbf", 'sigmoid',] #'poly'
    param_grid =  {"kernel": kernel,
                "C": [x/110 for x in range(1,1000,13)]+[x/10 for x in range(10,150,13)],
               "gamma": [x/1080 for x in range(1,1300,93)]+[x/7 for x in range(10,200,13)],
               "degree": [3],"coef0": [0.618,0.312,0.2,0.8,0.5,0.1,0.9,0.25,0.75]}
    grid = RandomizedSearchCV(SVC(), param_grid,n_iter=n_iter, cv=6,n_jobs=n_jobs)
    grid.fit(X_train[yinzi], y_train)

    clf_best = grid.best_estimator_

    pre = clf_best.predict(X_train[yinzi])
    X_train = pd.DataFrame(X_train)
    X_train.loc[:, '预测值'] = pd.Series(pre, index=X_train.index)

    new  = X_train[['预测值',res_col]].sample(int((X_train.shape[0])*0.618))
    sj_acc_score = accuracy_score(new[res_col], new['预测值'])
    sj_ba_score = balanced_accuracy_score(new[res_col], new['预测值'])

    print('svc，随机遍历最优超参数:', [grid.best_score_,sj_acc_score,sj_ba_score])
    return clf_best, X_train, [grid.best_score_,sj_acc_score,sj_ba_score]

def bag_svc_grid(X_train=pd.DataFrame(),yinzi=[],y_train=pd.DataFrame(),n_iter=5,n_jobs=2):
    from sklearn.svm import SVC
    from sklearn.ensemble import BaggingClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    res_col = str(y_train.name)
    # clfb = BaggingClassifier(base_estimator=SVC(),n_estimators=n_iter,max_samples=0.6,max_features=0.4,bootstrap=True,
    #                     bootstrap_features=True, random_state=0)

    param_grid = {"n_estimators": [x  for x in range(50, 110, 10)],
                  "base_estimator": [SVC(gamma=x) for x in [x/1080 for x in range(1,1300,93)]+[x/7 for x in range(10,200,13)]+['auto_deprecated']],
                  "max_samples": [x / 10 for x in range(5,8)] ,
                  "max_features": [x / 10 for x in range(3,7)] ,}
    grid = RandomizedSearchCV(BaggingClassifier(SVC()), param_grid, n_iter=n_iter, cv =5, n_jobs=n_jobs)
    grid.fit(X_train[yinzi], y_train)

    clf_best = grid.best_estimator_

    # clfb = clf_best.fit(X_train[yinzi], y_train)
    pre = clf_best.predict(X_train[yinzi])
    X_train = pd.DataFrame(X_train)
    X_train.loc[:,'预测值'] = pd.Series(pre,index=X_train.index)

    new = X_train[['预测值', res_col]].sample(int((X_train.shape[0]) * 0.618))
    sj_acc_score = accuracy_score(new[res_col], new['预测值'])
    sj_ba_score = balanced_accuracy_score(new[res_col], new['预测值'])

    print('bag_svc，随机遍历最优超参数:', [grid.best_score_,sj_acc_score,sj_ba_score])
    return clf_best,X_train,[grid.best_score_,sj_acc_score,sj_ba_score]

def polynomial_Logistic_classify_grid(X_train=[],yinzi=[],y_train=[], n_iter=10,n_jobs=2):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import BaggingClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score


    # 构建多项式特征
    polynomial_features = PolynomialFeatures(degree=3,include_bias=False)
    Logistic_Regression = LogisticRegression()
    # 添加队列
    pipeline_model = Pipeline([("polynomial_features", polynomial_features),
                         ("Logistic_Regression", Logistic_Regression)])
    baglogi = BaggingClassifier(base_estimator = pipeline_model(),
                      n_estimators = n_iter,
                      max_samples=0.7,
                      max_features=0.7,
                      bootstrap=True,
                      bootstrap_features=True,
                      n_jobs=n_jobs,
                      )
    baglogi = baglogi.fit(X_train[yinzi], y_train)

    score0 = baglogi.score(X_train[yinzi], y_train)

    pre = baglogi.predict(X_train[yinzi])
    X_train = pd.DataFrame(X_train)
    X_train.loc[:, '预测值'] = pd.Series(pre, index=X_train.index)

    new = X_train[['预测值', '预测周期真实分类']].sample(int((X_train.shape[0]) * 0.618))
    sj_acc_score = accuracy_score(new['预测周期真实分类'], new['预测值'])
    sj_ba_score = balanced_accuracy_score(new['预测周期真实分类'], new['预测值'])


    print('bag_logi，随机遍历最优超参数:', [score0, sj_acc_score, sj_ba_score])


    return pipeline_model,X_train ,[score0, sj_acc_score, sj_ba_score]
