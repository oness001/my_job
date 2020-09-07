#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20 0020 16:48
# @Author  : Hadrianl 
# @File    : __init__.py


from pathlib import Path

from vnpy.trader.app import BaseApp

from .engine import StrategyReviewEngine, APP_NAME


class StrategyReviewApp(BaseApp):
    """"""

    app_name = APP_NAME
    app_module = __module__
    app_path = Path(__file__).parent
    display_name = "IB策略执行回顾"
    engine_class = StrategyReviewEngine
    widget_name = "StrategyReviewer"
    icon_name = "reviewer.ico"