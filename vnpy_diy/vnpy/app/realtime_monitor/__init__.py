#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
@author:Hadrianl

"""

from pathlib import Path

from vnpy.trader.app import BaseApp

from .engine import VisualEngine, APP_NAME


class VisualizationApp(BaseApp):
    """"""

    app_name = APP_NAME
    app_module = __module__
    app_path = Path(__file__).parent
    display_name = "IB可视化"
    engine_class = VisualEngine
    widget_name = "CandleChartWidget"
    icon_name = "visulization.ico"