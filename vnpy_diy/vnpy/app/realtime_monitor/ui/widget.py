#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
@author:Hadrianl
THANKS FOR th github project https://github.com/moonnejs/uiKLine
"""

from vnpy.event import EventEngine, Event
from vnpy.trader.engine import MainEngine
import datetime as dt

from vnpy.trader.event import EVENT_TRADE, EVENT_ORDER, EVENT_TICK, EVENT_CONTRACT
from functools import partial

from vnpy.trader.constant import Direction, Interval, Exchange
from vnpy.trader.object import HistoryRequest, BarData, ContractData, TickData, TradeData
from vnpy.trader.ui import QtWidgets, QtCore, QtGui
from ..engine import APP_NAME


DEFAULT_MA = [5, 10, 30, 60]
DEFAULT_MA_COLOR = ['r', 'b', 'g', 'y']

from .baseQtItems import InfoWidget, MarketDataChartWidget
from vnpy.trader.object import OrderData

class CandleChartWidget(QtWidgets.QWidget):
    signal_update_tick = QtCore.pyqtSignal(TickData)
    signal_update_trade = QtCore.pyqtSignal(TradeData)
    signal_update_order = QtCore.pyqtSignal(OrderData)
    def __init__(self, main_engine: MainEngine, event_engine: EventEngine):
        """"""
        super().__init__()
        self.main_engine = main_engine
        self.event_engine = event_engine
        self.visual_engine = main_engine.get_engine(APP_NAME)
        self.contracts = {}
        self.vt_symbol = None
        self.init_ui()
        self.event_engine.register(EVENT_CONTRACT, self.add_contract)
        self.register_signal()

    def init_ui(self):
        """"""
        self.setWindowTitle("K线图表")
        self.resize(1400, 800)
        self.chart = MarketDataChartWidget()

        self.contract_combo = QtWidgets.QComboBox()
        self.contracts = {c.vt_symbol:c for c in self.main_engine.get_all_contracts()}
        self.contract_combo.addItems(self.contracts.keys())
        self.contract_combo.setCurrentIndex(-1)
        self.contract_combo.currentTextChanged.connect(self.change_contract)

        self.interval_combo = QtWidgets.QComboBox()
        self.interval_combo.addItems([Interval.MINUTE.value])
        self.indicator_combo = QtWidgets.QComboBox()
        self.indicator_combo.addItems([n for n in self.chart.indicators.keys()])
        self.indicator_combo.currentTextChanged.connect(self.chart.change_indicator)

        self.previous_btn = QtWidgets.QPushButton("←")
        self.previous_btn.released.connect(partial(self.update_previous_bar, 300))

        form = QtWidgets.QFormLayout()
        form.addRow("合约", self.contract_combo)
        form.addRow("周期", self.interval_combo)
        form.addRow("指标", self.indicator_combo)
        form.addRow(self.previous_btn)

        self.tick_info = InfoWidget()

        # Set layout
        box = QtWidgets.QHBoxLayout()
        vbox = QtWidgets.QVBoxLayout()
        vbox.addLayout(form)
        vbox.addWidget(self.chart)

        infoBox = QtWidgets.QVBoxLayout()
        infoBox.addStretch(1)
        infoBox.addLayout(self.tick_info)
        infoBox.addStretch(5)

        box.addLayout(vbox)
        box.addLayout(infoBox)
        box.setStretchFactor(vbox, 8)
        box.setStretchFactor(infoBox, 1)

        self.setLayout(box)


    def change_contract(self, vt_symbol):
        self.unregister_event()
        self.chart.clear_all()

        self.vt_symbol = vt_symbol
        contract: ContractData  = self.contracts.get(self.vt_symbol)
        if contract:
            interval = Interval(self.interval_combo.currentText())

            his_data = self.visual_engine.get_historical_data(contract, '', 600, interval)
            trade_data = self.visual_engine.get_trades(contract)
            order_data = self.visual_engine.get_orders(contract)

            self.register_event()
            self.chart.update_all(his_data, trade_data, order_data)

    def update_previous_bar(self, n):
        contract: ContractData = self.contracts.get(self.vt_symbol)
        if contract:
            his_data = self.chart._manager.get_all_bars()
            first_bar = his_data[0]
            first_bar_time = first_bar.datetime
            vt_symbol = self.vt_symbol
            self.chart.clear_all()
            self.vt_symbol = vt_symbol
            interval = Interval(self.interval_combo.currentText())
            total_minutes = {Interval.MINUTE: 1, Interval.HOUR: 60}[interval] * n
            req = HistoryRequest(contract.symbol, contract.exchange,
                                 start=first_bar_time - dt.timedelta(minutes=total_minutes),
                                 end=first_bar_time, interval=interval)

            pre_his_data = self.main_engine.query_history(req, contract.gateway_name)
            trade_data = [t for t in self.main_engine.get_all_trades() if t.vt_symbol == self.vt_symbol]
            order_data = [o for o in self.main_engine.get_all_orders() if o.vt_symbol == self.vt_symbol]

            for data in pre_his_data[::-1]:
                if data.datetime < first_bar_time:
                    his_data.insert(0, data)

            right_ix = self.chart._right_ix + len(pre_his_data)
            bar_count = self.chart._bar_count
            self.chart.update_all(his_data, trade_data, order_data)

            self.chart._right_ix = right_ix
            self.chart._bar_count = bar_count
            self.chart._update_x_range()

    def add_contract(self, event: Event):
        c = event.data
        self.contracts[c.vt_symbol] = c
        self.contract_combo.addItem(c.vt_symbol)

    def register_signal(self):
        self.signal_update_tick.connect(self.chart.update_tick)
        self.signal_update_tick.connect(self.chart.update_tick_line)
        self.signal_update_tick.connect(self.tick_info.update_tick)
        self.signal_update_trade.connect(self.chart.update_trade)
        self.signal_update_order.connect(self.chart.update_order)

    def process_tick_event(self, event: Event):
        tick = event.data
        self.signal_update_tick.emit(tick)

    def process_trade_event(self, event: Event):
        trade = event.data
        self.signal_update_trade.emit(trade)

    def process_order_event(self, event: Event):
        order = event.data
        if order.vt_symbol == self.vt_symbol:
            self.signal_update_order.emit(order)

    def register_event(self):
        if self.vt_symbol is not None:
            self.event_engine.register(EVENT_TICK + self.vt_symbol, self.process_tick_event)
            self.event_engine.register(EVENT_TRADE + self.vt_symbol, self.process_trade_event)
            self.event_engine.register(EVENT_ORDER, self.process_order_event)

    def unregister_event(self):
        if self.vt_symbol is not None:
            self.event_engine.unregister(EVENT_TICK + self.vt_symbol, self.process_tick_event)
            self.event_engine.unregister(EVENT_TRADE + self.vt_symbol, self.process_trade_event)
            self.event_engine.unregister(EVENT_ORDER + self.vt_symbol, self.process_order_event)

    def closeEvent(self, QCloseEvent):
        self.unregister_event()
        self.chart.clear_all()
        self.tick_info.clear_all()
        self.contract_combo.setCurrentIndex(-1)