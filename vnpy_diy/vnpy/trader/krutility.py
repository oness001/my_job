#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/26 0026 13:37
# @Author  : Hadrianl 
# @File    : krutility

import json
from pathlib import Path
from typing import Callable
from decimal import Decimal
from dataclasses import dataclass
import numpy as np
import talib

from .object import BarData, TickData
from .constant import Exchange, Interval
from mongoengine import DateTimeField, Document, FloatField, StringField, ListField
import datetime
from typing import List, Sequence
from vnpy.trader.database import database_manager


class ArrayManager(object):
    """
    For:
    1. time series container of bar data
    2. calculating technical indicator value
    """

    def __init__(self, size=100):
        """Constructor"""
        self.count = 0
        self.size = size
        self.inited = False

        self.datetime_array = np.full(size, np.nan, dtype=datetime.datetime)
        self.open_array = np.full(size, np.nan, dtype=np.float64)
        self.high_array = np.full(size, np.nan, dtype=np.float64)
        self.low_array = np.full(size, np.nan, dtype=np.float64)
        self.close_array = np.full(size, np.nan, dtype=np.float64)
        self.volume_array = np.full(size, np.nan, dtype=np.float64)

    def update_bar(self, bar):
        """
        Update new bar data into array manager.
        """
        self.count += 1
        if not self.inited and self.count >= self.size:
            self.inited = True

        self.datetime_array[:-1] = self.datetime_array[1:]
        self.open_array[:-1] = self.open_array[1:]
        self.high_array[:-1] = self.high_array[1:]
        self.low_array[:-1] = self.low_array[1:]
        self.close_array[:-1] = self.close_array[1:]
        self.volume_array[:-1] = self.volume_array[1:]

        self.datetime_array[-1] = bar.datetime
        self.open_array[-1] = bar.open_price
        self.high_array[-1] = bar.high_price
        self.low_array[-1] = bar.low_price
        self.close_array[-1] = bar.close_price
        self.volume_array[-1] = bar.volume

    @property
    def datetime(self):
        """
        Get datetime series
        """
        return self.datetime_array

    @property
    def open(self):
        """
        Get open price time series.
        """
        return self.open_array

    @property
    def high(self):
        """
        Get high price time series.
        """
        return self.high_array

    @property
    def low(self):
        """
        Get low price time series.
        """
        return self.low_array

    @property
    def close(self):
        """
        Get close price time series.
        """
        return self.close_array

    @property
    def volume(self):
        """
        Get trading volume time series.
        """
        return self.volume_array

    def sma(self, n, array=False):
        """
        Simple moving average.
        """
        result = talib.SMA(self.close, n)
        if array:
            return result
        return result[-1]

    def ma(self, n, array=False):
        """
        Simple moving average.
        """
        result = talib.MA(self.close, n)
        if array:
            return result
        return result[-1]

    def std(self, n, array=False):
        """
        Standard deviation
        """
        result = talib.STDDEV(self.close, n)
        if array:
            return result
        return result[-1]

    def cci(self, n, array=False):
        """
        Commodity Channel Index (CCI).
        """
        result = talib.CCI(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def atr(self, n, array=False):
        """
        Average True Range (ATR).
        """
        result = talib.ATR(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def rsi(self, n, array=False):
        """
        Relative Strenght Index (RSI).
        """
        result = talib.RSI(self.close, n)
        if array:
            return result
        return result[-1]

    def macd(self, fast_period, slow_period, signal_period, array=False):
        """
        MACD.
        """
        macd, signal, hist = talib.MACD(
            self.close, fast_period, slow_period, signal_period
        )
        if array:
            return macd, signal, hist
        return macd[-1], signal[-1], hist[-1]

    def adx(self, n, array=False):
        """
        ADX.
        """
        result = talib.ADX(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def boll(self, n, dev, array=False):
        """
        Bollinger Channel.
        """
        mid = self.sma(n, array)
        std = self.std(n, array)

        up = mid + std * dev
        down = mid - std * dev

        return up, down

    def keltner(self, n, dev, array=False):
        """
        Keltner Channel.
        """
        mid = self.sma(n, array)
        atr = self.atr(n, array)

        up = mid + atr * dev
        down = mid - atr * dev

        return up, down

    def donchian(self, n, array=False):
        """
        Donchian Channel.
        """
        up = talib.MAX(self.high, n)
        down = talib.MIN(self.low, n)

        if array:
            return up, down
        return up[-1], down[-1]

    def aroon(self, n, array=False):
        """
        Aroon indicator.
        """
        aroon_up, aroon_down = talib.AROON(self.high, self.low, n)

        if array:
            return aroon_up, aroon_down
        return aroon_up[-1], aroon_down[-1]


@dataclass
class TraitData:
    symbol: str
    exchange: Exchange
    start: datetime
    end: datetime
    interval: Interval
    data: List[float]

    def __post_init__(self):
        """"""
        self.vt_symbol = f"{self.symbol}.{self.exchange.value}"

class DbTraitData(Document):
    """
    Candlestick bar data for database storage.

    Index is defined unique with datetime, interval, symbol
    """

    symbol: str = StringField()
    exchange: str = StringField()
    start: datetime = DateTimeField()
    end: datetime = DateTimeField()
    interval: str = StringField()

    data: list = ListField(FloatField())

    meta = {
        "indexes": [
            {
                "fields": ("symbol", "exchange", "start", "end"),
            }
        ]
    }

    @staticmethod
    def from_trait(trait: TraitData):
        """
        Generate DbBarData object from BarData.
        """
        db_trait = DbTraitData()

        db_trait.symbol = trait.symbol
        db_trait.exchange = trait.exchange.value
        db_trait.start = trait.start
        db_trait.end  = trait.end
        db_trait.interval = trait.interval.value
        db_trait.data = trait.data

        return db_trait

    def to_trait(self):
        """
        Generate BarData object from DbBarData.
        """
        trait = TraitData(
            symbol=self.symbol,
            exchange=Exchange(self.exchange),
            start=self.start,
            end=self.end,
            interval=Interval(self.interval),
            data=self.data
        )
        return trait


def load_trait_data(symbol, exchange: Exchange, interval: Interval=None):
    params = {'symbol': symbol, 'exchange': exchange.value}

    if interval:
        params['interval'] = interval

    data = DbTraitData.objects(**params)

    return [t.to_trait() for t in data]

def save_trait_data(trait_datas: Sequence[TraitData]):

    for t in trait_datas:
        updates = database_manager.to_update_param(t)
        updates.pop("set__vt_symbol")
        (
            DbTraitData.objects(
                symbol=t.symbol, exchange=t.exchange.value, interval=t.interval.value, start=t.start, end=t.end
            ).update_one(upsert=True, **updates)
        )
