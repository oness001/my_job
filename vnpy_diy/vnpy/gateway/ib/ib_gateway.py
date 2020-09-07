"""
IB Symbol Rules

SPY-USD-STK   SMART
EUR-USD-CASH  IDEALPRO
XAUUSD-USD-CMDTY  SMART
ES-202002-USD-FUT  GLOBEX
SI-202006-1000-USD-FUT  NYMEX
ES-2020006-C-2430-50-USD-FOP  GLOBEX
"""


from copy import copy
from datetime import datetime
from dateutil import parser
from queue import Empty
from threading import Thread, Condition, Lock
from typing import Optional
import shelve
from tzlocal import get_localzone

from ibapi import comm
from ibapi.client import EClient
from ibapi.common import MAX_MSG_LEN, NO_VALID_ID, OrderId, TickAttrib, TickerId
from ibapi.contract import Contract, ContractDetails
from ibapi.execution import Execution, ExecutionFilter
from ibapi.order import Order
from ibapi.order_state import OrderState
from ibapi.ticktype import TickType, TickTypeEnum
from ibapi.wrapper import EWrapper
from ibapi.errors import BAD_LENGTH
from ibapi.common import BarData as IbBarData
from dataclasses import dataclass

from vnpy.trader.gateway import BaseGateway
from vnpy.trader.object import (
    TickData,
    OrderData,
    TradeData,
    PositionData,
    AccountData,
    ContractData,
    BarData,
    OrderRequest,
    CancelRequest,
    SubscribeRequest,
    HistoryRequest,
    BaseData
)
from vnpy.trader.constant import (
    Product,
    OrderType,
    Direction,
    Exchange,
    Currency,
    Status,
    OptionType,
    Interval
)
from vnpy.trader.utility import get_file_path


ORDERTYPE_VT2IB = {
    OrderType.LIMIT: "LMT",
    OrderType.MARKET: "MKT",
    OrderType.STOP: "STP"
}
ORDERTYPE_IB2VT = {v: k for k, v in ORDERTYPE_VT2IB.items()}

DIRECTION_VT2IB = {Direction.LONG: "BUY", Direction.SHORT: "SELL"}
DIRECTION_IB2VT = {v: k for k, v in DIRECTION_VT2IB.items()}
DIRECTION_IB2VT["BOT"] = Direction.LONG
DIRECTION_IB2VT["SLD"] = Direction.SHORT

EXCHANGE_VT2IB = {
    Exchange.SMART: "SMART",
    Exchange.NYMEX: "NYMEX",
    Exchange.GLOBEX: "GLOBEX",
    Exchange.IDEALPRO: "IDEALPRO",
    Exchange.CME: "CME",
    Exchange.ICE: "ICE",
    Exchange.SEHK: "SEHK",
    Exchange.HKFE: "HKFE",
    Exchange.CFE: "CFE",
    Exchange.NYSE: "NYSE",
    Exchange.NASDAQ: "NASDAQ",
    Exchange.ARCA: "ARCA",
    Exchange.EDGEA: "EDGEA",
    Exchange.ISLAND: "ISLAND",
    Exchange.BATS: "BATS",
    Exchange.IEX: "IEX",
    Exchange.IBKRATS: "IBKRATS",
}
EXCHANGE_IB2VT = {v: k for k, v in EXCHANGE_VT2IB.items()}

STATUS_IB2VT = {
    "ApiPending": Status.SUBMITTING,
    "PendingSubmit": Status.SUBMITTING,
    "PreSubmitted": Status.NOTTRADED,
    "Submitted": Status.NOTTRADED,
    "ApiCancelled": Status.CANCELLED,
    "Cancelled": Status.CANCELLED,
    "Filled": Status.ALLTRADED,
    "Inactive": Status.REJECTED,
}

PRODUCT_IB2VT = {
    "STK": Product.EQUITY,
    "CASH": Product.FOREX,
    "CMDTY": Product.SPOT,
    "FUT": Product.FUTURES,
    "OPT": Product.OPTION,
    "FOT": Product.OPTION
}

OPTION_VT2IB = {OptionType.CALL: "CALL", OptionType.PUT: "PUT"}

CURRENCY_VT2IB = {
    Currency.USD: "USD",
    Currency.CNY: "CNY",
    Currency.HKD: "HKD",
}

TICKFIELD_IB2VT = {
    0: "bid_volume_1",
    1: "bid_price_1",
    2: "ask_price_1",
    3: "ask_volume_1",
    4: "last_price",
    5: "last_volume",
    6: "high_price",
    7: "low_price",
    8: "volume",
    9: "pre_close",
    14: "open_price",
    22: "open_interest"
}

ACCOUNTFIELD_IB2VT = {
    "NetLiquidationByCurrency": "balance",
    "NetLiquidation": "balance",
    "UnrealizedPnL": "positionProfit",
    "AvailableFunds": "available",
    "MaintMarginReq": "margin",
}

INTERVAL_VT2IB = {
    Interval.MINUTE: "1 min",
    Interval.HOUR: "1 hour",
    Interval.DAILY: "1 day",
    Interval.WEEKLY: "1 week"
}

EVENT_ERROR = "eError"

@dataclass
class IBError(BaseData):
    reqId: int = -1
    code: int = None
    content: str = ""

JOIN_SYMBOL = "-"


class IbGateway(BaseGateway):
    """"""

    default_setting = {
        "TWS地址": "127.0.0.1",
        "TWS端口": 7497,
        "客户号": 1,
        "交易账户": ""
    }

    exchanges = list(EXCHANGE_VT2IB.keys())

    def __init__(self, event_engine):
        """"""
        super().__init__(event_engine, "IB")

        self.api = IbApi(self)

    def connect(self, setting: dict):
        """
        Start gateway connection.
        """
        host = setting["TWS地址"]
        port = setting["TWS端口"]
        clientid = setting["客户号"]
        account = setting["交易账户"]

        self.api.connect(host, port, clientid, account)

    def close(self):
        """
        Close gateway connection.
        """
        self.api.close()

    def subscribe(self, req: SubscribeRequest):
        """
        Subscribe tick data update.
        """
        self.api.subscribe(req)

    def send_order(self, req: OrderRequest):
        """
        Send a new order.
        """
        return self.api.send_order(req)

    def cancel_order(self, req: CancelRequest):
        """
        Cancel an existing order.
        """
        self.api.cancel_order(req)

    def query_account(self):
        """
        Query account balance.
        """
        pass

    def query_position(self):
        """
        Query holding positions.
        """
        return self.api.query_position()

    def query_history(self, req: HistoryRequest):
        """"""
        return self.api.query_history(req)


class IbApi(EWrapper):
    """"""
    data_filename = "ib_contract_data.db"
    data_filepath = str(get_file_path(data_filename))

    local_tz = get_localzone()

    def __init__(self, gateway: BaseGateway):
        """"""
        super().__init__()

        self.gateway = gateway
        self.gateway_name = gateway.gateway_name

        self.status = False

        self.reqid = 0
        self.orderid = 0
        self.clientid = 0
        self.account = ""
        self.ticks = {}
        self.orders = {}
        self.accounts = {}
        self.contracts = {}

        self.tick_exchange = {}

        self.history_req = None
        self.history_condition = Condition()
        self.history_buf = []

        self.position_condition = Condition()
        self.position_buf = []

        self.client = IbClient(self)
        self.thread = Thread(target=self.client.run)
        self.orderLock = Lock()

    def connectAck(self):  # pylint: disable=invalid-name
        """
        Callback when connection is established.
        """
        self.status = True
        self.gateway.write_log("IB TWS连接成功")

        self.load_contract_data()

    def connectionClosed(self):  # pylint: disable=invalid-name
        """
        Callback when connection is closed.
        """
        self.status = False
        self.gateway.write_log("IB TWS连接断开")

    def nextValidId(self, orderId: int):  # pylint: disable=invalid-name
        """
        Callback of next valid orderid.
        """
        super().nextValidId(orderId)

        if not self.orderid:
            self.orderid = orderId

    def currentTime(self, time: int):  # pylint: disable=invalid-name
        """
        Callback of current server time of IB.
        """
        super().currentTime(time)

        dt = datetime.fromtimestamp(time)
        time_string = dt.strftime("%Y-%m-%d %H:%M:%S.%f")

        msg = f"服务器时间: {time_string}"
        self.gateway.write_log(msg)

    def error(
        self, reqId: TickerId, errorCode: int, errorString: str
    ):  # pylint: disable=invalid-name
        """
        Callback of error caused by specific request.
        """
        super().error(reqId, errorCode, errorString)

        msg = f"信息通知，代码：{errorCode}，内容: {errorString}, ReqID={reqId}"
        self.gateway.write_log(msg)
        e = IBError(reqId=reqId, code=errorCode, content=errorString, gateway_name="IB")
        self.gateway.on_event(EVENT_ERROR, e)

    def tickPrice(  # pylint: disable=invalid-name
        self, reqId: TickerId, tickType: TickType, price: float, attrib: TickAttrib
    ):
        """
        Callback of tick price update.
        """
        super().tickPrice(reqId, tickType, price, attrib)

        if tickType not in TICKFIELD_IB2VT:
            return

        tick = self.ticks[reqId]
        name = TICKFIELD_IB2VT[tickType]
        setattr(tick, name, price)

        # Update name into tick data.
        contract = self.contracts.get(tick.vt_symbol, None)
        if contract:
            tick.name = contract.name

        # Forex and spot product of IDEALPRO has no tick time and last price.
        # We need to calculate locally.
        exchange = self.tick_exchange[reqId]
        if exchange is Exchange.IDEALPRO:
            tick.last_price = (tick.bid_price_1 + tick.ask_price_1) / 2
            tick.datetime = datetime.now(self.local_tz)
        self.gateway.on_tick(copy(tick))

    def tickSize(
        self, reqId: TickerId, tickType: TickType, size: int
    ):  # pylint: disable=invalid-name
        """
        Callback of tick volume update.
        """
        super().tickSize(reqId, tickType, size)

        if tickType not in TICKFIELD_IB2VT:
            return

        tick = self.ticks[reqId]
        name = TICKFIELD_IB2VT[tickType]
        setattr(tick, name, size)

        self.gateway.on_tick(copy(tick))

    def tickString(
        self, reqId: TickerId, tickType: TickType, value: str
    ):  # pylint: disable=invalid-name
        """
        Callback of tick string update.
        """
        super().tickString(reqId, tickType, value)

        if tickType != TickTypeEnum.LAST_TIMESTAMP:
            return

        tick = self.ticks[reqId]
        dt = datetime.fromtimestamp(int(value))
        tick.datetime = dt.replace(tzinfo=self.local_tz)

        self.gateway.on_tick(copy(tick))

    def orderStatus(  # pylint: disable=invalid-name
        self,
        orderId: OrderId,
        status: str,
        filled: float,
        remaining: float,
        avgFillPrice: float,
        permId: int,
        parentId: int,
        lastFillPrice: float,
        clientId: int,
        whyHeld: str,
        mktCapPrice: float,
    ):
        """
        Callback of order status update.
        """
        super().orderStatus(
            orderId,
            status,
            filled,
            remaining,
            avgFillPrice,
            permId,
            parentId,
            lastFillPrice,
            clientId,
            whyHeld,
            mktCapPrice,
        )

        orderid = str(orderId)
        order = self.orders.get(orderid, None)
        if order:
            order.status = STATUS_IB2VT.get(status, '')  # FIXME: PendingCancel is not included
            order.traded = filled

            self.gateway.on_order(copy(order))

    def openOrder(  # pylint: disable=invalid-name
        self,
        orderId: OrderId,
        ib_contract: Contract,
        ib_order: Order,
        orderState: OrderState,
    ):
        """
        Callback when opening new order.
        """
        super().openOrder(
            orderId, ib_contract, ib_order, orderState
        )

        orderid = str(orderId)
        if ib_order.orderType in ['LMT', 'MKT']:
            orderType = ORDERTYPE_IB2VT[ib_order.orderType]
        elif ib_order.orderType.startswith('STP'):
            orderType = ORDERTYPE_IB2VT['STP']
        else:
            orderType = ib_order.orderType

        order = OrderData(
            symbol=str(ib_contract.conId),
            exchange=EXCHANGE_IB2VT.get(
                ib_contract.exchange, ib_contract.exchange),
            type=orderType,
            orderid=orderid,
            direction=DIRECTION_IB2VT[ib_order.action],
            price=ib_order.lmtPrice,
            volume=ib_order.totalQuantity,
            gateway_name=self.gateway_name,
        )

        if order.type == OrderType.LIMIT:
            order.price = ib_order.lmtPrice
        elif order.type == OrderType.STOP:
            order.price = ib_order.auxPrice

        self.orders[orderid] = order
        self.gateway.on_order(copy(order))

    def updateAccountValue(  # pylint: disable=invalid-name
        self, key: str, val: str, currency: str, accountName: str
    ):
        """
        Callback of account update.
        """
        super().updateAccountValue(key, val, currency, accountName)

        if not currency or key not in ACCOUNTFIELD_IB2VT:
            return

        accountid = f"{accountName}.{currency}"
        account = self.accounts.get(accountid, None)
        if not account:
            account = AccountData(accountid=accountid,
                                  gateway_name=self.gateway_name)
            self.accounts[accountid] = account

        name = ACCOUNTFIELD_IB2VT[key]
        setattr(account, name, float(val))

    def accountUpdateMulti(self, reqId:int, account:str, modelCode:str,
                            key:str, value:str, currency:str):
        self.updateAccountValue(key, value, currency, account)

    def updatePortfolio(  # pylint: disable=invalid-name
        self,
        contract: Contract,
        position: float,
        marketPrice: float,
        marketValue: float,
        averageCost: float,
        unrealizedPNL: float,
        realizedPNL: float,
        accountName: str,
    ):
        """
        Callback of position update.
        """
        super().updatePortfolio(
            contract,
            position,
            marketPrice,
            marketValue,
            averageCost,
            unrealizedPNL,
            realizedPNL,
            accountName,
        )

        if contract.exchange:
            exchange = EXCHANGE_IB2VT.get(contract.exchange, None)
        elif contract.primaryExchange:
            exchange = EXCHANGE_IB2VT.get(contract.primaryExchange, None)
        else:
            exchange = Exchange.SMART   # Use smart routing for default

        if not exchange:
            msg = f"存在不支持的交易所持仓{contract.conId} {contract.exchange} {contract.primaryExchange}"
            self.gateway.write_log(msg)
            return

        try:
            ib_size = int(contract.multiplier)
        except ValueError:
            ib_size = 1
        price = averageCost / ib_size

        pos = PositionData(
            symbol=generate_symbol(contract),
            exchange=exchange,
            direction=Direction.NET,
            volume=position,
            price=price,
            pnl=unrealizedPNL,
            gateway_name=self.gateway_name,
        )
        self.gateway.on_position(pos)

    def updateAccountTime(self, timeStamp: str):  # pylint: disable=invalid-name
        """
        Callback of account update time.
        """
        super().updateAccountTime(timeStamp)
        for account in self.accounts.values():
            self.gateway.on_account(copy(account))

    def contractDetails(self, reqId: int, contractDetails: ContractDetails):  # pylint: disable=invalid-name
        """
        Callback of contract data update.
        """
        super().contractDetails(reqId, contractDetails)

        # Generate symbol from ib contract details
        ib_contract = contractDetails.contract
        if not ib_contract.multiplier:
            ib_contract.multiplier = 1

        symbol = generate_symbol(ib_contract)

        # Generate contract
        contract = ContractData(
            symbol=symbol,
            exchange=EXCHANGE_IB2VT[ib_contract.exchange],
            name=contractDetails.longName,
            product=PRODUCT_IB2VT[ib_contract.secType],
            size=ib_contract.multiplier,
            pricetick=contractDetails.minTick,
            net_position=True,
            expiry=parser.parse(contractDetails.contract.lastTradeDateOrContractMonth),
            history_data=True,
            stop_supported=True,
            gateway_name=self.gateway_name,
        )

        if contract.vt_symbol not in self.contracts:
            self.gateway.on_contract(contract)

            self.contracts[contract.vt_symbol] = contract
            self.save_contract_data()

    def execDetails(
        self, reqId: int, contract: Contract, execution: Execution
    ):  # pylint: disable=invalid-name
        """
        Callback of trade data update.
        """
        super().execDetails(reqId, contract, execution)

        dt = datetime.strptime(execution.time, "%Y%m%d  %H:%M:%S")
        dt = dt.replace(tzinfo=self.local_tz)

        trade = TradeData(
            symbol=generate_symbol(contract),
            exchange=EXCHANGE_IB2VT.get(contract.exchange, contract.exchange),
            orderid=str(execution.orderId),
            tradeid=str(execution.execId),
            direction=DIRECTION_IB2VT[execution.side],
            price=execution.price,
            volume=execution.shares,
            datetime=dt,
            orderRef=execution.orderRef,
            gateway_name=self.gateway_name,
        )

        self.gateway.on_trade(trade)

    def managedAccounts(self, accountsList: str):
        """
        Callback of all sub accountid.
        """
        super().managedAccounts(accountsList)

        if not self.account:
            for account_code in accountsList.split(","):
                self.account = account_code

        self.gateway.write_log(f"当前使用的交易账号为{self.account}")
        self.client.reqAccountUpdates(True, self.account)

    def historicalData(self, reqId: int, ib_bar: IbBarData):
        """
        Callback of history data update.
        """
        dt = datetime.strptime(ib_bar.date, "%Y%m%d %H:%M:%S")
        dt = dt.replace(tzinfo=self.local_tz)

        bar = BarData(
            symbol=self.history_req.symbol,
            exchange=self.history_req.exchange,
            datetime=dt,
            interval=self.history_req.interval,
            volume=ib_bar.volume,
            open_price=ib_bar.open,
            high_price=ib_bar.high,
            low_price=ib_bar.low,
            close_price=ib_bar.close,
            gateway_name=self.gateway_name
        )

        self.history_buf.append(bar)

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """
        Callback of history data finished.
        """
        self.history_condition.acquire()
        self.history_condition.notify()
        self.history_condition.release()

    def position(self, account:str, contract:Contract, position:float,
                 avgCost:float):
        if contract.exchange:
            exchange = EXCHANGE_IB2VT.get(contract.exchange, None)
        elif contract.primaryExchange:
            exchange = EXCHANGE_IB2VT.get(contract.primaryExchange, None)
        else:
            exchange = Exchange.HKFE   # FIXME: position with no exhange!!!!

        if not exchange:
            msg = f"存在不支持的交易所持仓{contract.conId} {contract.exchange} {contract.primaryExchange}"
            self.gateway.write_log(msg)
            return


        pos = PositionData(
            symbol=str(contract.conId),
            exchange=exchange,
            direction=Direction.NET,
            volume=position,
            price=avgCost,
            # pnl=unrealizedPNL,
            gateway_name=self.gateway_name,
        )
        self.position_buf.append(pos)
        # self.gateway.on_position(pos)

    def positionEnd(self):
        """
        Callback of position data finished.
        """
        self.position_condition.acquire()
        self.position_condition.notify()
        self.position_condition.release()

    def connect(self, host: str, port: int, clientid: int, account: str):
        """
        Connect to TWS.
        """
        if self.status:
            return

        self.clientid = clientid
        self.account = account
        self.client.connect(host, port, clientid)
        self.thread.start()

        self.client.reqCurrentTime()
        if self.clientid == 0:
            self.client.reqAutoOpenOrders(True)

        self.reqid += 1
        executionFilter = ExecutionFilter()
        self.client.reqExecutions(self.reqid, executionFilter)
        self.client.reqOpenOrders()
        self.client.reqPositions()

    def close(self):
        """
        Disconnect to TWS.
        """
        if not self.status:
            return

        self.status = False
        self.client.disconnect()

    def subscribe(self, req: SubscribeRequest):
        """
        Subscribe tick data update.
        """
        if not self.status:
            return

        if req.exchange not in EXCHANGE_VT2IB:
            self.gateway.write_log(f"不支持的交易所{req.exchange}")
            return

        # Extract ib contract detail
        ib_contract = generate_ib_contract(req.symbol, req.exchange)
        if not ib_contract:
            self.gateway.write_log("代码解析失败，请检查格式是否正确")
            return

        # Get contract data from TWS.
        self.reqid += 1
        self.client.reqContractDetails(self.reqid, ib_contract)

        # Subscribe tick data and create tick object buffer.
        self.reqid += 1
        self.client.reqMktData(self.reqid, ib_contract, "", False, False, [])

        tick = TickData(
            symbol=req.symbol,
            exchange=req.exchange,
            datetime=datetime.now(self.local_tz),
            gateway_name=self.gateway_name,
        )
        self.ticks[self.reqid] = tick
        self.tick_exchange[self.reqid] = req.exchange

    def send_order(self, req: OrderRequest):
        """
        Send a new order.
        """
        if not self.status:
            return ""

        if req.exchange not in EXCHANGE_VT2IB:
            self.gateway.write_log(f"不支持的交易所：{req.exchange}")
            return ""

        if req.type not in ORDERTYPE_VT2IB:
            self.gateway.write_log(f"不支持的价格类型：{req.type}")
            return ""

        ib_contract = generate_ib_contract(req.symbol, req.exchange)
        if not ib_contract:
            return ""

        ib_order = Order()
        ib_order.clientId = self.clientid
        ib_order.action = DIRECTION_VT2IB[req.direction]
        ib_order.orderType = ORDERTYPE_VT2IB[req.type]
        ib_order.totalQuantity = req.volume
        ib_order.outsideRth = True
        ib_order.orderRef = req.orderRef
        ib_order.account = self.account

        if req.type == OrderType.LIMIT:
            ib_order.lmtPrice = req.price
        elif req.type == OrderType.STOP:
            ib_order.auxPrice = req.price

        self.orderLock.acquire()
        self.orderid += 1
        ib_order.orderId = self.orderid
        self.client.placeOrder(self.orderid, ib_contract, ib_order)
        # self.client.reqIds(1)
        order = req.create_order_data(str(self.orderid), self.gateway_name)
        self.orderLock.release()

        self.gateway.on_order(order)
        return order.vt_orderid

    def cancel_order(self, req: CancelRequest):
        """
        Cancel an existing order.
        """
        if not self.status:
            return

        self.client.cancelOrder(int(req.orderid))

    def query_history(self, req: HistoryRequest):
        """"""
        self.history_req = req

        self.reqid += 1

        ib_contract = generate_ib_contract(req.symbol, req.exchange)

        if req.end:
            end = req.end
            end_str = end.strftime("%Y%m%d %H:%M:%S")
        else:
            end = datetime.now(get_localzone())
            end_str = ""

        delta = end - req.start
        # days = min(delta.days, 180)     # IB only provides 6-month data
        # duration = f"{days} D"
        duration = self.timedelta2durationStr(delta)
        bar_size = INTERVAL_VT2IB[req.interval]

        if req.exchange == Exchange.IDEALPRO:
            bar_type = "MIDPOINT"
        else:
            bar_type = "TRADES"

        self.client.reqHistoricalData(
            self.reqid,
            ib_contract,
            end_str,
            duration,
            bar_size,
            bar_type,
            False,
            1,
            False,
            []
        )

        self.history_condition.acquire()    # Wait for async data return
        self.history_condition.wait()
        self.history_condition.release()

        # print(self.history_buf[0], req.start)
        # history = [h for h in self.history_buf if h.datetime >=req.start]
        history = self.history_buf
        self.history_buf = []       # Create new buffer list
        self.history_req = None

        return history

    def query_position(self):
        self.position_buf = []
        self.client.reqPositions()

        self.position_condition.acquire()
        self.position_condition.wait()
        self.position_condition.release()

        position = self.position_buf
        self.position_buf = []

        return position

    @staticmethod
    def timedelta2durationStr(delta):
        total_seconds = delta.total_seconds()
        if total_seconds < 86400:
            durationStr = f'{int(total_seconds //60 * 60  + 60)} S'
        elif total_seconds < 86400 * 30:
            durationStr = f'{int(min(delta.days + 1, 30))} D'
        elif total_seconds < 86400 * 30 * 6:
            durationStr = f'{int(min(delta.days // 30 + 1, 6))} M'
        else:
            durationStr = f'{int(delta.days // (30 * 12) + 1)} Y'

        return durationStr

    def load_contract_data(self):
        """"""
        f = shelve.open(self.data_filepath)
        self.contracts = f.get("contracts", {})
        f.close()

        for contract in self.contracts.values():
            self.gateway.on_contract(contract)

        self.gateway.write_log("本地缓存合约信息加载成功")

    def save_contract_data(self):
        """"""
        f = shelve.open(self.data_filepath)
        f["contracts"] = self.contracts
        f.close()


class IbClient(EClient):
    """"""

    def run(self):
        """
        Reimplement the original run message loop of eclient.

        Remove all unnecessary try...catch... and allow exceptions to interrupt loop.
        """
        while not self.done and self.isConnected():
            try:
                text = self.msg_queue.get(block=True, timeout=0.2)

                if len(text) > MAX_MSG_LEN:
                    errorMsg = "%s:%d:%s" % (BAD_LENGTH.msg(), len(text), text)
                    self.wrapper.error(
                        NO_VALID_ID, BAD_LENGTH.code(), errorMsg
                    )
                    self.disconnect()
                    break

                fields = comm.read_fields(text)
                self.decoder.interpret(fields)
            except Empty:
                pass


def generate_ib_contract(symbol: str, exchange: Exchange) -> Optional[Contract]:
    """"""
    try:
        fields = symbol.split(JOIN_SYMBOL)

        ib_contract = Contract()
        ib_contract.exchange = EXCHANGE_VT2IB[exchange]
        ib_contract.secType = fields[-1]
        ib_contract.currency = fields[-2]
        ib_contract.symbol = fields[0]

        if ib_contract.secType in ["FUT", "OPT", "FOP"]:
            ib_contract.lastTradeDateOrContractMonth = fields[1]

        if ib_contract.secType == "FUT":
            if len(fields) == 5:
                ib_contract.multiplier = int(fields[2])

        if ib_contract.secType in ["OPT", "FOP"]:
            ib_contract.right = fields[2]
            ib_contract.strike = float(fields[3])
            ib_contract.multiplier = int(fields[4])
    except IndexError:
        ib_contract = None

    return ib_contract


def generate_symbol(ib_contract: Contract) -> str:
    """"""
    fields = [ib_contract.symbol]

    if ib_contract.secType in ["FUT", "OPT", "FOP"]:
        fields.append(ib_contract.lastTradeDateOrContractMonth)

    if ib_contract.secType in ["OPT", "FOP"]:
        fields.append(ib_contract.right)
        fields.append(str(ib_contract.strike))
        fields.append(str(ib_contract.multiplier))

    fields.append(ib_contract.currency)
    fields.append(ib_contract.secType)

    symbol = JOIN_SYMBOL.join(fields)

    return symbol
