"""
"""

import sys
import pytz
from datetime import datetime
from typing import Dict, List

from vnpy.api.uft import (
    FUTURES_LICENSE,
    OPTION_LICENSE,
    MdApi,
    TdApi,
    HS_EI_CFFEX,
    HS_EI_SHFE,
    HS_EI_DCE,
    HS_EI_CZCE,
    HS_EI_INE,
    HS_EI_SZSE,
    HS_EI_SSE,
    HS_OS_Reported,
    HS_OS_Abandoned,
    HS_OS_PartsTraded,
    HS_OS_Traded,
    HS_OS_Canceled,
    HS_OS_CanceledWithPartsTraded,
    HS_D_Buy,
    HS_D_Sell,
    HS_CT_Limit,
    HS_CT_Market,
    HS_OF_Open,
    HS_OF_Close,
    HS_OF_CloseToday,
    HS_PTYPE_Futures,
    HS_PTYPE_OptFutu,
    HS_PTYPE_OptStock,
    HS_PTYPE_Combination,
    HS_OT_CallOptions,
    HS_OT_PutOptions,
)
from vnpy.trader.constant import (
    Direction,
    Offset,
    Exchange,
    OrderType,
    Product,
    Status,
    OptionType
)
from vnpy.trader.gateway import BaseGateway
from vnpy.trader.object import (
    TickData,
    OrderData,
    TradeData,
    PositionData,
    AccountData,
    ContractData,
    OrderRequest,
    CancelRequest,
    SubscribeRequest,
)
from vnpy.trader.utility import get_folder_path
from vnpy.trader.event import EVENT_TIMER
from vnpy.event import EventEngine


STATUS_UFT2VT: Dict[str, Status] = {
    HS_OS_Reported: Status.NOTTRADED,
    HS_OS_Abandoned: Status.REJECTED,
    HS_OS_PartsTraded: Status.PARTTRADED,
    HS_OS_Traded: Status.ALLTRADED,
    HS_OS_Canceled: Status.CANCELLED,
    HS_OS_CanceledWithPartsTraded: Status.CANCELLED
}

DIRECTION_VT2UFT: Dict[Direction, str] = {
    Direction.LONG: HS_D_Buy,
    Direction.SHORT: HS_D_Sell
}
DIRECTION_UFT2VT: Dict[str, Direction] = {v: k for k, v in DIRECTION_VT2UFT.items()}
# DIRECTION_UFT2VT[HS_PT_Right] = Direction.LONG
# DIRECTION_UFT2VT[HS_PT_Voluntary] = Direction.SHORT

ORDERTYPE_VT2UFT: Dict[OrderType, str] = {
    OrderType.LIMIT: HS_CT_Limit,
    OrderType.MARKET: HS_CT_Market
}
ORDERTYPE_UFT2VT: Dict[str, OrderType] = {v: k for k, v in ORDERTYPE_VT2UFT.items()}

OFFSET_VT2UFT: Dict[Offset, str] = {
    Offset.OPEN: HS_OF_Open,
    Offset.CLOSE: HS_OF_Close,
    Offset.CLOSETODAY: HS_OF_CloseToday,
}
OFFSET_UFT2VT: Dict[str, Offset] = {v: k for k, v in OFFSET_VT2UFT.items()}

EXCHANGE_UFT2VT: Dict[str, Exchange] = {
    HS_EI_CFFEX: Exchange.CFFEX,
    HS_EI_SHFE: Exchange.SHFE,
    HS_EI_CZCE: Exchange.CZCE,
    HS_EI_DCE: Exchange.DCE,
    HS_EI_INE: Exchange.INE,
    HS_EI_SSE: Exchange.SSE,
    HS_EI_SZSE: Exchange.SZSE,
}
EXCHANGE_VT2UFT: Dict[Exchange, str] = {v: k for k, v in EXCHANGE_UFT2VT.items()}

PRODUCT_UFT2VT: Dict[str, Product] = {
    HS_PTYPE_Futures: Product.FUTURES,
    HS_PTYPE_OptFutu: Product.OPTION,
    HS_PTYPE_OptStock: Product.OPTION,
    HS_PTYPE_Combination: Product.SPREAD
}

OPTIONTYPE_UFT2VT: Dict[str, OptionType] = {
    HS_OT_CallOptions: OptionType.CALL,
    HS_OT_PutOptions: OptionType.PUT
}

MAX_FLOAT = sys.float_info.max
CHINA_TZ = pytz.timezone("Asia/Shanghai")

symbol_name_map = {}
symbol_size_map = {}


class UftGateway(BaseGateway):
    """
    VN Trader Gateway for UFT .
    """

    default_setting: Dict[str, str] = {
        "用户名": "",
        "密码": "",
        "服务器地址": "",
        "服务器类型": ["期货", "ETF期权"],
        "产品名称": "",
        "授权编码": "",
        "产品信息": ""
    }

    exchanges: List[Exchange] = list(EXCHANGE_UFT2VT.values())

    def __init__(self, event_engine: EventEngine):
        """Constructor"""
        super().__init__(event_engine, "UFT")

        self.td_api = UftTdApi(self)
        self.md_api = UftMdApi(self)

    def connect(self, setting: dict) -> None:
        """"""
        userid = setting["用户名"]
        password = setting["密码"]
        address = setting["服务器地址"]
        server = setting["服务器类型"]
        appid = setting["产品名称"]
        auth_code = setting["授权编码"]

        if not address.startswith("tcp://"):
            address = "tcp://" + address

        if server == "期货":
            server_license = FUTURES_LICENSE
        else:
            server_license = OPTION_LICENSE

        self.td_api.connect(
            address,
            server_license,
            userid,
            password,
            auth_code,
            appid
        )
        self.md_api.connect(
            address,
            server_license
        )

        self.init_query()

    def subscribe(self, req: SubscribeRequest) -> None:
        """"""
        self.md_api.subscribe(req)

    def send_order(self, req: OrderRequest) -> str:
        """"""
        return self.td_api.send_order(req)

    def cancel_order(self, req: CancelRequest) -> None:
        """"""
        self.td_api.cancel_order(req)

    def query_account(self) -> None:
        """"""
        self.td_api.query_account()

    def query_position(self) -> None:
        """"""
        self.td_api.query_position()

    def close(self) -> None:
        """"""
        self.td_api.close()
        self.md_api.close()

    def write_error(self, msg: str, error: dict) -> None:
        """"""
        error_id = error["ErrorID"]
        error_msg = error["ErrorMsg"]
        msg = f"{msg}，代码：{error_id}，信息：{error_msg}"
        self.write_log(msg)

    def process_timer_event(self, event) -> None:
        """"""
        self.count += 1
        if self.count < 2:
            return
        self.count = 0

        func = self.query_functions.pop(0)
        func()
        self.query_functions.append(func)

    def init_query(self) -> None:
        """"""
        self.count = 0
        self.query_functions = [self.query_account, self.query_position]
        self.event_engine.register(EVENT_TIMER, self.process_timer_event)


class UftMdApi(MdApi):
    """"""

    def __init__(self, gateway: UftGateway):
        """Constructor"""
        super(UftMdApi, self).__init__()

        self.gateway: UftGateway = gateway
        self.gateway_name: str = gateway.gateway_name

        self.reqid: int = 0

        self.connect_status: bool = False

        self.userid: str = ""
        self.password: str = ""

    def onFrontConnected(self) -> None:
        """
        Callback when front server is connected.
        """
        self.gateway.write_log("行情服务器连接成功")

    def onFrontDisconnected(self, reason: int) -> None:
        """
        Callback when front server is disconnected.
        """
        self.gateway.write_log(f"行情服务器连接断开，原因{reason}")

    def onRspDepthMarketDataSubscribe(
        self,
        error: dict,
        reqid: int,
        last: bool
    ) -> None:
        """"""
        if not error or not error["ErrorID"]:
            return

        self.gateway.write_error("行情订阅失败", error)

    def onRtnDepthMarketData(self, data: dict) -> None:
        """
        Callback of tick data update.
        """
        symbol = data["InstrumentID"]
        exchange = EXCHANGE_UFT2VT[data["ExchangeID"]]
        if not exchange:
            return

        timestamp = f"{data['TradingDay']} {data['UpdateTime']}000"
        dt = datetime.strptime(timestamp, "%Y%m%d %H%M%S%f")
        dt = dt.replace(tzinfo=CHINA_TZ)

        tick = TickData(
            symbol=symbol,
            exchange=exchange,
            datetime=dt,
            name=symbol_name_map[symbol],
            volume=data["TradeVolume"],
            open_interest=data["OpenInterest"],
            last_price=data["LastPrice"],
            limit_up=data["UpperLimitPrice"],
            limit_down=data["LowerLimitPrice"],
            open_price=adjust_price(data["OpenPrice"]),
            high_price=adjust_price(data["HighestPrice"]),
            low_price=adjust_price(data["LowestPrice"]),
            pre_close=adjust_price(data["PreClosePrice"]),
            bid_price_1=adjust_price(data["BidPrice1"]),
            ask_price_1=adjust_price(data["AskPrice1"]),
            bid_volume_1=data["BidVolume1"],
            ask_volume_1=data["AskVolume1"],
            gateway_name=self.gateway_name
        )

        if data["BidVolume2"] or data["AskVolume2"]:
            tick.bid_price_2 = adjust_price(data["BidPrice2"])
            tick.bid_price_3 = adjust_price(data["BidPrice3"])
            tick.bid_price_4 = adjust_price(data["BidPrice4"])
            tick.bid_price_5 = adjust_price(data["BidPrice5"])

            tick.ask_price_2 = adjust_price(data["AskPrice2"])
            tick.ask_price_3 = adjust_price(data["AskPrice3"])
            tick.ask_price_4 = adjust_price(data["AskPrice4"])
            tick.ask_price_5 = adjust_price(data["AskPrice5"])

            tick.bid_volume_2 = adjust_price(data["BidVolume2"])
            tick.bid_volume_3 = adjust_price(data["BidVolume3"])
            tick.bid_volume_4 = adjust_price(data["BidVolume4"])
            tick.bid_volume_5 = adjust_price(data["BidVolume5"])

            tick.ask_volume_2 = adjust_price(data["AskVolume2"])
            tick.ask_volume_3 = adjust_price(data["AskVolume3"])
            tick.ask_volume_4 = adjust_price(data["AskVolume4"])
            tick.ask_volume_5 = adjust_price(data["AskVolume5"])

        self.gateway.on_tick(tick)

    def connect(self, address: str, server_license: str) -> None:
        """
        Start connection to server.
        """
        if not self.connect_status:
            path = get_folder_path(self.gateway_name.lower())
            self.newMdApi(str(path) + "\\Md")

            self.registerFront(address)

            self.init(
                server_license,
                "",
                "",
                "",
                ""
            )

            self.connect_status = True

    def subscribe(self, req: SubscribeRequest) -> None:
        """
        Subscribe to tick data update.
        """
        if self.connect_status:
            uft_req = {
                "ExchangeID": req.exchange.value,
                "InstrumentID": req.symbol
            }

            self.reqid += 1
            self.reqDepthMarketDataSubscribe(uft_req, self.reqid)

    def close(self) -> None:
        """
        Close the connection.
        """
        if self.connect_status:
            self.exit()


class UftTdApi(TdApi):
    """"""

    def __init__(self, gateway: UftGateway):
        """Constructor"""
        super(UftTdApi, self).__init__()

        self.gateway: UftGateway = gateway
        self.gateway_name: str = gateway.gateway_name

        self.reqid: int = 0
        self.order_ref: int = 0

        self.connect_status: bool = False
        self.login_status: bool = False
        self.auth_staus: bool = False
        self.login_failed: bool = False

        self.userid: str = ""
        self.password: str = ""
        self.auth_code: str = ""
        self.appid: str = ""

        self.frontid: int = 0
        self.sessionid: int = 0

        self.positions: Dict[str, PositionData] = {}
        self.orders: Dict[str, OrderData] = {}

    def onFrontConnected(self) -> None:
        """"""
        self.gateway.write_log("交易服务器连接成功")

        if self.auth_code:
            self.authenticate()
        else:
            self.login()

    def onFrontDisconnected(self, reason: int) -> None:
        """"""
        self.login_status = False
        self.gateway.write_log(f"交易服务器连接断开，原因{reason}")

    def onRspAuthenticate(
        self,
        data: dict,
        error: dict,
        reqid: int,
        last: bool
    ) -> None:
        """"""
        if not error["ErrorID"]:
            self.auth_staus = True
            self.gateway.write_log("交易服务器授权验证成功")
            self.login()
        else:
            self.gateway.write_error("交易服务器授权验证失败", error)

    def onRspUserLogin(
        self,
        data: dict,
        error: dict,
        reqid: int,
        last: bool
    ) -> None:
        """"""
        if not error["ErrorID"]:
            self.sessionid = data["SessionID"]
            self.login_status = True
            self.gateway.write_log("交易服务器登录成功")

            self.reqid += 1
            self.reqQryInstrument({}, self.reqid)

            self.query_order()
            self.query_trade()
        else:
            self.login_failed = True

            self.gateway.write_error("交易服务器登录失败", error)

    def query_order(self) -> None:
        """"""
        self.reqid += 1
        self.reqQryOrder({}, self.reqid)

    def onRspQryOrder(
        self,
        data: dict,
        error: dict,
        reqid: int,
        last: bool
    ) -> None:
        """"""
        if not data and last:
            self.gateway.write_log("委托信息查询成功")
            return

        self.update_order(data)

    def onRspQryTrade(
        self,
        data: dict,
        error: dict,
        reqid: int,
        last: bool
    ) -> None:
        """"""
        if not data and last:
            self.gateway.write_log("成交信息查询成功")
            return

        self.update_trade(data)

    def query_trade(self) -> None:
        """"""
        self.reqid += 1
        self.reqQryTrade({}, self.reqid)

    def onRspErrorOrderInsert(
        self,
        data: dict,
        error: dict,
        reqid: int,
        last: bool
    ) -> None:
        """"""
        self.gateway.write_error("委托下单失败", error)

        order_ref = data["OrderRef"]
        orderid = f"{self.sessionid}_{order_ref}"

        symbol = data["InstrumentID"]
        exchange = EXCHANGE_UFT2VT[data["ExchangeID"]]

        order = OrderData(
            symbol=symbol,
            exchange=exchange,
            orderid=orderid,
            direction=DIRECTION_UFT2VT[data["Direction"]],
            offset=OFFSET_UFT2VT.get(data["OffsetFlag"], Offset.NONE),
            price=data["OrderPrice"],
            volume=data["OrderVolume"],
            status=Status.REJECTED,
            gateway_name=self.gateway_name
        )
        self.gateway.on_order(order)

        self.gateway.write_error("交易委托失败", error)

    def onRspOrderAction(self, data: dict, error: dict, reqid: int, last: bool) -> None:
        """"""
        if error["ErrorID"]:
            self.gateway.write_error("交易撤单失败", error)

    def onRspQueryMaxOrderVolume(self, data: dict, error: dict, reqid: int, last: bool) -> None:
        """"""
        pass

    def onRspQryPosition(
        self,
        data: dict,
        error: dict,
        reqid: int,
        last: bool
    ) -> None:
        """"""
        if not data and last:
            for position in self.positions.values():
                self.gateway.on_position(position)

            self.positions.clear()
            return

        # Check if contract data received
        if data["InstrumentID"] in symbol_name_map:
            # Get buffered position object
            key = f"{data['InstrumentID'], data['Direction']}"
            position = self.positions.get(key, None)
            if not position:
                position = PositionData(
                    symbol=data["InstrumentID"],
                    exchange=EXCHANGE_UFT2VT[data["ExchangeID"]],
                    direction=DIRECTION_UFT2VT[data["Direction"]],
                    gateway_name=self.gateway_name
                )
                self.positions[key] = position

            # For SHFE and INE position data update
            if position.exchange in [Exchange.SHFE, Exchange.INE]:
                if data["YdPositionVolume"] and not data["TodayPositionVolume"]:
                    position.yd_volume = data["PositionVolume"]
            # For other exchange position data update
            else:
                position.yd_volume = data["PositionVolume"] - data["TodayPositionVolume"]

            # Get contract size (spread contract has no size value)
            size = symbol_size_map.get(position.symbol, 0)

            # Calculate previous position cost
            cost = position.price * position.volume * size

            # Update new position volume
            position.volume += data["PositionVolume"]
            position.pnl += data["PositionProfit"]

            # Calculate average position price
            if position.volume and size:
                cost += data["PositionCost"]
                position.price = cost / (position.volume * size)

            position.frozen += data["CloseFrozenVolume"]

    def onRspQryTradingAccount(
        self,
        data: dict,
        error: dict,
        reqid: int,
        last: bool
    ) -> None:
        """"""
        if not data:
            return

        account = AccountData(
            accountid=data["AccountID"],
            balance=data["FrozenBalance"] + data["AvailableBalance"],
            frozen=data["FrozenBalance"],
            gateway_name=self.gateway_name
        )

        self.gateway.on_account(account)

    def onRspQryInstrument(
        self,
        data: dict,
        error: dict,
        reqid: int,
        last: bool
    ) -> None:
        """
        Callback of instrument query.
        """
        if not data and last:
            self.gateway.write_log("合约信息查询成功")
            return

        product = PRODUCT_UFT2VT.get(data["ProductType"], None)
        if product:
            contract = ContractData(
                symbol=data["InstrumentID"],
                exchange=EXCHANGE_UFT2VT[data["ExchangeID"]],
                name=data["InstrumentName"],
                product=product,
                size=data["VolumeMultiple"],
                pricetick=data["PriceTick"],
                gateway_name=self.gateway_name
            )

            # For option only
            if contract.product == Product.OPTION:
                # Remove C/P suffix of CZCE option product name
                if contract.exchange == Exchange.CZCE:
                    contract.option_portfolio = data["ProductID"][:-1]
                else:
                    contract.option_portfolio = data["ProductID"]

                contract.option_underlying = data["UnderlyingInstrID"]
                contract.option_type = OPTIONTYPE_UFT2VT.get(data["OptionsType"], None)
                contract.option_strike = data["ExercisePrice"]
                contract.option_index = str(data["ExercisePrice"])
                contract.option_expiry = datetime.strptime(str(data["ExpireDate"]), "%Y%m%d")

            self.gateway.on_contract(contract)

            symbol_name_map[contract.symbol] = contract.name
            symbol_size_map[contract.symbol] = contract.size

    def onRtnOrder(self, data: dict) -> None:
        """
        Callback of order status update.
        """
        self.update_order(data)

    def update_order(self, data: dict) -> None:
        """"""
        sessionid = data["SessionID"]
        order_ref = data["OrderRef"]
        orderid = f"{sessionid}_{order_ref}"

        order = self.orders.get(orderid, None)
        insert_time = generate_time(data["InsertTime"])
        timestamp = f"{data['InsertDate']} {insert_time}"
        dt = datetime.strptime(timestamp, "%Y%m%d %H:%M:%S")
        dt = dt.replace(tzinfo=CHINA_TZ)

        if not order:
            order = OrderData(
                symbol=data["InstrumentID"],
                exchange=EXCHANGE_UFT2VT[data["ExchangeID"]],
                orderid=orderid,
                type=ORDERTYPE_UFT2VT[data["OrderCommand"]],
                direction=DIRECTION_UFT2VT[data["Direction"]],
                offset=OFFSET_UFT2VT[data["OffsetFlag"]],
                price=data["OrderPrice"],
                volume=data["OrderVolume"],
                traded=data["TradeVolume"],
                status=STATUS_UFT2VT.get(data["OrderStatus"], Status.SUBMITTING),
                datetime=dt,
                gateway_name=self.gateway_name
            )
            self.orders[orderid] = order
        else:
            order.traded = data["OrderVolume"]
            order.status = STATUS_UFT2VT.get(data["OrderStatus"], Status.SUBMITTING)

        self.gateway.on_order(order)

    def onRtnTrade(self, data: dict) -> None:
        """
        Callback of trade status update.
        """
        self.update_trade(data)

    def update_trade(self, data: dict) -> None:
        """"""
        symbol = data["InstrumentID"]
        exchange = EXCHANGE_UFT2VT[data["ExchangeID"]]
        sessionid = data["SessionID"]
        order_ref = data["OrderRef"]
        orderid = f"{sessionid}_{order_ref}"

        order = self.orders.get(orderid, None)
        if order:
            order.traded += data["TradeVolume"]

            if order.traded < order.volume:
                order.status = Status.PARTTRADED
            else:
                order.status = Status.ALLTRADED

            self.gateway.on_order(order)

        trade_time = generate_time(data["TradeTime"])
        timestamp = f"{data['TradeDate']} {trade_time}"
        dt = datetime.strptime(timestamp, "%H:%M:%S")
        dt = dt.replace(tzinfo=CHINA_TZ)

        trade = TradeData(
            symbol=symbol,
            exchange=exchange,
            orderid=orderid,
            tradeid=data["TradeID"],
            direction=DIRECTION_UFT2VT[data["Direction"]],
            offset=OFFSET_UFT2VT[data["OffsetFlag"]],
            price=data["TradePrice"],
            volume=data["TradeVolume"],
            datetime=dt,
            gateway_name=self.gateway_name
        )
        self.gateway.on_trade(trade)

    def connect(
        self,
        address: str,
        server_license: str,
        userid: str,
        password: str,
        auth_code: str,
        appid: str
    ) -> None:
        """
        Start connection to server.
        """
        self.userid = userid
        self.password = password
        self.auth_code = auth_code
        self.appid = appid

        if not self.connect_status:
            path = get_folder_path(self.gateway_name.lower())
            self.newTradeApi(str(path) + "\\Td")

            self.rgisterSubModel("1")  # ??

            self.registerFront(address)
            self.init(
                server_license,
                "",
                "",
                "",
                ""
            )

            self.connect_status = True
        else:
            self.authenticate()

    def authenticate(self):
        """
        Authenticate with auth_code and appid.
        """
        req = {
            "AccountID": self.userid,
            "Password": self.password,
            "AuthCode": self.auth_code,
            "AppID": self.appid
        }

        self.reqid += 1
        self.reqAuthenticate(req, self.reqid)

    def login(self) -> None:
        """
        Login onto server.
        """
        if self.login_failed:
            return

        req = {
            "AccountID": self.userid,
            "Password": self.password,
            "UserApplicationType": "q",
            "UserApplicationInfo": "",
            "MacAddress": "",
            "IPAddress": "",

        }

        self.reqid += 1
        self.reqUserLogin(req, self.reqid)

    def send_order(self, req: OrderRequest) -> str:
        """
        Send new order.
        """
        if req.offset not in OFFSET_VT2UFT:
            self.gateway.write_log("请选择开平方向")
            return ""

        self.order_ref += 1

        uft_req = {
            "OrderRef": str(self.order_ref),
            "ExchangeID": EXCHANGE_VT2UFT[req.exchange],
            "InstrumentID": req.symbol,
            "Direction": DIRECTION_VT2UFT[req.direction],
            "OffsetFlag": OFFSET_VT2UFT[req.offset],
            "HedgeType": "0",
            "OrderPrice": req.price,
            "OrderVolume": int(req.volume),
            "OrderCommand": ORDERTYPE_VT2UFT[req.type],
            "SwapOrderFlag": "0"
        }

        self.reqid += 1
        self.reqOrderInsert(uft_req, self.reqid)

        orderid = f"{self.sessionid}_{self.order_ref}"
        order = req.create_order_data(orderid, self.gateway_name)
        self.gateway.on_order(order)

        return order.vt_orderid

    def cancel_order(self, req: CancelRequest) -> None:
        """
        Cancel existing order.
        """
        sessionid, order_ref = req.orderid.split("_")

        uft_req = {
            # "InstrumentID": req.symbol,
            # "ExchangeID": EXCHANGE_VT2UFT[req.exchange],
            "SessionID": int(sessionid),
            "OrderRef": order_ref
        }

        self.reqid += 1
        self.reqOrderAction(uft_req, self.reqid)

    def query_account(self) -> None:
        """
        Query account balance data.
        """
        self.reqid += 1
        self.reqQryTradingAccount({}, self.reqid)

    def query_position(self) -> None:
        """
        Query position holding data.
        """
        self.reqid += 1
        self.reqQryPosition({}, self.reqid)

    def close(self):
        """"""
        if self.connect_status:
            self.exit()


def adjust_price(price: float) -> float:
    """"""
    if price == MAX_FLOAT:
        price = 0
    return price


def generate_time(data: int) -> str:
    """"""
    buf = str(data)
    if len(buf) < 6:
        buf = "0" + buf

    hour = buf[:2]
    minute = buf[2:4]
    second = buf[4:]
    return f"{hour}:{minute}:{second}"
