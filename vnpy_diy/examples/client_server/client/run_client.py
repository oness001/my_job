from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.gateway.rpc import RpcGateway
from vnpy.app.cta_strategy import CtaStrategyApp


def main():
    """"""

    event_engine = EventEngine()

    main_engine = MainEngine(event_engine)

    main_engine.add_gateway(RpcGateway)
    main_engine.add_app(CtaStrategyApp)

    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()



if __name__ == "__main__":
    main()
