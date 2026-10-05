import asyncio
import json
import threading
import websockets

try:
    from services.market_api import get_live_market_data
except ImportError:
    from .market_api import get_live_market_data


class LivePriceFeed:

    def __init__(self):
        self.prices = {
            "BTC": 107287.80,
            "ETH": 2526.44,
        }
        self.running = False
        self.thread = None

        # Immediate initialization from REST API / Market Data
        try:
            mkt_data = get_live_market_data()
            if mkt_data.get("bitcoin", {}).get("usd"):
                self.prices["BTC"] = float(mkt_data["bitcoin"]["usd"])
            if mkt_data.get("ethereum", {}).get("usd"):
                self.prices["ETH"] = float(mkt_data["ethereum"]["usd"])
        except Exception:
            pass

    async def _connect(self):

        url = (
            "wss://stream.binance.com:9443/stream"
            "?streams=btcusdt@ticker/ethusdt@ticker"
        )

        while self.running:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=20
                ) as websocket:

                    while self.running:
                        message = await websocket.recv()
                        data = json.loads(message)
                        ticker = data.get("data", {})
                        symbol = ticker.get("s")
                        price = ticker.get("c")

                        if symbol == "BTCUSDT" and price:
                            self.prices["BTC"] = float(price)
                        elif symbol == "ETHUSDT" and price:
                            self.prices["ETH"] = float(price)
            except Exception:
                # Re-sync via REST API if WebSocket is offline or blocked
                try:
                    mkt_data = get_live_market_data()
                    if mkt_data.get("bitcoin", {}).get("usd"):
                        self.prices["BTC"] = float(mkt_data["bitcoin"]["usd"])
                    if mkt_data.get("ethereum", {}).get("usd"):
                        self.prices["ETH"] = float(mkt_data["ethereum"]["usd"])
                except Exception:
                    pass

                await asyncio.sleep(5)

    def _run(self):

        asyncio.run(self._connect())

    def start(self):

        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._run,
            daemon=True
        )

        self.thread.start()

    def get_prices(self):

        return self.prices.copy()

    def stop(self):

        self.running = False
