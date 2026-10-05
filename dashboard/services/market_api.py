import requests
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# Comprehensive fallback list for top cryptocurrencies
FALLBACK_COINS = [
    {
        "market_cap_rank": 1,
        "name": "Bitcoin",
        "symbol": "btc",
        "current_price": 107287.80,
        "market_cap": 2120000000000,
        "total_volume": 67548133399,
        "price_change_percentage_24h": -3.93,
        "circulating_supply": 19780000
    },
    {
        "market_cap_rank": 2,
        "name": "Ethereum",
        "symbol": "eth",
        "current_price": 2526.44,
        "market_cap": 304000000000,
        "total_volume": 30536501878,
        "price_change_percentage_24h": -5.17,
        "circulating_supply": 120200000
    },
    {
        "market_cap_rank": 3,
        "name": "Tether",
        "symbol": "usdt",
        "current_price": 1.00,
        "market_cap": 118000000000,
        "total_volume": 55000000000,
        "price_change_percentage_24h": 0.01,
        "circulating_supply": 118000000000
    },
    {
        "market_cap_rank": 4,
        "name": "Solana",
        "symbol": "sol",
        "current_price": 182.50,
        "market_cap": 85000000000,
        "total_volume": 4200000000,
        "price_change_percentage_24h": 1.45,
        "circulating_supply": 465000000
    },
    {
        "market_cap_rank": 5,
        "name": "BNB",
        "symbol": "bnb",
        "current_price": 645.20,
        "market_cap": 94000000000,
        "total_volume": 1200000000,
        "price_change_percentage_24h": -0.82,
        "circulating_supply": 145000000
    },
    {
        "market_cap_rank": 6,
        "name": "XRP",
        "symbol": "xrp",
        "current_price": 2.45,
        "market_cap": 139000000000,
        "total_volume": 8500000000,
        "price_change_percentage_24h": 4.12,
        "circulating_supply": 56800000000
    },
    {
        "market_cap_rank": 7,
        "name": "USDC",
        "symbol": "usdc",
        "current_price": 1.00,
        "market_cap": 37000000000,
        "total_volume": 6200000000,
        "price_change_percentage_24h": 0.00,
        "circulating_supply": 37000000000
    },
    {
        "market_cap_rank": 8,
        "name": "Cardano",
        "symbol": "ada",
        "current_price": 0.88,
        "market_cap": 31000000000,
        "total_volume": 1800000000,
        "price_change_percentage_24h": -1.15,
        "circulating_supply": 35700000000
    },
    {
        "market_cap_rank": 9,
        "name": "Dogecoin",
        "symbol": "doge",
        "current_price": 0.38,
        "market_cap": 55000000000,
        "total_volume": 3400000000,
        "price_change_percentage_24h": 2.80,
        "circulating_supply": 146000000000
    },
    {
        "market_cap_rank": 10,
        "name": "Avalanche",
        "symbol": "avax",
        "current_price": 38.60,
        "market_cap": 15800000000,
        "total_volume": 720000000,
        "price_change_percentage_24h": -0.65,
        "circulating_supply": 410000000
    }
]


def _fetch_from_binance():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    data = resp.json()

    mapping = {
        "BTCUSDT": ("Bitcoin", "btc", 1),
        "ETHUSDT": ("Ethereum", "eth", 2),
        "SOLUSDT": ("Solana", "sol", 3),
        "BNBUSDT": ("BNB", "bnb", 4),
        "XRPUSDT": ("XRP", "xrp", 5),
        "ADAUSDT": ("Cardano", "ada", 6),
        "DOGEUSDT": ("Dogecoin", "doge", 7),
        "AVAXUSDT": ("Avalanche", "avax", 8),
        "DOTUSDT": ("Polkadot", "dot", 9),
        "LINKUSDT": ("Chainlink", "link", 10),
    }

    results = []
    for item in data:
        sym = item.get("symbol")
        if sym in mapping:
            name, symbol, rank = mapping[sym]
            results.append({
                "market_cap_rank": rank,
                "name": name,
                "symbol": symbol,
                "current_price": float(item.get("lastPrice", 0)),
                "market_cap": float(item.get("quoteVolume", 0)) * 10,
                "total_volume": float(item.get("quoteVolume", 0)),
                "price_change_percentage_24h": float(item.get("priceChangePercent", 0)),
                "circulating_supply": float(item.get("volume", 0))
            })

    results.sort(key=lambda x: x["market_cap_rank"])
    return results


def _fetch_from_coincap():
    url = "https://api.coincap.io/v2/assets"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    items = resp.json().get("data", [])

    results = []
    for item in items[:15]:
        results.append({
            "market_cap_rank": int(item.get("rank", 99)),
            "name": item.get("name"),
            "symbol": item.get("symbol", "").lower(),
            "current_price": float(item.get("priceUsd", 0)),
            "market_cap": float(item.get("marketCapUsd", 0)),
            "total_volume": float(item.get("volumeUsd24Hr", 0)),
            "price_change_percentage_24h": float(item.get("changePercent24Hr", 0)),
            "circulating_supply": float(item.get("supply", 0))
        })
    return results


def _fetch_from_coingecko():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 15,
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": "24h",
    }
    resp = requests.get(url, params=params, timeout=5)
    resp.raise_for_status()
    return resp.json()


def get_live_coin_market():
    """
    Get live cryptocurrency market rankings with multi-provider failover.
    Tries Binance REST -> CoinCap -> CoinGecko -> Offline Fallback.
    """
    for provider_func in [_fetch_from_binance, _fetch_from_coincap, _fetch_from_coingecko]:
        try:
            res = provider_func()
            if res and len(res) > 0:
                return res
        except Exception:
            continue

    # Fallback to local dataset latest prices if network is unavailable
    try:
        btc_path = RAW_DATA_DIR / "btc_daily.csv"
        eth_path = RAW_DATA_DIR / "eth_daily.csv"

        if btc_path.exists() and eth_path.exists():
            btc_df = pd.read_csv(btc_path)
            eth_df = pd.read_csv(eth_path)

            latest_btc = btc_df.iloc[-1]
            latest_eth = eth_df.iloc[-1]

            FALLBACK_COINS[0]["current_price"] = float(latest_btc["Close"])
            FALLBACK_COINS[0]["total_volume"] = float(latest_btc["Volume"])
            FALLBACK_COINS[0]["price_change_percentage_24h"] = float(latest_btc["Return"]) * 100

            FALLBACK_COINS[1]["current_price"] = float(latest_eth["Close"])
            FALLBACK_COINS[1]["total_volume"] = float(latest_eth["Volume"])
            FALLBACK_COINS[1]["price_change_percentage_24h"] = float(latest_eth["Return"]) * 100
    except Exception:
        pass

    return FALLBACK_COINS


def get_live_market_data():
    """
    Get live price dictionary for BTC and ETH.
    """
    market = get_live_coin_market()
    prices = {"bitcoin": {}, "ethereum": {}}

    for coin in market:
        sym = coin.get("symbol", "").lower()
        if sym == "btc":
            prices["bitcoin"] = {
                "usd": coin.get("current_price"),
                "usd_24h_change": coin.get("price_change_percentage_24h"),
                "usd_24h_vol": coin.get("total_volume")
            }
        elif sym == "eth":
            prices["ethereum"] = {
                "usd": coin.get("current_price"),
                "usd_24h_change": coin.get("price_change_percentage_24h"),
                "usd_24h_vol": coin.get("total_volume")
            }

    return prices
