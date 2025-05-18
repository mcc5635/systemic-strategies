import yaml
import pandas as pd
from datetime import datetime, timedelta
from coinbase.rest import RESTClient
import time
import requests
import numpy as np
import os

class SOLUSDCEnv:
    """
    Environment for SOL/USDC strategy.
    Handles data loading and preprocessing.
    """
    def __init__(self):
        pass

    def load_ohlcv_from_coinbase(self, start: int, end: int) -> pd.DataFrame:
        with open('conf/conf_coinbase_perpetual.yml', 'r') as f:
            config = yaml.safe_load(f)
        api_key = config['coinbase_perpetual_api_key']
        api_secret = config['coinbase_perpetual_api_secret']
        client = RESTClient(api_key=api_key, api_secret=api_secret)
        product_id = 'SOL-USD'  # Adjusted for SOL/USDC
        MAX_CANDLES = 300
        interval = 60  # 1 minute in seconds
        all_candles = []
        batch_start = start
        total = end - start
        while batch_start < end:
            batch_end = min(batch_start + interval * MAX_CANDLES, end)
            percent = 100 * (batch_start - start) / total if total > 0 else 0
            print(f"[PROGRESS] Fetching: {datetime.utcfromtimestamp(batch_start).strftime('%Y-%m-%d %H:%M:%S')} ({percent:.2f}% complete)")
            print(f"[DEBUG] Requesting candles from API with params: product_id={product_id}, start={batch_start}, end={batch_end}, granularity=ONE_MINUTE")
            try:
                candles_response = client.get_public_candles(
                    product_id=product_id,
                    start=batch_start,
                    end=batch_end,
                    granularity="ONE_MINUTE"
                )
                candles = getattr(candles_response, 'candles', None)
            except Exception as e:
                print(f"[ERROR] Exception during API call: {e}")
                break
            if candles is None or not candles:
                print(f"[WARN] No candles returned for batch {batch_start} to {batch_end}")
                batch_start = batch_end
                continue
            batch_df = pd.DataFrame([{
                'timestamp': pd.to_datetime(candle['start'], unit='s'),
                'open': float(candle['open']),
                'high': float(candle['high']),
                'low': float(candle['low']),
                'close': float(candle['close']),
                'volume': float(candle['volume'])
            } for candle in candles if candle is not None])
            all_candles.append(batch_df)
            batch_start = batch_end
            time.sleep(0.25)
        if not all_candles:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap'])
        df = pd.concat(all_candles, ignore_index=True)
        if df.empty:
            return df
        df = df.sort_values('timestamp')
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap']]

    def load_funding_rate_from_binance(self, start: int, end: int, symbol: str = 'SOLUSDT') -> pd.DataFrame:
        start_ms = start * 1000
        end_ms = end * 1000
        url = f'https://fapi.binance.com/fapi/v1/fundingRate'
        params = {
            'symbol': symbol,
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': 1000
        }
        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[ERROR] Exception fetching funding rates from Binance: {e}")
            return pd.DataFrame(columns=['timestamp', 'funding_rate'])
        if not data:
            return pd.DataFrame(columns=['timestamp', 'funding_rate'])
        df = pd.DataFrame([
            {
                'timestamp': pd.to_datetime(item['fundingTime'], unit='ms'),
                'funding_rate': float(item['fundingRate'])
            }
            for item in data if 'fundingTime' in item and 'fundingRate' in item
        ])
        df = df.sort_values('timestamp')
        return df[['timestamp', 'funding_rate']]

    def load_oi_from_binance(self, start: int, end: int, symbol: str = 'SOLUSDT', period: str = '5m') -> pd.DataFrame:
        start_ms = start * 1000
        end_ms = end * 1000
        url = 'https://fapi.binance.com/futures/data/openInterestHist'
        params = {
            'symbol': symbol,
            'period': period,
            'limit': 200,
            'startTime': start_ms,
            'endTime': end_ms
        }
        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[ERROR] Exception fetching OI from Binance: {e}")
            return pd.DataFrame(columns=['timestamp', 'open_interest'])
        if not data:
            return pd.DataFrame(columns=['timestamp', 'open_interest'])
        df = pd.DataFrame([
            {
                'timestamp': pd.to_datetime(item['timestamp'], unit='ms'),
                'open_interest': float(item['sumOpenInterest'])
            }
            for item in data if 'timestamp' in item and 'sumOpenInterest' in item
        ])
        df = df.sort_values('timestamp')
        return df[['timestamp', 'open_interest']]

    def load_cvd_from_binance(self, start: int, end: int, symbol: str = 'SOLUSDT') -> pd.DataFrame:
        start_ms = start * 1000
        end_ms = end * 1000
        url = 'https://fapi.binance.com/fapi/v1/aggTrades'
        all_trades = []
        last_trade_id = None
        while True:
            params = {
                'symbol': symbol,
                'startTime': start_ms,
                'endTime': end_ms,
                'limit': 1000
            }
            if last_trade_id is not None:
                params['fromId'] = last_trade_id + 1
            try:
                resp = requests.get(url, params=params)
                resp.raise_for_status()
                trades = resp.json()
            except Exception as e:
                print(f"[ERROR] Exception fetching trades from Binance: {e}")
                break
            if not trades:
                break
            all_trades.extend(trades)
            if len(trades) < 1000:
                break
            last_trade_id = trades[-1]['a']
            start_ms = trades[-1]['T'] + 1
            if start_ms > end_ms:
                break
        if not all_trades:
            return pd.DataFrame(columns=['timestamp', 'cvd'])
        df = pd.DataFrame([
            {
                'timestamp': pd.to_datetime(trade['T'], unit='ms'),
                'quantity': float(trade['q']),
                'is_buyer_maker': trade['m']
            }
            for trade in all_trades if 'T' in trade and 'q' in trade and 'm' in trade
        ])
        if df.empty:
            return pd.DataFrame(columns=['timestamp', 'cvd'])
        df['signed_qty'] = np.where(df['is_buyer_maker'], -df['quantity'], df['quantity'])
        df.set_index('timestamp', inplace=True)
        cvd_1m = df['signed_qty'].resample('1T').sum().fillna(0).cumsum()
        result = cvd_1m.reset_index().rename(columns={'signed_qty': 'cvd'})
        return result[['timestamp', 'cvd']]

    def load_ohlcv_from_geckoterminal(self, start, end, pool_address=None, timeframe=None):
        now = int(time.time())
        if start > now or end > now:
            print(f"[WARN] Requested date range includes future timestamps. Now: {now}, Start: {start}, End: {end}")
        default_pool_address = '8sLbNZoA1cfnvMJLPfp98ZLAnFSYCFApfJKMbiXNLwxj'  # SOL/USDC pool
        if pool_address is None:
            pool_address = default_pool_address
        print(f"[INFO] Using GeckoTerminal Raydium pool address: {pool_address}")
        timeframes = ['hour', 'day', 'minute'] if timeframe is None else [timeframe]
        params = {'from': start, 'to': end}
        headers = {'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}
        for tf in timeframes:
            url = f"https://api.geckoterminal.com/api/v2/networks/solana/pools/{pool_address}/ohlcv/{tf}"
            print(f"[DEBUG] GeckoTerminal request URL: {url}")
            print(f"[DEBUG] GeckoTerminal request params: {params}")
            try:
                resp = requests.get(url, headers=headers, params=params)
                print(f"[DEBUG] GeckoTerminal raw response: {resp.text}")
                if resp.status_code == 400:
                    print(f"[ERROR] 400 Bad Request. Response: {resp.text}")
                    continue
                resp.raise_for_status()
                data = resp.json()
                if (
                    isinstance(data.get('data'), dict) and
                    'attributes' in data['data'] and
                    'ohlcv_list' in data['data']['attributes']
                ):
                    ohlcv_list = data['data']['attributes']['ohlcv_list']
                    ohlcv_data = []
                    for entry in ohlcv_list:
                        ohlcv_data.append({
                            'timestamp': pd.to_datetime(entry[0], unit='s'),
                            'open': float(entry[1]),
                            'high': float(entry[2]),
                            'low': float(entry[3]),
                            'close': float(entry[4]),
                            'volume': float(entry[5])
                        })
                    df = pd.DataFrame(ohlcv_data)
                    print(f"[INFO] Successfully parsed GeckoTerminal ohlcv_list for {tf} with {len(df)} rows.")
                    return df[["timestamp","open","high","low","close","volume"]]
                else:
                    print(f"[ERROR] Unexpected data format: {data}")
                    continue
            except Exception as e:
                print(f"[ERROR] Exception fetching OHLCV from GeckoTerminal ({tf}): {e}")
                continue
        print("[ERROR] All timeframes failed for GeckoTerminal OHLCV.")
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"]) 