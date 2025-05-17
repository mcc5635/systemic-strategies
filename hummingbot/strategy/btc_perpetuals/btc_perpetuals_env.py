import yaml
import pandas as pd
from datetime import datetime, timedelta
from coinbase.rest import RESTClient
import time
import requests
import numpy as np
import os

class BTCPerpetualsEnv:
    """
    Environment and data handling for BTC Perpetuals strategy.
    """
    def load_data(self, source: str):
        """Placeholder for loading data from a given source."""
        pass

    def preprocess(self, data):
        """Placeholder for preprocessing data (cleaning, alignment, normalization)."""
        pass

    def load_ohlcv_from_coinbase(self, start: int, end: int) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for BTC-USD from Coinbase Advanced Trade API using the coinbase-advanced-py SDK.
        Supports batching for long time windows (Coinbase limit: <350 candles per request).
        Args:
            start: UNIX timestamp (seconds since epoch)
            end: UNIX timestamp (seconds since epoch)
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, vwap
        """
        # Load API keys
        with open('conf/conf_coinbase_perpetual.yml', 'r') as f:
            config = yaml.safe_load(f)
        api_key = config['coinbase_perpetual_api_key']
        api_secret = config['coinbase_perpetual_api_secret']
        # Initialize RESTClient
        client = RESTClient(api_key=api_key, api_secret=api_secret)
        product_id = 'BTC-USD'  # or 'BTC-USD-PERP' if available
        MAX_CANDLES = 300
        interval = 60  # 1 minute in seconds
        all_candles = []
        batch_start = start
        while batch_start < end:
            batch_end = min(batch_start + interval * MAX_CANDLES, end)
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
            time.sleep(0.25)  # avoid rate limits
        if not all_candles:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap'])
        df = pd.concat(all_candles, ignore_index=True)
        if df.empty:
            return df
        df = df.sort_values('timestamp')
        # Calculate VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap']]

    def load_funding_rate_from_binance(self, start: int, end: int, symbol: str = 'BTCUSDT') -> pd.DataFrame:
        """
        Fetch historical funding rates for a given symbol from Binance Futures API.
        Args:
            start: UNIX timestamp (seconds since epoch)
            end: UNIX timestamp (seconds since epoch)
            symbol: Binance futures symbol (default: BTCUSDT)
        Returns:
            DataFrame with columns: timestamp, funding_rate
        """
        # Binance API expects ms timestamps
        start_ms = start * 1000
        end_ms = end * 1000
        url = f'https://fapi.binance.com/fapi/v1/fundingRate'
        params = {
            'symbol': symbol,
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': 1000  # max per request
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

    def load_oi_from_binance(self, start: int, end: int, symbol: str = 'BTCUSDT', period: str = '5m') -> pd.DataFrame:
        """
        Fetch historical open interest (OI) for a given symbol from Binance Futures API.
        Args:
            start: UNIX timestamp (seconds since epoch)
            end: UNIX timestamp (seconds since epoch)
            symbol: Binance futures symbol (default: BTCUSDT)
            period: OI interval (default: '5m')
        Returns:
            DataFrame with columns: timestamp, open_interest
        Note:
            Binance OI period minimum is 5m. To align with 1m OHLCV, forward-fill or interpolate as needed.
        """
        start_ms = start * 1000
        end_ms = end * 1000
        url = 'https://fapi.binance.com/futures/data/openInterestHist'
        params = {
            'symbol': symbol,
            'period': period,
            'limit': 200,  # max per request
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

    def load_cvd_from_hyperliquid(self, start: int, end: int, symbol: str = 'BTC-PERP') -> pd.DataFrame:
        """
        Fetch trades from Hyperliquid REST API for BTC-PERP, aggregate to 1-minute intervals, and calculate CVD.
        Args:
            start: UNIX timestamp (seconds since epoch)
            end: UNIX timestamp (seconds since epoch)
            symbol: Hyperliquid symbol (default: 'BTC-PERP')
        Returns:
            DataFrame with columns: timestamp, cvd (1-minute intervals)
        """
        # Hyperliquid API endpoint for trades (example, update if needed)
        url = f'https://api.hyperliquid.xyz/info'
        params = {
            'type': 'trades',
            'symbol': symbol,
            'startTime': start * 1000,  # ms
            'endTime': end * 1000      # ms
        }
        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[ERROR] Exception fetching trades from Hyperliquid: {e}")
            return pd.DataFrame(columns=['timestamp', 'cvd'])
        # Expecting data as a list of trades with 'timestamp', 'size', 'side' (or similar)
        if not data or 'trades' not in data:
            print(f"[ERROR] No trade data returned from Hyperliquid.")
            return pd.DataFrame(columns=['timestamp', 'cvd'])
        trades = data['trades']
        # Build DataFrame
        df = pd.DataFrame([
            {
                'timestamp': pd.to_datetime(trade['timestamp'], unit='ms'),
                'size': float(trade['size']),
                'side': trade['side']  # 'buy' or 'sell'
            }
            for trade in trades if 'timestamp' in trade and 'size' in trade and 'side' in trade
        ])
        if df.empty:
            return pd.DataFrame(columns=['timestamp', 'cvd'])
        # Assign +size for buy, -size for sell
        df['signed_size'] = np.where(df['side'].str.lower() == 'buy', df['size'], -df['size'])
        # Resample to 1-minute intervals, summing signed_size
        df.set_index('timestamp', inplace=True)
        cvd_1m = df['signed_size'].resample('1T').sum().fillna(0).cumsum()
        result = cvd_1m.reset_index().rename(columns={'signed_size': 'cvd'})
        return result[['timestamp', 'cvd']]

    def load_cvd_from_binance(self, start: int, end: int, symbol: str = 'BTCUSDT') -> pd.DataFrame:
        """
        Fetch trades from Binance for BTCUSDT, aggregate to 1-minute intervals, and calculate CVD.
        Args:
            start: UNIX timestamp (seconds since epoch)
            end: UNIX timestamp (seconds since epoch)
            symbol: Binance symbol (default: 'BTCUSDT')
        Returns:
            DataFrame with columns: timestamp, cvd (1-minute intervals)
        """
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
            # Update start_ms to the last trade's timestamp + 1ms to avoid overlap
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
        # Assign +quantity for buy (aggressor is buyer, m=False), -quantity for sell (m=True)
        df['signed_qty'] = np.where(df['is_buyer_maker'], -df['quantity'], df['quantity'])
        # Resample to 1-minute intervals, summing signed_qty
        df.set_index('timestamp', inplace=True)
        cvd_1m = df['signed_qty'].resample('1T').sum().fillna(0).cumsum()
        result = cvd_1m.reset_index().rename(columns={'signed_qty': 'cvd'})
        return result[['timestamp', 'cvd']]

    def load_net_flow_from_numia(self, start, end, asset='BTC'):
        """
        Fetches on-chain net flow data for BTC from Numia API, aggregates to 1-minute intervals.
        Args:
            start (int): Start timestamp (UNIX seconds)
            end (int): End timestamp (UNIX seconds)
            asset (str): Asset symbol, default 'BTC'
        Returns:
            pd.DataFrame: Columns ['timestamp', 'net_flow'] (1-min intervals, UTC)
        """
        api_key = os.environ.get('NUMIA_API_KEY')
        if not api_key:
            raise ValueError('NUMIA_API_KEY not set in environment')
        url = 'https://api.numia.xyz/v2/netflow'
        params = {
            'symbol': asset,
            'interval': '1m',
            'start_time': start,
            'end_time': end
        }
        headers = {'x-api-key': api_key}
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Error fetching Numia net flow: {e}")
            return pd.DataFrame(columns=['timestamp', 'net_flow'])
        # Expecting data['data'] to be a list of dicts with 'timestamp' and 'net_flow'
        if 'data' not in data or not isinstance(data['data'], list):
            print("Numia API response missing 'data' field or not a list")
            return pd.DataFrame(columns=['timestamp', 'net_flow'])
        df = pd.DataFrame(data['data'])
        if 'timestamp' not in df or 'net_flow' not in df:
            print("Numia data missing required columns")
            return pd.DataFrame(columns=['timestamp', 'net_flow'])
        # Convert timestamp to int (if needed) and sort
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df = df.sort_values('timestamp')
        # Ensure 1-min frequency, fill missing with 0
        df = df.set_index('timestamp').resample('1min').sum().fillna(0).reset_index()
        return df[['timestamp', 'net_flow']]

if __name__ == '__main__':
    env = BTCPerpetualsEnv()
    # Coinbase API: max 300 candles, 1m granularity = 5h window, both times in past, 5s cushion
    SPAN = 60 * 300  # 5 hours in seconds
    end = int(time.time()) - 5  # 5-second cushion to ensure not in future
    start = end - SPAN
    df = env.load_ohlcv_from_coinbase(start, end)
    print(df.head())
    print(df.tail())
    # Test funding rate fetch from Binance
    funding_df = env.load_funding_rate_from_binance(start, end)
    print(funding_df.head())
    print(funding_df.tail())
    oi_df = env.load_oi_from_binance(start, end)
    print(oi_df.head())
    print(oi_df.tail())
    # cvd_df = env.load_cvd_from_hyperliquid(start, end)
    # print(cvd_df.head())
    # print(cvd_df.tail())
    cvd_df = env.load_cvd_from_binance(start, end)
    print(cvd_df.head())
    print(cvd_df.tail())
    # Test Numia net flow fetch
    # net_flow_df = env.load_net_flow_from_numia(start, end)
    # print(net_flow_df.head())
    # print(net_flow_df.tail()) 