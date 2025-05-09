import asyncio
import unittest
from decimal import Decimal
from typing import Dict
from unittest.mock import MagicMock, patch

from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.funding_info import FundingInfo
from hummingbot.scripts.funding_rate_optimizer import FundingRateOptimizer
from hummingbot.scripts.funding_rate_optimizer_config import FundingRateOptimizerConfig


class TestFundingRateOptimizer(unittest.TestCase):
    def setUp(self):
        self.config = FundingRateOptimizerConfig(
            connectors={"binance_perpetual", "bybit_perpetual"},
            trading_pairs={"BTC-USDT", "ETH-USDT"},
            leverage=5,
            position_size_usd=Decimal("1000"),
            min_funding_rate=Decimal("0.001"),
            max_position_age=24 * 60 * 60,
            order_type="MARKET",
            slippage_tolerance=Decimal("0.002"),
            stop_loss_pct=Decimal("0.05"),
            take_profit_pct=Decimal("0.02"),
            max_concurrent_positions=3,
            funding_rate_update_interval=60,
            price_update_interval=10,
        )
        self.strategy = FundingRateOptimizer(self.config)
        
        # Mock connectors
        self.strategy.connectors = {
            "binance_perpetual": MagicMock(),
            "bybit_perpetual": MagicMock(),
        }
        
    def test_initialization(self):
        """Test strategy initialization"""
        self.assertEqual(self.strategy.config.leverage, 5)
        self.assertEqual(self.strategy.config.position_size_usd, Decimal("1000"))
        self.assertEqual(len(self.strategy._active_positions), 0)
        self.assertEqual(len(self.strategy._funding_rates), 0)
        self.assertEqual(len(self.strategy._prices), 0)
        
    @patch("hummingbot.scripts.funding_rate_optimizer.FundingRateOptimizer.set_position_mode")
    @patch("hummingbot.scripts.funding_rate_optimizer.FundingRateOptimizer.set_leverage")
    async def test_on_start(self, mock_set_leverage, mock_set_position_mode):
        """Test strategy startup"""
        await self.strategy.on_start()
        
        # Check if position mode and leverage were set for all connectors and trading pairs
        self.assertEqual(mock_set_position_mode.call_count, len(self.config.connectors))
        self.assertEqual(
            mock_set_leverage.call_count,
            len(self.config.connectors) * len(self.config.trading_pairs)
        )
        
    async def test_update_funding_rates(self):
        """Test funding rate updates"""
        # Mock funding info responses
        mock_funding_info = FundingInfo(
            trading_pair="BTC-USDT",
            index_price=Decimal("50000"),
            mark_price=Decimal("50100"),
            next_funding_utc_timestamp=1234567890,
            rate=Decimal("0.001")
        )
        
        for connector in self.strategy.connectors.values():
            connector.get_funding_info.return_value = mock_funding_info
            
        await self.strategy.update_funding_rates()
        
        # Verify funding rates were updated
        for trading_pair in self.config.trading_pairs:
            for connector_name in self.config.connectors:
                self.assertIn(trading_pair, self.strategy._funding_rates)
                self.assertIn(connector_name, self.strategy._funding_rates[trading_pair])
                self.assertEqual(
                    self.strategy._funding_rates[trading_pair][connector_name].rate,
                    Decimal("0.001")
                )
                
    async def test_update_prices(self):
        """Test price updates"""
        # Mock price responses
        for connector in self.strategy.connectors.values():
            connector.get_price.return_value = Decimal("50000")
            
        await self.strategy.update_prices()
        
        # Verify prices were updated
        for trading_pair in self.config.trading_pairs:
            for connector_name in self.config.connectors:
                self.assertIn(trading_pair, self.strategy._prices)
                self.assertIn(connector_name, self.strategy._prices[trading_pair])
                self.assertEqual(
                    self.strategy._prices[trading_pair][connector_name],
                    Decimal("50000")
                )
                
    async def test_check_and_execute_opportunities(self):
        """Test opportunity detection and execution"""
        # Set up mock funding rates with a clear opportunity
        self.strategy._funding_rates = {
            "BTC-USDT": {
                "binance_perpetual": FundingInfo(
                    trading_pair="BTC-USDT",
                    index_price=Decimal("50000"),
                    mark_price=Decimal("50000"),
                    next_funding_utc_timestamp=1234567890,
                    rate=Decimal("0.0001")  # Low funding rate
                ),
                "bybit_perpetual": FundingInfo(
                    trading_pair="BTC-USDT",
                    index_price=Decimal("50000"),
                    mark_price=Decimal("50000"),
                    next_funding_utc_timestamp=1234567890,
                    rate=Decimal("0.002")  # High funding rate
                )
            }
        }
        
        # Set up mock prices
        self.strategy._prices = {
            "BTC-USDT": {
                "binance_perpetual": Decimal("50000"),
                "bybit_perpetual": Decimal("50000")
            }
        }
        
        # Mock order execution
        for connector in self.strategy.connectors.values():
            connector.buy.return_value = "mock_order_id"
            connector.sell.return_value = "mock_order_id"
            
        await self.strategy.check_and_execute_opportunities()
        
        # Verify position was opened
        self.assertEqual(len(self.strategy._active_positions), 1)
        position = self.strategy._active_positions["BTC-USDT"]
        self.assertEqual(position["long_venue"], "binance_perpetual")  # Lower funding rate
        self.assertEqual(position["short_venue"], "bybit_perpetual")  # Higher funding rate
        
    async def test_monitor_positions(self):
        """Test position monitoring and management"""
        # Set up a mock position
        self.strategy._active_positions = {
            "BTC-USDT": {
                "long_venue": "binance_perpetual",
                "short_venue": "bybit_perpetual",
                "long_entry_price": Decimal("50000"),
                "short_entry_price": Decimal("50000"),
                "amount": Decimal("0.02"),
                "timestamp": self.strategy.current_timestamp - 1000,
                "funding_payments": Decimal("0")
            }
        }
        
        # Set up current prices that would trigger take profit
        self.strategy._prices = {
            "BTC-USDT": {
                "binance_perpetual": Decimal("51000"),  # +2% on long
                "bybit_perpetual": Decimal("49000")     # +2% on short
            }
        }
        
        # Set up funding rates
        self.strategy._funding_rates = {
            "BTC-USDT": {
                "binance_perpetual": FundingInfo(
                    trading_pair="BTC-USDT",
                    index_price=Decimal("50000"),
                    mark_price=Decimal("50000"),
                    next_funding_utc_timestamp=1234567890,
                    rate=Decimal("0.0001")
                ),
                "bybit_perpetual": FundingInfo(
                    trading_pair="BTC-USDT",
                    index_price=Decimal("50000"),
                    mark_price=Decimal("50000"),
                    next_funding_utc_timestamp=1234567890,
                    rate=Decimal("0.002")
                )
            }
        }
        
        # Mock order execution for position closing
        for connector in self.strategy.connectors.values():
            connector.buy.return_value = "mock_order_id"
            connector.sell.return_value = "mock_order_id"
            
        await self.strategy.monitor_positions()
        
        # Verify position was closed due to take profit
        self.assertEqual(len(self.strategy._active_positions), 0)
        
    def test_format_status(self):
        """Test status formatting"""
        # Set up a mock position
        self.strategy._active_positions = {
            "BTC-USDT": {
                "long_venue": "binance_perpetual",
                "short_venue": "bybit_perpetual",
                "long_entry_price": Decimal("50000"),
                "short_entry_price": Decimal("50000"),
                "amount": Decimal("0.02"),
                "timestamp": self.strategy.current_timestamp - 3600,  # 1 hour old
                "funding_payments": Decimal("0.001")
            }
        }
        
        # Set up current prices
        self.strategy._prices = {
            "BTC-USDT": {
                "binance_perpetual": Decimal("50500"),
                "bybit_perpetual": Decimal("49500")
            }
        }
        
        status = self.strategy.format_status()
        
        # Verify status contains key information
        self.assertIn("BTC-USDT", status)
        self.assertIn("binance_perpetual", status)
        self.assertIn("bybit_perpetual", status)
        self.assertIn("PnL", status)
        self.assertIn("Age", status) 