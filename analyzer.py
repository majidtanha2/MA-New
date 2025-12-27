# core/analyzer.py
"""
تحلیل‌گر پیشرفته بازار با قابلیت‌های کامل
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import talib
from typing import Dict, List, Tuple, Optional
import json
import os
import warnings
warnings.filterwarnings('ignore')

class AdvancedMarketAnalyzer:
    """تحلیل‌گر پیشرفته بازار"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.connection = None
        self.cache = {}
        
        # نمادهای اصلی
        self.symbols = {
            'forex': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD', 
                     'NZDUSD', 'USDCHF', 'EURGBP', 'EURJPY', 'GBPJPY'],
            'commodities': ['XAUUSD', 'XAGUSD'],
            'indices': ['US30', 'SPX500', 'NAS100'],
            'crypto': ['BTCUSD', 'ETHUSD']
        }
        
        # اندیکاتورها
        self.indicators = {
            'trend': ['SMA', 'EMA', 'MACD', 'ADX', 'Ichimoku'],
            'momentum': ['RSI', 'Stochastic', 'WilliamsR', 'CCI', 'MFI'],
            'volatility': ['BBANDS', 'ATR', 'Keltner'],
            'volume': ['OBV', 'Volume', 'AD']
        }
        
        print("✅ Advanced Market Analyzer initialized")
    
    def connect_to_mt5(self) -> bool:
        """اتصال به MT5"""
        try:
            if mt5.initialize():
                self.connection = True
                print("✅ Connected to MT5")
                return True
            return False
        except Exception as e:
            print(f"❌ MT5 connection error: {e}")
            return False
    
    def get_symbol_data(self, symbol: str, timeframe: str, bars: int = 500) -> Optional[pd.DataFrame]:
        """دریافت داده‌های نماد"""
        try:
            # تبدیل timeframe به فرمت MT5
            tf_map = {
                'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1,
                'MN1': mt5.TIMEFRAME_MN1
            }
            
            tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            # دریافت داده‌ها
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
            
            if rates is None:
                print(f"❌ No data for {symbol}")
                return None
            
            # تبدیل به DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # محاسبه باز شدن، بسته شدن، بالاترین، پایین‌ترین
            df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            }, inplace=True)
            
            # محاسبه قیمت‌های اضافی
            df['Typical'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['Median'] = (df['High'] + df['Low']) / 2
            
            return df
            
        except Exception as e:
            print(f"❌ Error getting data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """محاسبه اندیکاتورهای تکنیکال"""
        if df is None or df.empty:
            return {}
        
        indicators = {}
        
        # 1. اندیکاتورهای روند
        indicators['SMA_20'] = talib.SMA(df['Close'], timeperiod=20).iloc[-1]
        indicators['SMA_50'] = talib.SMA(df['Close'], timeperiod=50).iloc[-1]
        indicators['SMA_200'] = talib.SMA(df['Close'], timeperiod=200).iloc[-1]
        
        indicators['EMA_9'] = talib.EMA(df['Close'], timeperiod=9).iloc[-1]
        indicators['EMA_21'] = talib.EMA(df['Close'], timeperiod=21).iloc[-1]
        indicators['EMA_50'] = talib.EMA(df['Close'], timeperiod=50).iloc[-1]
        
        # 2. MACD
        macd, macd_signal, macd_hist = talib.MACD(df['Close'], 
                                                  fastperiod=12, 
                                                  slowperiod=26, 
                                                  signalperiod=9)
        indicators['MACD'] = macd.iloc[-1]
        indicators['MACD_Signal'] = macd_signal.iloc[-1]
        indicators['MACD_Histogram'] = macd_hist.iloc[-1]
        
        # 3. RSI
        indicators['RSI'] = talib.RSI(df['Close'], timeperiod=14).iloc[-1]
        
        # 4. Stochastic
        slowk, slowd = talib.STOCH(df['High'], df['Low'], df['Close'],
                                   fastk_period=14, slowk_period=3,
                                   slowk_matype=0, slowd_period=3, slowd_matype=0)
        indicators['Stochastic_K'] = slowk.iloc[-1]
        indicators['Stochastic_D'] = slowd.iloc[-1]
        
        # 5. Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['Close'], 
                                            timeperiod=20, 
                                            nbdevup=2, 
                                            nbdevdn=2)
        indicators['BB_Upper'] = upper.iloc[-1]
        indicators['BB_Middle'] = middle.iloc[-1]
        indicators['BB_Lower'] = lower.iloc[-1]
        indicators['BB_Percent'] = ((df['Close'].iloc[-1] - lower.iloc[-1]) / 
                                   (upper.iloc[-1] - lower.iloc[-1]))
        
        # 6. ATR (نوسان)
        indicators['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], 
                                     timeperiod=14).iloc[-1]
        
        # 7. حجم
        if 'Volume' in df.columns:
            indicators['Volume_SMA'] = talib.SMA(df['Volume'], timeperiod=20).iloc[-1]
            indicators['Volume_Ratio'] = (df['Volume'].iloc[-1] / 
                                         indicators['Volume_SMA'] if indicators['Volume_SMA'] > 0 else 1)
        
        # 8. تشخیص روند
        sma_20 = indicators['SMA_20']
        sma_50 = indicators['SMA_50']
        
        if sma_20 > sma_50:
            indicators['Trend'] = 'صعودی'
            indicators['Trend_Strength'] = abs(sma_20 - sma_50) / sma_50 * 100
        elif sma_20 < sma_50:
            indicators['Trend'] = 'نزولی'
            indicators['Trend_Strength'] = abs(sma_20 - sma_50) / sma_50 * 100
        else:
            indicators['Trend'] = 'خنثی'
            indicators['Trend_Strength'] = 0
        
        # 9. سیگنال‌های ترکیبی
        signals = []
        
        # سیگنال RSI
        if indicators['RSI'] < 30:
            signals.append(('RSI', 'خرید', 2))
        elif indicators['RSI'] > 70:
            signals.append(('RSI', 'فروش', 2))
        
        # سیگنال Stochastic
        if indicators['Stochastic_K'] < 20:
            signals.append(('Stochastic', 'خرید', 1))
        elif indicators['Stochastic_K'] > 80:
            signals.append(('Stochastic', 'فروش', 1))
        
        # سیگنال MACD
        if indicators['MACD'] > indicators['MACD_Signal']:
            signals.append(('MACD', 'خرید', 2))
        elif indicators['MACD'] < indicators['MACD_Signal']:
            signals.append(('MACD', 'فروش', 2))
        
        # سیگنال بولینگر
        if indicators['BB_Percent'] < 0.2:
            signals.append(('BB', 'خرید', 1))
        elif indicators['BB_Percent'] > 0.8:
            signals.append(('BB', 'فروش', 1))
        
        # محاسبه سیگنال نهایی
        buy_signals = [s for s in signals if s[1] == 'خرید']
        sell_signals = [s for s in signals if s[1] == 'فروش']
        
        buy_score = sum([s[2] for s in buy_signals])
        sell_score = sum([s[2] for s in sell_signals])
        
        if buy_score > sell_score:
            indicators['Signal'] = 'خرید'
            indicators['Signal_Score'] = buy_score - sell_score
        elif sell_score > buy_score:
            indicators['Signal'] = 'فروش'
            indicators['Signal_Score'] = sell_score - buy_score
        else:
            indicators['Signal'] = 'خنثی'
            indicators['Signal_Score'] = 0
        
        indicators['Signals_List'] = signals
        
        return indicators
    
    def analyze_multiple_symbols(self, symbols: List[str], timeframe: str = 'H1') -> Dict:
        """تحلیل چند نماد همزمان"""
        results = {}
        
        for symbol in symbols:
            print(f"🔍 Analyzing {symbol}...")
            
            df = self.get_symbol_data(symbol, timeframe)
            
            if df is not None and not df.empty:
                indicators = self.calculate_technical_indicators(df)
                results[symbol] = {
                    'indicators': indicators,
                    'price': df['Close'].iloc[-1],
                    'change': ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / 
                              df['Close'].iloc[-2] * 100),
                    'high': df['High'].max(),
                    'low': df['Low'].min(),
                    'volume': df['Volume'].mean() if 'Volume' in df.columns else 0
                }
        
        return results
    
    def generate_signals(self, analysis_results: Dict) -> List[Dict]:
        """تولید سیگنال‌های معاملاتی"""
        signals = []
        
        for symbol, data in analysis_results.items():
            indicators = data['indicators']
            
            if 'Signal' in indicators and indicators['Signal'] != 'خنثی':
                signal = {
                    'symbol': symbol,
                    'signal': indicators['Signal'],
                    'score': indicators.get('Signal_Score', 0),
                    'price': data['price'],
                    'time': datetime.now().isoformat(),
                    'confidence': min(100, indicators.get('Signal_Score', 0) * 20),
                    'reasons': [f"{s[0]}: {s[1]}" for s in indicators.get('Signals_List', [])],
                    'risk_level': self.calculate_risk_level(indicators)
                }
                
                # فیلتر بر اساس اعتماد
                if signal['confidence'] >= 60:  # حداقل 60% اعتماد
                    signals.append(signal)
        
        # مرتب‌سازی بر اساس امتیاز
        signals.sort(key=lambda x: x['score'], reverse=True)
        
        return signals
    
    def calculate_risk_level(self, indicators: Dict) -> str:
        """محاسبه سطح ریسک"""
        risk_score = 0
        
        # RSI در حالت اشباع
        if indicators.get('RSI', 50) < 30 or indicators.get('RSI', 50) > 70:
            risk_score += 1
        
        # Stochastic در حالت اشباع
        if (indicators.get('Stochastic_K', 50) < 20 or 
            indicators.get('Stochastic_K', 50) > 80):
            risk_score += 1
        
        # نوسان بالا
        if indicators.get('ATR', 0) > indicators.get('ATR', 1) * 1.5:
            risk_score += 1
        
        # حجم غیرعادی
        if indicators.get('Volume_Ratio', 1) > 2:
            risk_score += 1
        
        if risk_score >= 3:
            return 'بالا'
        elif risk_score >= 2:
            return 'متوسط'
        else:
            return 'پایین'
    
    def create_candlestick_chart(self, df: pd.DataFrame, title: str = '') -> Dict:
        """ایجاد نمودار کندل استیک"""
        if df is None or df.empty:
            return {}
        
        # انتخاب داده‌های اخیر
        df_recent = df.tail(100)
        
        chart_data = {
            'x': df_recent.index.tolist(),
            'open': df_recent['Open'].tolist(),
            'high': df_recent['High'].tolist(),
            'low': df_recent['Low'].tolist(),
            'close': df_recent['Close'].tolist(),
            'type': 'candlestick',
            'name': title,
            'increasing': {'line': {'color': '#2ECC71'}},
            'decreasing': {'line': {'color': '#E74C3C'}}
        }
        
        return chart_data
    
    def get_market_sentiment(self) -> Dict:
        """دریافت سنتیمنت کلی بازار"""
        # تحلیل چند نماد اصلی
        major_symbols = ['EURUSD', 'XAUUSD', 'US30', 'BTCUSD']
        results = self.analyze_multiple_symbols(major_symbols)
        
        bullish = 0
        bearish = 0
        neutral = 0
        
        for symbol, data in results.items():
            trend = data['indicators'].get('Trend', 'خنثی')
            if trend == 'صعودی':
                bullish += 1
            elif trend == 'نزولی':
                bearish += 1
            else:
                neutral += 1
        
        total = len(results)
        
        sentiment = {
            'bullish': bullish,
            'bearish': bearish,
            'neutral': neutral,
            'total': total,
            'bullish_percent': (bullish / total * 100) if total > 0 else 0,
            'bearish_percent': (bearish / total * 100) if total > 0 else 0,
            'neutral_percent': (neutral / total * 100) if total > 0 else 0,
            'overall': 'صعودی' if bullish > bearish else 'نزولی' if bearish > bullish else 'خنثی'
        }
        
        return sentiment
    
    def save_analysis_report(self, results: Dict, filename: str = None):
        """ذخیره گزارش تحلیل"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'analysis_report_{timestamp}.json'
        
        # ایجاد پوشه گزارشات
        os.makedirs('data/reports', exist_ok=True)
        filepath = os.path.join('data/reports', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Analysis report saved: {filepath}")
        return filepath

# تست تحلیل‌گر
if __name__ == '__main__':
    analyzer = AdvancedMarketAnalyzer()
    
    if analyzer.connect_to_mt5():
        # تحلیل EURUSD
        df = analyzer.get_symbol_data('EURUSD', 'H1', 200)
        
        if df is not None:
            indicators = analyzer.calculate_technical_indicators(df)
            print(f"\n📊 تحلیل EURUSD:")
            for key, value in indicators.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        # تحلیل چند نماد
        symbols = ['EURUSD', 'XAUUSD', 'US30', 'BTCUSD']
        results = analyzer.analyze_multiple_symbols(symbols)
        
        print(f"\n🎯 سیگنال‌های تولید شده:")
        signals = analyzer.generate_signals(results)
        for signal in signals:
            print(f"  {signal['symbol']}: {signal['signal']} (اعتماد: {signal['confidence']}%)")
        
        # سنتیمنت بازار
        sentiment = analyzer.get_market_sentiment()
        print(f"\n🌐 سنتیمنت کلی بازار: {sentiment['overall']}")
        print(f"  صعودی: {sentiment['bullish_percent']:.1f}%")
        print(f"  نزولی: {sentiment['bearish_percent']:.1f}%")
        
        mt5.shutdown()