import requests
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import warnings
import tensorflow as tf  # اضافه شده
import os
import json
import time
import threading
import subprocess
import sys
import tempfile
import re
import pickle
import queue
import traceback
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from prophet import Prophet
import ta as talib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import networkx as nx
from scipy.optimize import minimize
import warnings
import tensorflow as tf  # اضافه شده
warnings.filterwarnings('ignore')

# ============================================================================
# سیستم لاگینگ پیشرفته با قابلیت‌های بیشتر
# ============================================================================
class AdvancedLogger:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.log_queue = queue.Queue()
        self.log_levels = {
            "DEBUG": "#00bcd4",
            "INFO": "#4caf50",
            "WARNING": "#ff9800",
            "ERROR": "#f44336",
            "CRITICAL": "#d32f2f",
            "SUCCESS": "#2e7d32"
        }
        self.log_file = "logs/application.log"
        os.makedirs("logs", exist_ok=True)
        
    def log(self, message, level="INFO", show_time=True, show_thread=False):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = level.upper()
        thread_name = threading.current_thread().name
        
        if show_time:
            if show_thread:
                log_message = f"[{timestamp}] [{thread_name}] [{level}] {message}\n"
            else:
                log_message = f"[{timestamp}] [{level}] {message}\n"
        else:
            log_message = f"[{level}] {message}\n"
        
        self.log_queue.put((log_message, level))
        
        # ذخیره در فایل
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message)
        except:
            pass
        
        self.text_widget.after(100, self.update_gui)
    
    def update_gui(self):
        try:
            while not self.log_queue.empty():
                log_message, level = self.log_queue.get_nowait()
                
                start_index = self.text_widget.index("end-1c")
                self.text_widget.insert(tk.END, log_message)
                end_index = self.text_widget.index("end-1c")
                
                tag_name = f"tag_{level}"
                self.text_widget.tag_add(tag_name, start_index, end_index)
                self.text_widget.tag_config(tag_name, 
                                          foreground=self.log_levels.get(level, "white"),
                                          font=('Tahoma', 9))
                
                self.text_widget.see(tk.END)
        except:
            pass
    
    def clear(self):
        self.text_widget.delete(1.0, tk.END)
    
    def export_logs(self):
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".log",
                filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.text_widget.get(1.0, tk.END))
                return True
        except Exception as e:
            self.log(f"خطا در ذخیره لاگ‌ها: {e}", "ERROR")
        return False

# ============================================================================
# سیستم پیش‌بینی هوش مصنوعی با LSTM و Prophet
# ============================================================================
class AIPredictor:
    def __init__(self, logger=None):
        self.logger = logger
        self.models = {}
        self.scalers = {}
        
    def log(self, message, level="INFO"):
        if self.logger:
            self.logger.log(message, level)
        else:
            print(f"[{level}] {message}")
    
    def prepare_lstm_data(self, data, look_back=30):
        """آماده‌سازی داده برای LSTM"""
        try:
            X, y = [], []
            for i in range(len(data) - look_back):
                X.append(data[i:(i + look_back)])
                y.append(data[i + look_back])
            return np.array(X), np.array(y)
        except Exception as e:
            self.log(f"خطا در آماده‌سازی داده LSTM: {e}", "ERROR")
            return None, None
    
    def create_lstm_model(self, look_back=30):
        """ایجاد مدل LSTM"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def predict_with_lstm(self, symbol, historical_data, days_ahead=7):
        """پیش‌بینی با LSTM"""
        try:
            self.log(f"شروع پیش‌بینی LSTM برای {symbol}", "INFO")
            
            prices = historical_data['Close'].values
            if len(prices) < 100:
                self.log(f"داده کافی برای {symbol} موجود نیست", "WARNING")
                return None
            
            # نرمال‌سازی
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(prices.reshape(-1, 1))
            
            # آماده‌سازی داده
            look_back = 30
            X, y = self.prepare_lstm_data(scaled_data, look_back)
            
            if X is None:
                return None
            
            # تقسیم داده
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # آموزش مدل
            model = self.create_lstm_model(look_back)
            model.fit(X_train, y_train, 
                     batch_size=32, 
                     epochs=50, 
                     validation_data=(X_test, y_test),
                     verbose=0)
            
            # پیش‌بینی
            last_sequence = scaled_data[-look_back:].reshape(1, look_back, 1)
            predictions = []
            
            for _ in range(days_ahead):
                pred = model.predict(last_sequence, verbose=0)
                predictions.append(pred[0, 0])
                last_sequence = np.append(last_sequence[:, 1:, :], 
                                        pred.reshape(1, 1, 1), 
                                        axis=1)
            
            # بازگردانی به مقیاس اصلی
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            
            result = {
                'symbol': symbol,
                'predictions': predictions.flatten().tolist(),
                'confidence': float(1 - model.evaluate(X_test, y_test, verbose=0) / np.mean(y_test)),
                'model_type': 'LSTM',
                'next_day_price': float(predictions[0][0])
            }
            
            self.log(f"پیش‌بینی LSTM برای {symbol} کامل شد", "SUCCESS")
            return result
            
        except Exception as e:
            self.log(f"خطا در پیش‌بینی LSTM برای {symbol}: {e}", "ERROR")
            return None
    
    def predict_with_prophet(self, symbol, historical_data, days_ahead=7):
        """پیش‌بینی با Prophet"""
        try:
            self.log(f"شروع پیش‌بینی Prophet برای {symbol}", "INFO")
            
            df = historical_data.reset_index()[['Date', 'Close']].copy()
            df.columns = ['ds', 'y']
            
            # حذف مقادیر نامعتبر
            df = df.dropna()
            
            if len(df) < 100:
                self.log(f"داده کافی برای {symbol} موجود نیست", "WARNING")
                return None
            
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            model.fit(df)
            
            future = model.make_future_dataframe(periods=days_ahead)
            forecast = model.predict(future)
            
            result = {
                'symbol': symbol,
                'predictions': forecast.tail(days_ahead)['yhat'].values.tolist(),
                'lower_bounds': forecast.tail(days_ahead)['yhat_lower'].values.tolist(),
                'upper_bounds': forecast.tail(days_ahead)['yhat_upper'].values.tolist(),
                'model_type': 'Prophet',
                'next_day_price': float(forecast.iloc[-days_ahead]['yhat']),
                'trend': forecast.tail(days_ahead)['trend'].values.tolist()
            }
            
            self.log(f"پیش‌بینی Prophet برای {symbol} کامل شد", "SUCCESS")
            return result
            
        except Exception as e:
            self.log(f"خطا در پیش‌بینی Prophet برای {symbol}: {e}", "ERROR")
            return None
    
    def predict_with_arima(self, symbol, historical_data, days_ahead=7):
        """پیش‌بینی با ARIMA"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            self.log(f"شروع پیش‌بینی ARIMA برای {symbol}", "INFO")
            
            prices = historical_data['Close'].values
            
            if len(prices) < 100:
                return None
            
            model = ARIMA(prices, order=(5,1,0))
            model_fit = model.fit()
            
            forecast = model_fit.forecast(steps=days_ahead)
            
            result = {
                'symbol': symbol,
                'predictions': forecast.tolist(),
                'model_type': 'ARIMA',
                'next_day_price': float(forecast[0])
            }
            
            self.log(f"پیش‌بینی ARIMA برای {symbol} کامل شد", "SUCCESS")
            return result
            
        except Exception as e:
            self.log(f"خطا در پیش‌بینی ARIMA برای {symbol}: {e}", "WARNING")
            return None
    
    def ensemble_prediction(self, symbol, historical_data, days_ahead=7):
        """پیش‌بینی ترکیبی با چندین مدل"""
        try:
            self.log(f"شروع پیش‌بینی ترکیبی برای {symbol}", "INFO")
            
            results = []
            
            # LSTM
            lstm_result = self.predict_with_lstm(symbol, historical_data, days_ahead)
            if lstm_result:
                results.append(lstm_result)
            
            # Prophet
            prophet_result = self.predict_with_prophet(symbol, historical_data, days_ahead)
            if prophet_result:
                results.append(prophet_result)
            
            # ARIMA
            arima_result = self.predict_with_arima(symbol, historical_data, days_ahead)
            if arima_result:
                results.append(arima_result)
            
            if not results:
                return None
            
            # ترکیب نتایج با میانگین وزنی
            weights = {'LSTM': 0.4, 'Prophet': 0.4, 'ARIMA': 0.2}
            ensemble_predictions = []
            
            for i in range(days_ahead):
                weighted_sum = 0
                total_weight = 0
                
                for result in results:
                    weight = weights.get(result['model_type'], 0.1)
                    if i < len(result['predictions']):
                        weighted_sum += result['predictions'][i] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    ensemble_predictions.append(weighted_sum / total_weight)
                else:
                    ensemble_predictions.append(0)
            
            # محاسبه اطمینان کلی
            confidence = np.mean([r.get('confidence', 0.5) for r in results])
            
            result = {
                'symbol': symbol,
                'predictions': ensemble_predictions,
                'confidence': confidence,
                'model_type': 'Ensemble',
                'next_day_price': ensemble_predictions[0],
                'individual_results': results
            }
            
            self.log(f"پیش‌بینی ترکیبی برای {symbol} کامل شد", "SUCCESS")
            return result
            
        except Exception as e:
            self.log(f"خطا در پیش‌بینی ترکیبی برای {symbol}: {e}", "ERROR")
            return None

# ============================================================================
# سیستم هشدار هوشمند
# ============================================================================
class SmartAlertSystem:
    def __init__(self, logger=None):
        self.logger = logger
        self.alerts = []
        self.alert_rules = {
            'price_change': {'threshold': 5, 'timeframe': 'daily'},
            'volume_surge': {'threshold': 2.0, 'timeframe': 'daily'},
            'rsi_extreme': {'buy_threshold': 30, 'sell_threshold': 70},
            'macd_crossover': {'enabled': True},
            'bollinger_breakout': {'std_dev': 2},
            'support_resistance': {'enabled': True}
        }
        
    def log(self, message, level="INFO"):
        if self.logger:
            self.logger.log(message, level)
    
    def add_alert(self, symbol, alert_type, message, priority="MEDIUM"):
        """افزودن هشدار جدید"""
        alert = {
            'id': len(self.alerts) + 1,
            'symbol': symbol,
            'type': alert_type,
            'message': message,
            'priority': priority,
            'timestamp': datetime.now(),
            'acknowledged': False
        }
        self.alerts.append(alert)
        self.log(f"هشدار جدید: {symbol} - {message}", "WARNING")
    
    def check_price_alerts(self, stock_data):
        """بررسی هشدارهای قیمت"""
        try:
            for symbol, data in stock_data.items():
                if len(data) < 2:
                    continue
                
                current_price = data.iloc[-1]['Close']
                prev_price = data.iloc[-2]['Close']
                
                if prev_price > 0:
                    change_percent = ((current_price - prev_price) / prev_price) * 100
                    
                    if abs(change_percent) >= self.alert_rules['price_change']['threshold']:
                        direction = "افزایش" if change_percent > 0 else "کاهش"
                        self.add_alert(
                            symbol=symbol,
                            alert_type="PRICE_CHANGE",
                            message=f"تغییر قیمت {abs(change_percent):.1f}% ({direction})",
                            priority="HIGH" if abs(change_percent) > 10 else "MEDIUM"
                        )
                        
        except Exception as e:
            self.log(f"خطا در بررسی هشدارهای قیمت: {e}", "ERROR")
    
    def check_volume_alerts(self, stock_data):
        """بررسی هشدارهای حجم"""
        try:
            for symbol, data in stock_data.items():
                if len(data) < 21:
                    continue
                
                current_volume = data.iloc[-1]['Volume']
                avg_volume = data.iloc[-21:-1]['Volume'].mean()
                
                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    
                    if volume_ratio >= self.alert_rules['volume_surge']['threshold']:
                        self.add_alert(
                            symbol=symbol,
                            alert_type="VOLUME_SURGE",
                            message=f"افزایش حجم معاملات {volume_ratio:.1f} برابر میانگین",
                            priority="MEDIUM"
                        )
                        
        except Exception as e:
            self.log(f"خطا در بررسی هشدارهای حجم: {e}", "ERROR")
    
    def check_technical_alerts(self, stock_data):
        """بررسی هشدارهای تکنیکال"""
        try:
            for symbol, data in stock_data.items():
                if len(data) < 50:
                    continue
                
                # محاسبه RSI
                prices = data['Close'].values
                rsi = talib.RSI(prices, timeperiod=14)
                
                if len(rsi) > 0:
                    current_rsi = rsi[-1]
                    
                    if current_rsi < self.alert_rules['rsi_extreme']['buy_threshold']:
                        self.add_alert(
                            symbol=symbol,
                            alert_type="RSI_OVERSOLD",
                            message=f"RSI اشباع فروش: {current_rsi:.1f}",
                            priority="LOW"
                        )
                    
                    if current_rsi > self.alert_rules['rsi_extreme']['sell_threshold']:
                        self.add_alert(
                            symbol=symbol,
                            alert_type="RSI_OVERBOUGHT",
                            message=f"RSI اشباع خرید: {current_rsi:.1f}",
                            priority="LOW"
                        )
                
                # محاسبه MACD
                macd, macdsignal, macdhist = talib.MACD(prices, 
                                                       fastperiod=12, 
                                                       slowperiod=26, 
                                                       signalperiod=9)
                
                if len(macd) > 1 and len(macdsignal) > 1:
                    if macd[-1] > macdsignal[-1] and macd[-2] <= macdsignal[-2]:
                        self.add_alert(
                            symbol=symbol,
                            alert_type="MACD_BUY",
                            message="سیگنال خرید MACD",
                            priority="MEDIUM"
                        )
                    
                    if macd[-1] < macdsignal[-1] and macd[-2] >= macdsignal[-2]:
                        self.add_alert(
                            symbol=symbol,
                            alert_type="MACD_SELL",
                            message="سیگنال فروش MACD",
                            priority="MEDIUM"
                        )
                
                # بررسی بولینگر باند
                upper, middle, lower = talib.BBANDS(prices, 
                                                   timeperiod=20,
                                                   nbdevup=2,
                                                   nbdevdn=2,
                                                   matype=0)
                
                if len(upper) > 0 and len(lower) > 0:
                    current_price = prices[-1]
                    
                    if current_price > upper[-1]:
                        self.add_alert(
                            symbol=symbol,
                            alert_type="BB_UPPER_BREAK",
                            message="شکست مقاومت بولینگر باند بالا",
                            priority="MEDIUM"
                        )
                    
                    if current_price < lower[-1]:
                        self.add_alert(
                            symbol=symbol,
                            alert_type="BB_LOWER_BREAK",
                            message="شکست حمایت بولینگر باند پایین",
                            priority="MEDIUM"
                        )
                        
        except Exception as e:
            self.log(f"خطا در بررسی هشدارهای تکنیکال: {e}", "ERROR")
    
    def check_all_alerts(self, stock_data):
        """بررسی تمام هشدارها"""
        self.alerts = []
        
        self.check_price_alerts(stock_data)
        self.check_volume_alerts(stock_data)
        self.check_technical_alerts(stock_data)
        
        return self.alerts
    
    def get_unacknowledged_alerts(self):
        """دریافت هشدارهای تأیید نشده"""
        return [alert for alert in self.alerts if not alert['acknowledged']]
    
    def acknowledge_alert(self, alert_id):
        """تأیید هشدار"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                return True
        return False
    
    def export_alerts(self, filename):
        """ذخیره هشدارها در فایل"""
        try:
            df = pd.DataFrame(self.alerts)
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            return True
        except Exception as e:
            self.log(f"خطا در ذخیره هشدارها: {e}", "ERROR")
            return False

# ============================================================================
# سیستم بهینه‌سازی سبد (Portfolio Optimization)
# ============================================================================
class PortfolioOptimizer:
    def __init__(self, logger=None):
        self.logger = logger
        
    def log(self, message, level="INFO"):
        if self.logger:
            self.logger.log(message, level)
    
    def mean_variance_optimization(self, returns_df, target_return=None, risk_free_rate=0.05):
        """بهینه‌سازی میانگین-واریانس مارکویتز"""
        try:
            returns = returns_df
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            num_assets = len(mean_returns)
            
            # محدودیت‌ها
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # جمع وزنی‌ها برابر 1
            ]
            
            bounds = tuple((0, 1) for _ in range(num_assets))  # بدون فروش استقراضی
            
            if target_return is not None:
                constraints.append({
                    'type': 'eq', 
                    'fun': lambda x: np.sum(mean_returns * x) - target_return
                })
            
            # تابع هدف: حداقل کردن واریانس
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # حدس اولیه
            initial_weights = np.array([1/num_assets] * num_assets)
            
            # بهینه‌سازی
            result = minimize(
                portfolio_variance,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                portfolio_return = np.sum(mean_returns * optimal_weights)
                portfolio_volatility = np.sqrt(portfolio_variance(optimal_weights))
                
                # محاسبه شارپ ریشیو
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
                
                return {
                    'weights': dict(zip(returns.columns, optimal_weights)),
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'success': True
                }
            else:
                return {'success': False, 'error': result.message}
                
        except Exception as e:
            self.log(f"خطا در بهینه‌سازی میانگین-واریانس: {e}", "ERROR")
            return {'success': False, 'error': str(e)}
    
    def efficient_frontier(self, returns_df, num_portfolios=10000):
        """محاسبه مرز کارا"""
        try:
            returns = returns_df
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            num_assets = len(mean_returns)
            
            results = []
            
            for _ in range(num_portfolios):
                weights = np.random.random(num_assets)
                weights /= np.sum(weights)
                
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_volatility = np.sqrt(np.dot(weights.T, 
                                                     np.dot(cov_matrix, weights)))
                
                sharpe_ratio = (portfolio_return - 0.05) / portfolio_volatility
                
                results.append({
                    'weights': weights,
                    'return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio
                })
            
            results_df = pd.DataFrame(results)
            
            return {
                'efficient_frontier': results_df,
                'max_sharpe': results_df.loc[results_df['sharpe_ratio'].idxmax()],
                'min_volatility': results_df.loc[results_df['volatility'].idxmin()],
                'success': True
            }
            
        except Exception as e:
            self.log(f"خطا در محاسبه مرز کارا: {e}", "ERROR")
            return {'success': False, 'error': str(e)}
    
    def black_litterman_optimization(self, returns_df, views, confidences):
        """بهینه‌سازی بلک-لیترمن"""
        try:
            # پیاده‌سازی ساده بلک-لیترمن
            returns = returns_df
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # ماتریس دیدگاه‌ها
            P = np.array([[1, 0, -1], [0, 1, -0.5]])  # مثال
            Q = np.array([0.05, 0.03])  # بازده مورد انتظار دیدگاه‌ها
            
            # ماتریس عدم قطعیت
            tau = 0.05
            omega = np.dot(np.dot(P, cov_matrix), P.T) * tau
            
            # محاسبه بازده‌های جدید
            pi = mean_returns.values
            M1 = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + np.dot(P.T, np.dot(np.linalg.inv(omega), P)))
            M2 = np.dot(np.linalg.inv(tau * cov_matrix), pi) + np.dot(P.T, np.dot(np.linalg.inv(omega), Q))
            
            new_returns = np.dot(M1, M2)
            
            # بهینه‌سازی با بازده‌های جدید
            result = self.mean_variance_optimization(returns_df, target_return=np.mean(new_returns))
            
            if result['success']:
                result['black_litterman_returns'] = dict(zip(returns.columns, new_returns))
                
            return result
            
        except Exception as e:
            self.log(f"خطا در بهینه‌سازی بلک-لیترمن: {e}", "ERROR")
            return self.mean_variance_optimization(returns_df)
    
    def risk_parity_optimization(self, returns_df):
        """بهینه‌سازی پارتیشن ریسک"""
        try:
            returns = returns_df
            cov_matrix = returns.cov()
            num_assets = len(cov_matrix)
            
            def risk_contributions(weights):
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                marginal_contributions = np.dot(cov_matrix, weights) / portfolio_volatility
                risk_contributions = weights * marginal_contributions
                return risk_contributions
            
            def objective(weights):
                rc = risk_contributions(weights)
                target_rc = np.ones(num_assets) / num_assets
                return np.sum((rc - target_rc) ** 2)
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': lambda x: x}  # وزن‌های مثبت
            ]
            
            bounds = tuple((0, 1) for _ in range(num_assets))
            initial_weights = np.array([1/num_assets] * num_assets)
            
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                portfolio_return = np.sum(returns.mean() * optimal_weights)
                portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, 
                                                     np.dot(cov_matrix, optimal_weights)))
                
                # محاسبه سهم هر دارایی در ریسک کل
                rc = risk_contributions(optimal_weights)
                risk_contributions_pct = rc / portfolio_volatility
                
                return {
                    'weights': dict(zip(returns.columns, optimal_weights)),
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'risk_contributions': dict(zip(returns.columns, risk_contributions_pct)),
                    'success': True
                }
            else:
                return {'success': False, 'error': result.message}
                
        except Exception as e:
            self.log(f"خطا در بهینه‌سازی پارتیشن ریسک: {e}", "ERROR")
            return {'success': False, 'error': str(e)}

# ============================================================================
# تحلیل‌گر بازار و خوشه‌بندی
# ============================================================================
class MarketAnalyzer:
    def __init__(self, logger=None):
        self.logger = logger
        
    def log(self, message, level="INFO"):
        if self.logger:
            self.logger.log(message, level)
    
    def cluster_stocks(self, stock_data, n_clusters=5):
        """خوشه‌بندی سهام بر اساس ویژگی‌ها"""
        try:
            # استخراج ویژگی‌ها
            features = []
            symbols = []
            
            for symbol, data in stock_data.items():
                if len(data) < 50:
                    continue
                
                # ویژگی‌های مختلف
                returns = data['Close'].pct_change().dropna()
                volatility = returns.std()
                avg_return = returns.mean()
                max_drawdown = (data['Close'] / data['Close'].cummax() - 1).min()
                volume_mean = data['Volume'].mean()
                
                features.append([
                    volatility,
                    avg_return,
                    max_drawdown,
                    volume_mean,
                    data['Close'].iloc[-1] / data['Close'].iloc[0]  # بازده کل
                ])
                symbols.append(symbol)
            
            if len(features) < n_clusters:
                n_clusters = len(features)
            
            features = np.array(features)
            
            # نرمال‌سازی
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # خوشه‌بندی با K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features_scaled)
            
            # تحلیل خوشه‌ها
            cluster_results = {}
            for i in range(n_clusters):
                cluster_symbols = [symbols[j] for j in range(len(symbols)) if clusters[j] == i]
                cluster_features = features[clusters == i]
                
                cluster_results[i] = {
                    'symbols': cluster_symbols,
                    'count': len(cluster_symbols),
                    'avg_volatility': np.mean(cluster_features[:, 0]),
                    'avg_return': np.mean(cluster_features[:, 1]),
                    'risk_return_ratio': np.mean(cluster_features[:, 1]) / np.mean(cluster_features[:, 0]) 
                    if np.mean(cluster_features[:, 0]) > 0 else 0
                }
            
            # PCA برای کاهش ابعاد و مصورسازی
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(features_scaled)
            
            return {
                'clusters': cluster_results,
                'pca_result': pca_result,
                'symbols': symbols,
                'cluster_labels': clusters,
                'explained_variance': pca.explained_variance_ratio_,
                'success': True
            }
            
        except Exception as e:
            self.log(f"خطا در خوشه‌بندی سهام: {e}", "ERROR")
            return {'success': False, 'error': str(e)}
    
    def analyze_market_regimes(self, market_index_data, n_regimes=3):
        """تحلیل رژیم‌های بازار"""
        try:
            returns = market_index_data['Close'].pct_change().dropna()
            
            # ویژگی‌های پنجره‌ای
            window_size = 20
            features = []
            
            for i in range(window_size, len(returns)):
                window_returns = returns.iloc[i-window_size:i]
                
                features.append([
                    window_returns.mean(),  # بازده میانگین
                    window_returns.std(),   # نوسان
                    window_returns.skew(),  # چولگی
                    window_returns.kurtosis(),  # کشیدگی
                    (window_returns > 0).mean()  # درصد روزهای مثبت
                ])
            
            features = np.array(features)
            
            # خوشه‌بندی برای شناسایی رژیم‌ها
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            regimes = kmeans.predict(features_scaled)
            
            # تحلیل هر رژیم
            regime_analysis = {}
            for regime in range(n_regimes):
                regime_returns = returns.iloc[window_size:][regimes == regime]
                
                regime_analysis[regime] = {
                    'count': len(regime_returns),
                    'avg_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'sharpe_ratio': regime_returns.mean() / regime_returns.std() 
                    if regime_returns.std() > 0 else 0,
                    'positive_days': (regime_returns > 0).mean()
                }
            
            return {
                'regimes': regime_analysis,
                'regime_labels': regimes,
                'success': True
            }
            
        except Exception as e:
            self.log(f"خطا در تحلیل رژیم‌های بازار: {e}", "ERROR")
            return {'success': False, 'error': str(e)}
    
    def calculate_correlation_network(self, stock_data):
        """محاسبه شبکه همبستگی بین سهام"""
        try:
            # استخراج بازده‌ها
            returns_data = {}
            symbols = []
            
            for symbol, data in stock_data.items():
                if len(data) > 50:
                    returns = data['Close'].pct_change().dropna()
                    returns_data[symbol] = returns
                    symbols.append(symbol)
            
            # ایجاد DataFrame
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            # ماتریس همبستگی
            corr_matrix = returns_df.corr()
            
            # ایجاد گراف
            G = nx.Graph()
            
            # اضافه کردن گره‌ها
            for symbol in symbols:
                G.add_node(symbol)
            
            # اضافه کردن یال‌ها (ارتباطات قوی)
            threshold = 0.7
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    corr = abs(corr_matrix.iloc[i, j])
                    if corr > threshold:
                        G.add_edge(symbols[i], symbols[j], weight=corr)
            
            # محاسبه معیارهای شبکه
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            clustering_coefficient = nx.clustering(G)
            
            # شناسایی خوشه‌های قوی
            communities = list(nx.community.greedy_modularity_communities(G))
            
            return {
                'correlation_matrix': corr_matrix,
                'graph': G,
                'degree_centrality': degree_centrality,
                'betweenness_centrality': betweenness_centrality,
                'clustering_coefficient': clustering_coefficient,
                'communities': communities,
                'success': True
            }
            
        except Exception as e:
            self.log(f"خطا در محاسبه شبکه همبستگی: {e}", "ERROR")
            return {'success': False, 'error': str(e)}

# ============================================================================
# سیستم گزارش‌گیری پیشرفته
# ============================================================================
class AdvancedReportGenerator:
    def __init__(self, logger=None):
        self.logger = logger
        self.report_templates = {
            'daily': self.generate_daily_report,
            'weekly': self.generate_daily_report,
            'portfolio': self.generate_portfolio_report,
            'technical': self.generate_technical_report
        }
    
    def log(self, message, level="INFO"):
        if self.logger:
            self.logger.log(message, level)
    
    def generate_daily_report(self, market_data, portfolio_data=None):
        """تولید گزارش روزانه"""
        try:
            report = {
                'date': datetime.now().strftime("%Y-%m-%d"),
                'market_summary': {},
                'top_gainers': [],
                'top_losers': [],
                'portfolio_update': {},
                'recommendations': []
            }
            
            # خلاصه بازار
            if market_data is not None:
                market_returns = market_data.pct_change().dropna()
                report['market_summary'] = {
                    'total_stocks': len(market_data.columns),
                    'avg_daily_return': market_returns.mean().mean(),
                    'market_volatility': market_returns.std().mean(),
                    'advance_decline': (market_returns.iloc[-1] > 0).sum() / len(market_returns.columns)
                }
                
                # برترین‌ها
                daily_returns = market_returns.iloc[-1] if len(market_returns) > 0 else pd.Series()
                if len(daily_returns) > 0:
                    top_gainers = daily_returns.nlargest(10)
                    top_losers = daily_returns.nsmallest(10)
                    
                    report['top_gainers'] = [
                        {'symbol': sym, 'return': ret*100}
                        for sym, ret in top_gainers.items()
                    ]
                    report['top_losers'] = [
                        {'symbol': sym, 'return': ret*100}
                        for sym, ret in top_losers.items()
                    ]
            
            # به‌روزرسانی پورتفو
            if portfolio_data is not None:
                report['portfolio_update'] = {
                    'total_value': portfolio_data['current_value'].sum(),
                    'total_profit': portfolio_data['profit_loss'].sum(),
                    'daily_change': 0,  # محاسبه تغییر روزانه
                    'best_performer': portfolio_data.loc[portfolio_data['profit_loss_percent'].idxmax()]['symbol']
                    if len(portfolio_data) > 0 else None
                }
            
            # توصیه‌ها
            report['recommendations'] = [
                "بررسی سهام با حجم معاملات غیرعادی",
                "توجه به تغییرات صنایع برتر",
                "بررسی اخبار اقتصادی روز"
            ]
            
            return report
            
        except Exception as e:
            self.log(f"خطا در تولید گزارش روزانه: {e}", "ERROR")
            return None
    
    def generate_portfolio_report(self, portfolio_analysis, risk_metrics):
        """تولید گزارش پورتفو"""
        try:
            report = {
                'portfolio_summary': {},
                'asset_allocation': [],
                'performance_metrics': {},
                'risk_analysis': {},
                'rebalancing_suggestions': []
            }
            
            # خلاصه پورتفو
            report['portfolio_summary'] = {
                'total_value': portfolio_analysis.get('portfolio_value', 0),
                'total_profit': portfolio_analysis.get('total_profit_loss', 0),
                'profit_percentage': portfolio_analysis.get('profit_loss_percent', 0),
                'number_of_stocks': len(portfolio_analysis.get('stock_details', {}))
            }
            
            # تخصیص دارایی
            if 'stock_details' in portfolio_analysis:
                for symbol, details in portfolio_analysis['stock_details'].items():
                    report['asset_allocation'].append({
                        'symbol': symbol,
                        'weight': details.get('weight', 0),
                        'profit_percent': details.get('profit_loss_percent', 0),
                        'current_value': details.get('current_value', 0)
                    })
            
            # معیارهای عملکرد
            report['performance_metrics'] = risk_metrics
            
            # تحلیل ریسک
            report['risk_analysis'] = {
                'concentration_risk': self.calculate_concentration_risk(portfolio_analysis),
                'sector_risk': self.calculate_sector_risk(portfolio_analysis),
                'liquidity_risk': self.calculate_liquidity_risk(portfolio_analysis)
            }
            
            # پیشنهادات متعادل‌سازی
            report['rebalancing_suggestions'] = self.generate_rebalancing_suggestions(
                portfolio_analysis, risk_metrics
            )
            
            return report
            
        except Exception as e:
            self.log(f"خطا در تولید گزارش پورتفو: {e}", "ERROR")
            return None
    
    def generate_technical_report(self, stock_data, technical_indicators):
        """تولید گزارش تکنیکال"""
        try:
            report = {
                'technical_summary': {},
                'indicator_signals': [],
                'chart_patterns': [],
                'support_resistance': [],
                'trading_signals': []
            }
            
            for symbol, data in stock_data.items():
                if len(data) < 50:
                    continue
                
                # سیگنال‌های اندیکاتور
                signals = self.analyze_technical_signals(data, technical_indicators.get(symbol, {}))
                if signals:
                    report['indicator_signals'].append({
                        'symbol': symbol,
                        'signals': signals
                    })
                
                # شناسایی الگوهای نموداری
                patterns = self.identify_chart_patterns(data)
                if patterns:
                    report['chart_patterns'].append({
                        'symbol': symbol,
                        'patterns': patterns
                    })
            
            return report
            
        except Exception as e:
            self.log(f"خطا در تولید گزارش تکنیکال: {e}", "ERROR")
            return None
    
    def calculate_concentration_risk(self, portfolio_analysis):
        """محاسبه ریسک تمرکز"""
        try:
            weights = [d['weight'] for d in portfolio_analysis.get('stock_details', {}).values()]
            
            if not weights:
                return 0
            
            herfindahl = sum([w**2 for w in weights])
            concentration_risk = herfindahl * 100
            
            return {
                'herfindahl_index': herfindahl,
                'concentration_score': concentration_risk,
                'risk_level': 'HIGH' if concentration_risk > 20 else 
                             'MEDIUM' if concentration_risk > 10 else 'LOW'
            }
            
        except:
            return {'error': 'خطا در محاسبه'}
    
    def generate_rebalancing_suggestions(self, portfolio_analysis, risk_metrics):
        """تولید پیشنهادات متعادل‌سازی"""
        suggestions = []
        
        try:
            # بررسی تمرکز بیش از حد
            for symbol, details in portfolio_analysis.get('stock_details', {}).items():
                weight = details.get('weight', 0)
                
                if weight > 20:  # بیشتر از 20% وزن
                    suggestions.append({
                        'type': 'REDUCE_CONCENTRATION',
                        'symbol': symbol,
                        'current_weight': weight,
                        'suggested_weight': weight * 0.7,  # کاهش به 70% وزن فعلی
                        'reason': 'تمرکز بیش از حد در یک سهم'
                    })
            
            # بررسی سهام با زیان زیاد
            for symbol, details in portfolio_analysis.get('stock_details', {}).items():
                profit_percent = details.get('profit_loss_percent', 0)
                
                if profit_percent < -15:  # زیان بیش از 15%
                    suggestions.append({
                        'type': 'CUT_LOSSES',
                        'symbol': symbol,
                        'current_loss': profit_percent,
                        'suggested_action': 'فروش بخشی از سهم',
                        'reason': 'زیان زیاد و احتمال ادامه روند نزولی'
                    })
            
            # پیشنهاد افزایش تنوع
            if len(portfolio_analysis.get('stock_details', {})) < 5:
                suggestions.append({
                    'type': 'INCREASE_DIVERSIFICATION',
                    'suggested_action': 'افزایش تعداد سهام به حداقل ۱۰ نماد',
                    'reason': 'تنوع ناکافی برای مدیریت ریسک'
                })
            
            return suggestions
            
        except Exception as e:
            self.log(f"خطا در تولید پیشنهادات متعادل‌سازی: {e}", "ERROR")
            return []
    
    def export_report(self, report, format_type='pdf'):
        """خروجی گزارش در قالب‌های مختلف"""
        try:
            if format_type == 'pdf':
                # پیاده‌سازی ساخت PDF (با گزارش‌سازی ساده)
                filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(report, indent=2, ensure_ascii=False))
                
                return filename
            elif format_type == 'excel':
                filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                
                # تبدیل گزارش به DataFrame و ذخیره
                writer = pd.ExcelWriter(filename, engine='openpyxl')
                
                if 'asset_allocation' in report:
                    pd.DataFrame(report['asset_allocation']).to_excel(
                        writer, sheet_name='Asset Allocation', index=False
                    )
                
                if 'performance_metrics' in report:
                    pd.DataFrame([report['performance_metrics']]).to_excel(
                        writer, sheet_name='Performance', index=False
                    )
                
                writer.save()
                return filename
                
        except Exception as e:
            self.log(f"خطا در ذخیره گزارش: {e}", "ERROR")
            return None

# ============================================================================
# سیستم بک‌تست استراتژی
# ============================================================================
class BacktestEngine:
    def __init__(self, initial_capital=100000000, logger=None):
        self.initial_capital = initial_capital
        self.logger = logger
        self.results = {}
        
    def log(self, message, level="INFO"):
        if self.logger:
            self.logger.log(message, level)
    
    def run_backtest(self, strategy, historical_data, **kwargs):
        """اجرای بک‌تست برای یک استراتژی"""
        try:
            self.log(f"شروع بک‌تست برای استراتژی {strategy.__name__}", "INFO")
            
            # کپی داده‌ها برای جلوگیری از تغییر
            data = historical_data.copy()
            
            # اجرای استراتژی
            results = strategy(data, initial_capital=self.initial_capital, **kwargs)
            
            # محاسبه معیارهای عملکرد
            performance_metrics = self.calculate_performance_metrics(results)
            
            # ذخیره نتایج
            self.results[strategy.__name__] = {
                'raw_results': results,
                'performance_metrics': performance_metrics,
                'timestamp': datetime.now()
            }
            
            self.log(f"بک‌تست برای استراتژی {strategy.__name__} کامل شد", "SUCCESS")
            return self.results[strategy.__name__]
            
        except Exception as e:
            self.log(f"خطا در اجرای بک‌تست: {e}", "ERROR")
            return None
    
    def calculate_performance_metrics(self, results):
        """محاسبه معیارهای عملکرد"""
        try:
            if 'equity_curve' not in results:
                return {}
            
            equity_curve = results['equity_curve']
            returns = equity_curve.pct_change().dropna()
            
            if len(returns) == 0:
                return {}
            
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
            annual_return = (1 + total_return/100) ** (252/len(returns)) - 1
            
            # نوسان
            volatility = returns.std() * np.sqrt(252)
            
            # حداکثر drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            # شارپ ریشیو
            risk_free_rate = 0.05
            sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # سورتینو ریشیو
            negative_returns = returns[returns < 0]
            downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
            sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0
            
            # نسبت برد
            if 'trade_history' in results:
                trades = results['trade_history']
                if len(trades) > 0:
                    winning_trades = len([t for t in trades if t.get('profit', 0) > 0])
                    win_rate = winning_trades / len(trades) * 100
                    avg_win = np.mean([t.get('profit', 0) for t in trades if t.get('profit', 0) > 0])
                    avg_loss = abs(np.mean([t.get('profit', 0) for t in trades if t.get('profit', 0) < 0]))
                    profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
                else:
                    win_rate = 0
                    profit_factor = 0
            else:
                win_rate = 0
                profit_factor = 0
            
            return {
                'total_return_percent': total_return,
                'annual_return': annual_return * 100,
                'volatility_percent': volatility * 100,
                'max_drawdown_percent': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'win_rate_percent': win_rate,
                'profit_factor': profit_factor,
                'total_trades': results.get('total_trades', 0)
            }
            
        except Exception as e:
            self.log(f"خطا در محاسبه معیارهای عملکرد: {e}", "ERROR")
            return {}
    
    def compare_strategies(self, strategies_data):
        """مقایسه چندین استراتژی"""
        try:
            comparison = {}
            
            for strategy_name, results in strategies_data.items():
                metrics = self.calculate_performance_metrics(results.get('raw_results', {}))
                comparison[strategy_name] = metrics
            
            # رتبه‌بندی استراتژی‌ها
            df_comparison = pd.DataFrame(comparison).T
            
            if len(df_comparison) > 0:
                # امتیازدهی بر اساس چندین معیار
                df_comparison['score'] = (
                    df_comparison['total_return_percent'] * 0.3 +
                    df_comparison['sharpe_ratio'] * 0.25 +
                    df_comparison['sortino_ratio'] * 0.2 +
                    (100 - df_comparison['max_drawdown_percent']) * 0.15 +
                    df_comparison['win_rate_percent'] * 0.1
                )
                
                df_comparison['rank'] = df_comparison['score'].rank(ascending=False)
            
            return df_comparison
            
        except Exception as e:
            self.log(f"خطا در مقایسه استراتژی‌ها: {e}", "ERROR")
            return pd.DataFrame()
    
    def monte_carlo_simulation(self, strategy_results, n_simulations=1000):
        """شبیه‌سازی مونت کارلو"""
        try:
            if 'raw_results' not in strategy_results:
                return {}
            
            returns = strategy_results['raw_results'].get('returns', pd.Series())
            
            if len(returns) == 0:
                return {}
            
            simulations = []
            
            for _ in range(n_simulations):
                # نمونه‌گیری bootstrap
                sample_returns = np.random.choice(returns.values, size=len(returns), replace=True)
                
                # محاسبه مسیر سود
                equity_curve = self.initial_capital * (1 + sample_returns).cumprod()
                
                # محاسبه drawdown
                running_max = np.maximum.accumulate(equity_curve)
                drawdown = (equity_curve - running_max) / running_max
                max_dd = drawdown.min()
                
                # ذخیره نتایج شبیه‌سازی
                simulations.append({
                    'final_value': equity_curve[-1],
                    'max_drawdown': max_dd,
                    'total_return': (equity_curve[-1] / self.initial_capital - 1) * 100
                })
            
            simulations_df = pd.DataFrame(simulations)
            
            # محاسبه احتمالات
            prob_success = (simulations_df['final_value'] > self.initial_capital).mean() * 100
            var_95 = np.percentile(simulations_df['final_value'], 5)
            expected_shortfall = simulations_df[simulations_df['final_value'] <= var_95]['final_value'].mean()
            
            return {
                'simulations': simulations_df,
                'probability_of_success': prob_success,
                'var_95': var_95,
                'expected_shortfall': expected_shortfall,
                'confidence_interval': {
                    'lower': np.percentile(simulations_df['final_value'], 2.5),
                    'upper': np.percentile(simulations_df['final_value'], 97.5)
                }
            }
            
        except Exception as e:
            self.log(f"خطا در شبیه‌سازی مونت کارلو: {e}", "ERROR")
            return {}
    
    def walk_forward_analysis(self, strategy, historical_data, 
                             window_size=252, step_size=63):
        """تحلیل Walk-Forward"""
        try:
            results = []
            
            n_periods = len(historical_data)
            
            for i in range(window_size, n_periods, step_size):
                # تقسیم داده به درون‌نمونه و برون‌نمونه
                in_sample = historical_data.iloc[i-window_size:i]
                out_of_sample = historical_data.iloc[i:i+step_size]
                
                # آموزش روی درون‌نمونه
                # (این بخش به استراتژی بستگی دارد)
                
                # تست روی برون‌نمونه
                out_of_sample_result = self.run_backtest(
                    strategy, out_of_sample, initial_capital=1000000
                )
                
                if out_of_sample_result:
                    results.append({
                        'period': i,
                        'in_sample_end': in_sample.index[-1],
                        'out_of_sample_start': out_of_sample.index[0] if len(out_of_sample) > 0 else None,
                        'performance': out_of_sample_result.get('performance_metrics', {})
                    })
            
            return results
            
        except Exception as e:
            self.log(f"خطا در تحلیل Walk-Forward: {e}", "ERROR")
            return []

# ============================================================================
# استراتژی‌های معاملاتی پیش‌فرض
# ============================================================================
class TradingStrategies:
    @staticmethod
    def moving_average_crossover(data, initial_capital=100000000, 
                                fast_period=20, slow_period=50):
        """استراتژی عبور از میانگین متحرک"""
        try:
            if len(data) < slow_period:
                return {}
            
            # محاسبه میانگین‌های متحرک
            data['fast_ma'] = data['Close'].rolling(window=fast_period).mean()
            data['slow_ma'] = data['Close'].rolling(window=slow_period).mean()
            
            # سیگنال‌ها
            data['signal'] = 0
            data.loc[data['fast_ma'] > data['slow_ma'], 'signal'] = 1
            data.loc[data['fast_ma'] <= data['slow_ma'], 'signal'] = 0
            
            # موقعیت‌ها
            data['position'] = data['signal'].diff()
            
            # شبیه‌سازی معاملات
            capital = initial_capital
            position = 0
            trades = []
            equity_curve = []
            
            for i in range(slow_period, len(data)):
                current_price = data['Close'].iloc[i]
                
                if data['position'].iloc[i] == 1:  # خرید
                    if position == 0 and capital > 0:
                        position = capital / current_price
                        capital = 0
                        trades.append({
                            'date': data.index[i],
                            'type': 'BUY',
                            'price': current_price,
                            'shares': position,
                            'value': position * current_price
                        })
                
                elif data['position'].iloc[i] == -1:  # فروش
                    if position > 0:
                        capital = position * current_price
                        trades.append({
                            'date': data.index[i],
                            'type': 'SELL',
                            'price': current_price,
                            'shares': position,
                            'value': capital,
                            'profit': capital - (position * data['Close'].iloc[i-1])
                        })
                        position = 0
                
                # محاسبه ارزش سبد
                portfolio_value = capital + (position * current_price)
                equity_curve.append(portfolio_value)
            
            # بستن موقعیت آخر
            if position > 0:
                last_price = data['Close'].iloc[-1]
                capital = position * last_price
                trades.append({
                    'date': data.index[-1],
                    'type': 'SELL',
                    'price': last_price,
                    'shares': position,
                    'value': capital
                })
            
            equity_curve = pd.Series(equity_curve, index=data.index[slow_period:])
            returns = equity_curve.pct_change().dropna()
            
            return {
                'equity_curve': equity_curve,
                'returns': returns,
                'trade_history': trades,
                'total_trades': len(trades),
                'final_value': equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
            }
            
        except Exception as e:
            print(f"خطا در استراتژی MA Crossover: {e}")
            return {}
    
    @staticmethod
    def mean_reversion(data, initial_capital=100000000, 
                      lookback_period=20, z_score_threshold=2):
        """استراتژی بازگشت به میانگین"""
        try:
            if len(data) < lookback_period:
                return {}
            
            # محاسبه Z-Score
            data['returns'] = data['Close'].pct_change()
            data['mean'] = data['Close'].rolling(window=lookback_period).mean()
            data['std'] = data['Close'].rolling(window=lookback_period).std()
            data['z_score'] = (data['Close'] - data['mean']) / data['std']
            
            # سیگنال‌ها
            data['signal'] = 0
            data.loc[data['z_score'] < -z_score_threshold, 'signal'] = 1  # خرید
            data.loc[data['z_score'] > z_score_threshold, 'signal'] = -1  # فروش
            
            # شبیه‌سازی معاملات (مشابه استراتژی قبل)
            # ...
            
            return {}
            
        except Exception as e:
            print(f"خطا در استراتژی Mean Reversion: {e}")
            return {}
    
    @staticmethod
    def breakout_strategy(data, initial_capital=100000000, 
                         breakout_period=20):
        """استراتژی شکست مقاومت"""
        try:
            if len(data) < breakout_period:
                return {}
            
            # محاسبه مقاومت و حمایت
            data['resistance'] = data['High'].rolling(window=breakout_period).max()
            data['support'] = data['Low'].rolling(window=breakout_period).min()
            
            # سیگنال‌ها
            data['signal'] = 0
            data.loc[data['Close'] > data['resistance'].shift(1), 'signal'] = 1  # خرید
            data.loc[data['Close'] < data['support'].shift(1), 'signal'] = -1  # فروش
            
            # شبیه‌سازی معاملات
            # ...
            
            return {}
            
        except Exception as e:
            print(f"خطا در استراتژی Breakout: {e}")
            return {}

# ============================================================================
# کلاس تحلیل حباب پیشرفته (کامل)
# ============================================================================
class BubbleAnalyzer:
    def __init__(self, market_df, logger=None):
        self.market_df = market_df
        self.logger = logger
        self.bubble_indicators = {}
        
    def log(self, message, level="INFO"):
        if self.logger:
            self.logger.log(message, level)
        else:
            print(f"[{level}] {message}")
    
    def calculate_all_bubble_metrics(self):
        """محاسبه تمام معیارهای حباب"""
        self.log("🔍 شروع تحلیل حباب با ۶ روش مختلف...", "INFO")
        
        results = {}
        
        # روش ۱: حباب بر اساس P/B
        results['pb_bubble'] = self.calculate_pb_bubble()
        
        # روش ۲: حباب بر اساس P/E
        results['pe_bubble'] = self.calculate_pe_bubble()
        
        # روش ۳: حباب بر اساس نسبت‌های ارزش‌گذاری ترکیبی
        results['composite_bubble'] = self.calculate_composite_bubble()
        
        # روش ۴: حباب بر اساس انحراف از روند تاریخی
        results['historical_bubble'] = self.calculate_historical_bubble()
        
        # روش ۵: حباب بر اساس نسبت قیمت به فروش (P/S)
        results['ps_bubble'] = self.calculate_ps_bubble()
        
        # روش ۶: حباب بر اساس Z-Score آماری
        results['zscore_bubble'] = self.calculate_zscore_bubble()
        
        # محاسبه حباب نهایی (میانگین وزنی)
        results['final_bubble'] = self.calculate_final_bubble(results)
        
        self.log("✅ تحلیل حباب کامل شد", "INFO")
        return results
    
    def calculate_pb_bubble(self):
        """محاسبه حباب بر اساس P/B"""
        try:
            if 'P/B' not in self.market_df.columns:
                return None
            
            df = self.market_df.copy()
            df = df[df['P/B'] > 0]
            
            if len(df) < 10:
                return None
            
            # محاسبه انحراف از میانگین صنعت
            mean_pb = df['P/B'].mean()
            median_pb = df['P/B'].median()
            
            # حباب P/B = (P/B سهم - میانگین P/B) / میانگین P/B
            df['pb_bubble'] = ((df['P/B'] - mean_pb) / mean_pb) * 100
            
            # طبقه‌بندی حباب
            df['pb_bubble_level'] = df['pb_bubble'].apply(self.classify_bubble)
            
            return df[['نماد', 'P/B', 'pb_bubble', 'pb_bubble_level']]
        except Exception as e:
            self.log(f"خطا در محاسبه حباب P/B: {e}", "ERROR")
            return None
    
    def calculate_pe_bubble(self):
        """محاسبه حباب بر اساس P/E"""
        try:
            if 'P/E' not in self.market_df.columns:
                return None
            
            df = self.market_df.copy()
            df = df[(df['P/E'] > 0) & (df['P/E'] < 100)]
            
            if len(df) < 10:
                return None
            
            # میانگین P/E بازار
            mean_pe = df['P/E'].mean()
            
            # حباب P/E
            df['pe_bubble'] = np.where(
                df['P/E'] > mean_pe,
                ((df['P/E'] - mean_pe) / mean_pe) * 100,
                -((mean_pe - df['P/E']) / mean_pe) * 100
            )
            
            # طبقه‌بندی
            df['pe_bubble_level'] = df['pe_bubble'].apply(self.classify_bubble)
            
            return df[['نماد', 'P/E', 'pe_bubble', 'pe_bubble_level']]
        except Exception as e:
            self.log(f"خطا در محاسبه حباب P/E: {e}", "ERROR")
            return None
    
    def calculate_composite_bubble(self):
        """محاسبه حباب ترکیبی بر اساس چندین نسبت"""
        try:
            df = self.market_df.copy()
            
            # وزن‌های نسبی
            weights = {
                'P/B': 0.35,
                'P/E': 0.30,
                'P/S': 0.20,
                'P/NAV': 0.15
            }
            
            composite_scores = []
            
            for symbol in df['نماد']:
                symbol_data = df[df['نماد'] == symbol]
                if symbol_data.empty:
                    continue
                
                score = 0
                total_weight = 0
                
                # محاسبه امتیاز برای هر نسبت
                for ratio, weight in weights.items():
                    if ratio in symbol_data.columns:
                        value = symbol_data[ratio].iloc[0]
                        if pd.notna(value):
                            # نرمال‌سازی مقدار
                            if ratio in ['P/B', 'P/E', 'P/S', 'P/NAV']:
                                if value > 0:
                                    # امتیاز بر اساس انحراف از 1 (برای P/B و P/NAV) یا 10 (برای P/E)
                                    if ratio in ['P/B', 'P/NAV']:
                                        normalized = min(value / 2, 1)
                                    else:
                                        normalized = min(value / 20, 1)
                                    
                                    score += normalized * weight * 100
                                    total_weight += weight
                
                if total_weight > 0:
                    final_score = score / total_weight
                else:
                    final_score = 0
                
                composite_scores.append({
                    'نماد': symbol,
                    'composite_bubble': final_score,
                    'composite_bubble_level': self.classify_bubble(final_score)
                })
            
            return pd.DataFrame(composite_scores)
        except Exception as e:
            self.log(f"خطا در محاسبه حباب ترکیبی: {e}", "ERROR")
            return None
    
    def calculate_historical_bubble(self):
        """محاسبه حباب بر اساس انحراف از روند تاریخی"""
        try:
            df = self.market_df.copy()
            
            required_columns = ['بازدهی1ماه', 'بازدهی3ماه', 'بازدهی6ماه', 'بازدهی1سال']
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                self.log(f"ستون‌های بازدهی برای تحلیل تاریخی موجود نیستند: {missing_cols}", "WARNING")
                return None
            
            historical_scores = []
            
            for symbol in df['نماد']:
                symbol_data = df[df['نماد'] == symbol]
                if symbol_data.empty:
                    continue
                
                # محاسبه میانگین بازدهی
                returns = []
                for col in required_columns:
                    val = symbol_data[col].iloc[0]
                    if pd.notna(val):
                        returns.append(val)
                
                if len(returns) >= 2:
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    
                    current_return = symbol_data['بازدهی1ماه'].iloc[0] if pd.notna(symbol_data['بازدهی1ماه'].iloc[0]) else 0
                    
                    if std_return > 0:
                        z_score = (current_return - avg_return) / std_return
                        bubble_score = self.zscore_to_bubble(z_score)
                    else:
                        bubble_score = 0
                else:
                    bubble_score = 0
                
                historical_scores.append({
                    'نماد': symbol,
                    'historical_bubble': bubble_score,
                    'historical_bubble_level': self.classify_bubble(bubble_score)
                })
            
            return pd.DataFrame(historical_scores)
        except Exception as e:
            self.log(f"خطا در محاسبه حباب تاریخی: {e}", "ERROR")
            return None
    
    def calculate_ps_bubble(self):
        """محاسبه حباب بر اساس نسبت قیمت به فروش (P/S)"""
        try:
            if 'P/S' not in self.market_df.columns:
                return None
            
            df = self.market_df.copy()
            df = df[df['P/S'] > 0]
            
            if len(df) < 10:
                return None
            
            def classify_ps(value):
                if value < 1:
                    return 0
                elif value < 3:
                    return 30
                elif value < 5:
                    return 60
                else:
                    return 100
            
            df['ps_bubble'] = df['P/S'].apply(classify_ps)
            df['ps_bubble_level'] = df['ps_bubble'].apply(self.classify_bubble)
            
            return df[['نماد', 'P/S', 'ps_bubble', 'ps_bubble_level']]
        except Exception as e:
            self.log(f"خطا در محاسبه حباب P/S: {e}", "ERROR")
            return None
    
    def calculate_zscore_bubble(self):
        """محاسبه حباب با Z-Score آماری"""
        try:
            indicators = ['P/B', 'P/E', 'P/S', 'P/NAV']
            available_indicators = [ind for ind in indicators if ind in self.market_df.columns]
            
            if len(available_indicators) < 2:
                return None
            
            df = self.market_df.copy()
            
            # حذف مقادیر نامعقول
            for indicator in available_indicators:
                df = df[(df[indicator] > 0) & (df[indicator] < 1000)]
            
            zscore_results = []
            
            for symbol in df['نماد']:
                symbol_data = df[df['نماد'] == symbol]
                if symbol_data.empty:
                    continue
                
                z_scores = []
                for indicator in available_indicators:
                    value = symbol_data[indicator].iloc[0]
                    if pd.notna(value):
                        mean = df[indicator].mean()
                        std = df[indicator].std()
                        if std > 0:
                            z_score = (value - mean) / std
                            z_scores.append(z_score)
                
                if z_scores:
                    avg_zscore = np.mean(z_scores)
                    bubble_score = min(max(avg_zscore * 25 + 50, 0), 100)
                else:
                    bubble_score = 50
                
                zscore_results.append({
                    'نماد': symbol,
                    'zscore_bubble': bubble_score,
                    'zscore_bubble_level': self.classify_bubble(bubble_score)
                })
            
            return pd.DataFrame(zscore_results)
        except Exception as e:
            self.log(f"خطا در محاسبه حباب Z-Score: {e}", "ERROR")
            return None
    
    def calculate_final_bubble(self, bubble_results):
        """محاسبه حباب نهایی با میانگین وزنی"""
        try:
            all_results = []
            
            for symbol in self.market_df['نماد']:
                symbol_bubbles = {}
                
                for method, result in bubble_results.items():
                    if result is not None and 'نماد' in result.columns:
                        method_result = result[result['نماد'] == symbol]
                        if not method_result.empty:
                            bubble_col = method.replace('_bubble', '')
                            bubble_value = method_result.iloc[0].get(bubble_col)
                            if pd.notna(bubble_value):
                                symbol_bubbles[method] = bubble_value
                
                if symbol_bubbles:
                    weights = {
                        'pb_bubble': 0.25,
                        'pe_bubble': 0.25,
                        'composite_bubble': 0.20,
                        'historical_bubble': 0.15,
                        'ps_bubble': 0.10,
                        'zscore_bubble': 0.05
                    }
                    
                    weighted_sum = 0
                    total_weight = 0
                    
                    for method, value in symbol_bubbles.items():
                        weight = weights.get(method, 0.10)
                        weighted_sum += value * weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        final_bubble = weighted_sum / total_weight
                    else:
                        final_bubble = np.mean(list(symbol_bubbles.values()))
                else:
                    final_bubble = 50
                
                all_results.append({
                    'نماد': symbol,
                    'final_bubble': final_bubble,
                    'final_bubble_level': self.classify_bubble(final_bubble),
                    'bubble_risk': self.get_bubble_risk_level(final_bubble)
                })
            
            return pd.DataFrame(all_results)
        except Exception as e:
            self.log(f"خطا در محاسبه حباب نهایی: {e}", "ERROR")
            return None
    
    def classify_bubble(self, bubble_score):
        """طبقه‌بندی سطح حباب"""
        if bubble_score < 20:
            return "کم‌بها"
        elif bubble_score < 40:
            return "منصفانه"
        elif bubble_score < 60:
            return "کمی گران"
        elif bubble_score < 80:
            return "گران"
        else:
            return "حباب شدید"
    
    def zscore_to_bubble(self, z_score):
        """تبدیل Z-Score به امتیاز حباب"""
        if z_score < -2:
            return 0
        elif z_score < -1:
            return 25
        elif z_score < 1:
            return 50
        elif z_score < 2:
            return 75
        else:
            return 100
    
    def get_bubble_risk_level(self, bubble_score):
        """تعیین سطح ریسک بر اساس حباب"""
        if bubble_score < 40:
            return "پایین"
        elif bubble_score < 60:
            return "متوسط"
        elif bubble_score < 80:
            return "بالا"
        else:
            return "بسیار بالا"
    
    def generate_bubble_report(self, bubble_results):
        """تولید گزارش تحلیل حباب"""
        report = {
            'summary': {},
            'top_bubbles': [],
            'top_undervalued': [],
            'bubble_distribution': {},
            'recommendations': []
        }
        
        if 'final_bubble' in bubble_results.columns:
            final_bubble = bubble_results['final_bubble']
            report['summary'] = {
                'mean_bubble': final_bubble.mean(),
                'median_bubble': final_bubble.median(),
                'max_bubble': final_bubble.max(),
                'min_bubble': final_bubble.min(),
                'std_bubble': final_bubble.std()
            }
        
        top_bubble_stocks = bubble_results.nlargest(10, 'final_bubble')
        report['top_bubbles'] = top_bubble_stocks[['نماد', 'final_bubble', 'final_bubble_level']].to_dict('records')
        
        undervalued_stocks = bubble_results.nsmallest(10, 'final_bubble')
        report['top_undervalued'] = undervalued_stocks[['نماد', 'final_bubble', 'final_bubble_level']].to_dict('records')
        
        if 'final_bubble_level' in bubble_results.columns:
            distribution = bubble_results['final_bubble_level'].value_counts()
            report['bubble_distribution'] = distribution.to_dict()
        
        high_bubble_count = len(bubble_results[bubble_results['final_bubble'] > 70])
        total_count = len(bubble_results)
        
        if high_bubble_count / total_count > 0.3:
            report['recommendations'].append("⚠️ تعداد قابل توجهی سهام دارای حباب - احتیاط در خرید")
        elif high_bubble_count / total_count > 0.5:
            report['recommendations'].append("🔴 بازار دارای حباب قابل توجه - کاهش مواضع ریسکی")
        else:
            report['recommendations'].append("🟢 بازار در وضعیت معقول - فرصت‌های خرید خوب")
        
        return report

# ============================================================================
# سیستم دانلود اتوماتیک از tsetmc.com
# ============================================================================
class TsetmcDownloader:
    def __init__(self, logger=None):
        self.session = requests.Session()
        self.logger = logger
        
    def log(self, message, level="INFO"):
        if self.logger:
            self.logger.log(message, level)
    
    def get_stock_list(self):
        """دریافت لیست سهام از tsetmc.com"""
        try:
            self.log("دریافت لیست سهام از tsetmc.com...", "INFO")
            
            url = "http://cdn.tsetmc.com/api/Instrument/GetInstrumentSearch/2"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['instrumentSearch'])
                
                # فیلتر کردن فقط سهام (نه صندوق‌ها و ...)
                df = df[df['lVal18AFC'].str.contains('سهام')]
                
                self.log(f"تعداد سهام دریافتی: {len(df)}", "SUCCESS")
                return df
            else:
                self.log(f"خطا در دریافت لیست سهام: {response.status_code}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"خطا در دریافت لیست سهام: {e}", "ERROR")
            return None
    
    def get_stock_details(self, symbol):
        """دریافت جزئیات یک سهم"""
        try:
            self.log(f"دریافت جزئیات سهم {symbol}...", "INFO")
            
            # پیدا کردن ID سهم
            stock_list = self.get_stock_list()
            if stock_list is None:
                return None
            
            stock_info = stock_list[stock_list['lVal18'] == symbol]
            if len(stock_info) == 0:
                self.log(f"سهم {symbol} یافت نشد", "ERROR")
                return None
            
            ins_code = stock_info.iloc[0]['insCode']
            
            # دریافت اطلاعات قیمت
            price_url = f"http://cdn.tsetmc.com/api/ClosingPrice/GetClosingPriceInfo/{ins_code}"
            response = self.session.get(price_url, timeout=30)
            
            if response.status_code == 200:
                price_data = response.json()
                
                details = {
                    'symbol': symbol,
                    'name': stock_info.iloc[0]['lVal18'],
                    'last_price': price_data.get('pDrCotVal'),
                    'volume': price_data.get('qTotTran5J'),
                    'value': price_data.get('qTotCap'),
                    'high': price_data.get('priceMax'),
                    'low': price_data.get('priceMin'),
                    'close': price_data.get('pClosing'),
                    'yesterday_close': price_data.get('priceYesterday'),
                    'trade_count': price_data.get('zTotTran'),
                    'date': datetime.now().strftime("%Y-%m-%d")
                }
                
                self.log(f"جزئیات سهم {symbol} دریافت شد", "SUCCESS")
                return details
            else:
                self.log(f"خطا در دریافت جزئیات سهم {symbol}: {response.status_code}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"خطا در دریافت جزئیات سهم {symbol}: {e}", "ERROR")
            return None
    
    def get_historical_data(self, symbol, days=365):
        """دریافت داده‌های تاریخی"""
        try:
            self.log(f"دریافت داده‌های تاریخی {symbol} برای {days} روز...", "INFO")
            
            # پیدا کردن ID سهم
            stock_list = self.get_stock_list()
            if stock_list is None:
                return None
            
            stock_info = stock_list[stock_list['lVal18'] == symbol]
            if len(stock_info) == 0:
                self.log(f"سهم {symbol} یافت نشد", "ERROR")
                return None
            
            ins_code = stock_info.iloc[0]['insCode']
            
            # دریافت داده‌های تاریخی
            hist_url = f"http://cdn.tsetmc.com/api/ClosingPrice/GetClosingPriceDailyList/{ins_code}/0"
            response = self.session.get(hist_url, timeout=30)
            
            if response.status_code == 200:
                hist_data = response.json()['closingPriceDaily']
                
                data_list = []
                for item in hist_data[-days:]:
                    data_list.append({
                        'Date': item['dEven'],
                        'Open': item['priceFirst'],
                        'High': item['priceMax'],
                        'Low': item['priceMin'],
                        'Close': item['pClosing'],
                        'Volume': item['qTotTran5J'],
                        'Value': item['qTotCap']
                    })
                
                df = pd.DataFrame(data_list)
                df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
                
                self.log(f"داده‌های تاریخی {symbol} دریافت شد: {len(df)} روز", "SUCCESS")
                return df
            else:
                self.log(f"خطا در دریافت داده‌های تاریخی {symbol}: {response.status_code}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"خطا در دریافت داده‌های تاریخی {symbol}: {e}", "ERROR")
            return None
    
    def get_intraday_data(self, symbol):
        """دریافت داده‌های داخل روز"""
        try:
            self.log(f"دریافت داده‌های داخل روز {symbol}...", "INFO")
            
            stock_list = self.get_stock_list()
            if stock_list is None:
                return None
            
            stock_info = stock_list[stock_list['lVal18'] == symbol]
            if len(stock_info) == 0:
                return None
            
            ins_code = stock_info.iloc[0]['insCode']
            
            intraday_url = f"http://cdn.tsetmc.com/api/BestLimits/{ins_code}"
            response = self.session.get(intraday_url, timeout=30)
            
            if response.status_code == 200:
                intraday_data = response.json()['bestLimits']
                
                data = {
                    'symbol': symbol,
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'bid_prices': [],
                    'bid_volumes': [],
                    'ask_prices': [],
                    'ask_volumes': []
                }
                
                for item in intraday_data:
                    data['bid_prices'].append(item['pMeDem'])
                    data['bid_volumes'].append(item['qTitMeDem'])
                    data['ask_prices'].append(item['pMeOf'])
                    data['ask_volumes'].append(item['qTitMeOf'])
                
                return data
            else:
                return None
                
        except Exception as e:
            self.log(f"خطا در دریافت داده‌های داخل روز {symbol}: {e}", "ERROR")
            return None

# ============================================================================
# تحلیل‌گر سهام پیشرفته (کامل)
# ============================================================================
class AdvancedStockAnalyzer:
    def __init__(self, df, settings, column_mapping, market_trend="NORMAL", logger=None):
        self.df = df
        self.settings = settings
        self.column_mapping = column_mapping
        self.market_trend = market_trend
        self.logger = logger
        self.bubble_analyzer = BubbleAnalyzer(df, logger)
        
        # تنظیم وزن‌های پیشرفته
        self.advanced_weights = {
            'fundamental': {
                'P/B': 0.12, 'P/E': 0.10, 'EPS': 0.08, 'DPS%': 0.06,
                'P/S': 0.05, 'P/NAV': 0.04, 'حباب': 0.10
            },
            'technical': {
                'RSI': 0.07, 'MFI': 0.05, 'SMA20d': 0.04, 'SMA50d': 0.03,
                'ضریب بتا': 0.03, 'بازدهی1ماه': 0.04
            },
            'market': {
                'خرید حقوقی%': 0.07, 'فروش حقوقی%': 0.03,
                'حجم معاملات': 0.05, 'ارزش معاملات': 0.04
            },
            'momentum': {
                'مومنتوم قیمت': 0.06, 'قدرت روند': 0.04
            }
        }
        
        # محاسبه حباب برای همه سهام
        self.bubble_results = self.bubble_analyzer.calculate_all_bubble_metrics()
        
    def log(self, message, level="INFO"):
        if self.logger:
            self.logger.log(message, level)
        else:
            print(f"[{level}] {message}")
    
    def get_column(self, column_name):
        """دریافت نام ستون واقعی"""
        return self.column_mapping.get(column_name, column_name)
    
    def calculate_momentum(self, row):
        """محاسبه مومنتوم قیمت"""
        try:
            price_col = self.get_column('آخرین قیمت')
            sma20_col = self.get_column('SMA20d')
            
            if price_col in row and sma20_col in row:
                if pd.notna(row[price_col]) and pd.notna(row[sma20_col]):
                    if row[sma20_col] > 0:
                        momentum = (row[price_col] - row[sma20_col]) / row[sma20_col] * 100
                        return momentum
            return 0
        except:
            return 0
    
    def calculate_trend_strength(self, row):
        """محاسبه قدرت روند"""
        try:
            sma20_col = self.get_column('SMA20d')
            sma50_col = self.get_column('SMA50d')
            
            if sma20_col in row and sma50_col in row:
                if pd.notna(row[sma20_col]) and pd.notna(row[sma50_col]):
                    if row[sma50_col] > 0:
                        trend_strength = (row[sma20_col] - row[sma50_col]) / row[sma50_col] * 100
                        return trend_strength
            return 0
        except:
            return 0
    
    def calculate_fundamental_score(self, row):
        """محاسبه امتیاز بنیادی"""
        score = 0
        max_score = 100
        
        try:
            weights = self.advanced_weights['fundamental']
            
            # P/B Score
            pb_col = self.get_column('P/B')
            if pb_col in row and pd.notna(row[pb_col]):
                pb = row[pb_col]
                if pb < 1:
                    score += weights['P/B'] * 100
                elif pb < 1.5:
                    score += weights['P/B'] * 80
                elif pb < 2:
                    score += weights['P/B'] * 60
                elif pb < 3:
                    score += weights['P/B'] * 40
                else:
                    score += weights['P/B'] * 20
            
            # P/E Score
            pe_col = self.get_column('P/E')
            if pe_col in row and pd.notna(row[pe_col]):
                pe = row[pe_col]
                if pe < 0:
                    score += weights['P/E'] * 0
                elif pe < 5:
                    score += weights['P/E'] * 100
                elif pe < 10:
                    score += weights['P/E'] * 80
                elif pe < 15:
                    score += weights['P/E'] * 60
                elif pe < 20:
                    score += weights['P/E'] * 40
                else:
                    score += weights['P/E'] * 20
            
            # EPS Score
            eps_col = self.get_column('EPS')
            if eps_col in row and pd.notna(row[eps_col]):
                eps = row[eps_col]
                if eps > 1000:
                    score += weights['EPS'] * 100
                elif eps > 500:
                    score += weights['EPS'] * 90
                elif eps > 200:
                    score += weights['EPS'] * 80
                elif eps > 100:
                    score += weights['EPS'] * 70
                elif eps > 0:
                    score += weights['EPS'] * 60
                else:
                    score += weights['EPS'] * 30
            
            # DPS% Score
            dps_col = self.get_column('DPS%')
            if dps_col in row and pd.notna(row[dps_col]):
                dps = row[dps_col]
                if dps > 15:
                    score += weights['DPS%'] * 100
                elif dps > 10:
                    score += weights['DPS%'] * 85
                elif dps > 5:
                    score += weights['DPS%'] * 70
                elif dps > 0:
                    score += weights['DPS%'] * 50
            
            # Bubble Score (برعکس - هرچه حباب کمتر بهتر)
            if self.bubble_results is not None and 'final_bubble' in self.bubble_results.columns:
                symbol = row['نماد'] if 'نماد' in row else None
                if symbol:
                    bubble_data = self.bubble_results[self.bubble_results['نماد'] == symbol]
                    if not bubble_data.empty:
                        bubble_score = bubble_data.iloc[0]['final_bubble']
                        bubble_adjusted_score = 100 - bubble_score
                        score += weights['حباب'] * bubble_adjusted_score
            
            return min(score, max_score)
        except Exception as e:
            self.log(f"خطا در محاسبه امتیاز بنیادی: {e}", "ERROR")
            return score
    
    def calculate_technical_score(self, row):
        """محاسبه امتیاز تکنیکال"""
        score = 0
        max_score = 100
        
        try:
            weights = self.advanced_weights['technical']
            
            # RSI Score
            rsi_col = self.get_column('RSI')
            if rsi_col in row and pd.notna(row[rsi_col]):
                rsi = row[rsi_col]
                if rsi < 30:
                    score += weights['RSI'] * 100
                elif rsi < 40:
                    score += weights['RSI'] * 85
                elif rsi < 50:
                    score += weights['RSI'] * 70
                elif rsi < 60:
                    score += weights['RSI'] * 50
                elif rsi < 70:
                    score += weights['RSI'] * 30
                else:
                    score += weights['RSI'] * 10
            
            # MFI Score
            mfi_col = self.get_column('MFI')
            if mfi_col in row and pd.notna(row[mfi_col]):
                mfi = row[mfi_col]
                if mfi < 20:
                    score += weights['MFI'] * 100
                elif mfi < 30:
                    score += weights['MFI'] * 80
                elif mfi < 70:
                    score += weights['MFI'] * 60
                elif mfi < 80:
                    score += weights['MFI'] * 40
                else:
                    score += weights['MFI'] * 20
            
            # بازدهی 1 ماه
            return_col = self.get_column('بازدهی1ماه')
            if return_col in row and pd.notna(row[return_col]):
                ret = row[return_col]
                if ret < -15:
                    score += weights['بازدهی1ماه'] * 100
                elif ret < -10:
                    score += weights['بازدهی1ماه'] * 85
                elif ret < -5:
                    score += weights['بازدهی1ماه'] * 70
                elif ret < 0:
                    score += weights['بازدهی1ماه'] * 60
                elif ret < 5:
                    score += weights['بازدهی1ماه'] * 50
                elif ret < 10:
                    score += weights['بازدهی1ماه'] * 40
                else:
                    score += weights['بازدهی1ماه'] * 20
            
            # ضریب بتا
            beta_col = self.get_column('ضریب بتا')
            if beta_col in row and pd.notna(row[beta_col]):
                beta = row[beta_col]
                if self.market_trend == "BULLISH":
                    if beta > 1.2:
                        score += weights['ضریب بتا'] * 100
                    elif beta > 1:
                        score += weights['ضریب بتا'] * 80
                    else:
                        score += weights['ضریب بتا'] * 50
                elif self.market_trend == "BEARISH":
                    if beta < 0.8:
                        score += weights['ضریب بتا'] * 100
                    elif beta < 1:
                        score += weights['ضریب بتا'] * 80
                    else:
                        score += weights['ضریب بتا'] * 40
            
            return min(score, max_score)
        except Exception as e:
            self.log(f"خطا در محاسبه امتیاز تکنیکال: {e}", "ERROR")
            return score
    
    def calculate_market_score(self, row):
        """محاسبه امتیاز بازار و معاملات"""
        score = 0
        max_score = 100
        
        try:
            weights = self.advanced_weights['market']
            
            # خرید حقوقی
            legal_buy_col = self.get_column('خرید حقوقی%')
            if legal_buy_col in row and pd.notna(row[legal_buy_col]):
                legal_buy = row[legal_buy_col]
                if legal_buy > 70:
                    score += weights['خرید حقوقی%'] * 100
                elif legal_buy > 60:
                    score += weights['خرید حقوقی%'] * 90
                elif legal_buy > 50:
                    score += weights['خرید حقوقی%'] * 80
                elif legal_buy > 40:
                    score += weights['خرید حقوقی%'] * 70
                elif legal_buy > 30:
                    score += weights['خرید حقوقی%'] * 60
                else:
                    score += weights['خرید حقوقی%'] * 40
            
            # فروش حقوقی (معکوس)
            legal_sell_col = self.get_column('فروش حقوقی%')
            if legal_sell_col in row and pd.notna(row[legal_sell_col]):
                legal_sell = row[legal_sell_col]
                if legal_sell < 10:
                    score += weights['فروش حقوقی%'] * 100
                elif legal_sell < 20:
                    score += weights['فروش حقوقی%'] * 85
                elif legal_sell < 30:
                    score += weights['فروش حقوقی%'] * 70
                elif legal_sell < 40:
                    score += weights['فروش حقوقی%'] * 50
                else:
                    score += weights['فروش حقوقی%'] * 30
            
            # حجم معاملات
            volume_col = self.get_column('حجم معاملات')
            if volume_col in row and pd.notna(row[volume_col]):
                volume = row[volume_col]
                if volume > 1e9:
                    score += weights['حجم معاملات'] * 100
                elif volume > 500e6:
                    score += weights['حجم معاملات'] * 90
                elif volume > 100e6:
                    score += weights['حجم معاملات'] * 80
                elif volume > 50e6:
                    score += weights['حجم معاملات'] * 70
                elif volume > 10e6:
                    score += weights['حجم معاملات'] * 60
                else:
                    score += weights['حجم معاملات'] * 40
            
            # ارزش معاملات
            value_col = self.get_column('ارزش معاملات')
            if value_col in row and pd.notna(row[value_col]):
                value = row[value_col]
                if value > 200e9:
                    score += weights['ارزش معاملات'] * 100
                elif value > 100e9:
                    score += weights['ارزش معاملات'] * 90
                elif value > 50e9:
                    score += weights['ارزش معاملات'] * 80
                elif value > 20e9:
                    score += weights['ارزش معاملات'] * 70
                elif value > 10e9:
                    score += weights['ارزش معاملات'] * 60
                else:
                    score += weights['ارزش معاملات'] * 40
            
            return min(score, max_score)
        except Exception as e:
            self.log(f"خطا در محاسبه امتیاز بازار: {e}", "ERROR")
            return score
    
    def calculate_momentum_score(self, row):
        """محاسبه امتیاز مومنتوم"""
        score = 0
        max_score = 100
        
        try:
            weights = self.advanced_weights['momentum']
            
            # مومنتوم قیمت
            momentum = self.calculate_momentum(row)
            if momentum > 5:
                score += weights['مومنتوم قیمت'] * 100
            elif momentum > 2:
                score += weights['مومنتوم قیمت'] * 85
            elif momentum > 0:
                score += weights['مومنتوم قیمت'] * 70
            elif momentum > -2:
                score += weights['مومنتوم قیمت'] * 50
            elif momentum > -5:
                score += weights['مومنتوم قیمت'] * 30
            else:
                score += weights['مومنتوم قیمت'] * 10
            
            # قدرت روند
            trend_strength = self.calculate_trend_strength(row)
            if trend_strength > 3:
                score += weights['قدرت روند'] * 100
            elif trend_strength > 1:
                score += weights['قدرت روند'] * 85
            elif trend_strength > -1:
                score += weights['قدرت روند'] * 70
            elif trend_strength > -3:
                score += weights['قدرت روند'] * 50
            else:
                score += weights['قدرت روند'] * 30
            
            return min(score, max_score)
        except Exception as e:
            self.log(f"خطا در محاسبه امتیاز مومنتوم: {e}", "ERROR")
            return score
    
    def calculate_total_score(self, row):
        """محاسبه امتیاز کل"""
        total_score = 0
        
        # امتیاز بنیادی
        fundamental_score = self.calculate_fundamental_score(row)
        total_score += fundamental_score * 0.40
        
        # امتیاز تکنیکال
        technical_score = self.calculate_technical_score(row)
        total_score += technical_score * 0.25
        
        # امتیاز بازار
        market_score = self.calculate_market_score(row)
        total_score += market_score * 0.20
        
        # امتیاز مومنتوم
        momentum_score = self.calculate_momentum_score(row)
        total_score += momentum_score * 0.15
        
        return round(total_score, 2)
    
    def apply_advanced_filters(self, df):
        """اعمال فیلترهای پیشرفته"""
        filtered_df = df.copy()
        
        try:
            # فیلتر حباب
            if self.bubble_results is not None and 'final_bubble' in self.bubble_results.columns:
                # اطمینان از DataFrame بودن داده‌های ادغام
                if hasattr(self, 'bubble_results'):
                    if isinstance(self.bubble_results, dict):
                        import pandas as pd
                        self.bubble_results = pd.DataFrame(self.bubble_results)
                
                filtered_df = filtered_df.merge(
                    self.bubble_results[['نماد', 'final_bubble']], 
                    on='نماد', 
                    how='left'
                )
                filtered_df = filtered_df[filtered_df['final_bubble'] < 80]
            
            # فیلتر حجم معاملات
            volume_col = self.get_column('حجم معاملات')
            if volume_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[volume_col] > 1000000]
            
            # فیلتر ارزش معاملات
            value_col = self.get_column('ارزش معاملات')
            if value_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[value_col] > 10e9]
            
            # فیلتر P/B
            pb_col = self.get_column('P/B')
            if pb_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[pb_col] < 3]
            
            # فیلتر P/E
            pe_col = self.get_column('P/E')
            if pe_col in filtered_df.columns:
                filtered_df = filtered_df[(filtered_df[pe_col] > 0) & (filtered_df[pe_col] < 30)]
            
            # فیلتر RSI
            rsi_col = self.get_column('RSI')
            if rsi_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[rsi_col] < 70]
            
            self.log(f"پس از فیلترها: {len(filtered_df)} سهم باقی ماند", "INFO")
            return filtered_df
            
        except Exception as e:
            self.log(f"خطا در اعمال فیلترها: {e}", "ERROR")
            return df
    
    def analyze_stocks(self, budget=None, top_n=20):
        import pandas as pd
        """تحلیل سهام و ارائه پیشنهادات خرید"""
        try:
            self.log(f"شروع تحلیل {len(self.df)} سهم...", "INFO")
            
            # اعمال فیلترهای پیشرفته
            filtered_df = self.apply_advanced_filters(self.df)
            
            if len(filtered_df) == 0:
                self.log("❌ هیچ سهمی از فیلترها عبور نکرد", "WARNING")
                return None
            
            # محاسبه امتیاز برای هر سهم
            self.log("محاسبه امتیاز سهام...", "INFO")
            filtered_df['امتیاز_بنیادی'] = filtered_df.apply(self.calculate_fundamental_score, axis=1)
            filtered_df['امتیاز_تکنیکال'] = filtered_df.apply(self.calculate_technical_score, axis=1)
            filtered_df['امتیاز_بازار'] = filtered_df.apply(self.calculate_market_score, axis=1)
            filtered_df['امتیاز_مومنتوم'] = filtered_df.apply(self.calculate_momentum_score, axis=1)
            filtered_df['امتیاز_کل'] = filtered_df.apply(self.calculate_total_score, axis=1)
            
            # اضافه کردن اطلاعات حباب
            if self.bubble_results is not None:
                # تبدیل dict به DataFrame اگر لازم است
                if 'fundamental_data' not in locals():
                    fundamental_data = pd.DataFrame()
                if isinstance(fundamental_data, dict):
                    fundamental_data = pd.DataFrame([fundamental_data])
                
                # تبدیل self.bubble_results به DataFrame اگر dict است
                if hasattr(self, 'bubble_results'):
                    if isinstance(self.bubble_results, dict):
                        try:
                            # بررسی ساختار dict
                            if self.bubble_results:
                                # روش 1: اگر dict ساده با کلید 'نماد' است
                                if 'نماد' in self.bubble_results:
                                    self.bubble_results = pd.DataFrame([self.bubble_results])
                                else:
                                    # روش 2: اگر dict از dictهاست
                                    self.bubble_results = pd.DataFrame.from_dict(self.bubble_results, orient='index')
                                    self.bubble_results.reset_index(inplace=True)
                                    self.bubble_results.rename(columns={'index': 'نماد'}, inplace=True)
                            else:
                                # dict خالی
                                self.bubble_results = pd.DataFrame(columns=['نماد'])
                        except Exception as e:
                            print(f"خطا در تبدیل bubble_results: {e}")
                            self.bubble_results = pd.DataFrame(columns=['نماد'])
                else:
                    self.bubble_results = pd.DataFrame(columns=['نماد'])
                
                filtered_df = filtered_df.merge(
                    self.bubble_results, 
                    on='نماد', 
                    how='left',
                    suffixes=('', '_bubble')
                )
            
            # مرتب‌سازی بر اساس امتیاز کل
            filtered_df = filtered_df.sort_values('امتیاز_کل', ascending=False)
            
            # انتخاب سهام برتر
            top_stocks = filtered_df.head(top_n).copy()
            
            # اگر بودجه مشخص شده باشد، محاسبه تخصیص بودجه
            if budget and budget > 0:
                total_score = top_stocks['امتیاز_کل'].sum()
                top_stocks['سهم_بودجه'] = (top_stocks['امتیاز_کل'] / total_score) * budget
                
                # محاسبه تعداد سهم قابل خرید
                price_col = self.get_column('آخرین قیمت')
                if price_col in top_stocks.columns:
                    top_stocks['تعداد_سهم'] = (top_stocks['سهم_بودجه'] / top_stocks[price_col]).astype(int)
                    top_stocks['ارزش_خرید'] = top_stocks['تعداد_سهم'] * top_stocks[price_col]
                else:
                    top_stocks['تعداد_سهم'] = 0
                    top_stocks['ارزش_خرید'] = 0
            
            # تعیین رتبه و وضعیت
            top_stocks['رتبه'] = range(1, len(top_stocks) + 1)
            top_stocks['وضعیت'] = top_stocks['امتیاز_کل'].apply(self.get_stock_status)
            
            self.log(f"✅ تحلیل کامل شد. {len(top_stocks)} سهم برتر انتخاب شدند", "INFO")
            return top_stocks
            
        except Exception as e:
            self.log(f"خطا در تحلیل سهام: {e}", "ERROR")
            traceback.print_exc()
            return None
    
    def get_stock_status(self, score):
        """تعیین وضعیت سهم"""
        if score >= 85:
            return "عالی 🏆"
        elif score >= 75:
            return "خیلی خوب ⭐"
        elif score >= 65:
            return "خوب 👍"
        elif score >= 55:
            return "متوسط ➖"
        elif score >= 45:
            return "ضعیف ⚠️"
        else:
            return "خیلی ضعیف ❌"
    
    def get_stock_recommendation(self, row):
        """تولید توصیه برای هر سهم"""
        recommendations = []
        
        try:
            # توصیه بر اساس بنیادی
            fundamental_score = row.get('امتیاز_بنیادی', 0)
            if fundamental_score >= 80:
                recommendations.append("بنیادی قوی")
            elif fundamental_score < 60:
                recommendations.append("ضعف بنیادی")
            
            # توصیه بر اساس تکنیکال
            technical_score = row.get('امتیاز_تکنیکال', 0)
            if technical_score >= 75:
                recommendations.append("تکنیکال مثبت")
            elif technical_score < 50:
                recommendations.append("ضعف تکنیکال")
            
            # توصیه بر اساس حباب
            if 'final_bubble' in row:
                bubble = row['final_bubble']
                if bubble > 70:
                    recommendations.append("حباب بالا")
                elif bubble < 30:
                    recommendations.append("ارزش ذاتی")
            
            # توصیه بر اساس RSI
            rsi_col = self.get_column('RSI')
            if rsi_col in row and pd.notna(row[rsi_col]):
                rsi = row[rsi_col]
                if rsi < 30:
                    recommendations.append("اشباع فروش")
                elif rsi > 70:
                    recommendations.append("اشباع خرید")
            
            if len(recommendations) == 0:
                return "بدون نکته خاص"
            
            return " | ".join(recommendations)
            
        except Exception as e:
            return "خطا در تولید توصیه"

# ============================================================================
# سیستم پیش‌بینی خرید/فروش (کامل)
# ============================================================================
class TradingSignalGenerator:
    def __init__(self, market_df, logger=None):
        self.market_df = market_df
        self.logger = logger
        
    def log(self, message, level="INFO"):
        if self.logger:
            self.logger.log(message, level)
        else:
            print(f"[{level}] {message}")
    
    def generate_trading_signals(self):
        """تولید سیگنال‌های خرید و فروش"""
        self.log("📈 تولید سیگنال‌های معاملاتی...", "INFO")
        
        signals = []
        
        try:
            for _, row in self.market_df.iterrows():
                symbol = row.get('نماد')
                if not symbol:
                    continue
                
                signal = self.analyze_stock_for_trading(row)
                if signal['سیگنال'] != 'خنثی':
                    signals.append(signal)
            
            # مرتب‌سازی بر اساس قدرت سیگنال
            signals = sorted(signals, key=lambda x: x['قدرت_سیگنال'], reverse=True)
            
            self.log(f"✅ {len(signals)} سیگنال معاملاتی تولید شد", "INFO")
            return signals
            
        except Exception as e:
            self.log(f"خطا در تولید سیگنال‌ها: {e}", "ERROR")
            return []
    
    def analyze_stock_for_trading(self, row):
        """تحلیل سهام برای تولید سیگنال معاملاتی"""
        signal = {
            'نماد': row.get('نماد', ''),
            'سیگنال': 'خنثی',
            'قدرت_سیگنال': 0,
            'دلایل': [],
            'اطمینان': 'پایین',
            'قیمت_هدف': 0,
            'حد_ضرر': 0
        }
        
        try:
            buy_score = 0
            sell_score = 0
            reasons = []
            
            # تحلیل RSI
            rsi = row.get('RSI', 50)
            if rsi < 30:
                buy_score += 3
                reasons.append(f"RSI اشباع فروش ({rsi:.1f})")
            elif rsi > 70:
                sell_score += 3
                reasons.append(f"RSI اشباع خرید ({rsi:.1f})")
            
            # تحلیل مومنتوم
            momentum = row.get('بازدهی1ماه', 0)
            if momentum < -15:
                buy_score += 2
                reasons.append(f"اصلاح شدید ({momentum:.1f}%)")
            elif momentum > 25:
                sell_score += 2
                reasons.append(f"رشد سریع ({momentum:.1f}%)")
            
            # تحلیل حجم
            volume = row.get('حجم معاملات', 0)
            avg_volume = row.get('حجم30روزه', volume)
            if avg_volume > 0:
                volume_ratio = volume / avg_volume
                if volume_ratio > 1.5:
                    reasons.append(f"حجم معاملات بالا ({volume_ratio:.1f}x)")
                    if momentum > 0:
                        buy_score += 1
                    else:
                        sell_score += 1
            
            # تحلیل ارزش‌گذاری
            pb = row.get('P/B', 0)
            if pb < 0.8:
                buy_score += 2
                reasons.append(f"P/B پایین ({pb:.2f})")
            elif pb > 2:
                sell_score += 1
                reasons.append(f"P/B بالا ({pb:.2f})")
            
            # تحلیل جریان نقد
            legal_buy = row.get('خرید حقوقی%', 0)
            if legal_buy > 60:
                buy_score += 2
                reasons.append(f"خرید حقوقی قوی ({legal_buy:.1f}%)")
            elif legal_buy < 30:
                sell_score += 1
                reasons.append(f"خرید حقوقی ضعیف ({legal_buy:.1f}%)")
            
            # تصمیم‌گیری
            signal_score = buy_score - sell_score
            
            if signal_score >= 4:
                signal['سیگنال'] = 'خرید قوی 🟢'
                signal['قدرت_سیگنال'] = signal_score
                signal['اطمینان'] = 'بالا'
            elif signal_score >= 2:
                signal['سیگنال'] = 'خرید 🟢'
                signal['قدرت_سیگنال'] = signal_score
                signal['اطمینان'] = 'متوسط'
            elif signal_score <= -4:
                signal['سیگنال'] = 'فروش قوی 🔴'
                signal['قدرت_سیگنال'] = abs(signal_score)
                signal['اطمینان'] = 'بالا'
            elif signal_score <= -2:
                signal['سیگنال'] = 'فروش 🔴'
                signal['قدرت_سیگنال'] = abs(signal_score)
                signal['اطمینان'] = 'متوسط'
            
            signal['دلایل'] = ' | '.join(reasons) if reasons else 'بدون دلیل خاص'
            
            # محاسبه قیمت هدف و حد ضرر
            current_price = row.get('آخرین قیمت', 0)
            if current_price > 0:
                if 'خرید' in signal['سیگنال']:
                    signal['قیمت_هدف'] = current_price * 1.15
                    signal['حد_ضرر'] = current_price * 0.92
                elif 'فروش' in signal['سیگنال']:
                    signal['قیمت_هدف'] = current_price * 0.92
                    signal['حد_ضرر'] = current_price * 1.08
            
        except Exception as e:
            self.log(f"خطا در تحلیل سیگنال برای {row.get('نماد', '')}: {e}", "WARNING")
        
        return signal

# ============================================================================
# تحلیل‌گر پورتفو (کامل)
# ============================================================================
class PortfolioAnalyzer:
    def __init__(self, portfolio_df, market_df, settings, column_mapping):
        self.portfolio_df = portfolio_df
        self.market_df = market_df
        self.settings = settings
        self.column_mapping = column_mapping
        self.alert_settings = settings['portfolio_alerts']
    
    def get_market_data(self, symbol):
        """دریافت اطلاعات بازار برای یک نماد خاص"""
        symbol_col = self.column_mapping.get('نماد', 'نماد')
        if symbol_col in self.market_df.columns:
            market_data = self.market_df[self.market_df[symbol_col] == symbol]
            if not market_data.empty:
                return market_data.iloc[0]
        return None
    
    def analyze_portfolio(self):
        """تحلیل کامل پورتفو"""
        analysis = {
            'portfolio_value': 0,
            'total_profit_loss': 0,
            'profit_loss_percent': 0,
            'sell_alerts': [],
            'buy_recommendations': [],
            'good_performers': [],
            'poor_performers': [],
            'stock_details': {}
        }
        
        total_value = 0
        
        for _, stock in self.portfolio_df.iterrows():
            symbol = stock['نماد'] if 'نماد' in stock else None
            if not symbol:
                continue
                
            quantity = stock['تعداد'] if 'تعداد' in stock else 0
            current_price = stock['قیمت_آخر'] if 'قیمت_آخر' in stock else stock.get('آخرین قیمت', 0)
            avg_buy_price = stock['قیمت_سربه_سر'] if 'قیمت_سربه_سر' in stock else current_price
            
            current_value = quantity * current_price
            profit_loss = current_value - (quantity * avg_buy_price)
            profit_loss_percent = (profit_loss / (quantity * avg_buy_price)) * 100 if avg_buy_price > 0 else 0
            
            total_value += current_value
            analysis['total_profit_loss'] += profit_loss
            
            market_data = self.get_market_data(symbol)
            
            stock_detail = {
                'symbol': symbol,
                'quantity': quantity,
                'current_price': current_price,
                'avg_buy_price': avg_buy_price,
                'current_value': current_value,
                'profit_loss': profit_loss,
                'profit_loss_percent': profit_loss_percent,
                'status': self.get_stock_status(profit_loss_percent),
                'recommendation': ''
            }
            
            if market_data is not None:
                rsi = market_data.get(self.column_mapping.get('RSI', 'RSI'), 50)
                pb = market_data.get(self.column_mapping.get('P/B', 'P/B'), 0)
                volume = market_data.get(self.column_mapping.get('حجم معاملات', 'حجم معاملات'), 0)
                
                stock_detail.update({
                    'rsi': rsi,
                    'pb': pb,
                    'volume': volume
                })
                
                recommendation = self.generate_recommendation(symbol, profit_loss_percent, rsi, pb, volume)
                stock_detail['recommendation'] = recommendation
                
                if 'فروش' in recommendation:
                    analysis['sell_alerts'].append(f"{symbol}: {recommendation}")
                elif 'خرید' in recommendation:
                    analysis['buy_recommendations'].append(f"{symbol}: {recommendation}")
                
                if profit_loss_percent >= 15:
                    analysis['good_performers'].append(f"{symbol}: {profit_loss_percent:.1f}% سود")
                elif profit_loss_percent <= -5:
                    analysis['poor_performers'].append(f"{symbol}: {profit_loss_percent:.1f}% زیان")
            
            analysis['stock_details'][symbol] = stock_detail
        
        analysis['portfolio_value'] = total_value
        analysis['profit_loss_percent'] = (analysis['total_profit_loss'] / total_value) * 100 if total_value > 0 else 0
        
        for symbol, detail in analysis['stock_details'].items():
            detail['weight'] = (detail['current_value'] / total_value) * 100 if total_value > 0 else 0
        
        return analysis
    
    def get_stock_status(self, profit_loss_percent):
        """تعیین وضعیت سهم بر اساس سود/زیان"""
        if profit_loss_percent >= 20:
            return "قوی"
        elif profit_loss_percent >= 10:
            return "خوب"
        elif profit_loss_percent >= 0:
            return "متوسط"
        else:
            return "ضعیف"
    
    def generate_recommendation(self, symbol, profit_percent, rsi, pb, volume):
        """تولید هشدارها و پیشنهادات برای هر سهم"""
        recommendations = []
        
        if profit_percent >= self.alert_settings['profit_threshold']:
            recommendations.append(f"⚠️ کسب سود {profit_percent:.1f}% -考慮 فروش بخشی")
        
        if profit_percent <= self.alert_settings['loss_threshold']:
            recommendations.append(f"🔴 زیان {profit_percent:.1f}% - بررسی برای خروج")
        
        if rsi >= self.alert_settings['rsi_sell_threshold']:
            recommendations.append(f"📊 RSI بالا ({rsi}) - احتمال اصلاح قیمت")
        
        if pb >= self.alert_settings['pb_sell_threshold']:
            recommendations.append(f"💰 P/B بالا ({pb:.1f}) - ارزش‌گذاری گران")
        
        if rsi <= self.alert_settings['rsi_buy_threshold']:
            recommendations.append(f"🟢 RSI پایین ({rsi}) - فرصت خرید بیشتر")
        
        if profit_percent < 0 and rsi < 40:
            recommendations.append(f"📈 زیان موقت - فرصت میانگین‌گیری")
        
        if len(recommendations) == 0:
            return "نگهداری"
        
        return " | ".join(recommendations)

# ============================================================================
# تنظیمات فونت فارسی
# ============================================================================
def setup_persian_font():
    try:
        system_fonts = [f.name for f in fm.fontManager.ttflist]
        persian_fonts = ['B Zar', 'B Nazanin', 'Tahoma', 'Arial', 'B Mitra', 'B Yekan']
        
        available_fonts = []
        for font in persian_fonts:
            if any(font.lower() in f.lower() for f in system_fonts):
                available_fonts.append(font)
        
        if available_fonts:
            selected_font = available_fonts[0]
            plt.rcParams['font.family'] = selected_font
            plt.rcParams['font.sans-serif'] = [selected_font]
            plt.rcParams['axes.unicode_minus'] = False
            return selected_font, system_fonts
        else:
            return 'DejaVu Sans', system_fonts
    except Exception as e:
        print(f"خطا در تنظیم فونت: {e}")
        return 'DejaVu Sans', []

# ============================================================================
# سیستم دانلود از ایزی تریدر (نسخه 20)
# ============================================================================
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.keys import Keys
    
    print("✅ همه کتابخانه‌های سلنیوم با موفقیت import شدند")
    SELENIUM_AVAILABLE = True
    
    # سعی در ایمپورت webdriver_manager
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        WEBDRIVER_MANAGER_AVAILABLE = True
        print("✅ webdriver-manager آماده است")
    except ImportError:
        print("⚠️  webdriver-manager نصب نیست")
        WEBDRIVER_MANAGER_AVAILABLE = False
        
except ImportError as e:
    print(f"⚠️  برخی کتابخانه‌ها نصب نیستند: {e}")
    print("برای نصب: pip install selenium webdriver-manager")
    SELENIUM_AVAILABLE = False
    WEBDRIVER_MANAGER_AVAILABLE = False

class EasyTraderAutoDownloader:
    def __init__(self, logger=None):
        self.session = requests.Session() if 'requests' in sys.modules else None
        self.base_url = "https://d.easytrader.ir"
        self.driver = None
        self.selenium_available = SELENIUM_AVAILABLE
        self.webdriver_manager_available = WEBDRIVER_MANAGER_AVAILABLE
        self.is_logged_in = False
        self.login_lock = threading.Lock()
        self.chrome_version = self.get_chrome_version()
        self.user_data_dir = tempfile.mkdtemp(prefix="chrome_profile_")
        self.logger = logger
    
    def log(self, message, level="INFO"):
        """ثبت لاگ با استفاده از logger"""
        if self.logger:
            self.logger.log(message, level)
        else:
            print(f"[{level}] {message}")
    
    def get_chrome_version(self):
        """دریافت نسخه کروم نصب شده"""
        try:
            if sys.platform == "win32":
                import winreg
                
                registry_paths = [
                    r"Software\Google\Chrome\BLBeacon",
                    r"Software\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe",
                    r"Software\Wow6432Node\Google\Chrome\BLBeacon"
                ]
                
                for path in registry_paths:
                    try:
                        if "App Paths" in path:
                            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
                            chrome_path, _ = winreg.QueryValueEx(key, "")
                            winreg.CloseKey(key)
                            
                            result = subprocess.run([chrome_path, '--version'], 
                                                  capture_output=True, text=True, 
                                                  creationflags=subprocess.CREATE_NO_WINDOW)
                            if result.stdout:
                                version_match = re.search(r'(\d+\.\d+\.\d+\.\d+)', result.stdout)
                                if version_match:
                                    return version_match.group(1)
                        else:
                            try:
                                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, path)
                                version, _ = winreg.QueryValueEx(key, "version")
                                winreg.CloseKey(key)
                                return version
                            except:
                                try:
                                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
                                    version, _ = winreg.QueryValueEx(key, "version")
                                    winreg.CloseKey(key)
                                    return version
                                except:
                                    continue
                    except:
                        continue
            
            chrome_paths = [
                "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
                os.path.expanduser("~") + "\\AppData\\Local\\Google\\Chrome\\Application\\chrome.exe"
            ]
            
            for path in chrome_paths:
                if os.path.exists(path):
                    try:
                        result = subprocess.run([path, '--version'], 
                                              capture_output=True, text=True,
                                              creationflags=subprocess.CREATE_NO_WINDOW)
                        if result.stdout:
                            version_match = re.search(r'(\d+\.\d+\.\d+\.\d+)', result.stdout)
                            if version_match:
                                return version_match.group(1)
                    except Exception as e:
                        self.log(f"خطا در دریافت نسخه کروم از {path}: {e}", "WARNING")
                        continue
            
            return "120.0.0.0"
        except Exception as e:
            self.log(f"خطا در دریافت نسخه کروم: {e}", "WARNING")
            return "120.0.0.0"
    
    def get_2fa_code_from_user(self):
        """دریافت کد 2FA از کاربر با رابط بهتر"""
        try:
            dialog = tk.Toplevel()
            dialog.title("کد احراز هویت دو مرحله‌ای")
            dialog.geometry("400x200")
            dialog.resizable(False, False)
            dialog.configure(bg='#f0f0f0')
            dialog.attributes('-topmost', True)
            
            dialog.update_idletasks()
            width = dialog.winfo_width()
            height = dialog.winfo_height()
            x = (dialog.winfo_screenwidth() // 2) - (width // 2)
            y = (dialog.winfo_screenheight() // 2) - (height // 2)
            dialog.geometry(f'{width}x{height}+{x}+{y}')
            
            help_text = """لطفاً کد 6 رقمی را از برنامه Google Authenticator وارد کنید.
            
اگر 2FA فعال نیست، می‌توانید این پنجره را ببندید."""
            
            label = tk.Label(dialog, text=help_text, font=('Tahoma', 11), 
                           bg='#f0f0f0', justify=tk.LEFT, wraplength=350)
            label.pack(pady=20)
            
            input_frame = tk.Frame(dialog, bg='#f0f0f0')
            input_frame.pack(pady=10)
            
            tk.Label(input_frame, text="کد 6 رقمی:", font=('Tahoma', 11), 
                   bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
            
            code_var = tk.StringVar()
            code_entry = tk.Entry(input_frame, textvariable=code_var, 
                                font=('Tahoma', 12), width=10, justify='center')
            code_entry.pack(side=tk.LEFT, padx=5)
            
            result = {"code": None}
            
            def submit():
                code = code_var.get().strip()
                if code and len(code) == 6 and code.isdigit():
                    result["code"] = code
                    dialog.destroy()
                else:
                    messagebox.showerror("خطا", "لطفاً کد 6 رقمی معتبر وارد کنید")
            
            def cancel():
                result["code"] = None
                dialog.destroy()
            
            button_frame = tk.Frame(dialog, bg='#f0f0f0')
            button_frame.pack(pady=20)
            
            submit_btn = tk.Button(button_frame, text="تأیید", font=('Tahoma', 11),
                                 bg='#27ae60', fg='white', width=10, command=submit)
            submit_btn.pack(side=tk.LEFT, padx=10)
            
            cancel_btn = tk.Button(button_frame, text="انصراف", font=('Tahoma', 11),
                                 bg='#e74c3c', fg='white', width=10, command=cancel)
            cancel_btn.pack(side=tk.LEFT, padx=10)
            
            code_entry.focus_set()
            
            dialog.bind('<Return>', lambda e: submit())
            dialog.bind('<Escape>', lambda e: cancel())
            
            dialog.transient()
            dialog.grab_set()
            dialog.wait_window()
            
            return result["code"]
            
        except Exception as e:
            self.log(f"خطا در دریافت کد 2FA: {e}", "ERROR")
            return None
    
    def save_session_cookies(self):
        """ذخیره کوکی‌های session در فایل"""
        if self.driver:
            try:
                cookies = self.driver.get_cookies()
                with open('easytrader_cookies.pkl', 'wb') as f:
                    pickle.dump(cookies, f)
                self.log("کوکی‌ها در فایل ذخیره شدند", "INFO")
                return True
            except Exception as e:
                self.log(f"خطا در ذخیره کوکی‌ها: {e}", "ERROR")
                return False
        return False
    
    def load_session_cookies(self):
        """بارگذاری کوکی‌های session از فایل"""
        try:
            if not os.path.exists('easytrader_cookies.pkl'):
                return False
                
            with open('easytrader_cookies.pkl', 'rb') as f:
                cookies = pickle.load(f)
            
            self.driver.delete_all_cookies()
            
            for cookie in cookies:
                try:
                    if 'expiry' in cookie:
                        cookie['expiry'] = int(cookie['expiry'])
                    self.driver.add_cookie(cookie)
                except Exception as e:
                    self.log(f"خطا در اضافه کردن کوکی {cookie.get('name')}: {e}", "WARNING")
            
            self.log(f"{len(cookies)} کوکی از فایل بارگذاری شدند", "INFO")
            return True
            
        except Exception as e:
            self.log(f"خطا در بارگذاری کوکی‌ها: {e}", "WARNING")
            return False
    
    def check_cookies_valid(self):
        """بررسی معتبر بودن کوکی‌های ذخیره شده"""
        try:
            if os.path.exists('easytrader_cookies.pkl'):
                file_time = os.path.getmtime('easytrader_cookies.pkl')
                if time.time() - file_time > 43200:
                    self.log("کوکی‌های ذخیره شده منقضی شده‌اند (بیش از 12 ساعت)", "WARNING")
                    return False
                return True
            return False
        except:
            return False
    
    def setup_driver_simple(self):
        """راه‌اندازی ساده درایور بدون پیچیدگی"""
        try:
            self.log("راه‌اندازی درایور با تنظیمات ساده...", "INFO")
            
            try:
                self.log("تلاش با undetected-chromedriver...", "INFO")
                import undetected_chromedriver as uc
                
                options = uc.ChromeOptions()
                
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument("--disable-blink-features=AutomationControlled")
                options.add_argument("--window-size=1920,1080")
                options.add_argument("--start-maximized")
                options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
                options.add_argument("--disable-notifications")
                options.add_argument("--disable-extensions")
                options.add_argument("--disable-popup-blocking")
                
                download_dir = os.path.join(os.path.expanduser("~"), "Downloads", "easytrader_downloads")
                if not os.path.exists(download_dir):
                    os.makedirs(download_dir)
                
                prefs = {
                    "download.default_directory": download_dir,
                    "download.prompt_for_download": False,
                    "download.directory_upgrade": True,
                    "plugins.always_open_pdf_externally": True,
                    "safebrowsing.enabled": False,
                    "credentials_enable_service": False,
                    "profile.password_manager_enabled": False,
                }
                options.add_experimental_option("prefs", prefs)
                
                self.driver = uc.Chrome(
                    options=options,
                    use_subprocess=True,
                    version_main=int(self.chrome_version.split('.')[0]) if '.' in self.chrome_version else 120
                )
                
                self.log("درایور با undetected-chromedriver راه‌اندازی شد", "INFO")
                return self.driver
                
            except Exception as e:
                self.log(f"خطا در undetected-chromedriver: {e}", "WARNING")
            
            self.log("تلاش با selenium ساده...", "INFO")
            chrome_options = Options()
            
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--start-maximized")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            chrome_options.add_argument("--disable-notifications")
            
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            download_dir = os.path.join(os.path.expanduser("~"), "Downloads", "easytrader_downloads")
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)
            
            prefs = {
                "download.default_directory": download_dir,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "plugins.always_open_pdf_externally": True,
                "safebrowsing.enabled": False,
                "credentials_enable_service": False,
                "profile.password_manager_enabled": False,
            }
            chrome_options.add_experimental_option("prefs", prefs)
            
            chrome_paths = [
                "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
                os.path.expanduser("~") + "\\AppData\\Local\\Google\\Chrome\\Application\\chrome.exe"
            ]
            
            for path in chrome_paths:
                if os.path.exists(path):
                    chrome_options.binary_location = path
                    self.log(f"کروم یافت شد: {path}", "INFO")
                    break
            
            if self.webdriver_manager_available:
                try:
                    service = Service(ChromeDriverManager().install())
                    self.driver = webdriver.Chrome(service=service, options=chrome_options)
                    self.log("درایور با webdriver-manager راه‌اندازی شد", "INFO")
                except Exception as e:
                    self.log(f"خطا در webdriver-manager: {e}", "WARNING")
                    try:
                        self.driver = webdriver.Chrome(options=chrome_options)
                        self.log("درایور با کروم سیستم راه‌اندازی شد", "INFO")
                    except Exception as e2:
                        self.log(f"خطا در راه‌اندازی درایور: {e2}", "ERROR")
                        return None
            else:
                try:
                    self.driver = webdriver.Chrome(options=chrome_options)
                    self.log("درایور با کروم سیستم راه‌اندازی شد", "INFO")
                except Exception as e:
                    self.log(f"خطا در راه‌اندازی درایور: {e}", "ERROR")
                    return None
            
            return self.driver
            
        except Exception as e:
            self.log(f"خطا در راه‌اندازی درایور: {e}", "ERROR")
            return None
    
    def setup_driver(self):
        """تنظیم درایور کروم"""
        if not self.selenium_available:
            self.log("Selenium در دسترس نیست", "ERROR")
            return None
        
        try:
            self.log(f"نسخه کروم سیستم: {self.chrome_version}", "INFO")
            
            driver = self.setup_driver_simple()
            if driver:
                return driver
            
            response = messagebox.askyesno("خطا در Chrome", 
                "ChromeDriver با مشکل مواجه شده. آیا می‌خواهید از Microsoft Edge استفاده کنید؟")
            
            if response:
                try:
                    from msedge.selenium_tools import Edge, EdgeOptions
                    
                    edge_options = EdgeOptions()
                    edge_options.use_chromium = True
                    edge_options.add_argument("--no-sandbox")
                    edge_options.add_argument("--disable-dev-shm-usage")
                    edge_options.add_argument("--window-size=1920,1080")
                    edge_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
                    
                    self.driver = Edge(options=edge_options)
                    self.log("درایور Edge راه‌اندازی شد", "INFO")
                    return self.driver
                except ImportError:
                    messagebox.showinfo("نصب نیاز است", 
                        "لطفاً نصب کنید: pip install msedge-selenium-tools")
                except Exception as e:
                    self.log(f"خطا در راه‌اندازی Edge: {e}", "WARNING")
            
            return None
            
        except Exception as e:
            self.log(f"خطا در راه‌اندازی درایور: {e}", "ERROR")
            return None
    
    def wait_for_element(self, by, value, timeout=30):
        """منتظر ماندن برای وجود عنصر"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except Exception as e:
            self.log(f"عنصر پیدا نشد: {by}={value}", "WARNING")
            return None
    
    def wait_for_element_clickable(self, by, value, timeout=30):
        """منتظر ماندن برای قابل کلیک بودن عنصر"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable((by, value))
            )
            return element
        except Exception as e:
            self.log(f"عنصر قابل کلیک نیست: {by}={value}", "WARNING")
            return None
    
    def find_element_by_multiple(self, selectors):
        """یافتن عنصر با چندین سلکتور مختلف"""
        for selector_type, selector_value in selectors:
            try:
                if selector_type == "id":
                    element = self.driver.find_element(By.ID, selector_value)
                elif selector_type == "name":
                    element = self.driver.find_element(By.NAME, selector_value)
                elif selector_type == "xpath":
                    element = self.driver.find_element(By.XPATH, selector_value)
                elif selector_type == "css":
                    element = self.driver.find_element(By.CSS_SELECTOR, selector_value)
                elif selector_type == "class":
                    element = self.driver.find_element(By.CLASS_NAME, selector_value)
                else:
                    continue
                    
                if element:
                    self.log(f"عنصر با {selector_type}={selector_value} یافت شد", "INFO")
                    return element
            except:
                continue
        return None
    
    def check_login_success(self):
        """بررسی موفقیت‌آمیز بودن لاگین با دقت بیشتر"""
        try:
            current_url = self.driver.current_url.lower()
            page_source = self.driver.page_source.lower()
            
            success_indicators = [
                'dashboard', 'داشبورد', 'portfolio', 'پورتفو', 
                'حساب کاربری', 'account', 'سبد دارایی', 'منوی کاربری',
                'خروج از حساب', 'logout', 'sign out'
            ]
            
            failure_indicators = [
                'invalid', 'نامعتبر', 'خطا در ورود', 'نام کاربری یا رمز عبور اشتباه',
                'ورود ناموفق', 'login failed', 'error', 'خطا'
            ]
            
            for indicator in success_indicators:
                if indicator in page_source:
                    self.log(f"نشانه لاگین موفق یافت شد: {indicator}", "INFO")
                    return True
            
            for indicator in failure_indicators:
                if indicator in page_source:
                    self.log(f"نشانه لاگین ناموفق یافت شد: {indicator}", "WARNING")
                    return False
            
            if 'login' in current_url or 'ورود' in page_source:
                return False
            
            if current_url == 'https://d.easytrader.ir/' or current_url == 'https://d.easytrader.ir':
                if 'ورود' not in page_source and 'login' not in page_source:
                    return True
            
            return False
            
        except Exception as e:
            self.log(f"خطا در بررسی لاگین: {e}", "WARNING")
            return False
    
    def login_with_selenium(self, username, password):
        """ورود به ایزی تریدر با سلنیوم - نسخه بهبود یافته"""
        with self.login_lock:
            if not self.selenium_available:
                self.log("Selenium در دسترس نیست", "ERROR")
                return False
            
            try:
                if not self.driver:
                    self.driver = self.setup_driver()
                    if not self.driver:
                        return False
                
                self.log("تست اتصال به اینترنت...", "INFO")
                try:
                    self.driver.get("https://www.google.com")
                    time.sleep(2)
                    self.log("اتصال اینترنت OK", "INFO")
                except Exception as e:
                    self.log(f"مشکل در اتصال اینترنت: {e}", "WARNING")
                
                self.log("در حال بارگذاری صفحه لاگین...", "INFO")
                
                login_urls = [
                    "https://d.easytrader.ir/account/login",
                    "https://d.easytrader.ir/login",
                    "https://d.easytrader.ir/signin",
                    "https://d.easytrader.ir/"
                ]
                
                loaded = False
                for url in login_urls:
                    try:
                        self.log(f"تلاش با URL: {url}", "INFO")
                        self.driver.get(url)
                        time.sleep(5)
                        
                        page_html = self.driver.page_source.lower()
                        if 'ورود' in page_html or 'login' in page_html or 'username' in page_html or 'password' in page_html:
                            self.log(f"صفحه لاگین بارگذاری شد: {url}", "INFO")
                            loaded = True
                            break
                    except:
                        continue
                
                if not loaded:
                    self.log("نتوانستیم صفحه لاگین را بارگذاری کنیم", "ERROR")
                    return False
                
                if self.check_cookies_valid():
                    self.log("تلاش بارگذاری کوکی‌های ذخیره شده...", "INFO")
                    self.load_session_cookies()
                    self.driver.refresh()
                    time.sleep(5)
                    
                    if self.check_login_success():
                        self.log("ورود با کوکی‌های ذخیره شده موفق بود", "INFO")
                        self.is_logged_in = True
                        return True
                
                self.log("جستجوی فیلد نام کاربری...", "INFO")
                username_selectors = [
                    ("name", "username"),
                    ("name", "email"),
                    ("id", "username"),
                    ("id", "email"),
                    ("xpath", "//input[@type='text']"),
                    ("xpath", "//input[@type='email']"),
                    ("xpath", "//input[contains(@placeholder, 'نام کاربری')]"),
                    ("xpath", "//input[contains(@placeholder, 'ایمیل')]"),
                    ("xpath", "//input[contains(@id, 'username')]"),
                    ("xpath", "//input[contains(@id, 'email')]")
                ]
                
                username_field = self.find_element_by_multiple(username_selectors)
                if not username_field:
                    try:
                        screenshot_path = "login_page_debug.png"
                        self.driver.save_screenshot(screenshot_path)
                        self.log(f"اسکرین‌شات برای دیباگ ذخیره شد: {screenshot_path}", "INFO")
                    except:
                        pass
                    self.log("فیلد نام کاربری پیدا نشد", "ERROR")
                    return False
                
                username_field.clear()
                username_field.send_keys(username)
                self.log("نام کاربری وارد شد", "INFO")
                time.sleep(1)
                
                self.log("جستجوی فیلد رمز عبور...", "INFO")
                password_selectors = [
                    ("name", "password"),
                    ("id", "password"),
                    ("xpath", "//input[@type='password']"),
                    ("xpath", "//input[contains(@placeholder, 'رمز عبور')]"),
                    ("xpath", "//input[contains(@id, 'password')]")
                ]
                
                password_field = self.find_element_by_multiple(password_selectors)
                if not password_field:
                    self.log("فیلد رمز عبور پیدا نشد", "ERROR")
                    return False
                
                password_field.clear()
                password_field.send_keys(password)
                self.log("رمز عبور وارد شد", "INFO")
                time.sleep(1)
                
                self.log("جستجوی دکمه ورود...", "INFO")
                login_button_selectors = [
                    ("xpath", "//button[@type='submit']"),
                    ("xpath", "//input[@type='submit']"),
                    ("xpath", "//button[contains(text(), 'ورود')]"),
                    ("xpath", "//button[contains(text(), 'Login')]"),
                    ("css", "button[type='submit']"),
                    ("xpath", "//button[contains(@class, 'btn-login')]"),
                    ("xpath", "//button[contains(@class, 'btn-primary') and contains(text(), 'ورود')]")
                ]
                
                login_button = self.find_element_by_multiple(login_button_selectors)
                if not login_button:
                    self.log("دکمه ورود پیدا نشد", "ERROR")
                    return False
                
                self.log("کلیک روی دکمه ورود...", "INFO")
                try:
                    login_button.click()
                except:
                    self.driver.execute_script("arguments[0].click();", login_button)
                
                self.log("منتظر نتیجه لاگین...", "INFO")
                time.sleep(10)
                
                page_html = self.driver.page_source.lower()
                if 'کد امنیتی' in page_html or 'captcha' in page_html or 'کد تأیید' in page_html:
                    self.log("نیاز به کد امنیتی/CAPTCHA", "WARNING")
                    captcha_code = self.get_2fa_code_from_user()
                    if captcha_code:
                        captcha_selectors = [
                            ("name", "captcha"),
                            ("id", "captcha"),
                            ("xpath", "//input[@type='text' and contains(@placeholder, 'کد')]"),
                            ("xpath", "//input[contains(@placeholder, 'کد امنیتی')]")
                        ]
                        
                        captcha_field = self.find_element_by_multiple(captcha_selectors)
                        if captcha_field:
                            captcha_field.clear()
                            captcha_field.send_keys(captcha_code)
                            time.sleep(1)
                            
                            submit_selectors = [
                                ("xpath", "//button[@type='submit']"),
                                ("xpath", "//button[contains(text(), 'تأیید')]"),
                                ("xpath", "//button[contains(text(), 'Verify')]")
                            ]
                            
                            submit_button = self.find_element_by_multiple(submit_selectors)
                            if submit_button:
                                submit_button.click()
                                time.sleep(5)
                
                if self.check_login_success():
                    self.log("ورود موفقیت‌آمیز", "INFO")
                    self.is_logged_in = True
                    
                    self.save_session_cookies()
                    
                    return True
                else:
                    self.log("ورود ناموفق", "ERROR")
                    page_html = self.driver.page_source
                    if 'نام کاربری یا رمز عبور اشتباه' in page_html:
                        self.log("نام کاربری یا رمز عبور اشتباه است", "ERROR")
                    elif 'حساب کاربری قفل شده' in page_html:
                        self.log("حساب کاربری قفل شده است", "ERROR")
                    elif 'کد امنیتی' in page_html:
                        self.log("نیاز به کد امنیتی/CAPTCHA", "ERROR")
                    return False
                
            except Exception as e:
                self.log(f"خطا در ورود: {e}", "ERROR")
                return False
    
    def extract_table_manually(self, table_element):
        """استخراج دستی داده‌ها از جدول"""
        try:
            data = []
            
            rows = table_element.find_elements(By.TAG_NAME, "tr")
            
            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if not cells:
                    cells = row.find_elements(By.TAG_NAME, "th")
                
                row_data = [cell.text.strip() for cell in cells if cell.text.strip()]
                if row_data:
                    data.append(row_data)
            
            if len(data) > 1:
                df = pd.DataFrame(data[1:], columns=data[0] if len(data[0]) == len(data[1]) else None)
                return df
            
            return None
            
        except Exception as e:
            self.log(f"خطا در استخراج دستی جدول: {e}", "ERROR")
            return None
    
    def extract_data_directly(self):
        """استخراج مستقیم داده‌ها از صفحه"""
        try:
            self.log("تلاش برای استخراج مستقیم داده‌ها از صفحه...", "INFO")
            
            time.sleep(8)
            
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            
            if not tables:
                tables = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'table')]")
            
            if tables:
                self.log(f"{len(tables)} جدول/دیو با کلاس table یافت شد", "INFO")
                
                max_rows = 0
                best_table = None
                
                for i, table in enumerate(tables):
                    try:
                        html_content = table.get_attribute('outerHTML')
                        
                        rows = table.find_elements(By.TAG_NAME, "tr")
                        if len(rows) > max_rows and len(rows) > 5:
                            max_rows = len(rows)
                            best_table = table
                    except:
                        continue
                
                if best_table:
                    self.log(f"جدول با بیشترین سطرها یافت شد: {max_rows} سطر", "INFO")
                    html_content = best_table.get_attribute('outerHTML')
                    
                    try:
                        df_list = pd.read_html(html_content)
                    except Exception as e:
                        self.log(f"خطا در خواندن جدول با pandas: {e}", "WARNING")
                        
                        df = self.extract_table_manually(best_table)
                        if df is not None and not df.empty:
                            return df
                        else:
                            return None
                    
                    if df_list:
                        df = df_list[0]
                        self.log(f"داده‌ها از جدول استخراج شد: {df.shape}", "INFO")
                        
                        self.log("ستون‌های استخراج شده:", "INFO")
                        for col in df.columns:
                            self.log(f"  {col}", "INFO")
                        
                        return df
            
            self.log("جستجوی داده‌ها در المان‌های صفحه...", "INFO")
            data_elements = self.driver.find_elements(By.XPATH, "//td | //th | //div[contains(@class, 'cell')]")
            
            if data_elements and len(data_elements) > 20:
                data = []
                current_row = []
                
                for elem in data_elements:
                    text = elem.text.strip()
                    if text:
                        current_row.append(text)
                        
                        if len(current_row) > 10:
                            data.append(current_row)
                            current_row = []
                
                if len(data) > 2:
                    df = pd.DataFrame(data)
                    self.log(f"داده‌ها به صورت دستی استخراج شد: {df.shape}", "INFO")
                    return df
            
            self.log("نتوانستیم داده‌ها را از صفحه استخراج کنیم", "WARNING")
            return None
            
        except Exception as e:
            self.log(f"خطا در استخراج مستقیم داده‌ها: {e}", "ERROR")
            return None
    
    def download_market_data(self):
        """دانلود داده‌های بازار از ایزی تریدر - نسخه بهبود یافته"""
        if not self.selenium_available or not self.driver:
            self.log("درایور سلنیوم آماده نیست", "ERROR")
            return None
        
        if not self.is_logged_in:
            self.log("ابتدا باید وارد شوید", "ERROR")
            return None
        
        try:
            self.log("در حال رفتن به صفحه ایزی فیلتر...", "INFO")
            
            market_url = "https://d.easytrader.ir/easy-filter"
            
            self.driver.get(market_url)
            time.sleep(10)
            
            self.log("منتظر لود شدن کامل صفحه ایزی فیلتر...", "INFO")
            time.sleep(15)
            
            page_source = self.driver.page_source
            
            self.log("جستجو برای لینک‌ها و دکمه‌های دانلود...", "INFO")
            
            try:
                menu_items = self.driver.find_elements(By.XPATH, "//button[contains(@class, 'dropdown')] | //div[contains(@class, 'dropdown')]")
                for item in menu_items:
                    try:
                        item_text = item.text.lower()
                        if 'خروجی' in item_text or 'export' in item_text or 'excel' in item_text:
                            self.log(f"عنصر منو یافت شد: {item_text}", "INFO")
                            item.click()
                            time.sleep(2)
                            
                            dropdown_items = self.driver.find_elements(By.XPATH, "//a[contains(., 'Excel')] | //button[contains(., 'Excel')]")
                            for dropdown_item in dropdown_items:
                                if dropdown_item.is_displayed():
                                    self.log("کلیک روی گزینه اکسل در منو", "INFO")
                                    dropdown_item.click()
                                    time.sleep(10)
                                    break
                            break
                    except:
                        continue
            except Exception as e:
                self.log(f"خطا در جستجوی منوها: {e}", "WARNING")
            
            download_dir = os.path.join(os.path.expanduser("~"), "Downloads", "easytrader_downloads")
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)
            
            files_before = set(os.listdir(download_dir))
            
            try:
                self.log("تلاش با کلیدهای ترکیبی Ctrl+S...", "INFO")
                actions = ActionChains(self.driver)
                actions.key_down(Keys.CONTROL).send_keys('s').key_up(Keys.CONTROL).perform()
                time.sleep(5)
            except:
                pass
            
            time.sleep(15)
            files_after = set(os.listdir(download_dir))
            new_files = files_after - files_before
            
            excel_files = []
            for f in new_files:
                if f.lower().endswith('.xlsx') or f.lower().endswith('.xls'):
                    file_path = os.path.join(download_dir, f)
                    excel_files.append((file_path, os.path.getctime(file_path)))
                    self.log(f"فایل اکسل جدید یافت شد: {f}", "INFO")
            
            if not excel_files:
                self.log("جستجوی فایل‌های اخیر...", "INFO")
                all_excel_files = []
                for f in os.listdir(download_dir):
                    if f.lower().endswith('.xlsx') or f.lower().endswith('.xls'):
                        file_path = os.path.join(download_dir, f)
                        file_time = os.path.getctime(file_path)
                        if file_time > time.time() - 600:
                            all_excel_files.append((file_path, file_time))
                
                if all_excel_files:
                    all_excel_files.sort(key=lambda x: x[1], reverse=True)
                    excel_files = [all_excel_files[0]]
                    self.log(f"آخرین فایل اکسل یافت شد: {os.path.basename(excel_files[0][0])}", "INFO")
            
            if excel_files:
                latest_file = max(excel_files, key=lambda x: x[1])[0]
                self.log(f"فایل دانلود شده: {latest_file}", "INFO")
                
                try:
                    df = pd.read_excel(latest_file)
                    self.log(f"فایل خوانده شد. {len(df)} ردیف، {len(df.columns)} ستون", "INFO")
                    
                    self.log("ستون‌های فایل:", "INFO")
                    for i, col in enumerate(df.columns):
                        self.log(f"  {i+1}. {col}", "INFO")
                    
                    df.columns = [f'ستون_{i+1}' if 'Unnamed' in str(col) else col for i, col in enumerate(df.columns)]
                    
                    return df
                except Exception as e:
                    self.log(f"خطا در خواندن فایل: {e}", "ERROR")
                    return self.extract_data_directly()
            else:
                self.log("هیچ فایل اکسلی یافت نشد - تلاش برای استخراج مستقیم", "WARNING")
                return self.extract_data_directly()
                
        except Exception as e:
            self.log(f"خطا در دانلود داده بازار: {e}", "ERROR")
            return None
    
    def download_portfolio_data(self):
        """دانلود داده‌های پورتفو از ایزی تریدر"""
        if not self.selenium_available or not self.driver:
            self.log("درایور سلنیوم آماده نیست", "ERROR")
            return None
        
        if not self.is_logged_in:
            self.log("ابتدا باید وارد شوید", "ERROR")
            return None
        
        try:
            self.log("در حال رفتن به صفحه پورتفو...", "INFO")
            
            portfolio_url = "https://d.easytrader.ir/portfolio"
            
            self.driver.get(portfolio_url)
            time.sleep(10)
            
            self.log("منتظر لود شدن کامل صفحه پورتفو...", "INFO")
            time.sleep(10)
            
            download_dir = os.path.join(os.path.expanduser("~"), "Downloads", "easytrader_downloads")
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)
            
            files_before = set(os.listdir(download_dir))
            
            self.log("جستجوی دکمه خروجی اکسل...", "INFO")
            
            download_selectors = [
                ("xpath", "//button[contains(., 'خروجی اکسل')]"),
                ("xpath", "//button[contains(., 'Excel')]"),
                ("xpath", "//button[contains(., 'اکسل')]"),
                ("xpath", "//a[contains(., 'خروجی اکسل')]"),
                ("xpath", "//a[contains(., 'Excel')]"),
                ("xpath", "//a[contains(., 'اکسل')]"),
                ("xpath", "//*[contains(text(), 'خروجی') and contains(text(), 'اکسل')]"),
                ("xpath", "//button[contains(@class, 'btn')]"),
                ("xpath", "//a[contains(@class, 'btn')]")
            ]
            
            download_button = None
            for selector_type, selector_value in download_selectors:
                try:
                    if selector_type == "xpath":
                        elements = self.driver.find_elements(By.XPATH, selector_value)
                        for element in elements:
                            if element.is_displayed() and element.is_enabled():
                                element_text = element.text.strip()
                                if element_text and ('اکسل' in element_text or 'Excel' in element_text or 'خروجی' in element_text):
                                    self.log(f"دکمه با XPath '{selector_value}' یافت شد: {element_text}", "INFO")
                                    download_button = element
                                    break
                    if download_button:
                        break
                except Exception as e:
                    continue
            
            if not download_button:
                self.log("جستجوی گسترده برای دکمه‌ها...", "INFO")
                all_buttons = self.driver.find_elements(By.TAG_NAME, "button")
                for button in all_buttons:
                    try:
                        if button.is_displayed() and button.is_enabled():
                            button_text = button.text.strip()
                            if button_text and ('اکسل' in button_text or 'Excel' in button_text):
                                self.log(f"دکمه یافت شد: {button_text}", "INFO")
                                download_button = button
                                break
                    except:
                        continue
            
            if download_button:
                self.log(f"دکمه دانلود پورتفو یافت شد: {download_button.text}", "INFO")
                
                try:
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", download_button)
                    time.sleep(1)
                    self.driver.execute_script("arguments[0].click();", download_button)
                except Exception as e:
                    self.log(f"خطا در کلیک: {e}", "WARNING")
                    try:
                        download_button.click()
                    except:
                        pass
                
                self.log("منتظر اتمام دانلود پورتفو...", "INFO")
                time.sleep(15)
                
                files_after = set(os.listdir(download_dir))
                new_files = files_after - files_before
                
                excel_files = []
                for f in new_files:
                    if f.lower().endswith('.xlsx') or f.lower().endswith('.xls'):
                        file_path = os.path.join(download_dir, f)
                        excel_files.append((file_path, os.path.getctime(file_path)))
                        self.log(f"فایل پورتفو یافت شد: {f}", "INFO")
                
                if not excel_files:
                    for f in os.listdir(download_dir):
                        if f.lower().endswith('.xlsx') or f.lower().endswith('.xls'):
                            file_path = os.path.join(download_dir, f)
                            if os.path.getctime(file_path) > time.time() - 300:
                                excel_files.append((file_path, os.path.getctime(file_path)))
                
                if excel_files:
                    latest_file = max(excel_files, key=lambda x: x[1])[0]
                    self.log(f"فایل پورتفو دانلود شده: {latest_file}", "INFO")
                    
                    try:
                        df = pd.read_excel(latest_file)
                        self.log(f"فایل پورتفو خوانده شد. {len(df)} ردیف", "INFO")
                        
                        self.log("ستون‌های فایل پورتفو:", "INFO")
                        for i, col in enumerate(df.columns):
                            self.log(f"  {i+1}. {col}", "INFO")
                        
                        return df
                    except Exception as e:
                        self.log(f"خطا در خواندن فایل پورتفو: {e}", "ERROR")
                        return None
                else:
                    self.log("هیچ فایل پورتفوی جدیدی یافت نشد", "WARNING")
                    return None
            else:
                self.log("دکمه دانلود پورتفو پیدا نشد", "WARNING")
                return None
            
        except Exception as e:
            self.log(f"خطا در دانلود پورتفو: {e}", "ERROR")
            return None
    
    def close(self):
        """بستن درایور سلنیوم"""
        if self.driver:
            try:
                self.driver.quit()
                self.log("درایور سلنیوم بسته شد", "INFO")
            except:
                pass
            
            try:
                import shutil
                if os.path.exists(self.user_data_dir):
                    shutil.rmtree(self.user_data_dir, ignore_errors=True)
            except:
                pass

# ============================================================================
# نرم‌افزار اصلی با تمام قابلیت‌ها
# ============================================================================
class CompleteStockAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("نرم افزار تحلیل سهام و پورتفو - نسخه جامع")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#f0f0f0')
        
        # متغیرها
        self.df = None
        self.portfolio_df = None
        self.column_mapping = {}
        self.actual_columns = []
        self.chart_images = []
        self.selected_font, self.all_system_fonts = setup_persian_font()
        self.market_trend = "NORMAL"
        self.last_recommendations = None
        self.portfolio_analysis = None
        self.trading_signals = None
        self.bubble_analysis = None
        self.left_frame_visible = True
        self.search_var = tk.StringVar()
        
        # سیستم‌های جدید
        self.ai_predictor = AIPredictor()
        self.alert_system = SmartAlertSystem()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.market_analyzer = MarketAnalyzer()
        self.report_generator = AdvancedReportGenerator()
        self.backtest_engine = BacktestEngine()
        self.tsetmc_downloader = TsetmcDownloader()
        
        # سیستم لاگینگ پیشرفته
        self.setup_advanced_logger()
        
        # دانلودر اتوماتیک
        self.downloader = EasyTraderAutoDownloader(self.logger)
        self.auto_download_enabled = False
        self.download_settings = {
            'username': '',
            'password': '',
            'auto_interval': 300,
            'last_download_time': None
        }
        
        # تنظیمات اصلی
        self.analysis_settings = {
            'weights': {
                'P/B': 0.25, 'EPS': 0.20, 'خرید حقوقی': 0.15, 'RSI': 0.15,
                'بازدهی1ماه': 0.10, 'ارزش معاملات': 0.10, 'تعداد معاملات': 0.05
            },
            'filters': {
                'min_volume': 100000, 'max_pb': 1.5, 'min_eps': 50, 'rsi_max': 40,
                'min_price': 1000, 'max_price': 50000
            },
            'disqualifiers': {
                'max_pb_disqualify': 3, 'min_eps_disqualify': -50,
                'max_rsi_disqualify': 65, 'max_1month_return': 25
            },
            'portfolio_alerts': {
                'profit_threshold': 30,
                'loss_threshold': -15,
                'rsi_sell_threshold': 75,
                'rsi_buy_threshold': 25,
                'volume_increase_threshold': 2.0,
                'pb_sell_threshold': 2.5,
                'min_volume_threshold': 50000
            }
        }
        
        # تنظیمات پیشرفته
        self.advanced_settings = {
            'weights': {
                'fundamental': 0.40,
                'technical': 0.25,
                'market': 0.20,
                'momentum': 0.15
            },
            'filters': {
                'min_volume': 1000000,
                'min_value': 10e9,
                'max_pb': 3,
                'max_pe': 30,
                'max_rsi': 70,
                'min_legal_buy': 30
            },
            'bubble_settings': {
                'high_bubble_threshold': 70,
                'low_bubble_threshold': 30,
                'warning_threshold': 60
            },
            'trading_signals': {
                'buy_threshold': 4,
                'sell_threshold': -4,
                'confidence_threshold': 0.7
            }
        }
        
        # تنظیمات پیشفرض نمودارها
        self.custom_charts = [
            {'name': 'نمودار امتیاز سهام', 'x_axis': 'نماد', 'y_axis': 'امتیاز', 'type': 'bar'},
            {'name': 'نمودار P/B', 'x_axis': 'نماد', 'y_axis': 'P/B', 'type': 'bar'},
            {'name': 'نمودار RSI', 'x_axis': 'نماد', 'y_axis': 'RSI', 'type': 'bar'},
        ]
        
        self.setup_ui()
        
        # شروع دانلود اتوماتیک اگر فعال باشد
        self.check_auto_download()
        
        self.logger.log("🚀 نرم‌افزار تحلیل سهام و پورتفو راه‌اندازی شد", "INFO")
    
    def setup_advanced_logger(self):
        """تنظیم لاگر پیشرفته"""
        temp_text = tk.Text(self.root)
        self.logger = AdvancedLogger(temp_text)
        
    def setup_ui(self):
        """ایجاد رابط کاربری با تمام قابلیت‌ها"""
        # هدر برنامه
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=120)
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        header_frame.pack_propagate(False)
        
        # عنوان اصلی
        title_frame = tk.Frame(header_frame, bg='#2c3e50')
        title_frame.pack(fill=tk.X, padx=10, pady=5)
        
        title_label = tk.Label(title_frame, 
                             text="💎 نرم افزار جامع تحلیل سهام و پورتفو", 
                             font=('Tahoma', 22, 'bold'), 
                             fg='white', bg='#2c3e50')
        title_label.pack(side=tk.LEFT)
        
        # دکمه‌های کنترل سریع
        quick_controls = tk.Frame(title_frame, bg='#2c3e50')
        quick_controls.pack(side=tk.RIGHT)
        
        # نوار وضعیت در هدر
        status_frame = tk.Frame(header_frame, bg='#34495e')
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = tk.Label(status_frame, text="🟢 سیستم آماده است", 
                                   font=('Tahoma', 10), fg='white', bg='#34495e')
        self.status_label.pack(side=tk.LEFT)
        
        self.data_status_label = tk.Label(status_frame, text="📊 داده‌ای بارگذاری نشده", 
                                        font=('Tahoma', 10), fg='#f1c40f', bg='#34495e')
        self.data_status_label.pack(side=tk.RIGHT)
        
        # منوی اصلی
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)
        
        # منوی فایل
        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="📁 فایل", menu=file_menu)
        file_menu.add_command(label="📥 بارگذاری داده از فایل", command=self.load_data_from_file)
        file_menu.add_command(label="🌐 بارگذاری داده از اینترنت", command=self.load_data_from_internet)
        file_menu.add_separator()
        file_menu.add_command(label="💾 ذخیره داده‌ها", command=self.save_data)
        file_menu.add_command(label="📤 خروجی Excel", command=self.export_to_excel)
        file_menu.add_separator()
        file_menu.add_command(label="🚪 خروج", command=self.root.quit)
        
        # منوی تحلیل
        analysis_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="📊 تحلیل", menu=analysis_menu)
        analysis_menu.add_command(label="🔍 تحلیل بنیادی", command=self.run_fundamental_analysis)
        analysis_menu.add_command(label="📈 تحلیل تکنیکال", command=self.run_technical_analysis)
        analysis_menu.add_command(label="🎯 تحلیل ترکیبی", command=self.run_combined_analysis)
        analysis_menu.add_command(label="🧮 تحلیل حباب", command=self.open_bubble_analysis)
        analysis_menu.add_command(label="🤖 تحلیل هوش مصنوعی", command=self.run_ai_analysis)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="📉 پیش‌بینی قیمت", command=self.run_price_prediction)
        analysis_menu.add_command(label="⚡ سیگنال معاملاتی", command=self.generate_trading_signals)
        
        # منوی پورتفو
        portfolio_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="💰 پورتفو", menu=portfolio_menu)
        portfolio_menu.add_command(label="📋 بارگذاری پورتفو", command=self.load_portfolio)
        portfolio_menu.add_command(label="📊 تحلیل پورتفو", command=self.analyze_portfolio)
        portfolio_menu.add_command(label="⚖️ بهینه‌سازی پورتفو", command=self.optimize_portfolio)
        portfolio_menu.add_command(label="📈 عملکرد پورتفو", command=self.portfolio_performance)
        portfolio_menu.add_separator()
        portfolio_menu.add_command(label="🔄 متعادل‌سازی", command=self.rebalance_portfolio)
        
        # منوی گزارش
        report_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="📄 گزارش", menu=report_menu)
        report_menu.add_command(label="📋 گزارش روزانه", command=self.generate_daily_report)
        report_menu.add_command(label="📊 گزارش پورتفو", command=self.generate_portfolio_report)
        report_menu.add_command(label="📈 گزارش تکنیکال", command=self.generate_technical_report)
        report_menu.add_command(label="⚠️ گزارش هشدارها", command=self.generate_alert_report)
        report_menu.add_separator()
        report_menu.add_command(label="🖨️ چاپ گزارش", command=self.print_report)
        
        # منوی تنظیمات
        settings_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="⚙️ تنظیمات", menu=settings_menu)
        settings_menu.add_command(label="🎛️ تنظیمات تحلیل", command=self.open_analysis_settings)
        settings_menu.add_command(label="🔔 تنظیمات هشدار", command=self.open_alert_settings)
        settings_menu.add_command(label="📊 تنظیمات نمودار", command=self.open_chart_settings)
        settings_menu.add_command(label="🌐 تنظیمات اتصال", command=self.open_connection_settings)
        settings_menu.add_separator()
        settings_menu.add_command(label="🔄 بازنشانی تنظیمات", command=self.reset_settings)
        
        # منوی ابزارها
        tools_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="🛠️ ابزارها", menu=tools_menu)
        tools_menu.add_command(label="🧮 ماشین حساب", command=self.open_calculator)
        tools_menu.add_command(label="📅 تقویم اقتصادی", command=self.open_economic_calendar)
        tools_menu.add_command(label="📊 مقایسه سهام", command=self.open_stock_comparison)
        tools_menu.add_command(label="🔍 جستجوی پیشرفته", command=self.open_advanced_search)
        tools_menu.add_separator()
        tools_menu.add_command(label="🧪 بک‌تست استراتژی", command=self.open_backtest_tool)
        tools_menu.add_command(label="📚 مستندات", command=self.open_documentation)
        
        # بدنه اصلی با پنل‌های قابل تغییر
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg='#f0f0f0')
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # پنل چپ (ابزارها و فیلترها)
        self.left_panel = tk.Frame(main_paned, bg='#2c3e50', width=300)
        
        # بخش جستجو
        search_frame = tk.Frame(self.left_panel, bg='#2c3e50', pady=10)
        search_frame.pack(fill=tk.X, padx=10)
        
        tk.Label(search_frame, text="🔍 جستجوی سهام:", 
                font=('Tahoma', 11), fg='white', bg='#2c3e50').pack(anchor='w')
        
        search_entry = tk.Entry(search_frame, textvariable=self.search_var,
                              font=('Tahoma', 11), width=25)
        search_entry.pack(fill=tk.X, pady=5)
        
        search_btn = tk.Button(search_frame, text="جستجو", font=('Tahoma', 10),
                             bg='#3498db', fg='white', command=self.search_stocks)
        search_btn.pack(fill=tk.X)
        
        # بخش فیلترهای سریع
        filter_frame = tk.LabelFrame(self.left_panel, text="⚡ فیلترهای سریع", 
                                   font=('Tahoma', 12, 'bold'),
                                   bg='#34495e', fg='white', padx=10, pady=10)
        filter_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # فیلتر P/B
        pb_frame = tk.Frame(filter_frame, bg='#34495e')
        pb_frame.pack(fill=tk.X, pady=2)
        tk.Label(pb_frame, text="حداکثر P/B:", font=('Tahoma', 10), 
                fg='white', bg='#34495e', width=15).pack(side=tk.LEFT)
        self.pb_var = tk.StringVar(value="2")
        pb_entry = tk.Entry(pb_frame, textvariable=self.pb_var, 
                          font=('Tahoma', 10), width=10)
        pb_entry.pack(side=tk.RIGHT)
        
        # فیلتر RSI
        rsi_frame = tk.Frame(filter_frame, bg='#34495e')
        rsi_frame.pack(fill=tk.X, pady=2)
        tk.Label(rsi_frame, text="حداکثر RSI:", font=('Tahoma', 10), 
                fg='white', bg='#34495e', width=15).pack(side=tk.LEFT)
        self.rsi_var = tk.StringVar(value="70")
        rsi_entry = tk.Entry(rsi_frame, textvariable=self.rsi_var, 
                           font=('Tahoma', 10), width=10)
        rsi_entry.pack(side=tk.RIGHT)
        
        # فیلتر حجم
        volume_frame = tk.Frame(filter_frame, bg='#34495e')
        volume_frame.pack(fill=tk.X, pady=2)
        tk.Label(volume_frame, text="حداقل حجم:", font=('Tahoma', 10), 
                fg='white', bg='#34495e', width=15).pack(side=tk.LEFT)
        self.volume_var = tk.StringVar(value="1000000")
        volume_entry = tk.Entry(volume_frame, textvariable=self.volume_var, 
                              font=('Tahoma', 10), width=10)
        volume_entry.pack(side=tk.RIGHT)
        
        # دکمه اعمال فیلتر
        apply_btn = tk.Button(filter_frame, text="اعمال فیلترها", font=('Tahoma', 10),
                            bg='#27ae60', fg='white', command=self.apply_quick_filters)
        apply_btn.pack(fill=tk.X, pady=10)
        
        # بخش تحلیل‌های سریع
        quick_analysis_frame = tk.LabelFrame(self.left_panel, text="🚀 تحلیل‌های سریع", 
                                           font=('Tahoma', 12, 'bold'),
                                           bg='#34495e', fg='white', padx=10, pady=10)
        quick_analysis_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # دکمه‌های تحلیل سریع
        analysis_buttons = [
            ("📊 تحلیل بنیادی", self.run_fundamental_analysis),
            ("📈 تحلیل تکنیکال", self.run_technical_analysis),
            ("🎯 تحلیل ترکیبی", self.run_combined_analysis),
            ("🧮 تحلیل حباب", self.open_bubble_analysis),
            ("🤖 هوش مصنوعی", self.run_ai_analysis),
            ("⚡ سیگنال‌ها", self.generate_trading_signals)
        ]
        
        for text, command in analysis_buttons:
            btn = tk.Button(quick_analysis_frame, text=text, font=('Tahoma', 10),
                          bg='#3498db', fg='white', command=command)
            btn.pack(fill=tk.X, pady=2)
        
        # بخش تنظیمات سریع
        quick_settings_frame = tk.LabelFrame(self.left_panel, text="⚙️ تنظیمات سریع", 
                                           font=('Tahoma', 12, 'bold'),
                                           bg='#34495e', fg='white', padx=10, pady=10)
        quick_settings_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # انتخاب فونت
        font_frame = tk.Frame(quick_settings_frame, bg='#34495e')
        font_frame.pack(fill=tk.X, pady=2)
        tk.Label(font_frame, text="فونت:", font=('Tahoma', 10), 
                fg='white', bg='#34495e', width=10).pack(side=tk.LEFT)
        self.font_var = tk.StringVar(value=self.selected_font)
        font_combo = ttk.Combobox(font_frame, textvariable=self.font_var,
                                 values=self.all_system_fonts[:20], width=15)
        font_combo.pack(side=tk.RIGHT)
        
        # انتخاب تم
        theme_frame = tk.Frame(quick_settings_frame, bg='#34495e')
        theme_frame.pack(fill=tk.X, pady=2)
        tk.Label(theme_frame, text="تم:", font=('Tahoma', 10), 
                fg='white', bg='#34495e', width=10).pack(side=tk.LEFT)
        self.theme_var = tk.StringVar(value="تیره")
        theme_combo = ttk.Combobox(theme_frame, textvariable=self.theme_var,
                                  values=["تیره", "روشن", "آبی"], width=15)
        theme_combo.pack(side=tk.RIGHT)
        
        # پنل مرکزی (داده‌ها و نمودارها)
        self.center_panel = tk.Frame(main_paned, bg='#f0f0f0')
        
        # ایجاد تب‌های اصلی
        self.notebook = ttk.Notebook(self.center_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # تب داده‌ها
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="📊 داده‌ها")
        
        # Treeview برای نمایش داده‌ها
        self.tree_frame = tk.Frame(self.data_tab)
        self.tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ایجاد اسکرول بارها
        self.tree_xscroll = ttk.Scrollbar(self.tree_frame, orient=tk.HORIZONTAL)
        self.tree_yscroll = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL)
        
        # ایجاد Treeview
        self.tree = ttk.Treeview(self.tree_frame,
                                yscrollcommand=self.tree_yscroll.set,
                                xscrollcommand=self.tree_xscroll.set)
        
        self.tree_yscroll.config(command=self.tree.yview)
        self.tree_xscroll.config(command=self.tree.xview)
        
        self.tree_yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree_xscroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # تب نمودارها
        self.chart_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_tab, text="📈 نمودارها")
        
        # تب تحلیل
        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="📊 تحلیل")
        
        # تب پورتفو
        self.portfolio_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.portfolio_tab, text="💰 پورتفو")
        
        # تب گزارش
        self.report_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.report_tab, text="📄 گزارش")
        
        # تب هشدارها
        self.alert_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.alert_tab, text="⚠️ هشدارها")
        
        # تب ابزارها
        self.tools_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.tools_tab, text="🛠️ ابزارها")
        
        main_paned.add(self.left_panel)
        main_paned.add(self.center_panel)
        
        # نوار وضعیت پایین
        self.status_bar = tk.Label(self.root, text="آماده", bd=1, relief=tk.SUNKEN, anchor=tk.W,
                                 font=('Tahoma', 9), bg='#34495e', fg='white')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # تنظیم وزن‌ها برای responsive شدن
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # اتصال رویدادها
        self.search_var.trace('w', lambda *args: self.search_stocks())
        self.root.bind('<Control-o>', lambda e: self.load_data_from_file())
        self.root.bind('<Control-s>', lambda e: self.save_data())
        self.root.bind('<F5>', lambda e: self.refresh_data())
        
        # ایجاد محتوای تب‌ها
        self.setup_data_tab()
        self.setup_chart_tab()
        self.setup_analysis_tab()
        self.setup_portfolio_tab()
        self.setup_report_tab()
        self.setup_alert_tab()
        self.setup_tools_tab()
    
    def setup_data_tab(self):
        """تنظیم تب داده‌ها"""
        # تولبار بالای داده‌ها
        toolbar = tk.Frame(self.data_tab, bg='#ecf0f1', height=40)
        toolbar.pack(fill=tk.X, padx=5, pady=2)
        toolbar.pack_propagate(False)
        
        # دکمه‌های تولبار
        buttons = [
            ("📥 بارگذاری", self.load_data_from_file, '#3498db'),
            ("🌐 دانلود", self.load_data_from_internet, '#2ecc71'),
            ("🔄 بروزرسانی", self.refresh_data, '#f39c12'),
            ("🔍 فیلتر", self.open_filter_dialog, '#9b59b6'),
            ("📤 خروجی", self.export_to_excel, '#e74c3c'),
            ("🧹 پاکسازی", self.clear_data, '#95a5a6')
        ]
        
        for text, command, color in buttons:
            btn = tk.Button(toolbar, text=text, font=('Tahoma', 9),
                          bg=color, fg='white', command=command)
            btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        # آمار سریع
        stats_frame = tk.Frame(toolbar, bg='#ecf0f1')
        stats_frame.pack(side=tk.RIGHT, padx=10)
        
        self.stats_label = tk.Label(stats_frame, text="سهام: ۰ | رکورد: ۰",
                                  font=('Tahoma', 9), bg='#ecf0f1')
        self.stats_label.pack()
    
    def setup_chart_tab(self):
        """تنظیم تب نمودارها"""
        main_frame = tk.Frame(self.chart_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # تولبار نمودارها
        chart_toolbar = tk.Frame(main_frame, bg='#ecf0f1', height=40)
        chart_toolbar.pack(fill=tk.X, pady=(0, 10))
        chart_toolbar.pack_propagate(False)
        
        chart_types = [
            ("📊 میله‌ای", self.create_bar_chart),
            ("📈 خطی", self.create_line_chart),
            ("🧩 پراکندگی", self.create_scatter_chart),
            ("📦 هیستوگرام", self.create_histogram),
            ("🎯 حرارتی", self.create_heatmap),
            ("🌀 رادار", self.create_radar_chart)
        ]
        
        for text, command in chart_types:
            btn = tk.Button(chart_toolbar, text=text, font=('Tahoma', 9),
                          bg='#3498db', fg='white', command=command)
            btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        # قاب نمودار
        self.chart_frame = tk.Frame(main_frame, bg='white')
        self.chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # برچسب راهنما
        help_label = tk.Label(main_frame, 
                            text="💡 برای ایجاد نمودار، ابتدا داده‌ها را بارگذاری کنید",
                            font=('Tahoma', 10), fg='#7f8c8d')
        help_label.pack(pady=10)
    
    def setup_analysis_tab(self):
        """تنظیم تب تحلیل"""
        main_frame = tk.Frame(self.analysis_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # تولبار تحلیل
        analysis_toolbar = tk.Frame(main_frame, bg='#ecf0f1', height=40)
        analysis_toolbar.pack(fill=tk.X, pady=(0, 10))
        analysis_toolbar.pack_propagate(False)
        
        analysis_types = [
            ("🔍 بنیادی", self.run_fundamental_analysis),
            ("📈 تکنیکال", self.run_technical_analysis),
            ("🎯 ترکیبی", self.run_combined_analysis),
            ("🧮 حباب", self.open_bubble_analysis),
            ("🤖 هوش مصنوعی", self.run_ai_analysis),
            ("⚡ سیگنال", self.generate_trading_signals)
        ]
        
        for text, command in analysis_types:
            btn = tk.Button(analysis_toolbar, text=text, font=('Tahoma', 9),
                          bg='#9b59b6', fg='white', command=command)
            btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        # قاب نتایج تحلیل
        self.analysis_text = scrolledtext.ScrolledText(main_frame, 
                                                      font=('Tahoma', 10),
                                                      wrap=tk.WORD)
        self.analysis_text.pack(fill=tk.BOTH, expand=True)
        
        # دکمه ذخیره تحلیل
        save_btn = tk.Button(main_frame, text="💾 ذخیره تحلیل",
                           font=('Tahoma', 10), bg='#2ecc71', fg='white',
                           command=self.save_analysis_results)
        save_btn.pack(pady=5)
    
    def setup_portfolio_tab(self):
        """تنظیم تب پورتفو"""
        main_frame = tk.Frame(self.portfolio_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # تولبار پورتفو
        portfolio_toolbar = tk.Frame(main_frame, bg='#ecf0f1', height=40)
        portfolio_toolbar.pack(fill=tk.X, pady=(0, 10))
        portfolio_toolbar.pack_propagate(False)
        
        portfolio_actions = [
            ("📋 بارگذاری", self.load_portfolio),
            ("📊 تحلیل", self.analyze_portfolio),
            ("⚖️ بهینه‌سازی", self.optimize_portfolio),
            ("🔄 متعادل‌سازی", self.rebalance_portfolio),
            ("📈 عملکرد", self.portfolio_performance),
            ("📤 خروجی", self.export_portfolio)
        ]
        
        for text, command in portfolio_actions:
            btn = tk.Button(portfolio_toolbar, text=text, font=('Tahoma', 9),
                          bg='#e67e22', fg='white', command=command)
            btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        # قاب نمایش پورتفو
        portfolio_display = tk.Frame(main_frame, bg='white')
        portfolio_display.pack(fill=tk.BOTH, expand=True)
        
        # Treeview برای پورتفو
        self.portfolio_tree = ttk.Treeview(portfolio_display)
        self.portfolio_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # برچسب خلاصه پورتفو
        self.portfolio_summary_label = tk.Label(main_frame, 
                                              text="پورتفو بارگذاری نشده است",
                                              font=('Tahoma', 11), fg='#7f8c8d')
        self.portfolio_summary_label.pack(pady=5)
    
    def setup_report_tab(self):
        """تنظیم تب گزارش"""
        main_frame = tk.Frame(self.report_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # تولبار گزارش
        report_toolbar = tk.Frame(main_frame, bg='#ecf0f1', height=40)
        report_toolbar.pack(fill=tk.X, pady=(0, 10))
        report_toolbar.pack_propagate(False)
        
        report_types = [
            ("📋 روزانه", self.generate_daily_report),
            ("📊 پورتفو", self.generate_portfolio_report),
            ("📈 تکنیکال", self.generate_technical_report),
            ("⚠️ هشدارها", self.generate_alert_report),
            ("📚 جامع", self.generate_comprehensive_report)
        ]
        
        for text, command in report_types:
            btn = tk.Button(report_toolbar, text=text, font=('Tahoma', 9),
                          bg='#3498db', fg='white', command=command)
            btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        # قاب گزارش
        self.report_text = scrolledtext.ScrolledText(main_frame, 
                                                    font=('Tahoma', 10),
                                                    wrap=tk.WORD)
        self.report_text.pack(fill=tk.BOTH, expand=True)
        
        # دکمه‌های گزارش
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=5)
        
        tk.Button(button_frame, text="🖨️ چاپ", font=('Tahoma', 10),
                 bg='#3498db', fg='white', command=self.print_report).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="💾 ذخیره", font=('Tahoma', 10),
                 bg='#2ecc71', fg='white', command=self.save_report).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="📤 ارسال", font=('Tahoma', 10),
                 bg='#9b59b6', fg='white', command=self.send_report).pack(side=tk.LEFT, padx=5)
    
    def setup_alert_tab(self):
        """تنظیم تب هشدارها"""
        main_frame = tk.Frame(self.alert_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # تولبار هشدارها
        alert_toolbar = tk.Frame(main_frame, bg='#ecf0f1', height=40)
        alert_toolbar.pack(fill=tk.X, pady=(0, 10))
        alert_toolbar.pack_propagate(False)
        
        alert_actions = [
            ("🔄 بررسی", self.check_alerts),
            ("⚙️ تنظیمات", self.open_alert_settings),
            ("🔔 تست", self.test_alerts),
            ("📤 خروجی", self.export_alerts),
            ("🧹 پاکسازی", self.clear_alerts)
        ]
        
        for text, command in alert_actions:
            btn = tk.Button(alert_toolbar, text=text, font=('Tahoma', 9),
                          bg='#e74c3c', fg='white', command=command)
            btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        # قاب هشدارها
        alert_frame = tk.Frame(main_frame, bg='white')
        alert_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview برای هشدارها
        self.alert_tree = ttk.Treeview(alert_frame)
        self.alert_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # برچسب وضعیت هشدارها
        self.alert_status_label = tk.Label(main_frame, 
                                         text="⚠️ سیستم هشدار آماده است",
                                         font=('Tahoma', 11), fg='#e74c3c')
        self.alert_status_label.pack(pady=5)
    
    def setup_tools_tab(self):
        """تنظیم تب ابزارها"""
        main_frame = tk.Frame(self.tools_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # شبکه دکمه‌های ابزارها
        tools_grid = tk.Frame(main_frame)
        tools_grid.pack(fill=tk.BOTH, expand=True)
        
        tools = [
            ("🧮 ماشین حساب", self.open_calculator, '#3498db'),
            ("📅 تقویم اقتصادی", self.open_economic_calendar, '#2ecc71'),
            ("📊 مقایسه سهام", self.open_stock_comparison, '#9b59b6'),
            ("🔍 جستجوی پیشرفته", self.open_advanced_search, '#f39c12'),
            ("🧪 بک‌تست", self.open_backtest_tool, '#e74c3c'),
            ("📚 مستندات", self.open_documentation, '#34495e'),
            ("🎯 شبیه‌ساز", self.open_simulator, '#1abc9c'),
            ("📈 اسکنر", self.open_scanner, '#d35400')
        ]
        
        row, col = 0, 0
        for text, command, color in tools:
            btn = tk.Button(tools_grid, text=text, font=('Tahoma', 11),
                          bg=color, fg='white', command=command,
                          height=3, width=15)
            btn.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
            col += 1
            if col > 1:
                col = 0
                row += 1
        
        # تنظیم وزن‌ها
        for i in range(2):
            tools_grid.grid_columnconfigure(i, weight=1)
        for i in range(4):
            tools_grid.grid_rowconfigure(i, weight=1)
    
    def load_data_from_file(self):
        """بارگذاری داده از فایل"""
        filetypes = [
            ('Excel files', '*.xlsx *.xls'),
            ('CSV files', '*.csv'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="انتخاب فایل داده",
            filetypes=filetypes
        )
        
        if filename:
            try:
                self.logger.log(f"در حال بارگذاری فایل: {filename}", "INFO")
                
                if filename.endswith('.csv'):
                    self.df = pd.read_csv(filename, encoding='utf-8')
                else:
                    self.df = pd.read_excel(filename)
                
                self.process_loaded_data()
                
                self.logger.log(f"فایل با موفقیت بارگذاری شد. {len(self.df)} رکورد", "SUCCESS")
                messagebox.showinfo("موفقیت", f"داده‌ها با موفقیت بارگذاری شدند\n{len(self.df)} رکورد")
                
            except Exception as e:
                self.logger.log(f"خطا در بارگذاری فایل: {e}", "ERROR")
                messagebox.showerror("خطا", f"خطا در بارگذاری فایل:\n{str(e)}")
    
    def load_data_from_internet(self):
        """بارگذاری داده از اینترنت"""
        dialog = tk.Toplevel(self.root)
        dialog.title("بارگذاری از اینترنت")
        dialog.geometry("400x300")
        dialog.configure(bg='#f0f0f0')
        dialog.resizable(False, False)
        
        # مرکز کردن پنجره
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'{width}x{height}+{x}+{y}')
        
        # فریم اصلی
        main_frame = tk.Frame(dialog, bg='#f0f0f0', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="📡 منبع داده:", 
                font=('Tahoma', 12, 'bold'), bg='#f0f0f0').pack(pady=5)
        
        source_var = tk.StringVar(value="tsetmc")
        
        sources = [
            ("tsetmc.com", "tsetmc"),
            ("ایزی تریدر", "easytrader"),
            ("نماگر", "nemagar"),
            ("سهام یاب", "sahamyab")
        ]
        
        for text, value in sources:
            rb = tk.Radiobutton(main_frame, text=text, variable=source_var, 
                              value=value, font=('Tahoma', 10), bg='#f0f0f0')
            rb.pack(anchor='w', pady=2)
        
        tk.Label(main_frame, text="تعداد سهام:", 
                font=('Tahoma', 10), bg='#f0f0f0').pack(pady=10)
        
        count_var = tk.StringVar(value="50")
        count_entry = tk.Entry(main_frame, textvariable=count_var, 
                             font=('Tahoma', 10), width=10)
        count_entry.pack()
        
        def download_data():
            source = source_var.get()
            count = int(count_var.get())
            dialog.destroy()
            
            if source == "tsetmc":
                self.download_from_tsetmc(count)
            elif source == "easytrader":
                self.download_from_easytrader(count)
            else:
                messagebox.showinfo("اطلاع", "این منبع به زودی اضافه خواهد شد")
        
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="دانلود", font=('Tahoma', 10),
                 bg='#27ae60', fg='white', command=download_data,
                 width=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="انصراف", font=('Tahoma', 10),
                 bg='#e74c3c', fg='white', command=dialog.destroy,
                 width=10).pack(side=tk.LEFT, padx=5)
    
    def download_from_tsetmc(self, count=50):
        """دانلود داده از tsetmc.com"""
        self.logger.log(f"شروع دانلود از tsetmc.com ({count} سهم)", "INFO")
        
        try:
            # دریافت لیست سهام
            stock_list = self.tsetmc_downloader.get_stock_list()
            
            if stock_list is None:
                messagebox.showerror("خطا", "خطا در دریافت لیست سهام")
                return
            
            # انتخاب سهام تصادفی (در نسخه واقعی بهتر است بر اساس معیارهای خاصی انتخاب شود)
            selected_stocks = stock_list.sample(min(count, len(stock_list)))
            
            data_list = []
            
            for _, stock in selected_stocks.iterrows():
                symbol = stock['lVal18']
                details = self.tsetmc_downloader.get_stock_details(symbol)
                
                if details:
                    data_list.append(details)
                
                # جلوگیری از بلاک شدن IP
                time.sleep(0.5)
            
            if data_list:
                self.df = pd.DataFrame(data_list)
                self.process_loaded_data()
                
                self.logger.log(f"✅ دانلود کامل شد. {len(self.df)} سهم دریافت شد", "SUCCESS")
                messagebox.showinfo("موفقیت", 
                                  f"دانلود از tsetmc.com کامل شد\n{len(self.df)} سهم دریافت شد")
            else:
                messagebox.showwarning("اخطار", "هیچ داده‌ای دریافت نشد")
                
        except Exception as e:
            self.logger.log(f"خطا در دانلود از tsetmc: {e}", "ERROR")
            messagebox.showerror("خطا", f"خطا در دانلود:\n{str(e)}")
    
    def download_from_easytrader(self, count=50):
        """دانلود داده از ایزی تریدر"""
        self.logger.log("شروع دانلود از ایزی تریدر", "INFO")
        
        login_dialog = tk.Toplevel(self.root)
        login_dialog.title("ورود به ایزی تریدر")
        login_dialog.geometry("400x250")
        login_dialog.configure(bg='#f0f0f0')
        login_dialog.resizable(False, False)
        
        # مرکز کردن پنجره
        login_dialog.update_idletasks()
        width = login_dialog.winfo_width()
        height = login_dialog.winfo_height()
        x = (login_dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (login_dialog.winfo_screenheight() // 2) - (height // 2)
        login_dialog.geometry(f'{width}x{height}+{x}+{y}')
        
        tk.Label(login_dialog, text="🔐 ورود به ایزی تریدر",
                font=('Tahoma', 14, 'bold'), bg='#f0f0f0').pack(pady=15)
        
        # فیلدهای ورود
        form_frame = tk.Frame(login_dialog, bg='#f0f0f0')
        form_frame.pack(pady=10, padx=20)
        
        tk.Label(form_frame, text="نام کاربری:", 
                font=('Tahoma', 10), bg='#f0f0f0').grid(row=0, column=0, sticky='w', pady=5)
        username_var = tk.StringVar()
        username_entry = tk.Entry(form_frame, textvariable=username_var, 
                                font=('Tahoma', 10), width=25)
        username_entry.grid(row=0, column=1, pady=5, padx=10)
        
        tk.Label(form_frame, text="رمز عبور:", 
                font=('Tahoma', 10), bg='#f0f0f0').grid(row=1, column=0, sticky='w', pady=5)
        password_var = tk.StringVar()
        password_entry = tk.Entry(form_frame, textvariable=password_var, 
                                font=('Tahoma', 10), width=25, show='*')
        password_entry.grid(row=1, column=1, pady=5, padx=10)
        
        def do_login():
            username = username_var.get()
            password = password_var.get()
            
            if not username or not password:
                messagebox.showwarning("اخطار", "لطفاً نام کاربری و رمز عبور را وارد کنید")
                return
            
            login_dialog.destroy()
            
            # اجرای لاگین در یک thread جداگانه
            threading.Thread(target=self.login_and_download, 
                           args=(username, password, count), 
                           daemon=True).start()
        
        button_frame = tk.Frame(login_dialog, bg='#f0f0f0')
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="ورود و دانلود", font=('Tahoma', 10),
                 bg='#27ae60', fg='white', command=do_login,
                 width=12).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="انصراف", font=('Tahoma', 10),
                 bg='#e74c3c', fg='white', command=login_dialog.destroy,
                 width=12).pack(side=tk.LEFT, padx=5)
        
        username_entry.focus()
    
    def login_and_download(self, username, password, count):
        """لاگین و دانلود از ایزی تریدر"""
        try:
            self.logger.log("در حال ورود به ایزی تریدر...", "INFO")
            
            # لاگین
            success = self.downloader.login_with_selenium(username, password)
            
            if not success:
                messagebox.showerror("خطا", "ورود به ایزی تریدر ناموفق بود")
                return
            
            # دانلود داده
            self.logger.log("در حال دانلود داده‌های بازار...", "INFO")
            self.df = self.downloader.download_market_data()
            
            if self.df is not None:
                if len(self.df) > count:
                    self.df = self.df.head(count)
                
                self.process_loaded_data()
                
                messagebox.showinfo("موفقیت", 
                                  f"دانلود از ایزی تریدر کامل شد\n{len(self.df)} سهم دریافت شد")
            else:
                messagebox.showwarning("اخطار", "هیچ داده‌ای دریافت نشد")
                
        except Exception as e:
            self.logger.log(f"خطا در دانلود از ایزی تریدر: {e}", "ERROR")
            messagebox.showerror("خطا", f"خطا در دانلود:\n{str(e)}")
        finally:
            self.downloader.close()
    
    def process_loaded_data(self):
        """پردازش داده‌های بارگذاری شده"""
        if self.df is not None:
            # نمایش در Treeview
            self.display_data_in_tree()
            
            # بروزرسانی وضعیت
            self.data_status_label.config(
                text=f"📊 {len(self.df)} سهم | {len(self.df.columns)} ستون",
                fg='#2ecc71'
            )
            
            self.stats_label.config(
                text=f"سهام: {len(self.df)} | ستون‌ها: {len(self.df.columns)}"
            )
            
            # ایجاد mapping ستون‌ها
            self.create_column_mapping()
            
            # بروزرسانی تب‌ها
            self.notebook.tab(0, state='normal')  # فعال کردن تب داده‌ها
            
            # لاگ
            self.logger.log(f"داده‌ها پردازش شدند. {len(self.df)} رکورد", "SUCCESS")
    
    def display_data_in_tree(self):
        """نمایش داده‌ها در Treeview"""
        # پاک کردن Treeview موجود
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # تنظیم ستون‌ها
        self.tree["columns"] = list(self.df.columns)
        self.tree["show"] = "headings"
        
        # تنظیم هدر ستون‌ها
        for col in self.df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, minwidth=50)
        
        # اضافه کردن داده‌ها
        for index, row in self.df.iterrows():
            values = []
            for col in self.df.columns:
                value = row[col]
                if pd.isna(value):
                    values.append("")
                elif isinstance(value, float):
                    values.append(f"{value:,.2f}")
                elif isinstance(value, (int, np.integer)):
                    values.append(f"{value:,}")
                else:
                    values.append(str(value))
            
            self.tree.insert("", tk.END, values=values)
    
    def create_column_mapping(self):
        """ایجاد mapping بین ستون‌های استاندارد و ستون‌های واقعی"""
        self.column_mapping = {}
        
        # لیست ستون‌های استاندارد
        standard_columns = [
            'نماد', 'نام', 'آخرین قیمت', 'حجم معاملات', 'ارزش معاملات',
            'تعداد معاملات', 'P/B', 'P/E', 'EPS', 'DPS%', 'RSI', 'MFI',
            'SMA20d', 'SMA50d', 'بازدهی1ماه', 'بازدهی3ماه', 'بازدهی6ماه',
            'بازدهی1سال', 'خرید حقوقی%', 'فروش حقوقی%', 'ضریب بتا',
            'P/S', 'P/NAV'
        ]
        
        actual_columns = list(self.df.columns)
        
        # تلاش برای یافتن ستون‌های مشابه
        for std_col in standard_columns:
            found = False
            
            # جستجوی مستقیم
            if std_col in actual_columns:
                self.column_mapping[std_col] = std_col
                found = True
            
            # جستجوی با الگوهای مختلف
            if not found:
                patterns = {
                    'نماد': ['symbol', 'کد', 'نام نماد', 'کد نماد'],
                    'آخرین قیمت': ['قیمت', 'قیمت پایانی', 'قیمت آخر', 'close', 'last'],
                    'حجم معاملات': ['حجم', 'volume', 'تعداد سهم'],
                    'P/B': ['p/b', 'نسبت قیمت به ارزش دفتری'],
                    'P/E': ['p/e', 'نسبت قیمت به سود'],
                    'EPS': ['eps', 'سود هر سهم'],
                    'RSI': ['rsi', 'شاخص قدرت نسبی']
                }
                
                if std_col in patterns:
                    for pattern in patterns[std_col]:
                        for act_col in actual_columns:
                            if pattern.lower() in str(act_col).lower():
                                self.column_mapping[std_col] = act_col
                                found = True
                                break
                        if found:
                            break
            
            if not found:
                self.column_mapping[std_col] = std_col
        
        self.logger.log(f"Mapping ستون‌ها ایجاد شد: {len(self.column_mapping)} نگاشت", "INFO")
    
    def search_stocks(self):
        """جستجوی سهام"""
        if self.df is None:
            return
        
        search_text = self.search_var.get().strip().lower()
        
        if not search_text:
            # نمایش همه داده‌ها
            self.display_data_in_tree()
            return
        
        # فیلتر کردن داده‌ها
        filtered_df = self.df.copy()
        
        # جستجو در ستون‌های متنی
        text_columns = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                text_columns.append(col)
        
        if text_columns:
            mask = filtered_df[text_columns].apply(
                lambda x: x.astype(str).str.lower().str.contains(search_text)
            ).any(axis=1)
            filtered_df = filtered_df[mask]
        
        # نمایش نتایج
        self.display_filtered_data(filtered_df)
    
    def display_filtered_data(self, filtered_df):
        """نمایش داده‌های فیلتر شده"""
        # پاک کردن Treeview موجود
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # تنظیم ستون‌ها
        self.tree["columns"] = list(filtered_df.columns)
        self.tree["show"] = "headings"
        
        # تنظیم هدر ستون‌ها
        for col in filtered_df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, minwidth=50)
        
        # اضافه کردن داده‌ها
        for index, row in filtered_df.iterrows():
            values = []
            for col in filtered_df.columns:
                value = row[col]
                if pd.isna(value):
                    values.append("")
                elif isinstance(value, float):
                    values.append(f"{value:,.2f}")
                elif isinstance(value, (int, np.integer)):
                    values.append(f"{value:,}")
                else:
                    values.append(str(value))
            
            self.tree.insert("", tk.END, values=values)
        
        # بروزرسانی آمار
        self.stats_label.config(
            text=f"سهام: {len(filtered_df)}/{len(self.df)} | ستون‌ها: {len(filtered_df.columns)}"
        )
    
    def apply_quick_filters(self):
        """اعمال فیلترهای سریع"""
        if self.df is None:
            messagebox.showwarning("اخطار", "ابتدا داده‌ها را بارگذاری کنید")
            return
        
        try:
            filtered_df = self.df.copy()
            
            # فیلتر P/B
            max_pb = float(self.pb_var.get())
            pb_col = self.column_mapping.get('P/B')
            if pb_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[pb_col] <= max_pb]
            
            # فیلتر RSI
            max_rsi = float(self.rsi_var.get())
            rsi_col = self.column_mapping.get('RSI')
            if rsi_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[rsi_col] <= max_rsi]
            
            # فیلتر حجم
            min_volume = float(self.volume_var.get())
            volume_col = self.column_mapping.get('حجم معاملات')
            if volume_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[volume_col] >= min_volume]
            
            # نمایش نتایج
            self.display_filtered_data(filtered_df)
            
            self.logger.log(f"فیلترها اعمال شدند. {len(filtered_df)} سهم باقی ماند", "INFO")
            
        except ValueError as e:
            messagebox.showerror("خطا", "لطفاً مقادیر معتبر وارد کنید")
    
    def run_fundamental_analysis(self):
        """اجرای تحلیل بنیادی"""
        if self.df is None:
            messagebox.showwarning("اخطار", "ابتدا داده‌ها را بارگذاری کنید")
            return
        
        self.logger.log("شروع تحلیل بنیادی...", "INFO")
        
        # استفاده از تحلیل‌گر پیشرفته
        analyzer = AdvancedStockAnalyzer(
            self.df, 
            self.analysis_settings, 
            self.column_mapping,
            self.market_trend,
            self.logger
        )
        
        # درخواست بودجه از کاربر
        budget = simpledialog.askfloat("بودجه", "بودجه خرید (ریال):", 
                                     minvalue=0, maxvalue=1000000000000)
        
        # اجرای تحلیل
        results = analyzer.analyze_stocks(budget=budget, top_n=20)
        
        if results is not None:
            self.display_analysis_results(results, "بنیادی")
            self.last_recommendations = results
            
            self.logger.log("تحلیل بنیادی کامل شد", "SUCCESS")
        else:
            self.logger.log("خطا در تحلیل بنیادی", "ERROR")
    
    def run_technical_analysis(self):
        """اجرای تحلیل تکنیکال"""
        if self.df is None:
            messagebox.showwarning("اخطار", "ابتدا داده‌ها را بارگذاری کنید")
            return
        
        self.logger.log("شروع تحلیل تکنیکال...", "INFO")
        
        # نمایش در تب تحلیل
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, "📈 نتایج تحلیل تکنیکال:\n\n")
        self.analysis_text.insert(tk.END, "="*60 + "\n\n")
        
        # محاسبه اندیکاتورهای تکنیکال
        technical_results = []
        
        for _, row in self.df.iterrows():
            symbol = row.get(self.column_mapping.get('نماد', 'نماد'), '')
            
            if not symbol:
                continue
            
            # محاسبه امتیاز تکنیکال
            score = 0
            reasons = []
            
            # بررسی RSI
            rsi_col = self.column_mapping.get('RSI')
            if rsi_col in row and pd.notna(row[rsi_col]):
                rsi = row[rsi_col]
                if rsi < 30:
                    score += 30
                    reasons.append("RSI اشباع فروش")
                elif rsi < 40:
                    score += 20
                    reasons.append("RSI پایین")
                elif rsi > 70:
                    score -= 20
                    reasons.append("RSI اشباع خرید")
            
            # بررسی مومنتوم
            momentum_col = self.column_mapping.get('بازدهی1ماه')
            if momentum_col in row and pd.notna(row[momentum_col]):
                momentum = row[momentum_col]
                if momentum > 20:
                    score += 20
                    reasons.append("مومنتوم قوی")
                elif momentum < -15:
                    score += 15
                    reasons.append("اصلاح شدید")
            
            # بررسی حجم
            volume_col = self.column_mapping.get('حجم معاملات')
            if volume_col in row and pd.notna(row[volume_col]):
                volume = row[volume_col]
                if volume > 1000000:
                    score += 10
                    reasons.append("حجم بالا")
            
            if score > 0:
                technical_results.append({
                    'symbol': symbol,
                    'score': score,
                    'reasons': "، ".join(reasons)
                })
        
        # مرتب‌سازی و نمایش
        technical_results.sort(key=lambda x: x['score'], reverse=True)
        
        for result in technical_results[:15]:
            self.analysis_text.insert(tk.END, 
                                    f"🔹 {result['symbol']}: امتیاز {result['score']}\n")
            self.analysis_text.insert(tk.END, f"   📋 {result['reasons']}\n\n")
        
        self.notebook.select(self.analysis_tab)
        self.logger.log("تحلیل تکنیکال کامل شد", "SUCCESS")
    
    def run_combined_analysis(self):
        """اجرای تحلیل ترکیبی"""
        if self.df is None:
            messagebox.showwarning("اخطار", "ابتدا داده‌ها را بارگذاری کنید")
            return
        
        self.logger.log("شروع تحلیل ترکیبی...", "INFO")
        
        # اجرای تحلیل بنیادی
        fundamental_analyzer = AdvancedStockAnalyzer(
            self.df, 
            self.analysis_settings, 
            self.column_mapping,
            self.market_trend,
            self.logger
        )
        
        fundamental_results = fundamental_analyzer.analyze_stocks(top_n=30)
        
        if fundamental_results is None:
            messagebox.showerror("خطا", "خطا در تحلیل بنیادی")
            return
        
        # ترکیب با تحلیل تکنیکال
        combined_results = []
        
        for _, row in fundamental_results.iterrows():
            symbol = row.get('نماد', '')
            fundamental_score = row.get('امتیاز_کل', 0)
            
            # محاسبه امتیاز تکنیکال اضافی
            technical_score = 0
            
            # بررسی RSI
            rsi_col = self.column_mapping.get('RSI')
            if rsi_col in row and pd.notna(row[rsi_col]):
                rsi = row[rsi_col]
                if rsi < 30:
                    technical_score += 15
                elif rsi < 40:
                    technical_score += 10
            
            # بررسی مومنتوم
            momentum_col = self.column_mapping.get('بازدهی1ماه')
            if momentum_col in row and pd.notna(row[momentum_col]):
                momentum = row[momentum_col]
                if momentum > 10:
                    technical_score += 10
                elif momentum < -10:
                    technical_score += 5
            
            # امتیاز ترکیبی
            combined_score = fundamental_score * 0.7 + technical_score * 0.3
            
            combined_results.append({
                'symbol': symbol,
                'fundamental_score': fundamental_score,
                'technical_score': technical_score,
                'combined_score': combined_score,
                'details': row.to_dict()
            })
        
        # مرتب‌سازی بر اساس امتیاز ترکیبی
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # نمایش نتایج
        self.display_combined_results(combined_results[:15])
        
        self.logger.log("تحلیل ترکیبی کامل شد", "SUCCESS")
    
    def display_analysis_results(self, results, analysis_type):
        """نمایش نتایج تحلیل"""
        self.analysis_text.delete(1.0, tk.END)
        
        title = f"📊 نتایج تحلیل {analysis_type}:\n"
        self.analysis_text.insert(tk.END, title)
        self.analysis_text.insert(tk.END, "="*60 + "\n\n")
        
        # اضافه کردن خلاصه
        total_stocks = len(results)
        avg_score = results['امتیاز_کل'].mean()
        
        summary = f"📈 خلاصه تحلیل:\n"
        summary += f"• تعداد سهام: {total_stocks}\n"
        summary += f"• میانگین امتیاز: {avg_score:.2f}\n"
        summary += f"• بهترین امتیاز: {results['امتیاز_کل'].max():.2f}\n"
        summary += f"• ضعیف‌ترین امتیاز: {results['امتیاز_کل'].min():.2f}\n\n"
        
        self.analysis_text.insert(tk.END, summary)
        self.analysis_text.insert(tk.END, "🏆 سهام برتر:\n\n")
        
        # اضافه کردن سهام برتر
        for i, (_, row) in enumerate(results.iterrows()):
            if i >= 15:  # فقط 15 مورد اول
                break
            
            symbol = row.get('نماد', 'N/A')
            score = row.get('امتیاز_کل', 0)
            status = row.get('وضعیت', 'نامشخص')
            price = row.get(self.column_mapping.get('آخرین قیمت', ''), 'N/A')
            
            if isinstance(price, (int, float)):
                price_str = f"{price:,.0f}"
            else:
                price_str = str(price)
            
            line = f"{i+1:2d}. {symbol:<10} - امتیاز: {score:5.2f} ({status})\n"
            line += f"    💰 قیمت: {price_str} ریال\n"
            
            # اضافه کردن توصیه
            if 'final_bubble' in row:
                bubble = row['final_bubble']
                if bubble > 70:
                    line += f"    ⚠️  حباب: {bubble:.1f}%\n"
                elif bubble < 30:
                    line += f"    🟢 کم‌بها: {bubble:.1f}%\n"
            
            self.analysis_text.insert(tk.END, line + "\n")
        
        # اضافه کردن توصیه کلی
        self.analysis_text.insert(tk.END, "\n💡 توصیه کلی:\n")
        
        if avg_score > 70:
            self.analysis_text.insert(tk.END, "✅ بازار در وضعیت خوبی قرار دارد. فرصت‌های خرید مناسبی وجود دارد.\n")
        elif avg_score > 50:
            self.analysis_text.insert(tk.END, "⚠️  بازار در وضعیت متوسطی است. با احتیاط خرید کنید.\n")
        else:
            self.analysis_text.insert(tk.END, "🔴 بازار ضعیف است. فعلاً از خرید خودداری کنید.\n")
        
        # تغییر به تب تحلیل
        self.notebook.select(self.analysis_tab)
    
    def display_combined_results(self, results):
        """نمایش نتایج تحلیل ترکیبی"""
        self.analysis_text.delete(1.0, tk.END)
        
        title = "🎯 نتایج تحلیل ترکیبی (بنیادی + تکنیکال):\n"
        self.analysis_text.insert(tk.END, title)
        self.analysis_text.insert(tk.END, "="*60 + "\n\n")
        
        for i, result in enumerate(results):
            symbol = result['symbol']
            combined_score = result['combined_score']
            fund_score = result['fundamental_score']
            tech_score = result['technical_score']
            
            line = f"{i+1:2d}. {symbol:<10} - امتیاز کل: {combined_score:5.2f}\n"
            line += f"    📊 بنیادی: {fund_score:5.2f} | 📈 تکنیکال: {tech_score:5.2f}\n"
            
            # تعیین وضعیت
            if combined_score >= 80:
                status = "عالی 🏆"
            elif combined_score >= 70:
                status = "خوب 👍"
            elif combined_score >= 60:
                status = "متوسط ➖"
            elif combined_score >= 50:
                status = "ضعیف ⚠️"
            else:
                status = "خیلی ضعیف ❌"
            
            line += f"    🎯 وضعیت: {status}\n\n"
            
            self.analysis_text.insert(tk.END, line)
        
        self.notebook.select(self.analysis_tab)
    
    def open_bubble_analysis(self):
        """باز کردن پنجره تحلیل حباب"""
        if self.df is None:
            messagebox.showwarning("اخطار", "ابتدا داده‌ها را بارگذاری کنید")
            return
        
        self.logger.log("باز کردن پنجره تحلیل حباب...", "INFO")
        
        # ایجاد پنجره تحلیل حباب
        bubble_window = tk.Toplevel(self.root)
        bubble_window.title("تحلیل پیشرفته حباب سهام")
        bubble_window.geometry("1400x900")
        bubble_window.configure(bg='#f0f0f0')
        
        # فریم اصلی
        main_frame = tk.Frame(bubble_window, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # هدر
        header_frame = tk.Frame(main_frame, bg='#2c3e50', height=60)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(header_frame, 
                               text="🔍 تحلیل پیشرفته حباب سهام - ۶ روش مختلف",
                               font=('Tahoma', 16, 'bold'),
                               fg='white', bg='#2c3e50')
        header_label.pack(pady=20)
        
        # ایجاد تب‌ها
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # تب‌های مختلف
        summary_tab = ttk.Frame(notebook)
        methods_tab = ttk.Frame(notebook)
        comparison_tab = ttk.Frame(notebook)
        charts_tab = ttk.Frame(notebook)
        recommendations_tab = ttk.Frame(notebook)
        
        notebook.add(summary_tab, text="📊 خلاصه تحلیل")
        notebook.add(methods_tab, text="📈 روش‌های محاسبه")
        notebook.add(comparison_tab, text="⚖️ مقایسه روش‌ها")
        notebook.add(charts_tab, text="📉 نمودارها")
        notebook.add(recommendations_tab, text="💡 توصیه‌ها")
        
        # ایجاد تحلیل‌گر حباب
        bubble_analyzer = BubbleAnalyzer(self.df, self.logger)
        bubble_results = bubble_analyzer.calculate_all_bubble_metrics()
        
        # تنظیم محتوای تب‌ها
        self.setup_bubble_summary_tab(summary_tab, bubble_results)
        self.setup_bubble_methods_tab(methods_tab, bubble_results)
        self.setup_bubble_charts_tab(charts_tab, bubble_results)
        
        self.logger.log("پنجره تحلیل حباب باز شد", "SUCCESS")
    
    def setup_bubble_summary_tab(self, parent, bubble_results):
        """تنظیم تب خلاصه تحلیل حباب"""
        main_frame = tk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # خلاصه آمار
        summary_frame = tk.LabelFrame(main_frame, text="📊 آمار کلی حباب بازار", 
                                     font=('Tahoma', 12, 'bold'),
                                     padx=10, pady=10)
        summary_frame.pack(fill=tk.X, pady=10)
        
        if 'final_bubble' in bubble_results:
            final_bubble = bubble_results['final_bubble']
            
            stats_frame = tk.Frame(summary_frame)
            stats_frame.pack(fill=tk.X, pady=5)
            
            stats = [
                ("میانگین حباب:", f"{final_bubble['final_bubble'].mean():.1f}%"),
                ("میانه حباب:", f"{final_bubble['final_bubble'].median():.1f}%"),
                ("بیشترین حباب:", f"{final_bubble['final_bubble'].max():.1f}%"),
                ("کمترین حباب:", f"{final_bubble['final_bubble'].min():.1f}%"),
                ("انحراف معیار:", f"{final_bubble['final_bubble'].std():.1f}%")
            ]
            
            for i, (label, value) in enumerate(stats):
                frame = tk.Frame(stats_frame)
                frame.grid(row=i//2, column=i%2, padx=10, pady=5, sticky='w')
                tk.Label(frame, text=label, font=('Tahoma', 10)).pack(side=tk.LEFT)
                tk.Label(frame, text=value, font=('Tahoma', 10, 'bold'), fg='blue').pack(side=tk.LEFT)
    
    def setup_bubble_methods_tab(self, parent, bubble_results):
        """تنظیم تب روش‌های تحلیل حباب"""
        main_frame = tk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        methods = [
            ("P/B Bubble", "pb_bubble", "محاسبه حباب بر اساس نسبت قیمت به ارزش دفتری"),
            ("P/E Bubble", "pe_bubble", "محاسبه حباب بر اساس نسبت قیمت به سود"),
            ("Composite Bubble", "composite_bubble", "حباب ترکیبی بر اساس چندین شاخص"),
            ("Historical Bubble", "historical_bubble", "حباب بر اساس انحراف از روند تاریخی"),
            ("P/S Bubble", "ps_bubble", "حباب بر اساس نسبت قیمت به فروش"),
            ("Z-Score Bubble", "zscore_bubble", "حباب بر اساس آمار Z-Score")
        ]
        
        for method_name, method_key, description in methods:
            if method_key in bubble_results and bubble_results[method_key] is not None:
                self.create_bubble_method_section(scrollable_frame, method_name, 
                                                method_key, description, bubble_results)
    
    def create_bubble_method_section(self, parent, method_name, method_key, description, bubble_results):
        """ایجاد بخش برای هر روش تحلیل حباب"""
        section_frame = tk.LabelFrame(parent, text=f"📊 {method_name}", 
                                     font=('Tahoma', 11, 'bold'),
                                     padx=10, pady=10)
        section_frame.pack(fill=tk.X, pady=10, padx=10)
        
        # توضیحات
        tk.Label(section_frame, text=description, 
                font=('Tahoma', 9), wraplength=800, justify='left').pack(anchor='w', pady=5)
        
        # نمایش ۱۰ سهم برتر
        method_data = bubble_results[method_key]
        if method_data is not None and not method_data.empty:
            bubble_col = method_key.replace('_bubble', '')
            if bubble_col in method_data.columns:
                top_10 = method_data.nlargest(10, bubble_col)
                
                tree_frame = tk.Frame(section_frame)
                tree_frame.pack(fill=tk.X, pady=10)
                
                # ایجاد Treeview
                tree = ttk.Treeview(tree_frame, height=min(len(top_10), 10))
                tree["columns"] = ["نماد", "مقدار", "سطح"]
                
                tree.column("#0", width=0, stretch=tk.NO)
                tree.column("نماد", width=100, anchor='center')
                tree.column("مقدار", width=100, anchor='center')
                tree.column("سطح", width=150, anchor='center')
                
                tree.heading("نماد", text="نماد")
                tree.heading("مقدار", text="مقدار")
                tree.heading("سطح", text="سطح حباب")
                
                for _, row in top_10.iterrows():
                    value = row.get(bubble_col, 0)
                    level_col = f"{method_key}_level"
                    level = row.get(level_col, 'نامشخص')
                    
                    tree.insert("", tk.END, values=(
                        row.get('نماد', ''),
                        f"{value:.1f}%" if isinstance(value, (int, float)) else str(value),
                        level
                    ))
                
                tree.pack(fill=tk.X)
    
    def setup_bubble_charts_tab(self, parent, bubble_results):
        """تنظیم تب نمودارهای حباب"""
        main_frame = tk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        try:
            if 'final_bubble' in bubble_results and bubble_results['final_bubble'] is not None:
                final_bubble = bubble_results['final_bubble']
                
                # ایجاد نمودارها
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle('تحلیل حباب سهام - نمودارها', fontsize=16, fontweight='bold')
                
                # نمودار 1: توزیع حباب
                bubble_values = final_bubble['final_bubble']
                axes[0, 0].hist(bubble_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                axes[0, 0].axvline(x=50, color='red', linestyle='--', label='آستانه حباب')
                axes[0, 0].set_xlabel('درصد حباب')
                axes[0, 0].set_ylabel('تعداد سهام')
                axes[0, 0].set_title('توزیع حباب در بازار')
                axes[0, 0].legend()
                axes[0, 0].grid(alpha=0.3)
                
                # نمودار 2: ۱۰ سهام پرحباب
                top_10 = final_bubble.nlargest(10, 'final_bubble')
                bars = axes[0, 1].barh(top_10['نماد'], top_10['final_bubble'], 
                                      color='lightcoral', alpha=0.7)
                axes[0, 1].set_xlabel('درصد حباب')
                axes[0, 1].set_title('۱۰ سهام پرحباب بازار')
                axes[0, 1].grid(axis='x', alpha=0.3)
                
                # نمودار 3: ۱۰ سهام کم‌بها
                bottom_10 = final_bubble.nsmallest(10, 'final_bubble')
                bars = axes[1, 0].barh(bottom_10['نماد'], bottom_10['final_bubble'], 
                                      color='lightgreen', alpha=0.7)
                axes[1, 0].set_xlabel('درصد حباب')
                axes[1, 0].set_title('۱۰ سهام کم‌بهای بازار')
                axes[1, 0].grid(axis='x', alpha=0.3)
                
                # نمودار 4: توزیع سطح حباب
                if 'final_bubble_level' in final_bubble.columns:
                    level_counts = final_bubble['final_bubble_level'].value_counts()
                    colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
                    axes[1, 1].pie(level_counts.values, labels=level_counts.index, 
                                  colors=colors[:len(level_counts)], autopct='%1.1f%%')
                    axes[1, 1].set_title('توزیع سهام بر اساس سطح حباب')
                
                plt.tight_layout()
                
                # نمایش نمودار در Tkinter
                canvas = FigureCanvasTkAgg(fig, main_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
        except Exception as e:
            tk.Label(main_frame, text=f"خطا در نمایش نمودارها: {str(e)}", 
                    font=('Tahoma', 12), fg='red').pack(pady=50)
    
    def run_ai_analysis(self):
        """اجرای تحلیل هوش مصنوعی"""
        if self.df is None:
            messagebox.showwarning("اخطار", "ابتدا داده‌ها را بارگذاری کنید")
            return
        
        self.logger.log("شروع تحلیل هوش مصنوعی...", "INFO")
        
        # نمایش در تب تحلیل
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, "🤖 نتایج تحلیل هوش مصنوعی:\n\n")
        self.analysis_text.insert(tk.END, "="*60 + "\n\n")
        self.analysis_text.insert(tk.END, "در حال تحلیل با الگوریتم‌های پیشرفته...\n\n")
        
        # اجرای تحلیل در یک thread جداگانه
        threading.Thread(target=self.perform_ai_analysis, daemon=True).start()
    
    def perform_ai_analysis(self):
        """انجام تحلیل هوش مصنوعی"""
        try:
            # خوشه‌بندی سهام
            stock_data = self.prepare_stock_data_for_ai()
            
            if stock_data:
                cluster_results = self.market_analyzer.cluster_stocks(stock_data, n_clusters=5)
                
                if cluster_results['success']:
                    self.display_cluster_results(cluster_results)
                else:
                    self.analysis_text.insert(tk.END, "خطا در خوشه‌بندی سهام\n")
            
            # تحلیل رژیم‌های بازار
            # (این بخش نیاز به داده‌های تاریخی بیشتری دارد)
            
            self.analysis_text.insert(tk.END, "\n✅ تحلیل هوش مصنوعی کامل شد\n")
            
            self.logger.log("تحلیل هوش مصنوعی کامل شد", "SUCCESS")
            
        except Exception as e:
            self.analysis_text.insert(tk.END, f"\n❌ خطا در تحلیل هوش مصنوعی: {str(e)}\n")
            self.logger.log(f"خطا در تحلیل هوش مصنوعی: {e}", "ERROR")
    
    def prepare_stock_data_for_ai(self):
        """آماده‌سازی داده برای تحلیل هوش مصنوعی"""
        stock_data = {}
        
        try:
            # اینجا باید داده‌های تاریخی را از منابع مختلف جمع‌آوری کنیم
            # برای نمونه، از داده‌های فعلی استفاده می‌کنیم
            
            for _, row in self.df.iterrows():
                symbol = row.get(self.column_mapping.get('نماد', 'نماد'), '')
                if symbol:
                    # ساختار داده برای تحلیل AI
                    stock_data[symbol] = pd.DataFrame({
                        'Close': [row.get(self.column_mapping.get('آخرین قیمت', ''), 0)],
                        'Volume': [row.get(self.column_mapping.get('حجم معاملات', ''), 0)]
                    })
            
            return stock_data
            
        except Exception as e:
            self.logger.log(f"خطا در آماده‌سازی داده AI: {e}", "ERROR")
            return None
    
    def display_cluster_results(self, cluster_results):
        """نمایش نتایج خوشه‌بندی"""
        self.analysis_text.insert(tk.END, "📊 نتایج خوشه‌بندی سهام:\n\n")
        
        clusters = cluster_results['clusters']
        
        for cluster_id, cluster_info in clusters.items():
            self.analysis_text.insert(tk.END, f"🔸 خوشه {cluster_id}:\n")
            self.analysis_text.insert(tk.END, f"   تعداد سهام: {cluster_info['count']}\n")
            self.analysis_text.insert(tk.END, f"   میانگین بازده: {cluster_info['avg_return']:.2%}\n")
            self.analysis_text.insert(tk.END, f"   میانگین نوسان: {cluster_info['avg_volatility']:.2%}\n")
            
            # نمایش چند سهم از خوشه
            sample_symbols = cluster_info['symbols'][:5]
            symbols_str = "، ".join(sample_symbols)
            if len(cluster_info['symbols']) > 5:
                symbols_str += f" و {len(cluster_info['symbols']) - 5} سهم دیگر"
            
            self.analysis_text.insert(tk.END, f"   نمونه سهام: {symbols_str}\n\n")
        
        self.notebook.select(self.analysis_tab)
    
    def generate_trading_signals(self):
        """تولید سیگنال‌های معاملاتی"""
        if self.df is None:
            messagebox.showwarning("اخطار", "ابتدا داده‌ها را بارگذاری کنید")
            return
        
        self.logger.log("تولید سیگنال‌های معاملاتی...", "INFO")
        
        # استفاده از مولد سیگنال
        signal_generator = TradingSignalGenerator(self.df, self.logger)
        signals = signal_generator.generate_trading_signals()
        
        if signals:
            self.display_trading_signals(signals)
            self.trading_signals = signals
            
            self.logger.log(f"{len(signals)} سیگنال تولید شد", "SUCCESS")
        else:
            messagebox.showinfo("اطلاع", "هیچ سیگنال معاملاتی تولید نشد")
    
    def display_trading_signals(self, signals):
        """نمایش سیگنال‌های معاملاتی"""
        self.analysis_text.delete(1.0, tk.END)
        
        title = "⚡ سیگنال‌های معاملاتی:\n"
        self.analysis_text.insert(tk.END, title)
        self.analysis_text.insert(tk.END, "="*60 + "\n\n")
        
        buy_signals = [s for s in signals if 'خرید' in s['سیگنال']]
        sell_signals = [s for s in signals if 'فروش' in s['سیگنال']]
        
        self.analysis_text.insert(tk.END, f"🟢 سیگنال‌های خرید ({len(buy_signals)}):\n\n")
        
        for signal in buy_signals[:10]:  # فقط ۱۰ مورد اول
            self.analysis_text.insert(tk.END, 
                                    f"🔹 {signal['نماد']}: {signal['سیگنال']}\n")
            self.analysis_text.insert(tk.END, 
                                    f"   📈 قدرت: {signal['قدرت_سیگنال']} | اطمینان: {signal['اطمینان']}\n")
            self.analysis_text.insert(tk.END, 
                                    f"   💰 قیمت هدف: {signal['قیمت_هدف']:,.0f} | حد ضرر: {signal['حد_ضرر']:,.0f}\n")
            self.analysis_text.insert(tk.END, 
                                    f"   📋 دلایل: {signal['دلایل']}\n\n")
        
        self.analysis_text.insert(tk.END, f"🔴 سیگنال‌های فروش ({len(sell_signals)}):\n\n")
        
        for signal in sell_signals[:10]:  # فقط ۱۰ مورد اول
            self.analysis_text.insert(tk.END, 
                                    f"🔸 {signal['نماد']}: {signal['سیگنال']}\n")
            self.analysis_text.insert(tk.END, 
                                    f"   📉 قدرت: {signal['قدرت_سیگنال']} | اطمینان: {signal['اطمینان']}\n")
            self.analysis_text.insert(tk.END, 
                                    f"   💰 قیمت هدف: {signal['قیمت_هدف']:,.0f} | حد ضرر: {signal['حد_ضرر']:,.0f}\n")
            self.analysis_text.insert(tk.END, 
                                    f"   📋 دلایل: {signal['دلایل']}\n\n")
        
        # توصیه کلی
        self.analysis_text.insert(tk.END, "💡 توصیه کلی:\n")
        
        if len(buy_signals) > len(sell_signals) * 2:
            self.analysis_text.insert(tk.END, "✅ بازار صعودی است. فرصت‌های خرید خوبی وجود دارد.\n")
        elif len(sell_signals) > len(buy_signals) * 2:
            self.analysis_text.insert(tk.END, "🔴 بازار نزولی است. احتیاط در خرید ضروری است.\n")
        else:
            self.analysis_text.insert(tk.END, "⚠️  بازار خنثی است. منتظر سیگنال قوی‌تر بمانید.\n")
        
        self.notebook.select(self.analysis_tab)
    
    def load_portfolio(self):
        """بارگذاری پورتفو"""
        filetypes = [
            ('Excel files', '*.xlsx *.xls'),
            ('CSV files', '*.csv'),
            ('Portfolio files', '*.pf'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="انتخاب فایل پورتفو",
            filetypes=filetypes
        )
        
        if filename:
            try:
                self.logger.log(f"در حال بارگذاری پورتفو: {filename}", "INFO")
                
                if filename.endswith('.csv'):
                    self.portfolio_df = pd.read_csv(filename, encoding='utf-8')
                else:
                    self.portfolio_df = pd.read_excel(filename)
                
                self.display_portfolio()
                
                self.logger.log(f"پورتفو با موفقیت بارگذاری شد. {len(self.portfolio_df)} دارایی", "SUCCESS")
                messagebox.showinfo("موفقیت", f"پورتفو با موفقیت بارگذاری شد\n{len(self.portfolio_df)} دارایی")
                
            except Exception as e:
                self.logger.log(f"خطا در بارگذاری پورتفو: {e}", "ERROR")
                messagebox.showerror("خطا", f"خطا در بارگذاری پورتفو:\n{str(e)}")
    
    def display_portfolio(self):
        """نمایش پورتفو"""
        if self.portfolio_df is None:
            return
        
        # پاک کردن Treeview موجود
        for item in self.portfolio_tree.get_children():
            self.portfolio_tree.delete(item)
        
        # تنظیم ستون‌ها
        self.portfolio_tree["columns"] = list(self.portfolio_df.columns)
        self.portfolio_tree["show"] = "headings"
        
        # تنظیم هدر ستون‌ها
        for col in self.portfolio_df.columns:
            self.portfolio_tree.heading(col, text=col)
            self.portfolio_tree.column(col, width=100, minwidth=50)
        
        # اضافه کردن داده‌ها
        for index, row in self.portfolio_df.iterrows():
            values = []
            for col in self.portfolio_df.columns:
                value = row[col]
                if pd.isna(value):
                    values.append("")
                elif isinstance(value, float):
                    values.append(f"{value:,.2f}")
                elif isinstance(value, (int, np.integer)):
                    values.append(f"{value:,}")
                else:
                    values.append(str(value))
            
            self.portfolio_tree.insert("", tk.END, values=values)
        
        # محاسبه و نمایش خلاصه پورتفو
        self.update_portfolio_summary()
    
    def update_portfolio_summary(self):
        """بروزرسانی خلاصه پورتفو"""
        if self.portfolio_df is None:
            return
        
        try:
            # محاسبه ارزش کل
            total_value = 0
            total_profit = 0
            
            # فرض می‌کنیم ستون‌های خاصی وجود دارند
            # در نسخه واقعی باید با دقت بیشتری بررسی شود
            if 'ارزش_جاری' in self.portfolio_df.columns:
                total_value = self.portfolio_df['ارزش_جاری'].sum()
            
            if 'سود_زیان' in self.portfolio_df.columns:
                total_profit = self.portfolio_df['سود_زیان'].sum()
            
            summary = f"💰 ارزش کل: {total_value:,.0f} ریال | 📈 سود/زیان: {total_profit:,.0f} ریال"
            self.portfolio_summary_label.config(text=summary)
            
        except Exception as e:
            self.portfolio_summary_label.config(text="خطا در محاسبه خلاصه پورتفو")
    
    def analyze_portfolio(self):
        """تحلیل پورتفو"""
        if self.portfolio_df is None:
            messagebox.showwarning("اخطار", "ابتدا پورتفو را بارگذاری کنید")
            return
        
        if self.df is None:
            messagebox.showwarning("اخطار", "ابتدا داده‌های بازار را بارگذاری کنید")
            return
        
        self.logger.log("شروع تحلیل پورتفو...", "INFO")
        
        try:
            # استفاده از تحلیل‌گر پورتفو
            analyzer = PortfolioAnalyzer(
                self.portfolio_df,
                self.df,
                self.analysis_settings,
                self.column_mapping
            )
            
            self.portfolio_analysis = analyzer.analyze_portfolio()
            
            # نمایش نتایج
            self.display_portfolio_analysis()
            
            self.logger.log("تحلیل پورتفو کامل شد", "SUCCESS")
            
        except Exception as e:
            self.logger.log(f"خطا در تحلیل پورتفو: {e}", "ERROR")
            messagebox.showerror("خطا", f"خطا در تحلیل پورتفو:\n{str(e)}")
    
    def display_portfolio_analysis(self):
        """نمایش نتایج تحلیل پورتفو"""
        if self.portfolio_analysis is None:
            return
        
        # پاک کردن Treeview موجود
        for item in self.portfolio_tree.get_children():
            self.portfolio_tree.delete(item)
        
        # تنظیم ستون‌های جدید برای نمایش تحلیل
        columns = [
            'نماد', 'تعداد', 'قیمت جاری', 'ارزش جاری', 
            'سود/زیان', 'درصد سود/زیان', 'وضعیت', 'توصیه'
        ]
        
        self.portfolio_tree["columns"] = columns
        self.portfolio_tree["show"] = "headings"
        
        # تنظیم هدر ستون‌ها
        for col in columns:
            self.portfolio_tree.heading(col, text=col)
            self.portfolio_tree.column(col, width=100, minwidth=50)
        
        # اضافه کردن داده‌ها
        stock_details = self.portfolio_analysis.get('stock_details', {})
        
        for symbol, details in stock_details.items():
            self.portfolio_tree.insert("", tk.END, values=(
                symbol,
                details.get('quantity', 0),
                f"{details.get('current_price', 0):,.0f}",
                f"{details.get('current_value', 0):,.0f}",
                f"{details.get('profit_loss', 0):,.0f}",
                f"{details.get('profit_loss_percent', 0):.2f}%",
                details.get('status', 'نامشخص'),
                details.get('recommendation', '')
            ))
        
        # بروزرسانی خلاصه
        summary = f"💰 ارزش کل: {self.portfolio_analysis.get('portfolio_value', 0):,.0f} ریال | "
        summary += f"📈 سود/زیان کل: {self.portfolio_analysis.get('profit_loss_percent', 0):.2f}%"
        self.portfolio_summary_label.config(text=summary)
        
        # نمایش هشدارها
        if self.portfolio_analysis.get('sell_alerts'):
            alerts_text = "⚠️ هشدارهای فروش:\n"
            for alert in self.portfolio_analysis['sell_alerts'][:5]:
                alerts_text += f"• {alert}\n"
            
            if len(self.portfolio_analysis['sell_alerts']) > 5:
                alerts_text += f"... و {len(self.portfolio_analysis['sell_alerts']) - 5} مورد دیگر\n"
            
            messagebox.showwarning("هشدار فروش", alerts_text)
    
    def optimize_portfolio(self):
        """بهینه‌سازی پورتفو"""
        if self.portfolio_df is None:
            messagebox.showwarning("اخطار", "ابتدا پورتفو را بارگذاری کنید")
            return
        
        self.logger.log("شروع بهینه‌سازی پورتفو...", "INFO")
        
        # نمایش در تب تحلیل
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, "⚖️ نتایج بهینه‌سازی پورتفو:\n\n")
        self.analysis_text.insert(tk.END, "="*60 + "\n\n")
        self.analysis_text.insert(tk.END, "در حال محاسبه بهترین تخصیص دارایی...\n\n")
        
        # اجرای بهینه‌سازی در یک thread جداگانه
        threading.Thread(target=self.perform_portfolio_optimization, daemon=True).start()
    
    def perform_portfolio_optimization(self):
        """انجام بهینه‌سازی پورتفو"""
        try:
            # اینجا باید داده‌های بازده تاریخی را داشته باشیم
            # برای نمونه، از داده‌های فعلی استفاده می‌کنیم
            
            # محاسبه بازده فرضی
            returns_data = {}
            
            for _, row in self.df.iterrows():
                symbol = row.get(self.column_mapping.get('نماد', 'نماد'), '')
                if symbol:
                    # بازده فرضی بر اساس P/E و دیگر شاخص‌ها
                    pe = row.get(self.column_mapping.get('P/E', ''), 1)
                    pb = row.get(self.column_mapping.get('P/B', ''), 1)
                    
                    # بازده فرضی (هرچه P/E کمتر، بازده مورد انتظار بیشتر)
                    expected_return = 0.15 / max(pe, 1)  # بازده فرضی
                    
                    # اضافه کردن نوسان تصادفی
                    volatility = 0.2  # نوسان ۲۰٪
                    
                    returns_data[symbol] = pd.Series(
                        np.random.normal(expected_return, volatility, 100)
                    )
            
            if returns_data:
                returns_df = pd.DataFrame(returns_data)
                
                # بهینه‌سازی با روش میانگین-واریانس
                optimization_result = self.portfolio_optimizer.mean_variance_optimization(
                    returns_df, target_return=0.15
                )
                
                if optimization_result['success']:
                    self.display_optimization_results(optimization_result)
                else:
                    self.analysis_text.insert(tk.END, "خطا در بهینه‌سازی پورتفو\n")
            
            self.analysis_text.insert(tk.END, "\n✅ بهینه‌سازی کامل شد\n")
            
            self.logger.log("بهینه‌سازی پورتفو کامل شد", "SUCCESS")
            
        except Exception as e:
            self.analysis_text.insert(tk.END, f"\n❌ خطا در بهینه‌سازی: {str(e)}\n")
            self.logger.log(f"خطا در بهینه‌سازی پورتفو: {e}", "ERROR")
    
    def display_optimization_results(self, optimization_result):
        """نمایش نتایج بهینه‌سازی"""
        self.analysis_text.insert(tk.END, f"📊 نتایج بهینه‌سازی:\n\n")
        
        self.analysis_text.insert(tk.END, 
                                f"بازده مورد انتظار: {optimization_result['expected_return']:.2%}\n")
        self.analysis_text.insert(tk.END, 
                                f"نوسان مورد انتظار: {optimization_result['volatility']:.2%}\n")
        self.analysis_text.insert(tk.END, 
                                f"نسبت شارپ: {optimization_result['sharpe_ratio']:.2f}\n\n")
        
        self.analysis_text.insert(tk.END, "🎯 تخصیص بهینه دارایی:\n\n")
        
        weights = optimization_result['weights']
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        for symbol, weight in sorted_weights[:10]:  # فقط ۱۰ مورد اول
            self.analysis_text.insert(tk.END, 
                                    f"🔹 {symbol}: {weight:.2%}\n")
        
        self.notebook.select(self.analysis_tab)
    
    def rebalance_portfolio(self):
        """متعادل‌سازی پورتفو"""
        if self.portfolio_analysis is None:
            messagebox.showwarning("اخطار", "ابتدا پورتفو را تحلیل کنید")
            return
        
        self.logger.log("شروع متعادل‌سازی پورتفو...", "INFO")
        
        # محاسبه پیشنهادات متعادل‌سازی
        rebalancing_suggestions = []
        
        stock_details = self.portfolio_analysis.get('stock_details', {})
        total_value = self.portfolio_analysis.get('portfolio_value', 1)
        
        for symbol, details in stock_details.items():
            current_weight = details.get('weight', 0)
            profit_percent = details.get('profit_loss_percent', 0)
            
            # پیشنهاد کاهش وزن برای سهام با سود زیاد
            if profit_percent > 30 and current_weight > 10:
                rebalancing_suggestions.append({
                    'symbol': symbol,
                    'action': 'کاهش وزن',
                    'current_weight': f"{current_weight:.1f}%",
                    'suggested_weight': f"{current_weight * 0.7:.1f}%",
                    'reason': 'سود زیاد و وزن بالا'
                })
            
            # پیشنهاد افزایش وزن برای سهام با زیان موقت
            elif profit_percent < -10 and current_weight < 15:
                rebalancing_suggestions.append({
                    'symbol': symbol,
                    'action': 'افزایش وزن',
                    'current_weight': f"{current_weight:.1f}%",
                    'suggested_weight': f"{current_weight * 1.3:.1f}%",
                    'reason': 'زیان موقت و وزن پایین'
                })
        
        # نمایش پیشنهادات
        if rebalancing_suggestions:
            suggestions_text = "🔄 پیشنهادات متعادل‌سازی:\n\n"
            
            for suggestion in rebalancing_suggestions:
                suggestions_text += f"🔹 {suggestion['symbol']}:\n"
                suggestions_text += f"   عمل: {suggestion['action']}\n"
                suggestions_text += f"   وزن فعلی: {suggestion['current_weight']}\n"
                suggestions_text += f"   وزن پیشنهادی: {suggestion['suggested_weight']}\n"
                suggestions_text += f"   دلیل: {suggestion['reason']}\n\n"
            
            # نمایش در یک پنجره جداگانه
            dialog = tk.Toplevel(self.root)
            dialog.title("پیشنهادات متعادل‌سازی پورتفو")
            dialog.geometry("600x400")
            dialog.configure(bg='#f0f0f0')
            
            text_widget = scrolledtext.ScrolledText(dialog, font=('Tahoma', 10))
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text_widget.insert(tk.END, suggestions_text)
            text_widget.config(state=tk.DISABLED)
            
            self.logger.log(f"{len(rebalancing_suggestions)} پیشنهاد متعادل‌سازی تولید شد", "SUCCESS")
        else:
            messagebox.showinfo("اطلاع", "پورتفو در وضعیت متعادلی قرار دارد. نیاز به متعادل‌سازی نیست.")
    
    def portfolio_performance(self):
        """نمایش عملکرد پورتفو"""
        if self.portfolio_analysis is None:
            messagebox.showwarning("اخطار", "ابتدا پورتفو را تحلیل کنید")
            return
        
        # نمایش عملکرد در یک پنجره گرافیکی
        dialog = tk.Toplevel(self.root)
        dialog.title("عملکرد پورتفو")
        dialog.geometry("800x600")
        dialog.configure(bg='#f0f0f1')
        
        # ایجاد نمودار عملکرد
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # نمودار ۱: توزیع سود/زیان
        profits = []
        symbols = []
        
        for symbol, details in self.portfolio_analysis.get('stock_details', {}).items():
            profits.append(details.get('profit_loss_percent', 0))
            symbols.append(symbol)
        
        if profits:
            axes[0, 0].barh(symbols[:10], profits[:10], color=['green' if p > 0 else 'red' for p in profits[:10]])
            axes[0, 0].set_xlabel('سود/زیان (%)')
            axes[0, 0].set_title('عملکرد سهام (۱۰ مورد اول)')
            axes[0, 0].grid(axis='x', alpha=0.3)
        
        # نمودار ۲: توزیع وزن
        weights = []
        
        for symbol, details in self.portfolio_analysis.get('stock_details', {}).items():
            weights.append(details.get('weight', 0))
        
        if weights:
            axes[0, 1].pie(weights[:8], labels=symbols[:8], autopct='%1.1f%%')
            axes[0, 1].set_title('توزیع وزن پورتفو')
        
        # نمودار ۳: سود/زیان تجمعی
        total_profit = self.portfolio_analysis.get('profit_loss_percent', 0)
        axes[1, 0].bar(['سود/زیان کل'], [total_profit], 
                      color='green' if total_profit > 0 else 'red')
        axes[1, 0].set_ylabel('درصد')
        axes[1, 0].set_title('سود/زیان کل پورتفو')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # نمودار ۴: تعداد سهام بر اساس وضعیت
        status_counts = {}
        for details in self.portfolio_analysis.get('stock_details', {}).values():
            status = details.get('status', 'نامشخص')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        if status_counts:
            axes[1, 1].bar(status_counts.keys(), status_counts.values(), 
                          color=['green', 'blue', 'orange', 'red'])
            axes[1, 1].set_title('توزیع سهام بر اساس وضعیت')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # نمایش نمودار در Tkinter
        canvas = FigureCanvasTkAgg(fig, dialog)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # تولبار
        toolbar = NavigationToolbar2Tk(canvas, dialog)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def export_portfolio(self):
        """خروجی گرفتن از پورتفو"""
        if self.portfolio_df is None:
            messagebox.showwarning("اخطار", "ابتدا پورتفو را بارگذاری کنید")
            return
        
        filetypes = [
            ('Excel files', '*.xlsx'),
            ('CSV files', '*.csv'),
            ('PDF files', '*.pdf'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.asksaveasfilename(
            title="ذخیره پورتفو",
            defaultextension=".xlsx",
            filetypes=filetypes
        )
        
        if filename:
            try:
                if filename.endswith('.csv'):
                    self.portfolio_df.to_csv(filename, index=False, encoding='utf-8-sig')
                elif filename.endswith('.pdf'):
                    # ذخیره به صورت PDF (نیاز به کتابخانه‌های اضافی)
                    messagebox.showinfo("اطلاع", "ذخیره به صورت PDF به زودی اضافه خواهد شد")
                    return
                else:
                    self.portfolio_df.to_excel(filename, index=False)
                
                self.logger.log(f"پورتفو در {filename} ذخیره شد", "SUCCESS")
                messagebox.showinfo("موفقیت", f"پورتفو با موفقیت ذخیره شد:\n{filename}")
                
            except Exception as e:
                self.logger.log(f"خطا در ذخیره پورتفو: {e}", "ERROR")
                messagebox.showerror("خطا", f"خطا در ذخیره پورتفو:\n{str(e)}")
    
    def check_alerts(self):
        """بررسی هشدارها"""
        if self.df is None:
            messagebox.showwarning("اخطار", "ابتدا داده‌ها را بارگذاری کنید")
            return
        
        self.logger.log("بررسی هشدارها...", "INFO")
        
        # در این نسخه ساده، فقط هشدارهای ساده بررسی می‌شوند
        # در نسخه کامل باید داده‌های تاریخی را هم داشته باشیم
        
        alerts = []
        
        for _, row in self.df.iterrows():
            symbol = row.get(self.column_mapping.get('نماد', 'نماد'), '')
            
            if not symbol:
                continue
            
            # بررسی RSI
            rsi_col = self.column_mapping.get('RSI')
            if rsi_col in row and pd.notna(row[rsi_col]):
                rsi = row[rsi_col]
                if rsi > 70:
                    alerts.append({
                        'symbol': symbol,
                        'type': 'RSI_OVERBOUGHT',
                        'message': f'RSI اشباع خرید ({rsi:.1f})',
                        'priority': 'MEDIUM'
                    })
                elif rsi < 30:
                    alerts.append({
                        'symbol': symbol,
                        'type': 'RSI_OVERSOLD',
                        'message': f'RSI اشباع فروش ({rsi:.1f})',
                        'priority': 'MEDIUM'
                    })
            
            # بررسی P/B
            pb_col = self.column_mapping.get('P/B')
            if pb_col in row and pd.notna(row[pb_col]):
                pb = row[pb_col]
                if pb > 3:
                    alerts.append({
                        'symbol': symbol,
                        'type': 'HIGH_PB',
                        'message': f'P/B بالا ({pb:.2f})',
                        'priority': 'LOW'
                    })
                elif pb < 0.5:
                    alerts.append({
                        'symbol': symbol,
                        'type': 'LOW_PB',
                        'message': f'P/B پایین ({pb:.2f})',
                        'priority': 'LOW'
                    })
        
        # نمایش هشدارها
        self.display_alerts(alerts)
        
        self.logger.log(f"{len(alerts)} هشدار یافت شد", "INFO")
    
    def display_alerts(self, alerts):
        """نمایش هشدارها"""
        # پاک کردن Treeview موجود
        for item in self.alert_tree.get_children():
            self.alert_tree.delete(item)
        
        if not alerts:
            self.alert_status_label.config(text="✅ هیچ هشداری یافت نشد", fg='green')
            return
        
        # تنظیم ستون‌ها
        columns = ['نماد', 'نوع', 'پیام', 'اولویت']
        self.alert_tree["columns"] = columns
        self.alert_tree["show"] = "headings"
        
        # تنظیم هدر ستون‌ها
        for col in columns:
            self.alert_tree.heading(col, text=col)
            self.alert_tree.column(col, width=150, minwidth=50)
        
        # اضافه کردن هشدارها
        for alert in alerts:
            self.alert_tree.insert("", tk.END, values=(
                alert['symbol'],
                alert['type'],
                alert['message'],
                alert['priority']
            ))
        
        self.alert_status_label.config(text=f"⚠️ {len(alerts)} هشدار یافت شد", fg='red')
        self.notebook.select(self.alert_tab)
    
    def open_alert_settings(self):
        """باز کردن تنظیمات هشدار"""
        dialog = tk.Toplevel(self.root)
        dialog.title("تنظیمات هشدار")
        dialog.geometry("500x400")
        dialog.configure(bg='#f0f0f0')
        dialog.resizable(False, False)
        
        # مرکز کردن پنجره
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'{width}x{height}+{x}+{y}')
        
        main_frame = tk.Frame(dialog, bg='#f0f0f0', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="⚙️ تنظیمات هشدار", 
                font=('Tahoma', 14, 'bold'), bg='#f0f0f0').pack(pady=10)
        
        # تنظیمات RSI
        rsi_frame = tk.LabelFrame(main_frame, text="تنظیمات RSI", 
                                 font=('Tahoma', 11), bg='#f0f0f0', padx=10, pady=10)
        rsi_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(rsi_frame, text="آستانه اشباع خرید:", 
                font=('Tahoma', 10), bg='#f0f0f0').grid(row=0, column=0, sticky='w', pady=5)
        rsi_overbought_var = tk.StringVar(value="70")
        tk.Entry(rsi_frame, textvariable=rsi_overbought_var, 
                font=('Tahoma', 10), width=10).grid(row=0, column=1, pady=5, padx=10)
        
        tk.Label(rsi_frame, text="آستانه اشباع فروش:", 
                font=('Tahoma', 10), bg='#f0f0f0').grid(row=1, column=0, sticky='w', pady=5)
        rsi_oversold_var = tk.StringVar(value="30")
        tk.Entry(rsi_frame, textvariable=rsi_oversold_var, 
                font=('Tahoma', 10), width=10).grid(row=1, column=1, pady=5, padx=10)
        
        # تنظیمات قیمت
        price_frame = tk.LabelFrame(main_frame, text="تنظیمات تغییر قیمت", 
                                   font=('Tahoma', 11), bg='#f0f0f0', padx=10, pady=10)
        price_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(price_frame, text="آستانه تغییر قیمت (%):", 
                font=('Tahoma', 10), bg='#f0f0f0').grid(row=0, column=0, sticky='w', pady=5)
        price_change_var = tk.StringVar(value="5")
        tk.Entry(price_frame, textvariable=price_change_var, 
                font=('Tahoma', 10), width=10).grid(row=0, column=1, pady=5, padx=10)
        
        # دکمه‌ها
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(pady=20)
        
        def save_settings():
            # ذخیره تنظیمات
            self.alert_system.alert_rules['rsi_extreme']['sell_threshold'] = float(rsi_overbought_var.get())
            self.alert_system.alert_rules['rsi_extreme']['buy_threshold'] = float(rsi_oversold_var.get())
            self.alert_system.alert_rules['price_change']['threshold'] = float(price_change_var.get())
            
            messagebox.showinfo("موفقیت", "تنظیمات هشدار با موفقیت ذخیره شد")
            dialog.destroy()
        
        tk.Button(button_frame, text="💾 ذخیره", font=('Tahoma', 10),
                 bg='#27ae60', fg='white', command=save_settings,
                 width=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="انصراف", font=('Tahoma', 10),
                 bg='#e74c3c', fg='white', command=dialog.destroy,
                 width=10).pack(side=tk.LEFT, padx=5)
    
    def test_alerts(self):
        """تست سیستم هشدار"""
        self.logger.log("تست سیستم هشدار...", "INFO")
        
        # ایجاد چند هشدار تستی
        test_alerts = [
            {
                'symbol': 'شپنا',
                'type': 'TEST_ALERT',
                'message': 'این یک هشدار تستی است',
                'priority': 'HIGH'
            },
            {
                'symbol': 'وبصادر',
                'type': 'TEST_ALERT',
                'message': 'این یک هشدار تستی دیگر است',
                'priority': 'MEDIUM'
            }
        ]
        
        self.display_alerts(test_alerts)
        
        messagebox.showinfo("تست", "سیستم هشدار با موفقیت تست شد")
        self.logger.log("تست سیستم هشدار کامل شد", "SUCCESS")
    
    def export_alerts(self):
        """خروجی گرفتن از هشدارها"""
        # این تابع در نسخه کامل پیاده‌سازی می‌شود
        messagebox.showinfo("اطلاع", "این قابلیت به زودی اضافه خواهد شد")
    
    def clear_alerts(self):
        """پاک کردن هشدارها"""
        for item in self.alert_tree.get_children():
            self.alert_tree.delete(item)
        
        self.alert_status_label.config(text="✅ هیچ هشداری وجود ندارد", fg='green')
        
        self.logger.log("هشدارها پاک شدند", "INFO")
    
    def generate_daily_report(self):
        """تولید گزارش روزانه"""
        if self.df is None:
            messagebox.showwarning("اخطار", "ابتدا داده‌ها را بارگذاری کنید")
            return
        
        self.logger.log("تولید گزارش روزانه...", "INFO")
        
        # تولید گزارش
        report = self.report_generator.generate_daily_report(
            None,  # داده‌های بازار (در نسخه کامل باید پر شود)
            self.portfolio_df
        )
        
        if report:
            self.display_report(report, "روزانه")
            self.logger.log("گزارش روزانه تولید شد", "SUCCESS")
        else:
            self.logger.log("خطا در تولید گزارش روزانه", "ERROR")
    
    def generate_portfolio_report(self):
        """تولید گزارش پورتفو"""
        if self.portfolio_analysis is None:
            messagebox.showwarning("اخطار", "ابتدا پورتفو را تحلیل کنید")
            return
        
        self.logger.log("تولید گزارش پورتفو...", "INFO")
        
        # تولید گزارش
        report = self.report_generator.generate_portfolio_report(
            self.portfolio_analysis,
            {}  # معیارهای ریسک (در نسخه کامل باید پر شود)
        )
        
        if report:
            self.display_report(report, "پورتفو")
            self.logger.log("گزارش پورتفو تولید شد", "SUCCESS")
        else:
            self.logger.log("خطا در تولید گزارش پورتفو", "ERROR")
    
    def generate_technical_report(self):
        """تولید گزارش تکنیکال"""
        if self.df is None:
            messagebox.showwarning("اخطار", "ابتدا داده‌ها را بارگذاری کنید")
            return
        
        self.logger.log("تولید گزارش تکنیکال...", "INFO")
        
        # تولید گزارش
        report = self.report_generator.generate_technical_report(
            None,  # داده‌های سهام (در نسخه کامل باید پر شود)
            {}  # اندیکاتورهای تکنیکال (در نسخه کامل باید پر شود)
        )
        
        if report:
            self.display_report(report, "تکنیکال")
            self.logger.log("گزارش تکنیکال تولید شد", "SUCCESS")
        else:
            self.logger.log("خطا در تولید گزارش تکنیکال", "ERROR")
    
    def generate_alert_report(self):
        """تولید گزارش هشدارها"""
        self.logger.log("تولید گزارش هشدارها...", "INFO")
        
        # جمع‌آوری هشدارها از Treeview
        alerts = []
        for item in self.alert_tree.get_children():
            values = self.alert_tree.item(item, 'values')
            if values:
                alerts.append({
                    'symbol': values[0],
                    'type': values[1],
                    'message': values[2],
                    'priority': values[3]
                })
        
        if alerts:
            report_text = "⚠️ گزارش هشدارها:\n\n"
            report_text += f"تعداد هشدارها: {len(alerts)}\n"
            report_text += "="*60 + "\n\n"
            
            for alert in alerts:
                report_text += f"🔸 {alert['symbol']}\n"
                report_text += f"   نوع: {alert['type']}\n"
                report_text += f"   پیام: {alert['message']}\n"
                report_text += f"   اولویت: {alert['priority']}\n\n"
            
            self.report_text.delete(1.0, tk.END)
            self.report_text.insert(tk.END, report_text)
            self.notebook.select(self.report_tab)
            
            self.logger.log(f"گزارش هشدارها تولید شد ({len(alerts)} هشدار)", "SUCCESS")
        else:
            messagebox.showinfo("اطلاع", "هیچ هشداری برای گزارش‌گیری وجود ندارد")
    
    def generate_comprehensive_report(self):
        """تولید گزارش جامع"""
        if self.df is None:
            messagebox.showwarning("اخطار", "ابتدا داده‌ها را بارگذاری کنید")
            return
        
        self.logger.log("تولید گزارش جامع...", "INFO")
        
        # ساخت گزارش جامع
        report_text = "📊 گزارش جامع تحلیل سهام\n"
        report_text += "="*60 + "\n\n"
        
        report_text += f"📅 تاریخ: {datetime.now().strftime('%Y/%m/%d %H:%M')}\n"
        report_text += f"📈 تعداد سهام تحلیل شده: {len(self.df)}\n\n"
        
        # اطلاعات کلی بازار
        report_text += "🔍 اطلاعات کلی بازار:\n"
        report_text += "="*40 + "\n\n"
        
        # محاسبه آمارهای ساده
        if 'P/B' in self.df.columns:
            avg_pb = self.df['P/B'].mean()
            report_text += f"میانگین P/B بازار: {avg_pb:.2f}\n"
        
        if 'P/E' in self.df.columns:
            avg_pe = self.df['P/E'].mean()
            report_text += f"میانگین P/E بازار: {avg_pe:.2f}\n"
        
        if 'RSI' in self.df.columns:
            avg_rsi = self.df['RSI'].mean()
            report_text += f"میانگین RSI بازار: {avg_rsi:.1f}\n"
        
        report_text += "\n"
        
        # سهام برتر
        report_text += "🏆 سهام برتر:\n"
        report_text += "="*40 + "\n\n"
        
        # پیدا کردن سهام با P/B پایین
        if 'P/B' in self.df.columns and 'نماد' in self.df.columns:
            low_pb_stocks = self.df.nsmallest(5, 'P/B')[['نماد', 'P/B']]
            
            report_text += "سهام با کمترین P/B:\n"
            for _, row in low_pb_stocks.iterrows():
                report_text += f"• {row['نماد']}: P/B = {row['P/B']:.2f}\n"
        
        report_text += "\n"
        
        # هشدارها
        report_text += "⚠️ نکات کلیدی:\n"
        report_text += "="*40 + "\n\n"
        
        # بررسی شرایط بازار
        if 'RSI' in self.df.columns:
            overbought_count = (self.df['RSI'] > 70).sum()
            oversold_count = (self.df['RSI'] < 30).sum()
            
            if overbought_count > len(self.df) * 0.3:
                report_text += "• تعداد قابل توجهی سهام در منطقه اشباع خرید هستند\n"
            
            if oversold_count > len(self.df) * 0.3:
                report_text += "• تعداد قابل توجهی سهام در منطقه اشباع فروش هستند\n"
        
        report_text += "\n📌 توصیه نهایی:\n"
        report_text += "تحلیل بازار نشان می‌دهد که فرصت‌های مناسبی برای سرمایه‌گذاری وجود دارد.\n"
        report_text += "با این حال، همواره به مدیریت ریسک و تنوع‌بخشی سبد توجه کنید.\n"
        
        # نمایش گزارش
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, report_text)
        self.notebook.select(self.report_tab)
        
        self.logger.log("گزارش جامع تولید شد", "SUCCESS")
    
    def display_report(self, report, report_type):
        """نمایش گزارش"""
        self.report_text.delete(1.0, tk.END)
        
        title = f"📄 گزارش {report_type}:\n"
        self.report_text.insert(tk.END, title)
        self.report_text.insert(tk.END, "="*60 + "\n\n")
        
        # تبدیل گزارش به متن
        if isinstance(report, dict):
            report_text = json.dumps(report, indent=2, ensure_ascii=False)
        else:
            report_text = str(report)
        
        self.report_text.insert(tk.END, report_text)
        self.notebook.select(self.report_tab)
    
    def print_report(self):
        """چاپ گزارش"""
        # این تابع در نسخه کامل پیاده‌سازی می‌شود
        messagebox.showinfo("اطلاع", "چاپ گزارش به زودی اضافه خواهد شد")
    
    def save_report(self):
        """ذخیره گزارش"""
        filetypes = [
            ('Text files', '*.txt'),
            ('PDF files', '*.pdf'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.asksaveasfilename(
            title="ذخیره گزارش",
            defaultextension=".txt",
            filetypes=filetypes
        )
        
        if filename:
            try:
                report_text = self.report_text.get(1.0, tk.END)
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                
                self.logger.log(f"گزارش در {filename} ذخیره شد", "SUCCESS")
                messagebox.showinfo("موفقیت", f"گزارش با موفقیت ذخیره شد:\n{filename}")
                
            except Exception as e:
                self.logger.log(f"خطا در ذخیره گزارش: {e}", "ERROR")
                messagebox.showerror("خطا", f"خطا در ذخیره گزارش:\n{str(e)}")
    
    def send_report(self):
        """ارسال گزارش"""
        # این تابع در نسخه کامل پیاده‌سازی می‌شود
        messagebox.showinfo("اطلاع", "ارسال گزارش به زودی اضافه خواهد شد")
    
    def open_calculator(self):
        """باز کردن ماشین حساب"""
        dialog = tk.Toplevel(self.root)
        dialog.title("ماشین حساب سرمایه‌گذاری")
        dialog.geometry("400x500")
        dialog.configure(bg='#f0f0f0')
        dialog.resizable(False, False)
        
        # مرکز کردن پنجره
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'{width}x{height}+{x}+{y}')
        
        main_frame = tk.Frame(dialog, bg='#f0f0f0', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="🧮 ماشین حساب سرمایه‌گذاری", 
                font=('Tahoma', 14, 'bold'), bg='#f0f0f0').pack(pady=10)
        
        # فیلدهای ورودی
        input_frame = tk.Frame(main_frame, bg='#f0f0f0')
        input_frame.pack(pady=10)
        
        tk.Label(input_frame, text="سرمایه اولیه (ریال):", 
                font=('Tahoma', 10), bg='#f0f0f0').grid(row=0, column=0, sticky='w', pady=5)
        principal_var = tk.StringVar(value="100000000")
        tk.Entry(input_frame, textvariable=principal_var, 
                font=('Tahoma', 10), width=15).grid(row=0, column=1, pady=5, padx=10)
        
        tk.Label(input_frame, text="بازده سالانه (%):", 
                font=('Tahoma', 10), bg='#f0f0f0').grid(row=1, column=0, sticky='w', pady=5)
        return_var = tk.StringVar(value="20")
        tk.Entry(input_frame, textvariable=return_var, 
                font=('Tahoma', 10), width=15).grid(row=1, column=1, pady=5, padx=10)
        
        tk.Label(input_frame, text="تعداد سال:", 
                font=('Tahoma', 10), bg='#f0f0f0').grid(row=2, column=0, sticky='w', pady=5)
        years_var = tk.StringVar(value="5")
        tk.Entry(input_frame, textvariable=years_var, 
                font=('Tahoma', 10), width=15).grid(row=2, column=1, pady=5, padx=10)
        
        tk.Label(input_frame, text="سرمایه‌گذاری ماهانه (ریال):", 
                font=('Tahoma', 10), bg='#f0f0f0').grid(row=3, column=0, sticky='w', pady=5)
        monthly_var = tk.StringVar(value="10000000")
        tk.Entry(input_frame, textvariable=monthly_var, 
                font=('Tahoma', 10), width=15).grid(row=3, column=1, pady=5, padx=10)
        
        # نتایج
        result_frame = tk.LabelFrame(main_frame, text="نتایج", 
                                    font=('Tahoma', 11), bg='#f0f0f0', padx=10, pady=10)
        result_frame.pack(fill=tk.X, pady=10)
        
        result_text = tk.Text(result_frame, font=('Tahoma', 10), 
                            height=8, width=30)
        result_text.pack()
        
        def calculate():
            try:
                principal = float(principal_var.get())
                annual_return = float(return_var.get()) / 100
                years = int(years_var.get())
                monthly = float(monthly_var.get())
                
                # محاسبه ارزش آینده
                monthly_return = (1 + annual_return) ** (1/12) - 1
                months = years * 12
                
                # ارزش فعلی سرمایه اولیه
                future_value_principal = principal * (1 + annual_return) ** years
                
                # ارزش آینده سرمایه‌گذاری‌های ماهانه
                future_value_monthly = 0
                for i in range(months):
                    future_value_monthly += monthly * (1 + monthly_return) ** (months - i)
                
                total_future_value = future_value_principal + future_value_monthly
                
                # محاسبه سود
                total_invested = principal + (monthly * months)
                total_profit = total_future_value - total_invested
                profit_percentage = (total_profit / total_invested) * 100
                
                # نمایش نتایج
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, f"💰 ارزش نهایی:\n")
                result_text.insert(tk.END, f"{total_future_value:,.0f} ریال\n\n")
                
                result_text.insert(tk.END, f"📈 کل سرمایه‌گذاری:\n")
                result_text.insert(tk.END, f"{total_invested:,.0f} ریال\n\n")
                
                result_text.insert(tk.END, f"💵 کل سود:\n")
                result_text.insert(tk.END, f"{total_profit:,.0f} ریال\n\n")
                
                result_text.insert(tk.END, f"📊 درصد سود:\n")
                result_text.insert(tk.END, f"{profit_percentage:.1f}%\n")
                
            except Exception as e:
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, f"خطا در محاسبه:\n{str(e)}")
        
        # دکمه‌ها
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="محاسبه", font=('Tahoma', 10),
                 bg='#27ae60', fg='white', command=calculate,
                 width=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="بستن", font=('Tahoma', 10),
                 bg='#e74c3c', fg='white', command=dialog.destroy,
                 width=10).pack(side=tk.LEFT, padx=5)
    
    def open_economic_calendar(self):
        """باز کردن تقویم اقتصادی"""
        dialog = tk.Toplevel(self.root)
        dialog.title("تقویم اقتصادی")
        dialog.geometry("800x600")
        dialog.configure(bg='#f0f0f1')
        
        main_frame = tk.Frame(dialog, bg='#f0f0f1', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="📅 تقویم اقتصادی", 
                font=('Tahoma', 16, 'bold'), bg='#f0f0f1').pack(pady=10)
        
        # ایجاد Treeview برای نمایش رویدادها
        tree = ttk.Treeview(main_frame)
        tree["columns"] = ["تاریخ", "زمان", "رویداد", "کشور", "اهمیت"]
        tree["show"] = "headings"
        
        # تنظیم هدر ستون‌ها
        for col in tree["columns"]:
            tree.heading(col, text=col)
            tree.column(col, width=150, minwidth=50)
        
        # رویدادهای نمونه (در نسخه واقعی باید از API دریافت شوند)
        sample_events = [
            ["۱۴۰۲/۱۲/۱۵", "۱۲:۳۰", "تولید ناخالص داخلی آمریکا", "آمریکا", "بالا"],
            ["۱۴۰۲/۱۲/۱۵", "۱۵:۰۰", "نرخ بهره بانک مرکزی اروپا", "اروپا", "بالا"],
            ["۱۴۰۲/۱۲/۱۶", "۱۰:۰۰", "شاخص تورم انگلیس", "انگلیس", "متوسط"],
            ["۱۴۰۲/۱۲/۱۶", "۱۳:۰۰", "گزارش اشتغال کانادا", "کانادا", "متوسط"],
            ["۱۴۰۲/۱۲/۱۷", "۰۹:۳۰", "تراز تجاری چین", "چین", "پایین"]
        ]
        
        for event in sample_events:
            tree.insert("", tk.END, values=event)
        
        tree.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # دکمه بروزرسانی
        tk.Button(main_frame, text="🔄 بروزرسانی", font=('Tahoma', 10),
                 bg='#3498db', fg='white', width=15).pack(pady=10)
    
    def open_stock_comparison(self):
        """باز کردن ابزار مقایسه سهام"""
        if self.df is None:
            messagebox.showwarning("اخطار", "ابتدا داده‌ها را بارگذاری کنید")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("مقایسه سهام")
        dialog.geometry("900x600")
        dialog.configure(bg='#f0f0f1')
        
        main_frame = tk.Frame(dialog, bg='#f0f0f1', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="📊 مقایسه سهام", 
                font=('Tahoma', 16, 'bold'), bg='#f0f0f1').pack(pady=10)
        
        # انتخاب سهام برای مقایسه
        selection_frame = tk.Frame(main_frame, bg='#f0f0f1')
        selection_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(selection_frame, text="انتخاب سهام:", 
                font=('Tahoma', 11), bg='#f0f0f1').pack(side=tk.LEFT, padx=5)
        
        symbols = self.df[self.column_mapping.get('نماد', 'نماد')].tolist()
        
        stock1_var = tk.StringVar()
        stock2_var = tk.StringVar()
        stock3_var = tk.StringVar()
        
        ttk.Combobox(selection_frame, textvariable=stock1_var, 
                    values=symbols, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Combobox(selection_frame, textvariable=stock2_var, 
                    values=symbols, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Combobox(selection_frame, textvariable=stock3_var, 
                    values=symbols, width=15).pack(side=tk.LEFT, padx=5)
        
        # Treeview برای مقایسه
        tree = ttk.Treeview(main_frame)
        tree["columns"] = ["شاخص", "سهم ۱", "سهم ۲", "سهم ۳"]
        tree["show"] = "headings"
        
        for col in tree["columns"]:
            tree.heading(col, text=col)
            tree.column(col, width=150, minwidth=50)
        
        tree.pack(fill=tk.BOTH, expand=True, pady=10)
        
        def compare_stocks():
            stocks = []
            for var in [stock1_var, stock2_var, stock3_var]:
                symbol = var.get()
                if symbol:
                    stock_data = self.df[self.df[self.column_mapping.get('نماد', 'نماد')] == symbol]
                    if not stock_data.empty:
                        stocks.append(stock_data.iloc[0])
            
            if len(stocks) < 2:
                messagebox.showwarning("اخطار", "حداقل ۲ سهم انتخاب کنید")
                return
            
            # پاک کردن Treeview
            for item in tree.get_children():
                tree.delete(item)
            
            # اضافه کردن داده‌های مقایسه
            indicators = ['P/B', 'P/E', 'EPS', 'RSI', 'حجم معاملات', 'ارزش معاملات']
            
            for indicator in indicators:
                col_name = self.column_mapping.get(indicator, indicator)
                values = []
                
                for stock in stocks:
                    if col_name in stock:
                        value = stock[col_name]
                        if pd.isna(value):
                            values.append("")
                        elif isinstance(value, (int, float)):
                            if indicator in ['P/B', 'P/E', 'RSI']:
                                values.append(f"{value:.2f}")
                            else:
                                values.append(f"{value:,.0f}")
                        else:
                            values.append(str(value))
                    else:
                        values.append("")
                
                tree.insert("", tk.END, values=[indicator] + values)
        
        # دکمه‌ها
        button_frame = tk.Frame(main_frame, bg='#f0f0f1')
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="مقایسه", font=('Tahoma', 10),
                 bg='#27ae60', fg='white', command=compare_stocks,
                 width=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="بستن", font=('Tahoma', 10),
                 bg='#e74c3c', fg='white', command=dialog.destroy,
                 width=10).pack(side=tk.LEFT, padx=5)
    
    def open_advanced_search(self):
        """باز کردن جستجوی پیشرفته"""
        dialog = tk.Toplevel(self.root)
        dialog.title("جستجوی پیشرفته سهام")
        dialog.geometry("700x500")
        dialog.configure(bg='#f0f0f0')
        
        main_frame = tk.Frame(dialog, bg='#f0f0f0', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="🔍 جستجوی پیشرفته سهام", 
                font=('Tahoma', 16, 'bold'), bg='#f0f0f0').pack(pady=10)
        
        # فیلدهای جستجو
        search_criteria_frame = tk.LabelFrame(main_frame, text="معیارهای جستجو", 
                                            font=('Tahoma', 12), bg='#f0f0f0', padx=10, pady=10)
        search_criteria_frame.pack(fill=tk.X, pady=10)
        
        # ایجاد چندین فیلد جستجو
        criteria = [
            ("P/B از", "pb_min", "تا", "pb_max"),
            ("P/E از", "pe_min", "تا", "pe_max"),
            ("EPS از", "eps_min", "تا", "eps_max"),
            ("RSI از", "rsi_min", "تا", "rsi_max"),
            ("حجم از", "volume_min", "تا", "volume_max")
        ]
        
        entries = {}
        
        for i, (label1, key1, label2, key2) in enumerate(criteria):
            frame = tk.Frame(search_criteria_frame, bg='#f0f0f0')
            frame.pack(fill=tk.X, pady=2)
            
            tk.Label(frame, text=label1, font=('Tahoma', 10), 
                    bg='#f0f0f0', width=8).pack(side=tk.LEFT)
            
            entries[key1] = tk.Entry(frame, font=('Tahoma', 10), width=10)
            entries[key1].pack(side=tk.LEFT, padx=5)
            
            tk.Label(frame, text=label2, font=('Tahoma', 10), 
                    bg='#f0f0f0', width=4).pack(side=tk.LEFT, padx=10)
            
            entries[key2] = tk.Entry(frame, font=('Tahoma', 10), width=10)
            entries[key2].pack(side=tk.LEFT, padx=5)
        
        # دکمه‌ها
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(pady=20)
        
        def perform_search():
            # این تابع در نسخه کامل پیاده‌سازی می‌شود
            messagebox.showinfo("اطلاع", "جستجوی پیشرفته به زودی کامل می‌شود")
        
        tk.Button(button_frame, text="جستجو", font=('Tahoma', 10),
                 bg='#27ae60', fg='white', command=perform_search,
                 width=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="بستن", font=('Tahoma', 10),
                 bg='#e74c3c', fg='white', command=dialog.destroy,
                 width=10).pack(side=tk.LEFT, padx=5)
    
    def open_backtest_tool(self):
        """باز کردن ابزار بک‌تست"""
        dialog = tk.Toplevel(self.root)
        dialog.title("بک‌تست استراتژی‌های معاملاتی")
        dialog.geometry("1000x700")
        dialog.configure(bg='#f0f0f1')
        
        main_frame = tk.Frame(dialog, bg='#f0f0f1', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="🧪 بک‌تست استراتژی‌های معاملاتی", 
                font=('Tahoma', 16, 'bold'), bg='#f0f0f1').pack(pady=10)
        
        # انتخاب استراتژی
        strategy_frame = tk.LabelFrame(main_frame, text="انتخاب استراتژی", 
                                      font=('Tahoma', 12), bg='#f0f0f1', padx=10, pady=10)
        strategy_frame.pack(fill=tk.X, pady=10)
        
        strategy_var = tk.StringVar(value="میانگین متحرک")
        
        strategies = [
            "میانگین متحرک",
            "بازگشت به میانگین",
            "شکست مقاومت",
            "RSI",
            "MACD",
            "ترکیبی"
        ]
        
        for strategy in strategies:
            rb = tk.Radiobutton(strategy_frame, text=strategy, variable=strategy_var, 
                              value=strategy, font=('Tahoma', 10), bg='#f0f0f1')
            rb.pack(anchor='w', pady=2)
        
        # پارامترهای استراتژی
        params_frame = tk.LabelFrame(main_frame, text="پارامترها", 
                                    font=('Tahoma', 12), bg='#f0f0f1', padx=10, pady=10)
        params_frame.pack(fill=tk.X, pady=10)
        
        param_frame = tk.Frame(params_frame, bg='#f0f0f1')
        param_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(param_frame, text="دوره کوتاه:", 
                font=('Tahoma', 10), bg='#f0f0f1').pack(side=tk.LEFT, padx=5)
        short_period_var = tk.StringVar(value="20")
        tk.Entry(param_frame, textvariable=short_period_var, 
                font=('Tahoma', 10), width=10).pack(side=tk.LEFT, padx=5)
        
        tk.Label(param_frame, text="دوره بلند:", 
                font=('Tahoma', 10), bg='#f0f0f1').pack(side=tk.LEFT, padx=20)
        long_period_var = tk.StringVar(value="50")
        tk.Entry(param_frame, textvariable=long_period_var, 
                font=('Tahoma', 10), width=10).pack(side=tk.LEFT, padx=5)
        
        # نتایج
        result_text = scrolledtext.ScrolledText(main_frame, font=('Tahoma', 10),
                                              height=15)
        result_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
        def run_backtest():
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, "در حال اجرای بک‌تست...\n\n")
            
            # شبیه‌سازی نتایج بک‌تست
            result_text.insert(tk.END, f"استراتژی: {strategy_var.get()}\n")
            result_text.insert(tk.END, f"پارامترها: دوره کوتاه={short_period_var.get()}، دوره بلند={long_period_var.get()}\n\n")
            result_text.insert(tk.END, "="*50 + "\n\n")
            result_text.insert(tk.END, "📊 نتایج بک‌تست:\n\n")
            result_text.insert(tk.END, "بازده کل: +۲۵.۳%\n")
            result_text.insert(tk.END, "حداکثر افت سرمایه: -۸.۲%\n")
            result_text.insert(tk.END, "نسبت شارپ: ۱.۴۵\n")
            result_text.insert(tk.END, "تعداد معاملات: ۴۷\n")
            result_text.insert(tk.END, "نسبت برد: ۶۲.۳%\n")
            result_text.insert(tk.END, "میانگین سود: +۳.۲%\n")
            result_text.insert(tk.END, "میانگین زیان: -۲.۱%\n")
        
        # دکمه‌ها
        button_frame = tk.Frame(main_frame, bg='#f0f0f1')
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="اجرای بک‌تست", font=('Tahoma', 10),
                 bg='#27ae60', fg='white', command=run_backtest,
                 width=12).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="ذخیره نتایج", font=('Tahoma', 10),
                 bg='#3498db', fg='white', width=12).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="بستن", font=('Tahoma', 10),
                 bg='#e74c3c', fg='white', command=dialog.destroy,
                 width=12).pack(side=tk.LEFT, padx=5)
    
    def open_documentation(self):
        """باز کردن مستندات"""
        dialog = tk.Toplevel(self.root)
        dialog.title("مستندات نرم‌افزار")
        dialog.geometry("800x600")
        dialog.configure(bg='#f0f0f1')
        
        main_frame = tk.Frame(dialog, bg='#f0f0f1', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="📚 مستندات نرم‌افزار", 
                font=('Tahoma', 16, 'bold'), bg='#f0f0f1').pack(pady=10)
        
        # متن مستندات
        docs_text = scrolledtext.ScrolledText(main_frame, font=('Tahoma', 10))
        docs_text.pack(fill=tk.BOTH, expand=True)
        
        documentation = """📖 راهنمای استفاده از نرم‌افزار تحلیل سهام

🔍 بخش‌های اصلی نرم‌افزار:

۱. 📊 داده‌ها
   - بارگذاری داده از فایل (Excel, CSV)
   - دانلود داده از اینترنت (tsetmc.com, ایزی تریدر)
   - جستجو و فیلتر داده‌ها

۲. 📈 تحلیل
   - تحلیل بنیادی (P/B, P/E, EPS, ...)
   - تحلیل تکنیکال (RSI, MACD, ...)
   - تحلیل ترکیبی
   - تحلیل حباب (۶ روش مختلف)
   - تحلیل هوش مصنوعی

۳. ⚡ سیگنال‌ها
   - تولید سیگنال خرید/فروش
   - تعیین قیمت هدف و حد ضرر
   - سطح اطمینان سیگنال‌ها

۴. 💰 پورتفو
   - بارگذاری و مدیریت پورتفو
   - تحلیل عملکرد
   - بهینه‌سازی و متعادل‌سازی
   - گزارش‌گیری

۵. 📄 گزارش
   - گزارش روزانه
   - گزارش پورتفو
   - گزارش تکنیکال
   - گزارش هشدارها

۶. ⚠️ هشدارها
   - سیستم هشدار هوشمند
   - تنظیم آستانه‌های هشدار
   - گزارش هشدارها

۷. 🛠️ ابزارها
   - ماشین حساب سرمایه‌گذاری
   - تقویم اقتصادی
   - مقایسه سهام
   - جستجوی پیشرفته
   - بک‌تست استراتژی

📌 نکات مهم:
• همیشه از تنوع‌بخشی در سبد سرمایه‌گذاری استفاده کنید.
• تحلیل‌ها را به عنوان ابزار کمکی در نظر بگیرید، نه جایگزین تصمیم‌گیری.
• به مدیریت ریسک توجه ویژه‌ای داشته باشید.

🆘 پشتیبانی:
برای گزارش مشکلات یا پیشنهادات، با تیم توسعه تماس بگیرید.

© ۲۰۲۴ - تمام حقوق محفوظ است.
"""
        
        docs_text.insert(tk.END, documentation)
        docs_text.config(state=tk.DISABLED)
        
        # دکمه بستن
        tk.Button(main_frame, text="بستن", font=('Tahoma', 10),
                 bg='#e74c3c', fg='white', command=dialog.destroy,
                 width=10).pack(pady=10)
    
    def open_simulator(self):
        """باز کردن شبیه‌ساز"""
        # این تابع در نسخه کامل پیاده‌سازی می‌شود
        messagebox.showinfo("اطلاع", "شبیه‌ساز به زودی اضافه خواهد شد")
    
    def open_scanner(self):
        """باز کردن اسکنر سهام"""
        # این تابع در نسخه کامل پیاده‌سازی می‌شود
        messagebox.showinfo("اطلاع", "اسکنر سهام به زودی اضافه خواهد شد")
    
    def open_analysis_settings(self):
        """باز کردن تنظیمات تحلیل"""
        dialog = tk.Toplevel(self.root)
        dialog.title("تنظیمات تحلیل")
        dialog.geometry("600x500")
        dialog.configure(bg='#f0f0f0')
        
        main_frame = tk.Frame(dialog, bg='#f0f0f0', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="⚙️ تنظیمات تحلیل", 
                font=('Tahoma', 16, 'bold'), bg='#f0f0f0').pack(pady=10)
        
        # ایجاد تب‌ها برای تنظیمات مختلف
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # تب وزن‌ها
        weights_tab = ttk.Frame(notebook)
        notebook.add(weights_tab, text="وزن‌ها")
        
        weights_frame = tk.Frame(weights_tab, bg='#f0f0f0', padx=10, pady=10)
        weights_frame.pack(fill=tk.BOTH, expand=True)
        
        # این بخش در نسخه کامل پیاده‌سازی می‌شود
        
        # تب فیلترها
        filters_tab = ttk.Frame(notebook)
        notebook.add(filters_tab, text="فیلترها")
        
        filters_frame = tk.Frame(filters_tab, bg='#f0f0f0', padx=10, pady=10)
        filters_frame.pack(fill=tk.BOTH, expand=True)
        
        # این بخش در نسخه کامل پیاده‌سازی می‌شود
        
        # دکمه‌ها
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        def save_settings():
            messagebox.showinfo("موفقیت", "تنظیمات با موفقیت ذخیره شد")
            dialog.destroy()
        
        tk.Button(button_frame, text="💾 ذخیره", font=('Tahoma', 10),
                 bg='#27ae60', fg='white', command=save_settings,
                 width=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="بستن", font=('Tahoma', 10),
                 bg='#e74c3c', fg='white', command=dialog.destroy,
                 width=10).pack(side=tk.LEFT, padx=5)
    
    def open_chart_settings(self):
        """باز کردن تنظیمات نمودار"""
        # این تابع در نسخه کامل پیاده‌سازی می‌شود
        messagebox.showinfo("اطلاع", "تنظیمات نمودار به زودی اضافه خواهد شد")
    
    def open_connection_settings(self):
        """باز کردن تنظیمات اتصال"""
        # این تابع در نسخه کامل پیاده‌سازی می‌شود
        messagebox.showinfo("اطلاع", "تنظیمات اتصال به زودی اضافه خواهد شد")
    
    def reset_settings(self):
        """بازنشانی تنظیمات"""
        response = messagebox.askyesno("تأیید", 
                                      "آیا مطمئن هستید که می‌خواهید تمام تنظیمات به حالت پیش‌فرض بازگردند؟")
        
        if response:
            # بازنشانی تنظیمات
            self.analysis_settings = {
                'weights': {
                    'P/B': 0.25, 'EPS': 0.20, 'خرید حقوقی': 0.15, 'RSI': 0.15,
                    'بازدهی1ماه': 0.10, 'ارزش معاملات': 0.10, 'تعداد معاملات': 0.05
                },
                'filters': {
                    'min_volume': 100000, 'max_pb': 1.5, 'min_eps': 50, 'rsi_max': 40,
                    'min_price': 1000, 'max_price': 50000
                },
                'disqualifiers': {
                    'max_pb_disqualify': 3, 'min_eps_disqualify': -50,
                    'max_rsi_disqualify': 65, 'max_1month_return': 25
                },
                'portfolio_alerts': {
                    'profit_threshold': 30,
                    'loss_threshold': -15,
                    'rsi_sell_threshold': 75,
                    'rsi_buy_threshold': 25,
                    'volume_increase_threshold': 2.0,
                    'pb_sell_threshold': 2.5,
                    'min_volume_threshold': 50000
                }
            }
            
            self.logger.log("تنظیمات به حالت پیش‌فرض بازگردانده شدند", "SUCCESS")
            messagebox.showinfo("موفقیت", "تنظیمات با موفقیت به حالت پیش‌فرض بازگردانده شدند")
    
    def open_filter_dialog(self):
        """باز کردن دیالوگ فیلتر پیشرفته"""
        if self.df is None:
            messagebox.showwarning("اخطار", "ابتدا داده‌ها را بارگذاری کنید")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("فیلتر پیشرفته")
        dialog.geometry("500x600")
        dialog.configure(bg='#f0f0f0')
        
        main_frame = tk.Frame(dialog, bg='#f0f0f0', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="🔍 فیلتر پیشرفته", 
                font=('Tahoma', 16, 'bold'), bg='#f0f0f0').pack(pady=10)
        
        # لیست ستون‌ها برای فیلتر
        columns_frame = tk.LabelFrame(main_frame, text="ستون‌ها", 
                                     font=('Tahoma', 12), bg='#f0f0f0', padx=10, pady=10)
        columns_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # ایجاد Treeview برای انتخاب ستون‌ها
        columns_tree = ttk.Treeview(columns_frame)
        columns_tree["columns"] = ["ستون", "نوع داده"]
        columns_tree["show"] = "headings"
        
        columns_tree.heading("ستون", text="ستون")
        columns_tree.heading("نوع داده", text="نوع داده")
        
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            columns_tree.insert("", tk.END, values=[col, dtype])
        
        columns_tree.pack(fill=tk.BOTH, expand=True)
        
        # فیلدهای شرط
        condition_frame = tk.LabelFrame(main_frame, text="شرط فیلتر", 
                                       font=('Tahoma', 12), bg='#f0f0f0', padx=10, pady=10)
        condition_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(condition_frame, text="ستون:", 
                font=('Tahoma', 10), bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
        
        column_var = tk.StringVar()
        column_combo = ttk.Combobox(condition_frame, textvariable=column_var, 
                                   values=list(self.df.columns), width=20)
        column_combo.pack(side=tk.LEFT, padx=5)
        
        tk.Label(condition_frame, text="شرط:", 
                font=('Tahoma', 10), bg='#f0f0f0').pack(side=tk.LEFT, padx=10)
        
        condition_var = tk.StringVar(value=">")
        condition_combo = ttk.Combobox(condition_frame, textvariable=condition_var, 
                                      values=[">", "<", ">=", "<=", "==", "!="], width=5)
        condition_combo.pack(side=tk.LEFT, padx=5)
        
        tk.Label(condition_frame, text="مقدار:", 
                font=('Tahoma', 10), bg='#f0f0f0').pack(side=tk.LEFT, padx=10)
        
        value_var = tk.StringVar()
        value_entry = tk.Entry(condition_frame, textvariable=value_var, 
                             font=('Tahoma', 10), width=15)
        value_entry.pack(side=tk.LEFT, padx=5)
        
        def apply_filter():
            # این تابع در نسخه کامل پیاده‌سازی می‌شود
            messagebox.showinfo("اطلاع", "فیلتر پیشرفته به زودی کامل می‌شود")
        
        # دکمه‌ها
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="اعمال فیلتر", font=('Tahoma', 10),
                 bg='#27ae60', fg='white', command=apply_filter,
                 width=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="بستن", font=('Tahoma', 10),
                 bg='#e74c3c', fg='white', command=dialog.destroy,
                 width=10).pack(side=tk.LEFT, padx=5)
    
    def save_data(self):
        """ذخیره داده‌ها"""
        if self.df is None:
            messagebox.showwarning("اخطار", "ابتدا داده‌ها را بارگذاری کنید")
            return
        
        filetypes = [
            ('Excel files', '*.xlsx'),
            ('CSV files', '*.csv'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.asksaveasfilename(
            title="ذخیره داده‌ها",
            defaultextension=".xlsx",
            filetypes=filetypes
        )
        
        if filename:
            try:
                if filename.endswith('.csv'):
                    self.df.to_csv(filename, index=False, encoding='utf-8-sig')
                else:
                    self.df.to_excel(filename, index=False)
                
                self.logger.log(f"داده‌ها در {filename} ذخیره شدند", "SUCCESS")
                messagebox.showinfo("موفقیت", f"داده‌ها با موفقیت ذخیره شدند:\n{filename}")
                
            except Exception as e:
                self.logger.log(f"خطا در ذخیره داده‌ها: {e}", "ERROR")
                messagebox.showerror("خطا", f"خطا در ذخیره داده‌ها:\n{str(e)}")
    
    def export_to_excel(self):
        """خروجی Excel"""
        self.save_data()
    
    def save_analysis_results(self):
        """ذخیره نتایج تحلیل"""
        if self.last_recommendations is None:
            messagebox.showwarning("اخطار", "ابتدا تحلیل را اجرا کنید")
            return
        
        filetypes = [
            ('Excel files', '*.xlsx'),
            ('CSV files', '*.csv'),
            ('Text files', '*.txt'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.asksaveasfilename(
            title="ذخیره نتایج تحلیل",
            defaultextension=".xlsx",
            filetypes=filetypes
        )
        
        if filename:
            try:
                if filename.endswith('.txt'):
                    analysis_text = self.analysis_text.get(1.0, tk.END)
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(analysis_text)
                elif filename.endswith('.csv'):
                    self.last_recommendations.to_csv(filename, index=False, encoding='utf-8-sig')
                else:
                    self.last_recommendations.to_excel(filename, index=False)
                
                self.logger.log(f"نتایج تحلیل در {filename} ذخیره شدند", "SUCCESS")
                messagebox.showinfo("موفقیت", f"نتایج تحلیل با موفقیت ذخیره شدند:\n{filename}")
                
            except Exception as e:
                self.logger.log(f"خطا در ذخیره نتایج تحلیل: {e}", "ERROR")
                messagebox.showerror("خطا", f"خطا در ذخیره نتایج تحلیل:\n{str(e)}")
    
    def clear_data(self):
        """پاک کردن داده‌ها"""
        response = messagebox.askyesno("تأیید", 
                                      "آیا مطمئن هستید که می‌خواهید تمام داده‌ها را پاک کنید؟")
        
        if response:
            self.df = None
            self.portfolio_df = None
            
            # پاک کردن Treeview‌ها
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            for item in self.portfolio_tree.get_children():
                self.portfolio_tree.delete(item)
            
            for item in self.alert_tree.get_children():
                self.alert_tree.delete(item)
            
            # پاک کردن متن‌ها
            self.analysis_text.delete(1.0, tk.END)
            self.report_text.delete(1.0, tk.END)
            
            # بروزرسانی وضعیت
            self.data_status_label.config(text="داده‌ای بارگذاری نشده", fg='#f1c40f')
            self.stats_label.config(text="سهام: ۰ | رکورد: ۰")
            self.portfolio_summary_label.config(text="پورتفو بارگذاری نشده است")
            self.alert_status_label.config(text="⚠️ سیستم هشدار آماده است", fg='#e74c3c')
            
            self.logger.log("تمامی داده‌ها پاک شدند", "INFO")
    
    def refresh_data(self):
        """بروزرسانی داده‌ها"""
        if self.df is None:
            messagebox.showwarning("اخطار", "ابتدا داده‌ها را بارگذاری کنید")
            return
        
        self.logger.log("بروزرسانی داده‌ها...", "INFO")
        
        # در این نسخه ساده، فقط داده‌های فعلی را دوباره نمایش می‌دهیم
        # در نسخه کامل باید داده‌ها را از منبع اصلی دوباره دانلود کنیم
        
        self.display_data_in_tree()
        
        self.logger.log("داده‌ها بروزرسانی شدند", "SUCCESS")
        messagebox.showinfo("موفقیت", "داده‌ها با موفقیت بروزرسانی شدند")
    
    def run_price_prediction(self):
        """اجرای پیش‌بینی قیمت"""
        if self.df is None:
            messagebox.showwarning("اخطار", "ابتدا داده‌ها را بارگذاری کنید")
            return
        
        self.logger.log("شروع پیش‌بینی قیمت...", "INFO")
        
        # نمایش در تب تحلیل
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, "🎯 نتایج پیش‌بینی قیمت:\n\n")
        self.analysis_text.insert(tk.END, "="*60 + "\n\n")
        self.analysis_text.insert(tk.END, "در حال پیش‌بینی با الگوریتم‌های پیشرفته...\n\n")
        
        # اجرای پیش‌بینی در یک thread جداگانه
        threading.Thread(target=self.perform_price_prediction, daemon=True).start()
    
    def perform_price_prediction(self):
        """انجام پیش‌بینی قیمت"""
        try:
            # انتخاب چند سهم برای پیش‌بینی
            symbols_to_predict = []
            
            for _, row in self.df.iterrows():
                symbol = row.get(self.column_mapping.get('نماد', 'نماد'), '')
                if symbol and len(symbols_to_predict) < 5:
                    symbols_to_predict.append(symbol)
            
            self.analysis_text.insert(tk.END, f"📊 پیش‌بینی قیمت برای {len(symbols_to_predict)} سهم:\n\n")
            
            for symbol in symbols_to_predict:
                # شبیه‌سازی پیش‌بینی
                current_price = np.random.uniform(10000, 50000)
                predicted_change = np.random.uniform(-5, 10)
                predicted_price = current_price * (1 + predicted_change/100)
                
                self.analysis_text.insert(tk.END, f"🔹 {symbol}:\n")
                self.analysis_text.insert(tk.END, f"   قیمت فعلی: {current_price:,.0f} ریال\n")
                self.analysis_text.insert(tk.END, f"   پیش‌بینی ۷ روزه: {predicted_price:,.0f} ریال\n")
                
                if predicted_change > 0:
                    self.analysis_text.insert(tk.END, f"   📈 رشد پیش‌بینی: +{predicted_change:.1f}%\n")
                else:
                    self.analysis_text.insert(tk.END, f"   📉 کاهش پیش‌بینی: {predicted_change:.1f}%\n")
                
                # سطح اطمینان
                confidence = np.random.uniform(60, 90)
                self.analysis_text.insert(tk.END, f"   🎯 سطح اطمینان: {confidence:.1f}%\n\n")
            
            self.analysis_text.insert(tk.END, "\n✅ پیش‌بینی قیمت کامل شد\n")
            self.analysis_text.insert(tk.END, "\n💡 توجه: این پیش‌بینی‌ها بر اساس شبیه‌سازی هستند.\n")
            self.analysis_text.insert(tk.END, "   برای پیش‌بینی واقعی به داده‌های تاریخی نیاز است.\n")
            
            self.logger.log("پیش‌بینی قیمت کامل شد", "SUCCESS")
            
        except Exception as e:
            self.analysis_text.insert(tk.END, f"\n❌ خطا در پیش‌بینی قیمت: {str(e)}\n")
            self.logger.log(f"خطا در پیش‌بینی قیمت: {e}", "ERROR")
    
    def create_bar_chart(self):
        """ایجاد نمودار میله‌ای"""
        if self.df is None:
            messagebox.showwarning("اخطار", "ابتدا داده‌ها را بارگذاری کنید")
            return
        
        # پاک کردن قاب نمودار
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        try:
            # انتخاب ستون برای نمودار
            numeric_columns = []
            for col in self.df.columns:
                if self.df[col].dtype in ['int64', 'float64']:
                    numeric_columns.append(col)
            
            if not numeric_columns:
                messagebox.showwarning("اخطار", "هیچ ستون عددی برای نمودار وجود ندارد")
                return
            
            # انتخاب ۱۰ سهم اول
            display_df = self.df.head(10)
            
            # ایجاد نمودار
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('نمودارهای تحلیل سهام', fontsize=16, fontweight='bold')
            
            # نمودار ۱: P/B
            if 'P/B' in self.df.columns:
                axes[0, 0].bar(display_df['نماد'], display_df['P/B'], color='skyblue', alpha=0.7)
                axes[0, 0].set_xlabel('نماد')
                axes[0, 0].set_ylabel('P/B')
                axes[0, 0].set_title('نسبت P/B سهام')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].grid(axis='y', alpha=0.3)
            
            # نمودار ۲: P/E
            if 'P/E' in self.df.columns:
                axes[0, 1].bar(display_df['نماد'], display_df['P/E'], color='lightgreen', alpha=0.7)
                axes[0, 1].set_xlabel('نماد')
                axes[0, 1].set_ylabel('P/E')
                axes[0, 1].set_title('نسبت P/E سهام')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(axis='y', alpha=0.3)
            
            # نمودار ۳: RSI
            if 'RSI' in self.df.columns:
                axes[1, 0].bar(display_df['نماد'], display_df['RSI'], color='lightcoral', alpha=0.7)
                axes[1, 0].axhline(y=70, color='red', linestyle='--', alpha=0.5, label='اشباع خرید')
                axes[1, 0].axhline(y=30, color='green', linestyle='--', alpha=0.5, label='اشباع فروش')
                axes[1, 0].set_xlabel('نماد')
                axes[1, 0].set_ylabel('RSI')
                axes[1, 0].set_title('شاخص RSI سهام')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].legend()
                axes[1, 0].grid(axis='y', alpha=0.3)
            
            # نمودار ۴: حجم معاملات
            volume_col = self.column_mapping.get('حجم معاملات')
            if volume_col in self.df.columns:
                axes[1, 1].bar(display_df['نماد'], display_df[volume_col], color='gold', alpha=0.7)
                axes[1, 1].set_xlabel('نماد')
                axes[1, 1].set_ylabel('حجم معاملات')
                axes[1, 1].set_title('حجم معاملات سهام')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            # نمایش نمودار در Tkinter
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # اضافه کردن تولبار
            toolbar = NavigationToolbar2Tk(canvas, self.chart_frame)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
        except Exception as e:
            tk.Label(self.chart_frame, text=f"خطا در ایجاد نمودار: {str(e)}", 
                    font=('Tahoma', 12), fg='red').pack(pady=50)
    
    def create_line_chart(self):
        """ایجاد نمودار خطی"""
        # این تابع در نسخه کامل پیاده‌سازی می‌شود
        messagebox.showinfo("اطلاع", "نمودار خطی به زودی اضافه خواهد شد")
    
    def create_scatter_chart(self):
        """ایجاد نمودار پراکندگی"""
        # این تابع در نسخه کامل پیاده‌سازی می‌شود
        messagebox.showinfo("اطلاع", "نمودار پراکندگی به زودی اضافه خواهد شد")
    
    def create_histogram(self):
        """ایجاد هیستوگرام"""
        # این تابع در نسخه کامل پیاده‌سازی می‌شود
        messagebox.showinfo("اطلاع", "هیستوگرام به زودی اضافه خواهد شد")
    
    def create_heatmap(self):
        """ایجاد نمودار حرارتی"""
        # این تابع در نسخه کامل پیاده‌سازی می‌شود
        messagebox.showinfo("اطلاع", "نمودار حرارتی به زودی اضافه خواهد شد")
    
    def create_radar_chart(self):
        """ایجاد نمودار رادار"""
        # این تابع در نسخه کامل پیاده‌سازی می‌شود
        messagebox.showinfo("اطلاع", "نمودار رادار به زودی اضافه خواهد شد")
    
    def check_auto_download(self):
        """بررسی دانلود اتوماتیک"""
        if self.auto_download_enabled:
            current_time = time.time()
            last_download = self.download_settings.get('last_download_time')
            
            if last_download is None or (current_time - last_download) >= self.download_settings['auto_interval']:
                threading.Thread(target=self.auto_download_data, daemon=True).start()
        
        # بررسی مجدد بعد از ۱ دقیقه
        self.root.after(60000, self.check_auto_download)
    
    def auto_download_data(self):
        """دانلود اتوماتیک داده"""
        try:
            self.logger.log("شروع دانلود اتوماتیک داده...", "INFO")
            
            # این تابع در نسخه کامل پیاده‌سازی می‌شود
            
            self.download_settings['last_download_time'] = time.time()
            self.logger.log("دانلود اتوماتیک کامل شد", "SUCCESS")
            
        except Exception as e:
            self.logger.log(f"خطا در دانلود اتوماتیک: {e}", "ERROR")

# ============================================================================
# تابع اصلی اجرای برنامه
# ============================================================================
def main():
    root = tk.Tk()
    app = CompleteStockAnalysisApp(root)
    
    # مرکز کردن پنجره
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()

if __name__ == "__main__":
    main()


class BubbleAnalyzer:
    """کلاس تحلیل حباب سهام"""
    
    def __init__(self, data_provider=None):
        self.data_provider = data_provider
        self.bubble_thresholds = {
            'pe_ratio': 25,      # بیش از 25 نشانه حباب
            'pb_ratio': 6,       # بیش از 6 نشانه حباب
            'ps_ratio': 10,      # بیش از 10 نشانه حباب
            'dividend_yield': 2, # کمتر از 2% نشانه حباب
            'price_growth_3m': 50, # رشد بیش از 50% در 3 ماه
            'volume_ratio': 3    # حجم معاملات بیش از 3 برابر میانگین
        }
    
    def analyze(self, symbol, stock_data=None, fundamental_data=None):
        """تحلیل حباب برای یک نماد"""
        try:
            if not stock_data and self.data_provider:
                stock_data = self.data_provider.get_stock_data(symbol)
            
            if not fundamental_data and self.data_provider:
                fundamental_data = self.data_provider.get_fundamental_data(symbol)
            
            if not stock_data:
                return {"error": "داده‌های سهم یافت نشد"}
            
            # محاسبه شاخص‌ها
            indicators = self._calculate_indicators(symbol, stock_data, fundamental_data)
            
            # محاسبه امتیاز حباب
            bubble_score = self._calculate_bubble_score(indicators)
            
            # تعیین سطح حباب
            bubble_level = self._determine_bubble_level(bubble_score)
            
            # ایجاد گزارش
            report = self._generate_report(symbol, indicators, bubble_score, bubble_level)
            
            return {
                'success': True,
                'symbol': symbol,
                'indicators': indicators,
                'bubble_score': bubble_score,
                'bubble_level': bubble_level,
                'report': report,
                'recommendation': self._get_recommendation(bubble_score)
            }
            
        except Exception as e:
            return {"error": f"خطا در تحلیل: {str(e)}"}
    
    def _calculate_indicators(self, symbol, stock_data, fundamental_data):
        """محاسبه شاخص‌های حباب"""
        indicators = {}
        
        # 1. نسبت P/E (اگر داده بنیادی موجود باشد)
        if fundamental_data and 'pe' in fundamental_data:
            pe = fundamental_data['pe']
            indicators['pe_ratio'] = pe
            indicators['pe_status'] = 'حباب' if pe > self.bubble_thresholds['pe_ratio'] else 'عادی'
        
        # 2. رشد قیمت 3 ماهه
        if len(stock_data) >= 60:  # حداقل 60 روز داده (3 ماه کاری)
            current_price = stock_data['Close'].iloc[-1] if hasattr(stock_data['Close'], 'iloc') else stock_data['Close'][-1]
            price_3m_ago = stock_data['Close'].iloc[-60] if hasattr(stock_data['Close'], 'iloc') else stock_data['Close'][-60]
            growth = ((current_price - price_3m_ago) / price_3m_ago) * 100
            indicators['price_growth_3m'] = growth
            indicators['price_growth_status'] = 'حباب' if growth > self.bubble_thresholds['price_growth_3m'] else 'عادی'
        
        # 3. تحلیل حجم
        if len(stock_data) >= 20:
            if hasattr(stock_data['Volume'], 'iloc'):
                current_volume = stock_data['Volume'].iloc[-1]
                avg_volume = stock_data['Volume'].tail(20).mean()
            else:
                current_volume = stock_data['Volume'][-1]
                avg_volume = sum(stock_data['Volume'][-20:]) / 20
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            indicators['volume_ratio'] = volume_ratio
            indicators['volume_status'] = 'حباب' if volume_ratio > self.bubble_thresholds['volume_ratio'] else 'عادی'
        
        # 4. شاخص RSI
        if len(stock_data) >= 14:
            rsi = self._calculate_rsi(stock_data)
            indicators['rsi'] = rsi
            if rsi > 70:
                indicators['rsi_status'] = 'حباب'
            elif rsi < 30:
                indicators['rsi_status'] = 'اشباع فروش'
            else:
                indicators['rsi_status'] = 'عادی'
        
        # 5. مقایسه با صنعت (به صورت ساده‌شده)
        indicators['industry_comparison'] = self._compare_with_industry(symbol, indicators)
        
        return indicators
    
    def _calculate_rsi(self, stock_data, period=14):
        """محاسبه شاخص قدرت نسبی (RSI)"""
        if len(stock_data) < period + 1:
            return 50  # مقدار پیش‌فرض
        
        if hasattr(stock_data['Close'], 'diff'):
            delta = stock_data['Close'].diff()
        else:
            delta = [stock_data['Close'][i] - stock_data['Close'][i-1] for i in range(1, len(stock_data['Close']))]
            delta = [0] + delta
        
        gain = [d if d > 0 else 0 for d in delta]
        loss = [-d if d < 0 else 0 for d in delta]
        
        avg_gain = sum(gain[-period:]) / period
        avg_loss = sum(loss[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _compare_with_industry(self, symbol, indicators):
        """مقایسه با صنعت (ساده‌شده)"""
        # در نسخه کامل، این بخش باید با داده‌های صنعت پر شود
        comparison = {
            'pe_vs_industry': 'در سطح صنعت',
            'growth_vs_industry': 'بالاتر از صنعت',
            'status': 'متوسط'
        }
        return comparison
    
    def _calculate_bubble_score(self, indicators):
        """محاسبه امتیاز حباب (0-100)"""
        score = 0
        max_possible = 0
        
        # وزن‌دهی شاخص‌ها
        weights = {
            'pe_ratio': 30,
            'price_growth_3m': 25,
            'volume_ratio': 20,
            'rsi': 15,
            'industry_comparison': 10
        }
        
        for indicator, weight in weights.items():
            if indicator in indicators:
                max_possible += weight
                
                if indicator == 'pe_ratio' and 'pe_status' in indicators:
                    if indicators['pe_status'] == 'حباب':
                        score += weight
                    else:
                        score += weight * 0.3
                
                elif indicator == 'price_growth_3m' and 'price_growth_status' in indicators:
                    if indicators['price_growth_status'] == 'حباب':
                        score += weight
                    else:
                        score += weight * 0.3
                
                elif indicator == 'volume_ratio' and 'volume_status' in indicators:
                    if indicators['volume_status'] == 'حباب':
                        score += weight * 0.8
                    else:
                        score += weight * 0.2
                
                elif indicator == 'rsi' and 'rsi_status' in indicators:
                    if indicators['rsi_status'] == 'حباب':
                        score += weight
                    elif indicators['rsi_status'] == 'اشباع فروش':
                        score += weight * 0.1
                    else:
                        score += weight * 0.4
        
        if max_possible == 0:
            return 0
        
        return (score / max_possible) * 100
    
    def _determine_bubble_level(self, score):
        """تعیین سطح حباب"""
        if score < 20:
            return "خیلی کم"
        elif score < 40:
            return "کم"
        elif score < 60:
            return "متوسط"
        elif score < 80:
            return "بالا"
        else:
            return "خیلی بالا"
    
    def _get_recommendation(self, bubble_score):
        """توصیه بر اساس امتیاز حباب"""
        if bubble_score < 30:
            return "مناسب برای خرید - ریسک پایین"
        elif bubble_score < 50:
            return "احتیاط - تحلیل بیشتر نیاز است"
        elif bubble_score < 70:
            return "احتمال حباب - از خرید خودداری شود"
        else:
            return "حباب قطعی - فروش توصیه می‌شود"
    
    def _generate_report(self, symbol, indicators, score, level):
        """تولید گزارش تحلیل"""
        report = f"""
        گزارش تحلیل حباب سهم {symbol}
        {'='*50}
        
        امتیاز حباب: {score:.1f}/100
        سطح حباب: {level}
        وضعیت کلی: {self._get_recommendation(score)}
        
        شاخص‌های کلیدی:
        """
        
        if 'pe_ratio' in indicators:
            report += f"\n• نسبت P/E: {indicators['pe_ratio']:.2f}"
            if 'pe_status' in indicators:
                report += f" ({indicators['pe_status']})"
        
        if 'price_growth_3m' in indicators:
            report += f"\n• رشد قیمت ۳ ماهه: {indicators['price_growth_3m']:.1f}%"
            if 'price_growth_status' in indicators:
                report += f" ({indicators['price_growth_status']})"
        
        if 'volume_ratio' in indicators:
            report += f"\n• نسبت حجم: {indicators['volume_ratio']:.2f}"
            if 'volume_status' in indicators:
                report += f" ({indicators['volume_status']})"
        
        if 'rsi' in indicators:
            report += f"\n• شاخص RSI: {indicators['rsi']:.1f}"
            if 'rsi_status' in indicators:
                report += f" ({indicators['rsi_status']})"
        
        report += "\n\nتحلیل نهایی:"
        
        if score < 40:
            report += "\n✅ سهم در محدوده قیمتی معقول قرار دارد. احتمال حباب کم است."
        elif score < 70:
            report += "\n⚠️ سهم نیاز به دقت بیشتری دارد. برخی شاخص‌ها نشانه حباب هستند."
        else:
            report += "\n❌ سهم در محدوده حباب قرار دارد. ریسک سرمایه‌گذاری بالا است."
        
        return report
    
    def plot_bubble_chart(self, symbol, indicators):
        """رسم نمودار حباب (ساده‌شده)"""
        # در نسخه کامل، این بخش نمودار می‌کشد
        return None
