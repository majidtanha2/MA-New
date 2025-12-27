# dashboard/complete_dashboard.py - نسخه اصلاح شده
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
import yaml

# اضافه کردن مسیرهای لازم
sys.path.append('..')

print("🚀 Trader Guardian - Complete Professional Dashboard")
print("=" * 60)

# بارگیری تنظیمات
def load_config():
    try:
        with open('../settings.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        return {
            'mt5': {'login': 228019286, 'server': 'Alpari-MT5'},
            'risk_limits': {
                'max_daily_loss_percent': 3.0,
                'max_trade_risk_percent': 1.0,
                'max_daily_trades': 10
            },
            'psychology': {
                'cooling_period_minutes': 30,
                'emotion_check_interval': 5
            },
            'trading': {
                'symbols': ['EURUSD', 'GBPUSD', 'XAUUSD', 'USDJPY'],
                'timeframes': ['M15', 'H1', 'H4']
            }
        }

config = load_config()

# ایجاد برنامه Dash
app = dash.Dash(__name__, title='Trader Guardian Pro', suppress_callback_exceptions=True)
app.title = "Trader Guardian Professional Dashboard"

# ==================== استایل‌ها ====================
styles = {
    'container': {
        'fontFamily': 'Tahoma, Arial, sans-serif',
        'direction': 'rtl',
        'backgroundColor': '#f8f9fa',
        'minHeight': '100vh'
    },
    'header': {
        'textAlign': 'center',
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'color': 'white',
        'padding': '20px',
        'borderRadius': '10px',
        'marginBottom': '20px'
    },
    'card': {
        'backgroundColor': 'white',
        'borderRadius': '10px',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
        'padding': '15px',
        'margin': '10px'
    },
    'metric': {
        'textAlign': 'center',
        'padding': '15px',
        'borderRadius': '8px',
        'margin': '5px',
        'color': 'white'
    }
}

# ==================== لایه اصلی ====================
app.layout = html.Div([
    # هدر اصلی
    html.Div([
        html.H1("🛡️ TRADER GUARDIAN PROFESSIONAL"),
        html.H4("سیستم کامل مدیریت ریسک و تحلیل بازار"),
        html.Div(id='header-time', style={'marginTop': '10px'})
    ], style=styles['header']),
    
    # تب‌های اصلی
    dcc.Tabs(id='main-tabs', value='tab-dashboard', children=[
        dcc.Tab(label='📊 داشبورد', value='tab-dashboard'),
        dcc.Tab(label='📈 تحلیل بازار', value='tab-analysis'),
        dcc.Tab(label='⚠ مدیریت ریسک', value='tab-risk'),
        dcc.Tab(label='🧠 روانشناسی', value='tab-psychology'),
        dcc.Tab(label='⚙️ تنظیمات', value='tab-settings'),
    ]),
    
    # محتوای تب‌ها
    html.Div(id='tabs-content', style={'padding': '20px'}),
    
    # کامپوننت‌های پنهان
    dcc.Interval(id='header-update', interval=10000, n_intervals=0),
    dcc.Store(id='account-store'),
    dcc.Store(id='settings-store', data=config),
    
    # فوتر
    html.Div([
        html.Hr(),
        html.P("Trader Guardian System v3.0 | Professional Edition | © 2024",
               style={'textAlign': 'center', 'color': '#6c757d'})
    ], style={'marginTop': '30px'})
], style=styles['container'])

# ==================== کالبک بروزرسانی هدر ====================
@app.callback(
    Output('header-time', 'children'),
    Input('header-update', 'n_intervals')
)
def update_header_time(n):
    return f"آخرین بروزرسانی: {datetime.now().strftime('%H:%M:%S')}"

# ==================== کالبک محتوای تب‌ها ====================
@app.callback(
    Output('tabs-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab_content(tab):
    if tab == 'tab-dashboard':
        return create_dashboard_tab()
    elif tab == 'tab-analysis':
        return create_analysis_tab()
    elif tab == 'tab-risk':
        return create_risk_tab()
    elif tab == 'tab-psychology':
        return create_psychology_tab()
    elif tab == 'tab-settings':
        return create_settings_tab()
    return html.Div("در حال بارگذاری...")

# ==================== تابع‌های ایجاد تب‌ها ====================

def create_dashboard_tab():
    """ایجاد محتوای تب داشبورد"""
    return html.Div([
        # کارت‌های اطلاعات
        html.Div([
            html.Div([
                html.H5("💰 موجودی حساب"),
                html.H3("$392.75", id='dashboard-balance'),
                html.P("+0.00% امروز")
            ], style={**styles['metric'], 'backgroundColor': '#27ae60', 'flex': '1', 'minWidth': '200px'}),
            
            html.Div([
                html.H5("⚠ ریسک روزانه"),
                html.H3("0.0%", id='dashboard-risk'),
                html.P("حداکثر: 3.0%")
            ], style={**styles['metric'], 'backgroundColor': '#e74c3c', 'flex': '1', 'minWidth': '200px'}),
            
            html.Div([
                html.H5("📊 سود/زیان"),
                html.H3("$0.00", id='dashboard-profit'),
                html.P("امروز")
            ], style={**styles['metric'], 'backgroundColor': '#3498db', 'flex': '1', 'minWidth': '200px'}),
            
            html.Div([
                html.H5("🧠 وضعیت روانی"),
                html.H3("متعادل", id='dashboard-psychology'),
                html.P("اعتماد: 75%")
            ], style={**styles['metric'], 'backgroundColor': '#9b59b6', 'flex': '1', 'minWidth': '200px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
        
        # نمودارها
        html.Div([
            html.Div([
                html.H5("📈 نمودار اکوئیتی"),
                dcc.Graph(id='dashboard-equity-chart'),
                dcc.Interval(id='dashboard-interval', interval=5000, n_intervals=0)
            ], style={**styles['card'], 'flex': '2', 'minWidth': '500px'}),
            
            html.Div([
                html.H5("📊 توزیع ریسک"),
                dcc.Graph(id='dashboard-risk-chart'),
                html.Button("بروزرسانی", id='dashboard-refresh', 
                          style={'marginTop': '10px', 'width': '100%'})
            ], style={**styles['card'], 'flex': '1', 'minWidth': '300px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        
        # هشدارها
        html.Div([
            html.H5("🚨 هشدارهای سیستم"),
            html.Div(id='dashboard-alerts')
        ], style={**styles['card'], 'marginTop': '20px'}),
    ])

def create_analysis_tab():
    """ایجاد محتوای تب تحلیل"""
    symbols = config.get('trading', {}).get('symbols', ['EURUSD', 'GBPUSD'])
    
    return html.Div([
        html.H4("📈 تحلیل تکنیکال بازار", style={'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                html.Label("انتخاب نماد:"),
                dcc.Dropdown(
                    id='analysis-symbol',
                    options=[{'label': s, 'value': s} for s in symbols],
                    value='EURUSD'
                )
            ], style={**styles['card'], 'flex': '1'}),
            
            html.Div([
                html.Label("انتخاب تایم‌فریم:"),
                dcc.Dropdown(
                    id='analysis-timeframe',
                    options=[
                        {'label': '15 دقیقه', 'value': 'M15'},
                        {'label': '1 ساعت', 'value': 'H1'},
                        {'label': '4 ساعت', 'value': 'H4'},
                        {'label': 'روزانه', 'value': 'D1'}
                    ],
                    value='H1'
                )
            ], style={**styles['card'], 'flex': '1'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px'}),
        
        html.Div([
            html.Button("🔍 اجرای تحلیل", id='analysis-run', 
                      style={'margin': '5px', 'backgroundColor': '#3498db', 'color': 'white'}),
            html.Button("📊 نمودار کندل", id='analysis-candle',
                      style={'margin': '5px', 'backgroundColor': '#2ecc71', 'color': 'white'}),
            html.Button("📈 اندیکاتورها", id='analysis-indicators',
                      style={'margin': '5px', 'backgroundColor': '#9b59b6', 'color': 'white'}),
        ], style={'textAlign': 'center', 'margin': '20px'}),
        
        html.Div([
            dcc.Graph(id='analysis-chart', style={'height': '500px'})
        ], style={**styles['card']}),
        
        html.Div([
            html.H5("📋 نتایج تحلیل"),
            html.Div(id='analysis-results')
        ], style={**styles['card'], 'marginTop': '20px'}),
    ])

def create_risk_tab():
    """ایجاد محتوای تب ریسک"""
    return html.Div([
        html.H4("⚠ مدیریت ریسک معاملات", style={'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                html.H5("وضعیت ریسک فعلی"),
                html.Div(id='risk-status-text'),
                dcc.Graph(id='risk-gauge', style={'height': '200px'})
            ], style={**styles['card'], 'flex': '1'}),
            
            html.Div([
                html.H5("محدودیت‌های فعال"),
                html.Ul([
                    html.Li(f"حداکثر ضرر روزانه: {config['risk_limits']['max_daily_loss_percent']}%"),
                    html.Li(f"حداکثر ریسک هر معامله: {config['risk_limits']['max_trade_risk_percent']}%"),
                    html.Li(f"حداکثر معاملات روزانه: {config['risk_limits']['max_daily_trades']}"),
                    html.Li("حداکثر drawdown: 5.0%"),
                ])
            ], style={**styles['card'], 'flex': '1'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px'}),
        
        html.Div([
            html.H5("🎯 محاسبه ریسک معامله"),
            html.Div([
                html.Label("نماد:"),
                dcc.Input(id='risk-symbol', value='EURUSD', type='text', 
                         style={'width': '100%', 'marginBottom': '10px'}),
                
                html.Label("حجم (لات):"),
                dcc.Input(id='risk-volume', value='0.1', type='number',
                         style={'width': '100%', 'marginBottom': '10px'}),
                
                html.Label("حد ضرر (پیپ):"),
                dcc.Input(id='risk-sl', value='20', type='number',
                         style={'width': '100%', 'marginBottom': '10px'}),
                
                html.Button("محاسبه ریسک", id='risk-calculate',
                          style={'width': '100%', 'backgroundColor': '#e74c3c', 'color': 'white'}),
                
                html.Div(id='risk-calculation', style={'marginTop': '15px', 'padding': '10px'})
            ])
        ], style={**styles['card'], 'marginTop': '20px'}),
        
        html.Div([
            html.H5("📋 تاریخچه تخلفات"),
            html.Div(id='risk-violations')
        ], style={**styles['card'], 'marginTop': '20px'}),
    ])

def create_psychology_tab():
    """ایجاد محتوای تب روانشناسی"""
    return html.Div([
        html.H4("🧠 مدیریت روانشناسی ترید", style={'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                html.H5("وضعیت روانی فعلی"),
                dcc.Graph(id='psychology-gauge', style={'height': '200px'}),
                html.Div(id='psychology-advice', style={'marginTop': '10px'})
            ], style={**styles['card'], 'flex': '1'}),
            
            html.Div([
                html.H5("تمرینات فعال"),
                html.Ul([
                    html.Li("🧘 تمرین تنفس 5-5-5"),
                    html.Li("📝 ثبت احساسات قبل از معامله"),
                    html.Li("⏸️ استراحت پس از 2 ضرر متوالی"),
                ]),
                html.Button("شروع تمرین تنفس", id='psychology-breathing',
                          style={'marginTop': '10px', 'width': '100%'})
            ], style={**styles['card'], 'flex': '1'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px'}),
        
        html.Div([
            html.H5("📊 تاریخچه احساسات"),
            dcc.Graph(id='psychology-history'),
            dcc.Interval(id='psychology-interval', interval=30000, n_intervals=0)
        ], style={**styles['card'], 'marginTop': '20px'}),
    ])

def create_settings_tab():
    """ایجاد محتوای تب تنظیمات"""
    return html.Div([
        html.H4("⚙️ تنظیمات سیستم", style={'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                html.H6("تنظیمات MT5"),
                html.Label("شماره حساب:"),
                dcc.Input(id='settings-login', value=config['mt5']['login'], 
                         type='number', style={'width': '100%', 'marginBottom': '10px'}),
                
                html.Label("سرور:"),
                dcc.Input(id='settings-server', value=config['mt5']['server'],
                         style={'width': '100%', 'marginBottom': '10px'}),
            ], style={**styles['card'], 'flex': '1'}),
            
            html.Div([
                html.H6("محدودیت‌های ریسک"),
                html.Label("حداکثر ضرر روزانه (%):"),
                dcc.Slider(id='settings-daily-loss', min=1, max=10, step=0.5,
                          value=config['risk_limits']['max_daily_loss_percent'],
                          marks={i: str(i) for i in range(1, 11, 2)}),
                
                html.Label("حداکثر ریسک هر معامله (%):"),
                dcc.Slider(id='settings-trade-risk', min=0.5, max=5, step=0.5,
                          value=config['risk_limits']['max_trade_risk_percent'],
                          marks={i: str(i) for i in range(1, 6)}),
            ], style={**styles['card'], 'flex': '1'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px'}),
        
        html.Div([
            html.H6("تنظیمات روانشناسی"),
            html.Label("مدت استراحت پس از تخلف (دقیقه):"),
            dcc.Slider(id='settings-cooling', min=5, max=120, step=5,
                      value=config['psychology']['cooling_period_minutes'],
                      marks={15: '15', 30: '30', 60: '60', 90: '90', 120: '120'}),
            
            html.Label("فاصله چک احساسات (دقیقه):"),
            dcc.Slider(id='settings-emotion-check', min=1, max=30, step=1,
                      value=config['psychology']['emotion_check_interval'],
                      marks={5: '5', 10: '10', 15: '15', 20: '20', 30: '30'}),
        ], style={**styles['card'], 'marginTop': '20px'}),
        
        html.Div([
            html.Button("💾 ذخیره تنظیمات", id='settings-save',
                      style={'margin': '5px', 'backgroundColor': '#27ae60', 'color': 'white'}),
            html.Button("🔄 بارگذاری پیش‌فرض", id='settings-default',
                      style={'margin': '5px', 'backgroundColor': '#3498db', 'color': 'white'}),
            html.Div(id='settings-feedback', style={'marginTop': '10px'})
        ], style={'textAlign': 'center', 'marginTop': '20px'}),
    ])

# ==================== کالبک‌های داشبورد ====================
@app.callback(
    [Output('dashboard-balance', 'children'),
     Output('dashboard-risk', 'children'),
     Output('dashboard-profit', 'children'),
     Output('dashboard-psychology', 'children')],
    [Input('dashboard-interval', 'n_intervals'),
     Input('dashboard-refresh', 'n_clicks')]
)
def update_dashboard_metrics(n_intervals, n_clicks):
    """بروزرسانی معیارهای داشبورد"""
    # اینجا می‌توانید از MT5 اطلاعات واقعی بگیرید
    return "$392.75", "0.0%", "$0.00", "متعادل"

@app.callback(
    Output('dashboard-equity-chart', 'figure'),
    Input('dashboard-interval', 'n_intervals')
)
def update_dashboard_chart(n):
    """بروزرسانی نمودار داشبورد"""
    # داده‌های واقعی‌تر - بدون cumsum
    dates = pd.date_range(end=datetime.now(), periods=24, freq='H')
    base_equity = 392.75
    equity = [base_equity + np.random.uniform(-5, 5) for _ in range(24)]
    
    fig = go.Figure(data=[
        go.Scatter(x=dates, y=equity, mode='lines',
                  name='Equity', line={'color': '#2E86C1', 'width': 3})
    ])
    
    fig.update_layout(
        title='تاریخچه اکوئیتی',
        xaxis_title='زمان',
        yaxis_title='مقدار ($)',
        template='plotly_white',
        height=350
    )
    
    return fig

@app.callback(
    Output('dashboard-risk-chart', 'figure'),
    [Input('dashboard-interval', 'n_intervals'),
     Input('dashboard-refresh', 'n_clicks')]
)
def update_risk_chart(n_intervals, n_clicks):
    """بروزرسانی نمودار ریسک"""
    categories = ['ضرر روزانه', 'ریسک معامله', 'تعداد معاملات', 'Drawdown']
    values = [0.5, 0.8, 3, 1.2]  # مقادیر نمونه
    
    fig = go.Figure(data=[
        go.Bar(x=categories, y=values, 
               marker_color=['#2ecc71', '#f39c12', '#3498db', '#e74c3c'])
    ])
    
    fig.update_layout(
        title='توزیع ریسک',
        yaxis_title='مقدار',
        template='plotly_white',
        height=300
    )
    
    return fig

# ==================== کالبک‌های تحلیل ====================
@app.callback(
    Output('analysis-chart', 'figure'),
    [Input('analysis-run', 'n_clicks'),
     Input('analysis-candle', 'n_clicks'),
     Input('analysis-indicators', 'n_clicks')],
    [State('analysis-symbol', 'value'),
     State('analysis-timeframe', 'value')]
)
def update_analysis_chart(run_clicks, candle_clicks, indicators_clicks, symbol, timeframe):
    """بروزرسانی نمودار تحلیل"""
    # داده‌های نمونه برای نمودار
    dates = pd.date_range(end=datetime.now(), periods=50, freq='H')
    prices = 1.1 + np.cumsum(np.random.randn(50) * 0.01)
    
    fig = go.Figure(data=[
        go.Scatter(x=dates, y=prices, mode='lines',
                  name=symbol, line={'color': '#3498db', 'width': 2})
    ])
    
    fig.update_layout(
        title=f'تحلیل {symbol} - {timeframe}',
        xaxis_title='زمان',
        yaxis_title='قیمت',
        template='plotly_white',
        height=450
    )
    
    return fig

@app.callback(
    Output('analysis-results', 'children'),
    Input('analysis-run', 'n_clicks'),
    [State('analysis-symbol', 'value'),
     State('analysis-timeframe', 'value')]
)
def update_analysis_results(n_clicks, symbol, timeframe):
    """بروزرسانی نتایج تحلیل"""
    if n_clicks is None or n_clicks == 0:
        return html.P("برای تحلیل کلیک کنید")
    
    # نتایج نمونه
    results = [
        html.P(f"📊 تحلیل {symbol} در تایم‌فریم {timeframe}"),
        html.P("📈 روند: صعودی ضعیف"),
        html.P("📉 RSI: 52 (خنثی)"),
        html.P("📊 حجم: طبیعی"),
        html.P("🎯 سیگنال: انتظار برای شکست مقاومت"),
    ]
    
    return html.Div(results)

# ==================== کالبک‌های ریسک ====================
@app.callback(
    Output('risk-status-text', 'children'),
    Input('risk-calculate', 'n_clicks')
)
def update_risk_status(n_clicks):
    """بروزرسانی وضعیت ریسک"""
    return html.Div([
        html.P("🟢 وضعیت ریسک: پایین"),
        html.P("📊 ضرر روزانه: 0.5%"),
        html.P("🎯 معاملات موفق: 65%"),
        html.P("⏱️ آخرین تخلف: هیچ"),
    ])

@app.callback(
    Output('risk-gauge', 'figure'),
    Input('risk-calculate', 'n_clicks')
)
def update_risk_gauge(n_clicks):
    """بروزرسانی گیج ریسک"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=25,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "سطح ریسک"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#2ecc71"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "red"}
            ]
        }
    ))
    
    fig.update_layout(height=200)
    return fig

@app.callback(
    Output('risk-calculation', 'children'),
    [Input('risk-calculate', 'n_clicks')],
    [State('risk-symbol', 'value'),
     State('risk-volume', 'value'),
     State('risk-sl', 'value')]
)
def calculate_risk(n_clicks, symbol, volume, sl_pips):
    """محاسبه ریسک معامله"""
    if n_clicks is None or n_clicks == 0:
        return "مقادیر را وارد و محاسبه کنید"
    
    try:
        volume = float(volume)
        sl_pips = float(sl_pips)
        
        # محاسبه ساده ریسک
        risk_amount = volume * sl_pips * 10  # محاسبه ساده
        risk_percent = (risk_amount / 392.75) * 100
        
        max_risk = config['risk_limits']['max_trade_risk_percent']
        
        if risk_percent <= max_risk:
            color = "#27ae60"
            status = "✅ مجاز"
        else:
            color = "#e74c3c"
            status = "❌ غیرمجاز"
        
        return html.Div([
            html.P(f"📊 نماد: {symbol}"),
            html.P(f"📈 حجم: {volume} لات"),
            html.P(f"⚠ حد ضرر: {sl_pips} پیپ"),
            html.P(f"💰 مقدار ریسک: ${risk_amount:.2f}"),
            html.P(f"📊 درصد ریسک: {risk_percent:.1f}%"),
            html.P(f"🛡️ حداکثر مجاز: {max_risk}%"),
            html.P(f"🎯 وضعیت: {status}", style={'color': color, 'fontWeight': 'bold'})
        ])
    except:
        return "خطا در محاسبه. مقادیر را بررسی کنید."

# ==================== کالبک‌های روانشناسی ====================
@app.callback(
    Output('psychology-gauge', 'figure'),
    Input('psychology-breathing', 'n_clicks')
)
def update_psychology_gauge(n_clicks):
    """بروزرسانی گیج روانشناسی"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=75,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "اعتماد به نفس"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#9b59b6"},
            'steps': [
                {'range': [0, 50], 'color': "#e74c3c"},
                {'range': [50, 80], 'color': "#f39c12"},
                {'range': [80, 100], 'color': "#2ecc71"}
            ]
        }
    ))
    
    fig.update_layout(height=200)
    return fig

@app.callback(
    Output('psychology-advice', 'children'),
    Input('psychology-breathing', 'n_clicks')
)
def update_psychology_advice(n_clicks):
    """بروزرسانی توصیه روانشناسی"""
    advice = [
        "🧘 وضعیت روانی شما متعادل است",
        "📊 اعتماد به نفس در سطح خوبی قرار دارد",
        "🎯 تصمیم‌گیری‌های منطقی داشته باشید",
        "⏸️ در صورت احساس اضطراب استراحت کنید"
    ]
    
    return html.Ul([html.Li(item) for item in advice])

@app.callback(
    Output('psychology-history', 'figure'),
    Input('psychology-interval', 'n_intervals')
)
def update_psychology_history(n):
    """بروزرسانی تاریخچه احساسات"""
    dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
    emotions = ['متعادل', 'مثبت', 'منفی', 'متعادل', 'مثبت', 'منفی', 'متعادل']
    scores = [75, 80, 40, 70, 85, 35, 75]
    
    fig = go.Figure(data=[
        go.Scatter(x=dates, y=scores, mode='lines+markers',
                  line={'color': '#9b59b6', 'width': 3},
                  marker={'size': 10})
    ])
    
    fig.update_layout(
        title='تاریخچه وضعیت روانی',
        xaxis_title='تاریخ',
        yaxis_title='امتیاز',
        template='plotly_white',
        height=300
    )
    
    return fig

# ==================== کالبک‌های تنظیمات ====================
@app.callback(
    Output('settings-feedback', 'children'),
    Input('settings-save', 'n_clicks'),
    [State('settings-login', 'value'),
     State('settings-server', 'value'),
     State('settings-daily-loss', 'value'),
     State('settings-trade-risk', 'value'),
     State('settings-cooling', 'value'),
     State('settings-emotion-check', 'value')]
)
def save_settings(n_clicks, login, server, daily_loss, trade_risk, cooling, emotion_check):
    """ذخیره تنظیمات"""
    if n_clicks is None or n_clicks == 0:
        return ""
    
    try:
        # به‌روزرسانی config
        config['mt5']['login'] = login
        config['mt5']['server'] = server
        config['risk_limits']['max_daily_loss_percent'] = daily_loss
        config['risk_limits']['max_trade_risk_percent'] = trade_risk
        config['psychology']['cooling_period_minutes'] = cooling
        config['psychology']['emotion_check_interval'] = emotion_check
        
        # ذخیره در فایل
        with open('../settings.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return html.Div("✅ تنظیمات با موفقیت ذخیره شد", 
                       style={'color': '#27ae60', 'padding': '10px', 'backgroundColor': '#d5f4e6'})
    except Exception as e:
        return html.Div(f"❌ خطا در ذخیره تنظیمات: {str(e)}",
                       style={'color': '#e74c3c', 'padding': '10px', 'backgroundColor': '#fadbd8'})

print("=" * 60)
print("✅ Dashboard created successfully!")
print("🌐 Access at: http://localhost:8050")
print("=" * 60)

# اجرای سرور
if __name__ == '__main__':
    app.run(debug=True, port=8050, host='0.0.0.0')