# dashboard/components/charts.py
"""
کامپوننت‌های نمودار برای داشبورد
"""

import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class ChartComponents:
    """کامپوننت‌های نمودار"""
    
    @staticmethod
    def create_equity_chart(equity_data: List[float], dates: List[datetime]) -> go.Figure:
        """ایجاد نمودار اکوئیتی ثابت"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity_data,
            mode='lines',
            name='اکوئیتی',
            line={'color': '#2E86C1', 'width': 3},
            fill='tozeroy',
            fillcolor='rgba(46, 134, 193, 0.1)'
        ))
        
        # خط میانگین
        mean_value = np.mean(equity_data) if equity_data else 0
        fig.add_hline(y=mean_value, 
                     line_dash="dash", 
                     line_color="green",
                     annotation_text=f"میانگین: {mean_value:.2f}")
        
        fig.update_layout(
            title={'text': 'تاریخچه اکوئیتی حساب', 'x': 0.5},
            xaxis={'title': 'زمان', 'showgrid': True},
            yaxis={'title': 'مقدار ($)', 'showgrid': True},
            hovermode='x unified',
            template='plotly_white',
            height=350,
            margin={'l': 50, 'r': 20, 't': 50, 'b': 50},
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_candlestick_chart(data: Dict, title: str = '') -> go.Figure:
        """ایجاد نمودار کندل استیک"""
        fig = go.Figure(data=[
            go.Candlestick(
                x=data.get('x', []),
                open=data.get('open', []),
                high=data.get('high', []),
                low=data.get('low', []),
                close=data.get('close', []),
                name='قیمت',
                increasing_line_color='#2ECC71',
                decreasing_line_color='#E74C3C'
            )
        ])
        
        fig.update_layout(
            title={'text': title, 'x': 0.5},
            xaxis={'title': 'زمان', 'rangeslider': {'visible': False}},
            yaxis={'title': 'قیمت', 'side': 'right'},
            template='plotly_white',
            height=450,
            margin={'l': 50, 'r': 50, 't': 50, 'b': 50}
        )
        
        return fig
    
    @staticmethod
    def create_risk_distribution_chart(risk_data: Dict) -> go.Figure:
        """ایجاد نمودار توزیع ریسک"""
        categories = list(risk_data.keys())
        values = list(risk_data.values())
        
        colors = ['#2ECC71', '#F1C40F', '#E74C3C', '#3498DB']
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors[:len(categories)],
                text=[f'{v:.1f}%' for v in values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title={'text': 'توزیع ریسک', 'x': 0.5},
            xaxis={'title': 'نوع ریسک'},
            yaxis={'title': 'درصد', 'range': [0, 100]},
            template='plotly_white',
            height=300
        )
        
        return fig
    
    @staticmethod
    def create_gauge_chart(value: float, title: str, min_val: float = 0, 
                          max_val: float = 100) -> go.Figure:
        """ایجاد نمودار عقربه‌ای"""
        # تعیین رنگ بر اساس مقدار
        if value < max_val * 0.5:
            color = "#2ECC71"
        elif value < max_val * 0.8:
            color = "#F1C40F"
        else:
            color = "#E74C3C"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 16}},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': color},
                'steps': [
                    {'range': [min_val, max_val * 0.5], 'color': "lightgreen"},
                    {'range': [max_val * 0.5, max_val * 0.8], 'color': "yellow"},
                    {'range': [max_val * 0.8, max_val], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_val * 0.9
                }
            }
        ))
        
        fig.update_layout(height=200, margin={'l': 20, 'r': 20, 't': 50, 'b': 20})
        return fig
    
    @staticmethod
    def create_pie_chart(labels: List[str], values: List[float], 
                        title: str = '') -> go.Figure:
        """ایجاد نمودار پای"""
        colors = ['#2ECC71', '#E74C3C', '#3498DB', '#9B59B6', '#F1C40F']
        
        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker_colors=colors[:len(labels)],
                textinfo='label+percent',
                hoverinfo='label+value+percent'
            )
        ])
        
        fig.update_layout(
            title={'text': title, 'x': 0.5},
            template='plotly_white',
            height=300,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_line_chart_with_indicators(x_data: List, y_data: List[Dict], 
                                         title: str = '') -> go.Figure:
        """ایجاد نمودار خطی با اندیکاتورها"""
        fig = go.Figure()
        
        for data in y_data:
            fig.add_trace(go.Scatter(
                x=x_data,
                y=data['values'],
                mode='lines',
                name=data['name'],
                line={'color': data.get('color', '#000000'), 
                     'width': data.get('width', 2)},
                opacity=data.get('opacity', 1.0)
            ))
        
        fig.update_layout(
            title={'text': title, 'x': 0.5},
            xaxis={'title': 'زمان'},
            yaxis={'title': 'مقدار'},
            template='plotly_white',
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_histogram(data: List[float], title: str = '', 
                        bins: int = 20) -> go.Figure:
        """ایجاد هیستوگرام"""
        fig = go.Figure(data=[
            go.Histogram(
                x=data,
                nbinsx=bins,
                marker_color='#3498DB',
                opacity=0.7
            )
        ])
        
        fig.update_layout(
            title={'text': title, 'x': 0.5},
            xaxis={'title': 'مقدار'},
            yaxis={'title': 'تعداد'},
            template='plotly_white',
            height=300
        )
        
        return fig