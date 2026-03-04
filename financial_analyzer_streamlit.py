import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import math

st.set_page_config(
    page_title="Advanced Financial Data Analyzer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    .css-1n76uvr {
        background-color: #f7fafc;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Advanced Financial Data Analyzer")
st.caption("Upload any financial datasets with Date, Open, High, Low, Close, Volume columns")

# File upload section
with st.container():
    st.subheader("Upload Datasets")
    uploaded_files = st.file_uploader(
        "Choose CSV files", 
        accept_multiple_files=True,
        type="csv"
    )
    
    # Dataset names
    datasets = {}
    if uploaded_files:
        for file in uploaded_files:
            # Extract ticker from filename (without extension)
            ticker = file.name.split('.')[0].upper()
            # Read CSV
            df = pd.read_csv(file)
            # Ensure Date column is datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                datasets[ticker] = df
            else:
                st.error(f"File {file.name} missing 'Date' column")
    
    # Display uploaded datasets
    if datasets:
        st.success(f"Uploaded {len(datasets)} datasets: {', '.join(datasets.keys())}")
    else:
        st.info("Please upload CSV files to begin analysis")

# Risk management parameters
with st.expander("Risk Management Parameters", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        initial_capital = st.number_input("Initial Capital ($)", value=10000, min_value=1000)
    with col2:
        risk_per_trade = st.number_input("Risk Per Trade (%)", value=2.0, min_value=0.1, step=0.5)
    with col3:
        stop_loss = st.number_input("Stop Loss (%)", value=5.0, min_value=0.1, step=0.5)
    with col4:
        take_profit = st.number_input("Take Profit (%)", value=10.0, min_value=0.1, step=0.5)

# Helper functions
def calculate_returns(data):
    returns = []
    closes = data['Close'].values
    for i in range(1, len(closes)):
        if closes[i] and closes[i-1]:
            returns.append((closes[i] - closes[i-1]) / closes[i-1] * 100)
    return returns

def calculate_ema(data, period):
    closes = data['Close'].values
    ema = []
    multiplier = 2 / (period + 1)
    
    ema.append(closes[0])
    for i in range(1, len(closes)):
        ema.append((closes[i] - ema[i-1]) * multiplier + ema[i-1])
    return ema

def calculate_rsi(data, period=14):
    closes = data['Close'].values
    rsi = []
    
    for i in range(0, period):
        rsi.append(None)
    
    for i in range(period, len(closes)):
        gains = 0
        losses = 0
        
        for j in range(i - period + 1, i + 1):
            change = closes[j] - closes[j-1]
            if change > 0:
                gains += change
            else:
                losses -= change
        
        if losses == 0:
            rsi.append(100)
        else:
            avg_gain = gains / period
            avg_loss = losses / period
            rs = avg_gain / avg_loss
            rsi.append(100 - (100 / (1 + rs)))
    
    return rsi

def calculate_macd(data):
    ema12 = calculate_ema(data, 12)
    ema26 = calculate_ema(data, 26)
    macd = [e12 - e26 for e12, e26 in zip(ema12, ema26)]
    
    # Create temporary data for signal line
    signal_data = pd.DataFrame({'Close': macd, 'Date': data['Date']})
    signal = calculate_ema(signal_data, 9)
    histogram = [m - s for m, s in zip(macd, signal)]
    
    return {'macd': macd, 'signal': signal, 'histogram': histogram}

def calculate_bollinger_bands(data, period=20):
    closes = data['Close'].values
    sma = []
    upper = []
    lower = []
    
    for i in range(0, period - 1):
        sma.append(None)
        upper.append(None)
        lower.append(None)
    
    for i in range(period - 1, len(closes)):
        slice_ = closes[i - period + 1:i + 1]
        mean = np.mean(slice_)
        std_dev = np.std(slice_)
        
        sma.append(mean)
        upper.append(mean + 2 * std_dev)
        lower.append(mean - 2 * std_dev)
    
    return {'sma': sma, 'upper': upper, 'lower': lower}

def calculate_stats(data):
    closes = data['Close'].dropna().values
    returns = calculate_returns(data)
    
    if not returns:
        return {}
    
    mean = np.mean(returns)
    variance = np.var(returns)
    volatility = math.sqrt(variance) * math.sqrt(252)
    
    sorted_returns = sorted(returns)
    max_return = max(returns)
    min_return = min(returns)
    median = sorted_returns[len(sorted_returns) // 2]
    
    skewness = np.mean([(r - mean) ** 3 for r in returns]) / (variance ** 1.5)
    kurtosis = np.mean([(r - mean) ** 4 for r in returns]) / (variance ** 2) - 3
    
    total_return = ((closes[-1] - closes[0]) / closes[0]) * 100
    sharpe = ((mean * 252) - 2) / volatility if volatility > 0 else 0
    
    # Calculate max drawdown
    peak = closes[0]
    max_dd = 0
    for price in closes:
        if price > peak:
            peak = price
        dd = ((price - peak) / peak) * 100
        if dd < max_dd:
            max_dd = dd
    
    return {
        'currentPrice': round(closes[-1], 2),
        'totalReturn': round(total_return, 2),
        'volatility': round(volatility, 2),
        'avgReturn': round(mean * 252, 2),
        'maxReturn': round(max_return, 2),
        'minReturn': round(min_return, 2),
        'medianReturn': round(median, 2),
        'sharpe': round(sharpe, 2),
        'maxDrawdown': round(max_dd, 2),
        'skewness': round(skewness, 2),
        'kurtosis': round(kurtosis, 2)
    }

# Analysis tabs
tab1, tab2, tab3, tab4 = st.tabs(["Basic Analysis", "Technical Indicators", "Trading Strategies", "Risk Management"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_type = st.selectbox(
            "Select Analysis",
            ["Overview", "Compare Returns", "Volatility", "Correlation", 
             "Candlestick", "Moving Averages", "Drawdown", "Distribution", 
             "Sharpe Ratio", "Volume", "Statistics", "Monthly Returns"]
        )
    
    if datasets:
        if analysis_type == "Overview":
            # Display stats cards
            st.subheader("Overview")
            cols = st.columns(len(datasets))
            for i, (ticker, data) in enumerate(datasets.items()):
                with cols[i]:
                    stats = calculate_stats(data)
                    st.metric(label=ticker, value=f"${stats.get('currentPrice', 'N/A')}")
                    st.text(f"Total Return: {stats.get('totalReturn', 'N/A')}%")
                    st.text(f"Volatility: {stats.get('volatility', 'N/A')}%")
                    st.text(f"Sharpe: {stats.get('sharpe', 'N/A')}")
                    st.text(f"Max DD: {stats.get('maxDrawdown', 'N/A')}%")
            
            # Price charts
            st.subheader("Price History")
            for ticker, data in datasets.items():
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['Close'],
                    mode='lines',
                    name=ticker,
                    line=dict(color='#667eea', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.1)'
                ))
                fig.update_layout(
                    title=f"{ticker} Price History",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode="x unified",
                    margin=dict(t=50, r=20, b=50, l=50)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Compare Returns":
            st.subheader("Cumulative Returns Comparison")
            fig = go.Figure()
            
            for ticker, data in datasets.items():
                closes = data['Close'].values
                dates = data['Date']
                
                cumulative_returns = [0]
                for i in range(1, len(closes)):
                    return_ = ((closes[i] - closes[0]) / closes[0]) * 100
                    cumulative_returns.append(return_)
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=cumulative_returns,
                    mode='lines',
                    name=ticker,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Cumulative Returns Comparison",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                hovermode="x unified",
                legend=dict(x=0, y=1),
                margin=dict(t=50, r=20, b=50, l=50)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Volatility":
            st.subheader("Rolling 30-Day Volatility")
            fig = go.Figure()
            
            for ticker, data in datasets.items():
                returns = calculate_returns(data)
                dates = data['Date'][1:]
                
                window = 30
                rolling_vol = []
                vol_dates = []
                
                for i in range(window, len(returns)):
                    window_returns = returns[i-window:i]
                    mean = np.mean(window_returns)
                    variance = np.var(window_returns)
                    vol = math.sqrt(variance) * math.sqrt(252) * 100
                    
                    rolling_vol.append(vol)
                    vol_dates.append(dates.iloc[i])
                
                fig.add_trace(go.Scatter(
                    x=vol_dates,
                    y=rolling_vol,
                    mode='lines',
                    name=ticker,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Rolling 30-Day Volatility",
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                hovermode="x unified",
                legend=dict(x=0, y=1),
                margin=dict(t=50, r=20, b=50, l=50)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Correlation":
            st.subheader("Correlation Matrix (Returns)")
            if len(datasets) < 2:
                st.warning("Need at least 2 datasets for correlation analysis")
            else:
                tickers = list(datasets.keys())
                returns_data = {}
                min_length = float('inf')
                
                for ticker in tickers:
                    returns = calculate_returns(datasets[ticker])
                    returns_data[ticker] = returns
                    min_length = min(min_length, len(returns))
                
                for ticker in tickers:
                    returns_data[ticker] = returns_data[ticker][-min_length:]
                
                n = len(tickers)
                corr_matrix = np.zeros((n, n))
                
                for i in range(n):
                    for j in range(n):
                        returns1 = returns_data[tickers[i]]
                        returns2 = returns_data[tickers[j]]
                        corr_matrix[i][j] = np.corrcoef(returns1, returns2)[0, 1]
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    x=tickers,
                    y=tickers,
                    colorscale='RdBu',
                    zmid=0,
                    text=[[f"{val:.2f}" for val in row] for row in corr_matrix],
                    texttemplate="%{text}",
                    textfont=dict(size=14),
                    colorbar=dict(title="Correlation")
                ))
                
                fig.update_layout(
                    title="Correlation Matrix (Returns)",
                    margin=dict(t=50, r=20, b=100, l=100),
                    width=min(800, st.session_state.get('width', 800)),
                    height=min(600, st.session_state.get('width', 800))
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Candlestick":
            st.subheader("Candlestick Charts (Last 90 Days)")
            for ticker, data in datasets.items():
                recent_data = data.tail(90)
                
                fig = go.Figure(data=[go.Candlestick(
                    x=recent_data['Date'],
                    open=recent_data['Open'],
                    high=recent_data['High'],
                    low=recent_data['Low'],
                    close=recent_data['Close'],
                    name=ticker,
                    increasing_line_color='#48bb78',
                    decreasing_line_color='#f56565'
                )])
                
                fig.update_layout(
                    title=f"{ticker} Candlestick Chart",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    xaxis_rangeslider_visible=False,
                    margin=dict(t=50, r=20, b=50, l=50)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Moving Averages":
            st.subheader("Price with Moving Averages")
            for ticker, data in datasets.items():
                dates = data['Date']
                closes = data['Close'].values
                
                # Calculate MAs
                ma20 = []
                ma50 = []
                ma200 = []
                
                for i in range(len(closes)):
                    if i >= 19:
                        ma20.append(np.mean(closes[i-19:i+1]))
                    else:
                        ma20.append(None)
                    
                    if i >= 49:
                        ma50.append(np.mean(closes[i-49:i+1]))
                    else:
                        ma50.append(None)
                    
                    if i >= 199:
                        ma200.append(np.mean(closes[i-199:i+1]))
                    else:
                        ma200.append(None)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=closes,
                    mode='lines',
                    name='Price',
                    line=dict(color='#2d3748', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=ma20,
                    mode='lines',
                    name='MA20',
                    line=dict(color='#667eea', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=ma50,
                    mode='lines',
                    name='MA50',
                    line=dict(color='#48bb78', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=ma200,
                    mode='lines',
                    name='MA200',
                    line=dict(color='#ed8936', width=2)
                ))
                
                fig.update_layout(
                    title=f"{ticker} Price with Moving Averages",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode="x unified",
                    legend=dict(x=0, y=1),
                    margin=dict(t=50, r=20, b=50, l=50)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Drawdown":
            st.subheader("Drawdown Analysis")
            fig = go.Figure()
            
            for ticker, data in datasets.items():
                dates = data['Date']
                closes = data['Close'].values
                
                drawdowns = []
                peak = closes[0]
                
                for price in closes:
                    if price > peak:
                        peak = price
                    drawdowns.append(((price - peak) / peak) * 100)
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=drawdowns,
                    mode='lines',
                    name=ticker,
                    fill='tozeroy',
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Drawdown Analysis",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode="x unified",
                legend=dict(x=0, y=1),
                margin=dict(t=50, r=20, b=50, l=50)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Distribution":
            st.subheader("Daily Returns Distribution")
            fig = go.Figure()
            
            for ticker, data in datasets.items():
                returns = calculate_returns(data)
                
                fig.add_trace(go.Histogram(
                    x=returns,
                    name=ticker,
                    opacity=0.7,
                    nbinsx=50
                ))
            
            fig.update_layout(
                title="Daily Returns Distribution",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                barmode='overlay',
                legend=dict(x=0, y=1),
                margin=dict(t=50, r=20, b=50, l=50)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Sharpe Ratio":
            st.subheader("Risk-Return Profile (Sharpe Ratio)")
            tickers = []
            returns = []
            volatilities = []
            sharpes = []
            
            for ticker, data in datasets.items():
                stats = calculate_stats(data)
                tickers.append(ticker)
                returns.append(stats.get('avgReturn', 0))
                volatilities.append(stats.get('volatility', 0))
                sharpes.append(stats.get('sharpe', 0))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=volatilities,
                y=returns,
                mode='markers+text',
                text=tickers,
                textposition='top center',
                marker=dict(
                    size=[max(10, s * 5) for s in sharpes],
                    color=sharpes,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio")
                )
            ))
            
            fig.update_layout(
                title="Risk-Return Profile (Sharpe Ratio)",
                xaxis_title="Volatility (%)",
                yaxis_title="Annual Return (%)",
                hovermode="closest",
                margin=dict(t=50, r=20, b=50, l=50)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Volume":
            st.subheader("Price & Volume Analysis")
            for ticker, data in datasets.items():
                dates = data['Date']
                closes = data['Close']
                volumes = data['Volume']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=closes,
                    mode='lines',
                    name='Price',
                    yaxis='y'
                ))
                fig.add_trace(go.Bar(
                    x=dates,
                    y=volumes,
                    name='Volume',
                    yaxis='y2',
                    marker=dict(color='#667eea', opacity=0.5)
                ))
                
                fig.update_layout(
                    title=f"{ticker} Price & Volume",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    yaxis2=dict(
                        title="Volume",
                        overlaying='y',
                        side='right'
                    ),
                    hovermode="x unified",
                    legend=dict(x=0, y=1),
                    margin=dict(t=50, r=100, b=50, l=50)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Statistics":
            st.subheader("Detailed Statistics")
            
            # Create a DataFrame for statistics
            stats_data = []
            tickers = list(datasets.keys())
            
            metrics = [
                'currentPrice', 'totalReturn', 'volatility', 'avgReturn',
                'maxReturn', 'minReturn', 'medianReturn', 'sharpe',
                'maxDrawdown', 'skewness', 'kurtosis'
            ]
            
            metric_names = {
                'currentPrice': 'Current Price',
                'totalReturn': 'Total Return (%)',
                'volatility': 'Volatility (%)',
                'avgReturn': 'Avg Annual Return (%)',
                'maxReturn': 'Max Daily Return (%)',
                'minReturn': 'Min Daily Return (%)',
                'medianReturn': 'Median Daily Return (%)',
                'sharpe': 'Sharpe Ratio',
                'maxDrawdown': 'Max Drawdown (%)',
                'skewness': 'Skewness',
                'kurtosis': 'Kurtosis'
            }
            
            for metric in metrics:
                row = {'Metric': metric_names[metric]}
                for ticker in tickers:
                    stats = calculate_stats(datasets[ticker])
                    row[ticker] = stats.get(metric, 'N/A')
                stats_data.append(row)
            
            df_stats = pd.DataFrame(stats_data)
            st.dataframe(df_stats, use_container_width=True)
        
        elif analysis_type == "Monthly Returns":
            st.subheader("Monthly Returns")
            
            for ticker, data in datasets.items():
                # Extract month and year
                data['Month'] = data['Date'].dt.strftime('%Y-%m')
                
                # Calculate monthly returns
                monthly_data = data.groupby('Month').agg({
                    'Close': ['first', 'last']
                }).reset_index()
                monthly_data.columns = ['Month', 'First', 'Last']
                monthly_data['Return'] = ((monthly_data['Last'] - monthly_data['First']) / monthly_data['First']) * 100
                
                # Plot
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=monthly_data['Month'],
                    y=monthly_data['Return'],
                    marker_color=["#48bb78" if r >= 0 else "#f56565" for r in monthly_data['Return']]
                ))
                
                fig.update_layout(
                    title=f"{ticker} Monthly Returns",
                    xaxis_title="Month",
                    yaxis_title="Return (%)",
                    margin=dict(t=50, r=20, b=100, l=50)
                )
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        indicator_type = st.selectbox(
            "Select Technical Indicator",
            ["RSI", "MACD", "EMA", "Bollinger Bands"]
        )
    
    if datasets:
        if indicator_type == "RSI":
            st.subheader("Relative Strength Index (RSI)")
            period = st.slider("RSI Period", min_value=1, max_value=30, value=14)
            
            for ticker, data in datasets.items():
                rsi = calculate_rsi(data, period)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=rsi,
                    mode='lines',
                    name=f'RSI ({period})',
                    line=dict(color='#667eea', width=2)
                ))
                # Add overbought/oversold lines
                fig.add_hline(y=70, line_dash="dash", line_color="#f56565", annotation_text="Overbought")
                fig.add_hline(y=30, line_dash="dash", line_color="#48bb78", annotation_text="Oversold")
                
                fig.update_layout(
                    title=f"{ticker} RSI ({period})",
                    xaxis_title="Date",
                    yaxis_title="RSI",
                    yaxis_range=[0, 100],
                    margin=dict(t=50, r=20, b=50, l=50)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif indicator_type == "MACD":
            st.subheader("Moving Average Convergence Divergence (MACD)")
            
            for ticker, data in datasets.items():
                macd_data = calculate_macd(data)
                
                fig = go.Figure()
                # MACD line
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=macd_data['macd'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='#667eea', width=2)
                ))
                # Signal line
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=macd_data['signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='#ed8936', width=2)
                ))
                # Histogram
                fig.add_trace(go.Bar(
                    x=data['Date'],
                    y=macd_data['histogram'],
                    name='Histogram',
                    marker_color=["#48bb78" if h >= 0 else "#f56565" for h in macd_data['histogram']]
                ))
                
                fig.update_layout(
                    title=f"{ticker} MACD",
                    xaxis_title="Date",
                    yaxis_title="MACD",
                    margin=dict(t=50, r=20, b=50, l=50)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif indicator_type == "EMA":
            st.subheader("Exponential Moving Average (EMA)")
            periods = st.multiselect(
                "EMA Periods",
                [5, 10, 20, 50, 100, 200],
                [20, 50, 200]
            )
            
            for ticker, data in datasets.items():
                fig = go.Figure()
                # Price line
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#2d3748', width=1)
                ))
                # EMA lines
                colors = ['#667eea', '#48bb78', '#ed8936', '#f56565', '#9f7aea', '#38b2ac']
                for i, period in enumerate(periods):
                    ema = calculate_ema(data, period)
                    fig.add_trace(go.Scatter(
                        x=data['Date'],
                        y=ema,
                        mode='lines',
                        name=f'EMA{period}',
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))
                
                fig.update_layout(
                    title=f"{ticker} Price with EMAs",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode="x unified",
                    legend=dict(x=0, y=1),
                    margin=dict(t=50, r=20, b=50, l=50)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif indicator_type == "Bollinger Bands":
            st.subheader("Bollinger Bands")
            period = st.slider("Bollinger Bands Period", min_value=5, max_value=50, value=20)
            
            for ticker, data in datasets.items():
                bb_data = calculate_bollinger_bands(data, period)
                
                fig = go.Figure()
                # Price line
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#2d3748', width=2)
                ))
                # Upper band
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=bb_data['upper'],
                    mode='lines',
                    name='Upper Band',
                    line=dict(color='#f56565', width=1, dash='dash')
                ))
                # Middle band (SMA)
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=bb_data['sma'],
                    mode='lines',
                    name='Middle Band (SMA)',
                    line=dict(color='#667eea', width=1, dash='dash')
                ))
                # Lower band
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=bb_data['lower'],
                    mode='lines',
                    name='Lower Band',
                    line=dict(color='#48bb78', width=1, dash='dash')
                ))
                # Fill between bands
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=bb_data['upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=bb_data['lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(102, 126, 234, 0.1)',
                    showlegend=False
                ))
                
                fig.update_layout(
                    title=f"{ticker} Bollinger Bands (Period: {period})",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode="x unified",
                    legend=dict(x=0, y=1),
                    margin=dict(t=50, r=20, b=50, l=50)
                )
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        strategy_type = st.selectbox(
            "Select Trading Strategy",
            ["MACD Strategy", "RSI Strategy", "MA Crossover", "Momentum"]
        )
    
    if datasets:
        if strategy_type == "MACD Strategy":
            st.subheader("MACD Trading Strategy")
            
            for ticker, data in datasets.items():
                macd_data = calculate_macd(data)
                
                # Generate signals
                signals = []
                for i in range(1, len(macd_data['macd'])):
                    if macd_data['macd'][i] > macd_data['signal'][i] and macd_data['macd'][i-1] <= macd_data['signal'][i-1]:
                        signals.append('Buy')
                    elif macd_data['macd'][i] < macd_data['signal'][i] and macd_data['macd'][i-1] >= macd_data['signal'][i-1]:
                        signals.append('Sell')
                    else:
                        signals.append('Hold')
                signals.insert(0, 'Hold')  # Add hold for first day
                
                # Create signal dataframe
                signal_df = pd.DataFrame({
                    'Date': data['Date'],
                    'Close': data['Close'],
                    'Signal': signals
                })
                
                # Plot with signals
                fig = go.Figure()
                # Price line
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#2d3748', width=2)
                ))
                # Buy signals
                buy_signals = signal_df[signal_df['Signal'] == 'Buy']
                fig.add_trace(go.Scatter(
                    x=buy_signals['Date'],
                    y=buy_signals['Close'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='#48bb78', size=10, symbol='triangle-up')
                ))
                # Sell signals
                sell_signals = signal_df[signal_df['Signal'] == 'Sell']
                fig.add_trace(go.Scatter(
                    x=sell_signals['Date'],
                    y=sell_signals['Close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color='#f56565', size=10, symbol='triangle-down')
                ))
                
                fig.update_layout(
                    title=f"{ticker} MACD Strategy Signals",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode="x unified",
                    legend=dict(x=0, y=1),
                    margin=dict(t=50, r=20, b=50, l=50)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show signal table (last 20 rows)
                st.subheader(f"{ticker} MACD Signals (Last 20)")
                st.dataframe(signal_df.tail(20), use_container_width=True)
        
        elif strategy_type == "RSI Strategy":
            st.subheader("RSI Trading Strategy")
            overbought = st.slider("Overbought Threshold", min_value=60, max_value=90, value=70)
            oversold = st.slider("Oversold Threshold", min_value=10, max_value=40, value=30)
            
            for ticker, data in datasets.items():
                rsi = calculate_rsi(data)
                
                # Generate signals
                signals = []
                for i in range(len(rsi)):
                    if rsi[i] is not None:
                        if rsi[i] < oversold:
                            signals.append('Buy')
                        elif rsi[i] > overbought:
                            signals.append('Sell')
                        else:
                            signals.append('Hold')
                    else:
                        signals.append('Hold')
                
                # Create signal dataframe
                signal_df = pd.DataFrame({
                    'Date': data['Date'],
                    'Close': data['Close'],
                    'RSI': rsi,
                    'Signal': signals
                })
                
                # Plot with signals
                fig = go.Figure()
                # Price line
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#2d3748', width=2),
                    yaxis='y'
                ))
                # RSI line
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=rsi,
                    mode='lines',
                    name='RSI',
                    line=dict(color='#667eea', width=2),
                    yaxis='y2'
                ))
                # Overbought/oversold lines
                fig.add_hline(y=overbought, line_dash="dash", line_color="#f56565", annotation_text="Overbought", yref="y2")
                fig.add_hline(y=oversold, line_dash="dash", line_color="#48bb78", annotation_text="Oversold", yref="y2")
                # Buy signals
                buy_signals = signal_df[signal_df['Signal'] == 'Buy']
                fig.add_trace(go.Scatter(
                    x=buy_signals['Date'],
                    y=buy_signals['Close'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='#48bb78', size=10, symbol='triangle-up'),
                    yaxis='y'
                ))
                # Sell signals
                sell_signals = signal_df[signal_df['Signal'] == 'Sell']
                fig.add_trace(go.Scatter(
                    x=sell_signals['Date'],
                    y=sell_signals['Close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color='#f56565', size=10, symbol='triangle-down'),
                    yaxis='y'
                ))
                
                fig.update_layout(
                    title=f"{ticker} RSI Strategy Signals",
                    xaxis_title="Date",
                    yaxis=dict(title="Price (USD)"),
                    yaxis2=dict(title="RSI", overlaying='y', side='right', range=[0, 100]),
                    hovermode="x unified",
                    legend=dict(x=0, y=1),
                    margin=dict(t=50, r=100, b=50, l=50)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show signal table (last 20 rows)
                st.subheader(f"{ticker} RSI Signals (Last 20)")
                st.dataframe(signal_df.tail(20), use_container_width=True)
        
        elif strategy_type == "MA Crossover":
            st.subheader("Moving Average Crossover Strategy")
            short_period = st.slider("Short MA Period", min_value=5, max_value=50, value=20)
            long_period = st.slider("Long MA Period", min_value=20, max_value=200, value=50)
            
            for ticker, data in datasets.items():
                # Calculate MAs
                short_ma = []
                long_ma = []
                
                for i in range(len(data)):
                    if i >= short_period - 1:
                        short_ma.append(data['Close'].iloc[i-short_period+1:i+1].mean())
                    else:
                        short_ma.append(None)
                    
                    if i >= long_period - 1:
                        long_ma.append(data['Close'].iloc[i-long_period+1:i+1].mean())
                    else:
                        long_ma.append(None)
                
                # Generate signals
                signals = []
                for i in range(1, len(short_ma)):
                    if short_ma[i] is not None and long_ma[i] is not None:
                        if short_ma[i] > long_ma[i] and short_ma[i-1] <= long_ma[i-1]:
                            signals.append('Buy')
                        elif short_ma[i] < long_ma[i] and short_ma[i-1] >= long_ma[i-1]:
                            signals.append('Sell')
                        else:
                            signals.append('Hold')
                    else:
                        signals.append('Hold')
                signals.insert(0, 'Hold')  # Add hold for first day
                
                # Create signal dataframe
                signal_df = pd.DataFrame({
                    'Date': data['Date'],
                    'Close': data['Close'],
                    f'Short MA ({short_period})': short_ma,
                    f'Long MA ({long_period})': long_ma,
                    'Signal': signals
                })
                
                # Plot with signals
                fig = go.Figure()
                # Price line
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#2d3748', width=1)
                ))
                # Short MA
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=short_ma,
                    mode='lines',
                    name=f'Short MA ({short_period})',
                    line=dict(color='#48bb78', width=2)
                ))
                # Long MA
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=long_ma,
                    mode='lines',
                    name=f'Long MA ({long_period})',
                    line=dict(color='#f56565', width=2)
                ))
                # Buy signals
                buy_signals = signal_df[signal_df['Signal'] == 'Buy']
                fig.add_trace(go.Scatter(
                    x=buy_signals['Date'],
                    y=buy_signals['Close'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='#48bb78', size=10, symbol='triangle-up')
                ))
                # Sell signals
                sell_signals = signal_df[signal_df['Signal'] == 'Sell']
                fig.add_trace(go.Scatter(
                    x=sell_signals['Date'],
                    y=sell_signals['Close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color='#f56565', size=10, symbol='triangle-down')
                ))
                
                fig.update_layout(
                    title=f"{ticker} MA Crossover Strategy Signals",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode="x unified",
                    legend=dict(x=0, y=1),
                    margin=dict(t=50, r=20, b=50, l=50)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show signal table (last 20 rows)
                st.subheader(f"{ticker} MA Crossover Signals (Last 20)")
                st.dataframe(signal_df.tail(20), use_container_width=True)
        
        elif strategy_type == "Momentum":
            st.subheader("Momentum Strategy")
            lookback_period = st.slider("Lookback Period", min_value=1, max_value=30, value=10)
            
            for ticker, data in datasets.items():
                # Calculate momentum
                momentum = []
                for i in range(len(data)):
                    if i >= lookback_period:
                        mom = (data['Close'].iloc[i] - data['Close'].iloc[i-lookback_period]) / data['Close'].iloc[i-lookback_period] * 100
                        momentum.append(mom)
                    else:
                        momentum.append(None)
                
                # Generate signals
                signals = []
                for i in range(len(momentum)):
                    if momentum[i] is not None:
                        if momentum[i] > 0:
                            signals.append('Buy')
                        else:
                            signals.append('Sell')
                    else:
                        signals.append('Hold')
                
                # Create signal dataframe
                signal_df = pd.DataFrame({
                    'Date': data['Date'],
                    'Close': data['Close'],
                    f'Momentum ({lookback_period} days)': momentum,
                    'Signal': signals
                })
                
                # Plot with signals
                fig = go.Figure()
                # Price line
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#2d3748', width=2),
                    yaxis='y'
                ))
                # Momentum line
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=momentum,
                    mode='lines',
                    name=f'Momentum ({lookback_period} days)',
                    line=dict(color='#667eea', width=2),
                    yaxis='y2'
                ))
                # Zero line
                fig.add_hline(y=0, line_dash="dash", line_color="#718096", yref="y2")
                # Buy signals
                buy_signals = signal_df[signal_df['Signal'] == 'Buy']
                fig.add_trace(go.Scatter(
                    x=buy_signals['Date'],
                    y=buy_signals['Close'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='#48bb78', size=10, symbol='triangle-up'),
                    yaxis='y'
                ))
                # Sell signals
                sell_signals = signal_df[signal_df['Signal'] == 'Sell']
                fig.add_trace(go.Scatter(
                    x=sell_signals['Date'],
                    y=sell_signals['Close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color='#f56565', size=10, symbol='triangle-down'),
                    yaxis='y'
                ))
                
                fig.update_layout(
                    title=f"{ticker} Momentum Strategy Signals",
                    xaxis_title="Date",
                    yaxis=dict(title="Price (USD)"),
                    yaxis2=dict(title=f"Momentum ({lookback_period} days, %)", overlaying='y', side='right'),
                    hovermode="x unified",
                    legend=dict(x=0, y=1),
                    margin=dict(t=50, r=100, b=50, l=50)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show signal table (last 20 rows)
                st.subheader(f"{ticker} Momentum Signals (Last 20)")
                st.dataframe(signal_df.tail(20), use_container_width=True)

with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        risk_type = st.selectbox(
            "Select Risk Management Tool",
            ["Position Sizing", "Risk/Reward", "Value at Risk", "Portfolio Risk"]
        )
    
    if datasets:
        if risk_type == "Position Sizing":
            st.subheader("Position Sizing")
            
            for ticker, data in datasets.items():
                current_price = data['Close'].iloc[-1]
                # Calculate position size
                risk_amount = initial_capital * (risk_per_trade / 100)
                position_size = risk_amount / (current_price * (stop_loss / 100))
                
                # Display results
                st.write(f"**{ticker} Position Sizing**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric("Risk Amount", f"${risk_amount:.2f}")
                with col3:
                    st.metric("Position Size", f"{position_size:.2f} shares")
                with col4:
                    st.metric("Position Value", f"${position_size * current_price:.2f}")
        
        elif risk_type == "Risk/Reward":
            st.subheader("Risk/Reward Analysis")
            
            for ticker, data in datasets.items():
                current_price = data['Close'].iloc[-1]
                stop_price = current_price * (1 - stop_loss / 100)
                take_price = current_price * (1 + take_profit / 100)
                risk_reward_ratio = take_profit / stop_loss
                
                # Display results
                st.write(f"**{ticker} Risk/Reward Analysis**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric("Stop Loss Price", f"${stop_price:.2f}")
                with col3:
                    st.metric("Take Profit Price", f"${take_price:.2f}")
                with col4:
                    st.metric("Risk/Reward Ratio", f"{risk_reward_ratio:.2f}")
        
        elif risk_type == "Value at Risk":
            st.subheader("Value at Risk (VaR)")
            confidence_level = st.slider("Confidence Level", min_value=90, max_value=99, value=95)
            
            for ticker, data in datasets.items():
                returns = calculate_returns(data)
                if returns:
                    # Calculate VaR
                    var = np.percentile(returns, 100 - confidence_level)
                    daily_var = initial_capital * (var / 100)
                    annual_var = daily_var * math.sqrt(252)
                    
                    # Display results
                    st.write(f"**{ticker} Value at Risk**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{confidence}% Daily VaR", f"${daily_var:.2f}")
                    with col2:
                        st.metric(f"{confidence}% Annual VaR", f"${annual_var:.2f}")
                    with col3:
                        st.metric(f"{confidence}% Daily Return", f"{var:.2f}%")
                else:
                    st.write(f"**{ticker}** - Insufficient data for VaR calculation")
        
        elif risk_type == "Portfolio Risk":
            st.subheader("Portfolio Risk Analysis")
            
            if len(datasets) < 2:
                st.warning("Need at least 2 datasets for portfolio risk analysis")
            else:
                # Calculate portfolio statistics
                tickers = list(datasets.keys())
                returns_data = {}
                min_length = float('inf')
                
                for ticker in tickers:
                    returns = calculate_returns(datasets[ticker])
                    returns_data[ticker] = returns
                    min_length = min(min_length, len(returns))
                
                # Align returns
                for ticker in tickers:
                    returns_data[ticker] = returns_data[ticker][-min_length:]
                
                # Create returns matrix
                returns_matrix = np.array([returns_data[ticker] for ticker in tickers]).T
                
                # Calculate portfolio metrics (equal weights)
                weights = np.ones(len(tickers)) / len(tickers)
                portfolio_returns = np.dot(returns_matrix, weights)
                portfolio_mean = np.mean(portfolio_returns)
                portfolio_volatility = np.std(portfolio_returns) * math.sqrt(252)
                portfolio_sharpe = ((portfolio_mean * 252) - 2) / portfolio_volatility if portfolio_volatility > 0 else 0
                
                # Calculate max drawdown
                portfolio_cumulative = np.cumprod(1 + portfolio_returns / 100)
                peak = portfolio_cumulative[0]
                max_dd = 0
                for value in portfolio_cumulative:
                    if value > peak:
                        peak = value
                    dd = ((value - peak) / peak) * 100
                    if dd < max_dd:
                        max_dd = dd
                
                # Display results
                st.write("**Portfolio Risk Metrics (Equal Weights)**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Annual Return", f"{(portfolio_mean * 252):.2f}%")
                with col2:
                    st.metric("Annual Volatility", f"{portfolio_volatility:.2f}%")
                with col3:
                    st.metric("Sharpe Ratio", f"{portfolio_sharpe:.2f}")
                with col4:
                    st.metric("Max Drawdown", f"{max_dd:.2f}%")
                
                # Plot portfolio cumulative returns
                dates = datasets[tickers[0]]['Date'].iloc[-min_length:]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=(portfolio_cumulative - 1) * 100,
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='#667eea', width=2)
                ))
                
                # Add individual assets for comparison
                for ticker in tickers:
                    asset_returns = returns_data[ticker]
                    asset_cumulative = np.cumprod(1 + np.array(asset_returns) / 100)
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=(asset_cumulative - 1) * 100,
                        mode='lines',
                        name=ticker,
                        line=dict(width=1, dash='dash')
                    ))
                
                fig.update_layout(
                    title="Portfolio vs Individual Assets Cumulative Returns",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return (%)",
                    hovermode="x unified",
                    legend=dict(x=0, y=1),
                    margin=dict(t=50, r=20, b=50, l=50)
                )
                st.plotly_chart(fig, use_container_width=True)
