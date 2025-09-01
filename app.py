import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Technical Analysis Stock Screener",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Technical Analysis Stock Screener")
st.markdown("**Filter stocks based on advanced technical analysis patterns and indicators**")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Stock list input
    st.subheader("Stock Selection")
    default_stocks = "AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,NFLX,AMD,INTC,CRM,ADBE,ORCL,PYPL,SHOP,UBER,LYFT,ZOOM,ROKU,SQ"
    stock_input = st.text_area(
        "Enter stock symbols (comma-separated):",
        value=default_stocks,
        height=100,
        help="Enter stock symbols separated by commas"
    )
    
    # Time period
    period = st.selectbox(
        "Analysis Period:",
        options=["1y", "6mo", "3mo", "2y"],
        index=0,
        help="Period for historical data analysis"
    )
    
    # RSI Configuration
    st.subheader("RSI Settings")
    rsi_period = st.number_input("RSI Period", value=14, min_value=5, max_value=50)
    rsi_lower = st.number_input("RSI Lower Bound", value=20, min_value=0, max_value=50)
    rsi_upper = st.number_input("RSI Upper Bound", value=25, min_value=0, max_value=50)
    
    # MACD Configuration
    st.subheader("MACD Settings")
    macd_fast = st.number_input("MACD Fast Period", value=12, min_value=5, max_value=30)
    macd_slow = st.number_input("MACD Slow Period", value=26, min_value=20, max_value=50)
    macd_signal = st.number_input("MACD Signal Period", value=9, min_value=5, max_value=20)
    
    # Volume threshold for breakouts
    volume_multiplier = st.number_input(
        "Volume Breakout Multiplier", 
        value=1.5, 
        min_value=1.0, 
        max_value=5.0, 
        step=0.1,
        help="Volume must be X times the average volume"
    )

# Helper functions
@st.cache_data
def get_stock_data(symbol, period):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        if len(data) < 50:  # Need minimum data points
            return None
        return data
    except:
        return None

def calculate_rsi(data, period=14):
    """Calculate RSI"""
    return ta.momentum.RSIIndicator(data['Close'], window=period).rsi()

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    macd_indicator = ta.trend.MACD(data['Close'], window_fast=fast, window_slow=slow, window_sign=signal)
    return macd_indicator.macd(), macd_indicator.macd_signal()

def detect_rounding_bottom(data, window=20):
    """Detect rounding bottom pattern"""
    if len(data) < window * 3:
        return False
    
    # Get recent lows
    lows = data['Low'].rolling(window=window).min()
    recent_data = data.tail(window * 2)
    
    # Check if price has been forming a curved bottom
    # Look for gradual increase in lows over time
    recent_lows = recent_data['Low'].values
    
    # Simple check: if recent lows show an upward trend
    if len(recent_lows) >= 10:
        first_half = np.mean(recent_lows[:len(recent_lows)//2])
        second_half = np.mean(recent_lows[len(recent_lows)//2:])
        return second_half > first_half
    
    return False

def detect_head_shoulders(data, window=20):
    """Detect Head and Shoulders pattern"""
    if len(data) < window * 3:
        return False
    
    # Get recent highs and lows
    recent_data = data.tail(window * 3)
    highs = recent_data['High'].values
    
    # Find local peaks
    peaks = []
    for i in range(1, len(highs) - 1):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            peaks.append((i, highs[i]))
    
    # Need at least 3 peaks for head and shoulders
    if len(peaks) >= 3:
        # Sort peaks by height
        peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
        highest_peak = peaks_sorted[0]
        
        # Check if highest peak is in the middle (head)
        left_peaks = [p for p in peaks if p[0] < highest_peak[0]]
        right_peaks = [p for p in peaks if p[0] > highest_peak[0]]
        
        return len(left_peaks) >= 1 and len(right_peaks) >= 1
    
    return False

def detect_breakout_with_volume(data, volume_multiplier=1.5, window=20):
    """Detect price breakout with high volume"""
    if len(data) < window + 5:
        return False
    
    recent_data = data.tail(5)
    historical_data = data.tail(window + 5).head(window)
    
    # Calculate average volume
    avg_volume = historical_data['Volume'].mean()
    recent_volume = recent_data['Volume'].max()
    
    # Check for volume breakout
    volume_breakout = recent_volume > avg_volume * volume_multiplier
    
    # Check for price breakout (breaking resistance)
    resistance_level = historical_data['High'].quantile(0.95)
    recent_high = recent_data['High'].max()
    price_breakout = recent_high > resistance_level
    
    return volume_breakout and price_breakout

def analyze_stock(symbol, period, rsi_period, rsi_lower, rsi_upper, 
                 macd_fast, macd_slow, macd_signal, volume_multiplier):
    """Perform complete technical analysis on a stock"""
    
    data = get_stock_data(symbol, period)
    if data is None:
        return None
    
    # Calculate indicators
    rsi = calculate_rsi(data, rsi_period)
    macd, macd_signal_line = calculate_macd(data, macd_fast, macd_slow, macd_signal)
    
    # Get latest values
    latest_rsi = rsi.iloc[-1] if not rsi.empty else None
    latest_macd = macd.iloc[-1] if not macd.empty else None
    latest_macd_signal = macd_signal_line.iloc[-1] if not macd_signal_line.empty else None
    
    # Check criteria
    criteria_met = []
    
    # 1. RSI in range
    if latest_rsi and rsi_lower <= latest_rsi <= rsi_upper:
        criteria_met.append("RSI Oversold")
    
    # 2. MACD crossover (MACD above signal line in recent periods)
    if latest_macd and latest_macd_signal:
        if len(macd) >= 2 and len(macd_signal_line) >= 2:
            prev_macd = macd.iloc[-2]
            prev_signal = macd_signal_line.iloc[-2]
            
            # Bullish crossover: MACD crosses above signal line
            if prev_macd <= prev_signal and latest_macd > latest_macd_signal:
                criteria_met.append("MACD Bullish Crossover")
    
    # 3. Rounding bottom
    if detect_rounding_bottom(data):
        criteria_met.append("Rounding Bottom")
    
    # 4. Head and shoulders
    if detect_head_shoulders(data):
        criteria_met.append("Head & Shoulders")
    
    # 5. Breakout with volume
    if detect_breakout_with_volume(data, volume_multiplier):
        criteria_met.append("Volume Breakout")
    
    if criteria_met:
        return {
            'symbol': symbol,
            'current_price': data['Close'].iloc[-1],
            'rsi': latest_rsi,
            'macd': latest_macd,
            'macd_signal': latest_macd_signal,
            'volume': data['Volume'].iloc[-1],
            'criteria_met': criteria_met,
            'data': data
        }
    
    return None

def create_stock_chart(stock_data, symbol):
    """Create an interactive chart for a stock"""
    data = stock_data['data']
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxis=True,
        vertical_spacing=0.02,
        subplot_titles=(f'{symbol} - Price & Volume', 'RSI', 'MACD'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price and volume
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Volume bars
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            yaxis='y2',
            opacity=0.3
        ),
        row=1, col=1
    )
    
    # RSI
    rsi = calculate_rsi(data)
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=rsi,
            name='RSI',
            line=dict(color='purple')
        ),
        row=2, col=1
    )
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=rsi_upper, line_dash="dash", line_color="orange", row=2, col=1)
    fig.add_hline(y=rsi_lower, line_dash="dash", line_color="orange", row=2, col=1)
    
    # MACD
    macd, macd_signal_line = calculate_macd(data)
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=macd,
            name='MACD',
            line=dict(color='blue')
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=macd_signal_line,
            name='MACD Signal',
            line=dict(color='red')
        ),
        row=3, col=1
    )
    
    # MACD histogram
    macd_histogram = macd - macd_signal_line
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=macd_histogram,
            name='MACD Histogram',
            opacity=0.6
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        title=f'{symbol} Technical Analysis',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )
    
    return fig

# Main analysis
if st.button("🔍 Analyze Stocks", type="primary"):
    stocks = [s.strip().upper() for s in stock_input.split(',') if s.strip()]
    
    if not stocks:
        st.error("Please enter at least one stock symbol")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i, symbol in enumerate(stocks):
            status_text.text(f'Analyzing {symbol}... ({i+1}/{len(stocks)})')
            progress_bar.progress((i + 1) / len(stocks))
            
            result = analyze_stock(
                symbol, period, rsi_period, rsi_lower, rsi_upper,
                macd_fast, macd_slow, macd_signal, volume_multiplier
            )
            
            if result:
                results.append(result)
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            st.success(f"Found {len(results)} stocks matching your criteria!")
            
            # Display results
            for result in results:
                with st.expander(f"📊 {result['symbol']} - {', '.join(result['criteria_met'])}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", f"${result['current_price']:.2f}")
                        st.metric("RSI", f"{result['rsi']:.2f}" if result['rsi'] else "N/A")
                    
                    with col2:
                        st.metric("MACD", f"{result['macd']:.4f}" if result['macd'] else "N/A")
                        st.metric("MACD Signal", f"{result['macd_signal']:.4f}" if result['macd_signal'] else "N/A")
                    
                    with col3:
                        st.metric("Volume", f"{result['volume']:,.0f}")
                        st.write("**Criteria Met:**")
                        for criterion in result['criteria_met']:
                            st.write(f"✅ {criterion}")
                    
                    # Display chart
                    chart = create_stock_chart(result, result['symbol'])
                    st.plotly_chart(chart, use_container_width=True)
        else:
            st.warning("No stocks found matching your criteria. Try adjusting the parameters.")

# Information section
with st.expander("ℹ️ How it works"):
    st.markdown("""
    This technical analysis screener filters stocks based on five key criteria:
    
    **1. RSI Index (20-25 range):** 
    - Identifies oversold conditions that may indicate buying opportunities
    - RSI below 30 is typically considered oversold
    
    **2. MACD Crossover:**
    - Detects when the MACD line crosses above the signal line
    - This is often considered a bullish signal
    
    **3. Rounding Bottom Pattern:**
    - Identifies stocks forming a curved bottom pattern
    - Indicates potential trend reversal from bearish to bullish
    
    **4. Head and Shoulders Pattern:**
    - Detects this classic reversal pattern
    - Can indicate trend changes
    
    **5. Volume Breakouts:**
    - Finds stocks breaking resistance levels with high volume
    - Volume confirms the strength of price movements
    
    **Note:** This tool is for educational purposes. Always do your own research and consider consulting with a financial advisor before making investment decisions.
    """)

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit • Data from Yahoo Finance*")
