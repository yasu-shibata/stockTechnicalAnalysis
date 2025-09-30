"""
Streamlit Stock Technical Analysis App
Multi-stock analyzer with MACD crossover tracking and sector comparison
"""

import streamlit as st
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Technical Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Matplotlib settings
plt.rcParams['font.family'] = 'sans-serif'

# Sector ETF mapping
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financial Services': 'XLF',
    'Consumer Cyclical': 'XLY',
    'Consumer Defensive': 'XLP',
    'Energy': 'XLE',
    'Industrials': 'XLI',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Utilities': 'XLU',
    'Communication Services': 'XLC',
    'Basic Materials': 'XLB'
}

# Cache functions
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data with caching"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        if not data.empty:
            return data.dropna(), None
        return None, "No data available"
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=3600)
def get_stock_info(symbol):
    """Get stock information with caching"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A')
        }
    except:
        return {'name': symbol, 'sector': 'N/A', 'industry': 'N/A'}

def calculate_macd_crossover_days(macd_series, signal_series):
    """Calculate days since MACD crossover"""
    days_since_crossover = []
    last_crossover_idx = None
    
    for i in range(len(macd_series)):
        if i == 0:
            days_since_crossover.append(np.nan)
            continue
        
        prev_diff = macd_series.iloc[i-1] - signal_series.iloc[i-1]
        curr_diff = macd_series.iloc[i] - signal_series.iloc[i]
        
        if (prev_diff <= 0 and curr_diff > 0) or (prev_diff >= 0 and curr_diff < 0):
            last_crossover_idx = i
            days_since_crossover.append(0)
        elif last_crossover_idx is not None:
            days_since_crossover.append(i - last_crossover_idx)
        else:
            days_since_crossover.append(np.nan)
    
    return pd.Series(days_since_crossover, index=macd_series.index)

def calculate_technical_indicators(data):
    """Calculate all technical indicators"""
    df = data.copy()
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # EMA
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    df['MACD_Crossover_Days'] = calculate_macd_crossover_days(df['MACD'], df['MACD_Signal'])
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Volume
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # ATR
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    return df

def calculate_signal_score(data):
    """Calculate overall signal score"""
    latest = data.iloc[-1]
    
    buy_signals = 0
    sell_signals = 0
    
    if latest['Close'] > latest['SMA_20']:
        buy_signals += 1
    else:
        sell_signals += 1
    
    if latest['RSI'] < 30:
        buy_signals += 2
    elif latest['RSI'] > 70:
        sell_signals += 2
    
    if latest['MACD'] > latest['MACD_Signal']:
        buy_signals += 1
    else:
        sell_signals += 1
    
    if buy_signals > sell_signals:
        return "BULLISH", "green"
    elif sell_signals > buy_signals:
        return "BEARISH", "red"
    else:
        return "NEUTRAL", "gray"

def plot_price_chart(data, symbol, info):
    """Create price chart with indicators"""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), 
                                              gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # Price chart
    ax1.plot(data.index, data['Close'], linewidth=2, label='Close', color='black')
    ax1.plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7, color='blue')
    ax1.plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.7, color='red')
    ax1.fill_between(data.index, data['BB_Upper'], data['BB_Lower'], alpha=0.1, color='gray')
    ax1.set_title(f"{symbol} - {info['name']}", fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Volume
    colors = ['green' if c >= o else 'red' for c, o in zip(data['Close'], data['Open'])]
    ax2.bar(data.index, data['Volume'], color=colors, alpha=0.7)
    ax2.plot(data.index, data['Volume_MA'], color='orange', linewidth=2, label='MA 20')
    ax2.set_ylabel('Volume')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # RSI
    ax3.plot(data.index, data['RSI'], color='purple', linewidth=2)
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7)
    ax3.axhline(y=30, color='blue', linestyle='--', alpha=0.7)
    ax3.fill_between(data.index, 30, 70, alpha=0.1, color='gray')
    ax3.set_ylabel('RSI')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    
    # MACD
    ax4.plot(data.index, data['MACD'], color='blue', linewidth=2, label='MACD')
    ax4.plot(data.index, data['MACD_Signal'], color='red', linewidth=2, label='Signal')
    colors_macd = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
    ax4.bar(data.index, data['MACD_Histogram'], alpha=0.3, color=colors_macd)
    ax4.axhline(y=0, color='black', alpha=0.5)
    ax4.set_ylabel('MACD')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_comparison_chart(data_dict, symbols):
    """Create comparison chart"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Normalized price comparison
    for symbol in symbols:
        if symbol in data_dict:
            data = data_dict[symbol]
            normalized = data['Close'] / data['Close'].iloc[0]
            ax1.plot(data.index, normalized, label=symbol, linewidth=2)
    
    ax1.set_title('Normalized Price Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Normalized Price (Base = 1.0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RSI comparison
    for symbol in symbols:
        if symbol in data_dict:
            data = data_dict[symbol]
            ax2.plot(data.index, data['RSI'], label=symbol, alpha=0.8)
    
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(y=30, color='blue', linestyle='--', alpha=0.7)
    ax2.set_title('RSI Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Main App
def main():
    st.title("📈 Stock Technical Analysis Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Analysis mode
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Single Stock", "Multiple Stocks", "Stock + Sector ETF"]
        )
        
        # Symbol input
        if analysis_mode == "Single Stock" or analysis_mode == "Stock + Sector ETF":
            symbol_input = st.text_input(
                "Stock Symbol",
                value="AAPL",
                help="Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)"
            ).upper()
            symbols = [symbol_input]
        else:
            symbols_text = st.text_area(
                "Stock Symbols (one per line)",
                value="AAPL\nMSFT\nGOOGL",
                help="Enter stock symbols, one per line"
            )
            symbols = [s.strip().upper() for s in symbols_text.split('\n') if s.strip()]
            symbols = symbols[:10]  # Limit to 10 stocks
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=dt.date.today() - dt.timedelta(days=365)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=dt.date.today()
            )
        
        # Analyze button
        analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)
    
    # Main content
    if analyze_button:
        if not symbols:
            st.error("Please enter at least one stock symbol")
            return
        
        # Fetch data
        with st.spinner(f"Fetching data for {len(symbols)} symbol(s)..."):
            data_dict = {}
            info_dict = {}
            errors = []
            
            progress_bar = st.progress(0)
            for idx, symbol in enumerate(symbols):
                data, error = fetch_stock_data(symbol, start_date, end_date)
                if data is not None:
                    data_dict[symbol] = calculate_technical_indicators(data)
                    info_dict[symbol] = get_stock_info(symbol)
                else:
                    errors.append(f"{symbol}: {error}")
                progress_bar.progress((idx + 1) / len(symbols))
            
            progress_bar.empty()
        
        # Handle errors
        if errors:
            with st.expander("⚠️ Errors", expanded=False):
                for error in errors:
                    st.warning(error)
        
        if not data_dict:
            st.error("No data could be fetched. Please check your symbols and try again.")
            return
        
        # Add sector ETF if requested
        if analysis_mode == "Stock + Sector ETF" and len(symbols) == 1:
            symbol = symbols[0]
            info = info_dict.get(symbol, {})
            sector = info.get('sector')
            
            if sector and sector in SECTOR_ETFS:
                etf_symbol = SECTOR_ETFS[sector]
                st.info(f"Adding sector ETF: {etf_symbol} ({sector})")
                
                etf_data, etf_error = fetch_stock_data(etf_symbol, start_date, end_date)
                if etf_data is not None:
                    data_dict[etf_symbol] = calculate_technical_indicators(etf_data)
                    info_dict[etf_symbol] = get_stock_info(etf_symbol)
                    symbols.append(etf_symbol)
        
        # Display analysis
        if analysis_mode == "Multiple Stocks" or len(symbols) > 1:
            # Comparison view
            st.header("📊 Comparison Analysis")
            
            # Summary table
            summary_data = []
            for symbol in symbols:
                if symbol in data_dict:
                    data = data_dict[symbol]
                    latest = data.iloc[-1]
                    prev = data.iloc[-2] if len(data) > 1 else latest
                    signal, _ = calculate_signal_score(data)
                    
                    crossover_str = ""
                    if not np.isnan(latest['MACD_Crossover_Days']):
                        cross_type = "Bull" if latest['MACD'] > latest['MACD_Signal'] else "Bear"
                        crossover_str = f"{int(latest['MACD_Crossover_Days'])}d {cross_type}"
                    
                    summary_data.append({
                        'Symbol': symbol,
                        'Price': f"${latest['Close']:.2f}",
                        'Change %': f"{((latest['Close']/prev['Close'])-1)*100:+.2f}%",
                        'RSI': f"{latest['RSI']:.1f}",
                        'MACD': f"{latest['MACD_Histogram']:.3f}",
                        'Crossover': crossover_str,
                        'Signal': signal
                    })
            
            df_summary = pd.DataFrame(summary_data)
            
            # Color code the Signal column
            def color_signal(val):
                if val in ['Strong Buy']:
                    return 'background-color: #006400; color: white'
                elif val in ['Buy']:
                    return 'background-color: #90EE90'
                elif val in ['Neutral']:
                    return 'background-color: #D3D3D3'
                elif val in ['Sell']:
                    return 'background-color: #FFA500'
                elif val in ['Strong Sell']:
                    return 'background-color: #FF0000; color: white'
                else:
                    return ''
            
            st.dataframe(
                df_summary.style.applymap(color_signal, subset=['Buy Signal']),
                use_container_width=True,
                hide_index=True
            )
            
            # Comparison chart
            st.subheader("📈 Price & RSI Comparison")
            fig_comparison = plot_comparison_chart(data_dict, symbols)
            st.pyplot(fig_comparison)
            plt.close()
            
        # Individual analysis
        st.header("🔍 Detailed Analysis")
        
        for symbol in symbols:
            if symbol not in data_dict:
                continue
            
            data = data_dict[symbol]
            info = info_dict.get(symbol, {})
            latest = data.iloc[-1]
            prev = data.iloc[-2] if len(data) > 1 else latest
            
            with st.expander(f"📊 {symbol} - {info['name']}", expanded=(len(symbols) == 1)):
                # Metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    price_change = ((latest['Close']/prev['Close'])-1)*100
                    st.metric(
                        "Price",
                        f"${latest['Close']:.2f}",
                        f"{price_change:+.2f}%"
                    )
                
                with col2:
                    st.metric("RSI", f"{latest['RSI']:.1f}")
                
                with col3:
                    st.metric("MACD", f"{latest['MACD']:.4f}")
                
                with col4:
                    if not np.isnan(latest['MACD_Crossover_Days']):
                        cross_days = int(latest['MACD_Crossover_Days'])
                        cross_type = "Bull" if latest['MACD'] > latest['MACD_Signal'] else "Bear"
                        st.metric("MACD Cross", f"{cross_days}d", cross_type)
                    else:
                        st.metric("MACD Cross", "N/A")
                
                with col5:
                    signal, color = calculate_signal_score(data)
                    st.metric("Signal", signal)
                
                # Analysis text
                st.markdown("#### 📋 Technical Summary")
                
                # Trend
                if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
                    trend = "Strong Uptrend"
                    trend_emoji = "🚀"
                elif latest['Close'] > latest['SMA_20']:
                    trend = "Uptrend"
                    trend_emoji = "📈"
                else:
                    trend = "Downtrend"
                    trend_emoji = "📉"
                
                st.write(f"{trend_emoji} **Trend:** {trend}")
                
                # RSI
                if latest['RSI'] > 70:
                    st.write(f"⚠️ **RSI:** Overbought ({latest['RSI']:.1f})")
                elif latest['RSI'] < 30:
                    st.write(f"🎯 **RSI:** Oversold ({latest['RSI']:.1f})")
                else:
                    st.write(f"➡️ **RSI:** Neutral ({latest['RSI']:.1f})")
                
                # MACD
                macd_status = "Bullish" if latest['MACD'] > latest['MACD_Signal'] else "Bearish"
                st.write(f"📊 **MACD:** {macd_status} (Histogram: {latest['MACD_Histogram']:.4f})")
                
                # Volume
                if latest['Volume_Ratio'] > 2.0:
                    st.write(f"🔥 **Volume:** High ({latest['Volume_Ratio']:.2f}x average)")
                elif latest['Volume_Ratio'] < 0.5:
                    st.write(f"📉 **Volume:** Low ({latest['Volume_Ratio']:.2f}x average)")
                
                # Chart
                st.markdown("#### 📈 Technical Chart")
                fig = plot_price_chart(data, symbol, info)
                st.pyplot(fig)
                plt.close()
    
    else:
        # Welcome message
        st.info("👈 Configure your analysis settings in the sidebar and click 'Analyze' to begin")
        
        st.markdown("""
        ### Features
        - Single or multi-stock analysis
        - **Buy Signal Scoring (0-10)**: Comprehensive buy timing analysis
        - MACD crossover tracking with days since crossover
        - Comprehensive technical indicators (RSI, Bollinger Bands, Volume, ATR)
        - Sector ETF comparison
        - Visual charts and metrics
        
        ### How to Use
        1. Select analysis mode in the sidebar
        2. Enter stock symbol(s)
        3. Choose date range
        4. Click 'Analyze'
        
        ### Buy Signal Scoring
        
        **Score Range:**
        - **7-10 (Strong Buy)**: Multiple bullish signals aligned
        - **5-6 (Buy)**: Generally positive momentum
        - **3-4 (Neutral)**: Mixed signals
        - **1-2 (Sell)**: Bearish indicators present
        - **0 (Strong Sell)**: Multiple bearish signals
        
        **Scoring Factors:**
        - RSI oversold (<30): +3 points
        - MACD bullish crossover: +2 points
        - Recent bullish cross (≤5 days): +1 point
        - Strong uptrend: +2 points
        - Near lower Bollinger Band: +2 points
        - (Negative points for bearish signals)
        
        ### Popular Symbols
        - Tech: AAPL, MSFT, GOOGL, AMZN, NVDA
        - ETFs: SPY, QQQ, VOO, VTI, VOOG
        - Sector ETFs: XLK, XLV, XLF, XLE, XLI
        """)

if __name__ == "__main__":
    main()