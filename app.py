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
        return {'name': symbol, 'sector': symbol, 'industry': 'N/A'}

def calculate_returns(data):
    """Calculate YTD and 1Y returns"""
    latest_date = data.index[-1]
    latest_price = data['Close'].iloc[-1]
    
    # YTD return
    # 最新日付のタイムゾーン情報を取得
    tz = latest_date.tz
    # 年初の日付を作成し、タイムゾーンを適用
    year_start = pd.Timestamp(latest_date.year, 1, 1, tz=tz)
    
    ytd_data = data[data.index >= year_start]
    if len(ytd_data) > 0:
        ytd_start_price = ytd_data['Close'].iloc[0]
        ytd_return = ((latest_price / ytd_start_price) - 1) * 100
    else:
        ytd_return = None
    
    # 1Y return
    one_year_ago = latest_date - pd.Timedelta(days=365)
    one_year_data = data[data.index >= one_year_ago]
    if len(one_year_data) > 0:
        one_year_start_price = one_year_data['Close'].iloc[0]
        one_year_return = ((latest_price / one_year_start_price) - 1) * 100
    else:
        one_year_return = None
    
    return ytd_return, one_year_return

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
        
        # Crossover detected
        if (prev_diff <= 0 and curr_diff > 0) or (prev_diff >= 0 and curr_diff < 0):
            last_crossover_idx = i
            days_since_crossover.append(0)
        elif last_crossover_idx is not None:
            days_since_crossover.append(i - last_crossover_idx)
        else:
            days_since_crossover.append(np.nan)
    
    return pd.Series(days_since_crossover, index=macd_series.index)

def calculate_technical_indicators(data):
    """Calculate all technical indicators including Bollinger Bands (BB)"""
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
    
    # Bollinger Bands (BB) - Window 20, Std Dev 2
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Middle'] = df['SMA_20'] # SMA_20と同じ
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # BB Width Calculation (for Squeeze/Expansion analysis)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    
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
    
    # SMA 20 Check
    if latest['Close'] > latest['SMA_20']:
        buy_signals += 1
    else:
        sell_signals += 1
    
    # RSI Check (Overbought/Oversold has higher weight)
    if latest['RSI'] < 30:
        buy_signals += 2
    elif latest['RSI'] > 70:
        sell_signals += 2
    
    # MACD Check
    if latest['MACD'] > latest['MACD_Signal']:
        buy_signals += 1
    else:
        sell_signals += 1
        
    # Bollinger Band Check (Band Walk/Overbought/Oversold)
    if latest['Close'] > latest['BB_Upper']:
        sell_signals += 1 # Overbought signal
    elif latest['Close'] < latest['BB_Lower']:
        buy_signals += 1 # Oversold signal
    
    if buy_signals > sell_signals:
        return "BULLISH", "green"
    elif sell_signals > buy_signals:
        return "BEARISH", "red"
    else:
        return "NEUTRAL", "gray"

def plot_price_chart(data, symbol, info):
    """Create price chart with indicators including BB"""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), 
                                              gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # Price chart with Moving Averages and Bollinger Bands
    ax1.plot(data.index, data['Close'], linewidth=2, label='Close', color='black')
    ax1.plot(data.index, data['SMA_20'], label='SMA 20 (BB Middle)', alpha=0.7, color='blue')
    ax1.plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.7, color='red')
    
    # Bollinger Bands
    ax1.plot(data.index, data['BB_Upper'], label='+2σ Band', linestyle=':', color='gray', alpha=0.7)
    ax1.plot(data.index, data['BB_Lower'], label='-2σ Band', linestyle=':', color='gray', alpha=0.7)
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

# --- New Bollinger Band Analysis Functionality ---
def analyze_bollinger_bands(data):
    """
    ボリンジャーバンドの現在の状態を分析し、テキストで返す
    - 価格がバンドのどこにあるか
    - ボラティリティの状態 (スクイーズ/エクスパンション)
    """
    latest = data.iloc[-1]
    
    # 必要なデータが存在しない場合は早期リターン
    if pd.isna(latest['BB_Upper']) or pd.isna(latest['BB_Lower']) or data['BB_Width'].isnull().all():
        return "❓ データ不足のため、ボリンジャーバンドの分析はできません。", "❓"
    
    # 1. ボラティリティの状態 (収縮 Squeeze / 拡大 Expansion) の判定
    # 直近のバンド幅を、過去50日間の平均バンド幅と比較
    latest_bb_width = latest['BB_Width']
    # 最新日を除いた過去の平均バンド幅 (直近50日間の平均で比較)
    bb_width_ma_50 = data['BB_Width'].iloc[-51:-1].mean()
    
    volatility_status = "通常"
    volatility_emoji = "➡️"
    bb_ratio_text = ""
    
    if not pd.isna(bb_width_ma_50):
        # バンド幅の比率を計算
        bb_ratio = latest_bb_width / bb_width_ma_50
        bb_ratio_text = f" ({bb_ratio:.2f}x 過去平均)"

        if bb_ratio < 0.8:
            volatility_status = "**収縮中 (Squeeze)**"
            volatility_emoji = "🤏"
        elif bb_ratio > 1.2:
            volatility_status = "**拡大中 (Expansion)**"
            volatility_emoji = "💥"
            
    # 2. 価格の位置の判定
    if latest['Close'] > latest['BB_Upper']:
        position_status = "上側バンド（+2σ）を**超えて推移**"
        position_emoji = "🔥"
        # Band Walkの可能性
        if (data['Close'].iloc[-5:] > data['BB_Upper'].iloc[-5:]).sum() >= 3:
            position_status += "（強い上昇トレンド継続 - バンドウォークの可能性）"
    elif latest['Close'] < latest['BB_Lower']:
        position_status = "下側バンド（-2σ）を**超えて推移**"
        position_emoji = "🥶"
        # Band Walkの可能性
        if (data['Close'].iloc[-5:] < data['BB_Lower'].iloc[-5:]).sum() >= 3:
            position_status += "（強い下降トレンド継続 - バンドウォークの可能性）"
    elif latest['Close'] > latest['BB_Middle']:
        position_status = "ミドルバンド（SMA 20）の**上方**で推移"
        position_emoji = "🟢"
    else: # latest['Close'] <= latest['BB_Middle']
        position_status = "ミドルバンド（SMA 20）の**下方**で推移"
        position_emoji = "🔴"

    bb_status_text = (
        f"価格は{position_status}。ボラティリティは{volatility_status}{bb_ratio_text}。"
    )
    bb_emoji = f"{position_emoji}{volatility_emoji}"
    
    return bb_status_text, bb_emoji

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
                value=dt.date.today() - dt.timedelta(days=730)
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
                    # Bollinger Bandに必要な過去50日間の平均バンド幅を計算するため、
                    # 少なくとも250日程度のデータが必要
                    if len(data) < 50:
                        errors.append(f"{symbol}: データ期間が短すぎます (50日以上推奨)")
                        continue
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
            st.error("No data could be fetched. Please check your symbols and date range.")
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
                    
                    # Calculate returns
                    ytd_return, one_year_return = calculate_returns(data)
                    
                    crossover_str = ""
                    if not np.isnan(latest['MACD_Crossover_Days']):
                        cross_type = "Bull" if latest['MACD'] > latest['MACD_Signal'] else "Bear"
                        crossover_str = f"{int(latest['MACD_Crossover_Days'])}d {cross_type}"
                    
                    # Bollinger Band Info for summary
                    bb_diff = (latest['Close'] - latest['BB_Middle']) / latest['BB_Width'] if not pd.isna(latest['BB_Width']) and latest['BB_Width'] != 0 else np.nan
                    bb_diff_str = f"{bb_diff:+.2f}σ" if not np.isnan(bb_diff) else "N/A"
                    
                    summary_data.append({
                        'Symbol': symbol,
                        'Price': f"${latest['Close']:.2f}",
                        'Change %': f"{((latest['Close']/prev['Close'])-1)*100:+.2f}%",
                        'YTD': f"{ytd_return:+.2f}%" if ytd_return is not None else "N/A",
                        '1Y': f"{one_year_return:+.2f}%" if one_year_return is not None else "N/A",
                        'RSI': f"{latest['RSI']:.1f}",
                        'MACD': f"{latest['MACD_Histogram']:.3f}",
                        'BB Diff': bb_diff_str, # BB情報をサマリーに追加
                        'Crossover': crossover_str,
                        'Signal': signal
                    })
            
            df_summary = pd.DataFrame(summary_data)
            
            # Color code the Signal column
            def color_signal(val):
                if val == 'BULLISH':
                    return 'background-color: #90EE90' # LightGreen
                elif val == 'BEARISH':
                    return 'background-color: #FFB6C6' # LightPink
                else:
                    return 'background-color: #D3D3D3' # LightGray
            
            st.dataframe(
                df_summary.style.applymap(color_signal, subset=['Signal']),
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
            
            # Calculate returns
            ytd_return, one_year_return = calculate_returns(data)
            
            with st.expander(f"📊 {symbol} - {info['name']}", expanded=(len(symbols) == 1)):
                # Metrics
                col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                
                with col1:
                    price_change = ((latest['Close']/prev['Close'])-1)*100
                    st.metric(
                        "Price",
                        f"${latest['Close']:.2f}",
                        f"{price_change:+.2f}%"
                    )
                
                with col2:
                    if ytd_return is not None:
                        st.metric("YTD", f"{ytd_return:+.2f}%")
                    else:
                        st.metric("YTD", "N/A")
                
                with col3:
                    if one_year_return is not None:
                        st.metric("1Y", f"{one_year_return:+.2f}%")
                    else:
                        st.metric("1Y", "N/A")
                
                with col4:
                    st.metric("RSI", f"{latest['RSI']:.1f}")
                
                with col5:
                    st.metric("MACD", f"{latest['MACD']:.4f}")
                
                with col6:
                    if not np.isnan(latest['MACD_Crossover_Days']):
                        cross_days = int(latest['MACD_Crossover_Days'])
                        cross_type = "Bull" if latest['MACD'] > latest['MACD_Signal'] else "Bear"
                        st.metric("MACD Cross", f"{cross_days}d", cross_type)
                    else:
                        st.metric("MACD Cross", "N/A")
                
                with col7:
                    signal, color = calculate_signal_score(data)
                    st.metric("Signal", signal)
                
                # Analysis text
                st.markdown("#### 📋 Technical Summary")
                
                # Performance
                perf_parts = []
                if ytd_return is not None:
                    perf_parts.append(f"YTD: {ytd_return:+.2f}%")
                if one_year_return is not None:
                    perf_parts.append(f"1Y: {one_year_return:+.2f}%")
                if perf_parts:
                    st.write(f"📊 **Performance:** {' | '.join(perf_parts)}")
                
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
                
                # --- 追加したボリンジャーバンド分析 ---
                bb_text, bb_emoji = analyze_bollinger_bands(data)
                st.write(f"{bb_emoji} **ボリンジャーバンド (BB):** {bb_text}")
                # -----------------------------------
                
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
        - 📊 Single or multi-stock analysis
        - 📈 MACD crossover tracking
        - 🎯 Comprehensive technical indicators (RSI, **Bollinger Bands**, Volume, ATR)
        - 📅 YTD and 1-year return tracking
        - 🔄 Sector ETF comparison
        - 📉 Visual charts and metrics
        
        ### How to Use
        1. Select analysis mode in the sidebar
        2. Enter stock symbol(s)
        3. Choose date range
        4. Click 'Analyze'
        
        ### Popular Symbols
        - Tech: AAPL, MSFT, GOOGL, AMZN, NVDA
        - ETFs: SPY, QQQ, VOO, VTI, VOOG
        - Sector ETFs: XLK, XLV, XLF, XLE, XLI
        """)

if __name__ == "__main__":
    main()
