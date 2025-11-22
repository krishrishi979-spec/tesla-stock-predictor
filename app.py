"""
Tesla Stock Prediction - Streamlit Web App
Run with: streamlit run app.py

Features:
- Real-time predictions using trained AI models
- Interactive charts and visualizations
- Technical analysis dashboard
- Model performance comparison
- Historical data analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tesla Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #e31937, #000000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .prediction-box-bearish {
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

# Get app directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(APP_DIR)

# Try multiple possible paths for data and models
DATA_PATHS = [
    os.path.join(APP_DIR, 'data', 'tesla_spx500_complete.csv'),
    os.path.join(PARENT_DIR, 'data', 'tesla_spx500_complete.csv'),
    'data/tesla_spx500_complete.csv',
    '../data/tesla_spx500_complete.csv'
]

MODELS_DIRS = [
    os.path.join(APP_DIR, 'models'),
    os.path.join(PARENT_DIR, 'models'),
    'models',
    '../models'
]


@st.cache_data(ttl=3600)
def load_data():
    """Load the stock data"""
    for path in DATA_PATHS:
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return df
    return None


def get_models_dir():
    """Get the models directory"""
    for path in MODELS_DIRS:
        if os.path.exists(path):
            return path
    return None


@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    models_dir = get_models_dir()
    
    if models_dir is None:
        return models
    
    # Load scalers
    try:
        models['scaler_X'] = joblib.load(os.path.join(models_dir, 'scalers_seq', 'scaler_X.pkl'))
        models['scaler_y'] = joblib.load(os.path.join(models_dir, 'scalers_seq', 'scaler_y.pkl'))
        models['feature_columns'] = joblib.load(os.path.join(models_dir, 'scalers_seq', 'feature_columns.pkl'))
    except:
        pass
    
    try:
        models['scaler_X_ml'] = joblib.load(os.path.join(models_dir, 'scalers_ml', 'scaler_X.pkl'))
        models['feature_columns_ml'] = joblib.load(os.path.join(models_dir, 'scalers_ml', 'feature_columns.pkl'))
    except:
        pass
    
    # Load XGBoost
    try:
        models['XGBoost'] = joblib.load(os.path.join(models_dir, 'xgboost_model.pkl'))
    except:
        pass
    
    # Load TensorFlow models
    try:
        from tensorflow.keras.models import load_model
        
        lstm_path = os.path.join(models_dir, 'lstm_model.keras')
        if os.path.exists(lstm_path):
            models['LSTM'] = load_model(lstm_path)
        
        gru_path = os.path.join(models_dir, 'gru_model.keras')
        if os.path.exists(gru_path):
            models['GRU'] = load_model(gru_path)
        
        transformer_path = os.path.join(models_dir, 'transformer_model.keras')
        if os.path.exists(transformer_path):
            models['Transformer'] = load_model(transformer_path)
    except ImportError:
        st.warning("TensorFlow not installed. Deep learning models unavailable.")
    except Exception as e:
        st.warning(f"Error loading TensorFlow models: {e}")
    
    return models


@st.cache_data
def load_model_comparison():
    """Load model comparison data"""
    models_dir = get_models_dir()
    if models_dir:
        path = os.path.join(models_dir, 'model_comparison.csv')
        if os.path.exists(path):
            return pd.read_csv(path, index_col=0)
    return None


def make_predictions(df, models):
    """Make predictions using all available models"""
    predictions = {}
    
    # Check if we have the necessary scalers
    if 'scaler_X' not in models or 'scaler_y' not in models or 'feature_columns' not in models:
        st.warning("Scalers not found. Cannot make predictions.")
        return predictions
    
    # Prepare sequence data for deep learning models
    try:
        features = models['feature_columns']
        last_60 = df.iloc[-60:][features].values
        X_scaled = models['scaler_X'].transform(last_60)
        X_seq = X_scaled.reshape(1, 60, -1)
        
        # LSTM
        if 'LSTM' in models:
            try:
                pred = models['LSTM'].predict(X_seq, verbose=0)
                price = float(models['scaler_y'].inverse_transform(pred)[0][0])
                predictions['LSTM'] = price
            except Exception as e:
                st.error(f"LSTM prediction failed: {e}")
        
        # GRU
        if 'GRU' in models:
            try:
                pred = models['GRU'].predict(X_seq, verbose=0)
                price = float(models['scaler_y'].inverse_transform(pred)[0][0])
                predictions['GRU'] = price
            except Exception as e:
                st.error(f"GRU prediction failed: {e}")
        
        # Transformer
        if 'Transformer' in models:
            try:
                pred = models['Transformer'].predict(X_seq, verbose=0)
                price = float(models['scaler_y'].inverse_transform(pred)[0][0])
                predictions['Transformer'] = price
            except Exception as e:
                st.error(f"Transformer prediction failed: {e}")
                
    except Exception as e:
        st.error(f"Error preparing sequence data: {e}")
    
    # XGBoost
    if 'XGBoost' in models and 'scaler_X_ml' in models and 'feature_columns_ml' in models:
        try:
            features_ml = models['feature_columns_ml']
            X_ml = df[features_ml].iloc[-1:].values
            X_scaled_ml = models['scaler_X_ml'].transform(X_ml)
            price = float(models['XGBoost'].predict(X_scaled_ml)[0])
            predictions['XGBoost'] = price
        except Exception as e:
            st.error(f"XGBoost prediction failed: {e}")
    
    return predictions


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def create_price_chart(df, days=365):
    """Create interactive price chart"""
    df_plot = df.tail(days)
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['Open'],
        high=df_plot['High'],
        low=df_plot['Low'],
        close=df_plot['Close'],
        name='TSLA',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    # Add moving averages
    if 'SMA_20' in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['SMA_20'],
            name='SMA 20', line=dict(color='orange', width=1)
        ))
    
    if 'SMA_50' in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['SMA_50'],
            name='SMA 50', line=dict(color='blue', width=1)
        ))
    
    fig.update_layout(
        title='Tesla (TSLA) Stock Price',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_dark',
        height=500,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_prediction_gauge(current_price, predicted_price):
    """Create a gauge chart for prediction"""
    change_pct = ((predicted_price - current_price) / current_price) * 100
    
    # Determine color based on prediction
    if change_pct > 2:
        color = "green"
    elif change_pct > 0:
        color = "lightgreen"
    elif change_pct > -2:
        color = "yellow"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_price,
        delta={'reference': current_price, 'relative': True, 'valueformat': '.2%'},
        title={'text': "Predicted Price"},
        number={'prefix': "$", 'valueformat': '.2f'},
        gauge={
            'axis': {'range': [current_price * 0.9, current_price * 1.1]},
            'bar': {'color': color},
            'steps': [
                {'range': [current_price * 0.9, current_price * 0.98], 'color': "lightcoral"},
                {'range': [current_price * 0.98, current_price * 1.02], 'color': "lightyellow"},
                {'range': [current_price * 1.02, current_price * 1.1], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': current_price
            }
        }
    ))
    
    fig.update_layout(height=300, template='plotly_dark')
    return fig


def create_model_comparison_chart(predictions, current_price):
    """Create bar chart comparing model predictions"""
    if not predictions:
        return None
    
    models = list(predictions.keys())
    prices = list(predictions.values())
    changes = [(p - current_price) / current_price * 100 for p in prices]
    colors = ['green' if c > 0 else 'red' for c in changes]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=models,
        y=prices,
        marker_color=colors,
        text=[f"${p:.2f}<br>({c:+.2f}%)" for p, c in zip(prices, changes)],
        textposition='outside'
    ))
    
    # Add current price line
    fig.add_hline(y=current_price, line_dash="dash", line_color="white",
                  annotation_text=f"Current: ${current_price:.2f}")
    
    fig.update_layout(
        title='Model Predictions Comparison',
        yaxis_title='Predicted Price (USD)',
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    
    return fig


def create_technical_indicators_chart(df, days=90):
    """Create technical indicators chart"""
    df_plot = df.tail(days)
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=('Price & Bollinger Bands', 'RSI', 'MACD', 'Volume')
    )
    
    # Price with Bollinger Bands
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'], name='Close',
                             line=dict(color='white', width=2)), row=1, col=1)
    
    if 'BB_Upper' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_Upper'], name='BB Upper',
                                 line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_Lower'], name='BB Lower',
                                 line=dict(color='gray', dash='dash'), fill='tonexty',
                                 fillcolor='rgba(128,128,128,0.2)'), row=1, col=1)
    
    # RSI
    if 'RSI' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name='RSI',
                                 line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'MACD' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD'], name='MACD',
                                 line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD_Signal'], name='Signal',
                                 line=dict(color='orange')), row=3, col=1)
        
        colors = ['green' if v >= 0 else 'red' for v in df_plot['MACD_Hist']]
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['MACD_Hist'], name='Histogram',
                            marker_color=colors), row=3, col=1)
    
    # Volume
    colors = ['green' if df_plot['Close'].iloc[i] >= df_plot['Open'].iloc[i] else 'red' 
              for i in range(len(df_plot))]
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], name='Volume',
                        marker_color=colors), row=4, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return fig


def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    # Select numeric columns for correlation
    numeric_cols = ['Close', 'Volume', 'RSI', 'MACD', 'ATR', 'Momentum_10']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if 'SPX_Close' in df.columns:
        available_cols.append('SPX_Close')
    
    if len(available_cols) < 2:
        return None
    
    corr = df[available_cols].corr()
    
    fig = px.imshow(
        corr,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        title='Feature Correlation Heatmap'
    )
    
    fig.update_layout(template='plotly_dark', height=500)
    return fig


# ============================================================
# MAIN APP
# ============================================================

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Tesla_Motors.svg/1200px-Tesla_Motors.svg.png", 
                 width=150)
        st.markdown("## 🚀 Tesla Stock Predictor")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🏠 Home", "🔮 Predictions", "📊 Technical Analysis", 
             "📈 Model Performance", "📋 Data Explorer", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        chart_days = st.slider("Chart History (days)", 30, 365, 180)
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        
        df = load_data()
        if df is not None:
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            
            st.metric("Current Price", f"${current_price:.2f}", 
                     f"{change:+.2f} ({change_pct:+.2f}%)")
            st.metric("Last Updated", df.index[-1].strftime('%Y-%m-%d'))
    
    # Main content
    if page == "🏠 Home":
        show_home_page(df, chart_days)
    elif page == "🔮 Predictions":
        show_predictions_page(df)
    elif page == "📊 Technical Analysis":
        show_technical_page(df, chart_days)
    elif page == "📈 Model Performance":
        show_performance_page()
    elif page == "📋 Data Explorer":
        show_data_page(df)
    elif page == "ℹ️ About":
        show_about_page()


def show_home_page(df, chart_days):
    """Show home page"""
    st.markdown('<h1 class="main-header">📈 Tesla Stock Predictor</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem;'>AI-Powered Stock Price Forecasting System</p>", 
                unsafe_allow_html=True)
    
    if df is None:
        st.error("❌ Data not found! Please ensure the data file exists.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    high_52w = df['High'].tail(252).max()
    low_52w = df['Low'].tail(252).min()
    avg_volume = df['Volume'].tail(20).mean()
    
    with col1:
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100
        st.metric("Current Price", f"${current_price:.2f}", f"{change_pct:+.2f}%")
    
    with col2:
        st.metric("52-Week High", f"${high_52w:.2f}")
    
    with col3:
        st.metric("52-Week Low", f"${low_52w:.2f}")
    
    with col4:
        st.metric("Avg Volume (20d)", f"{avg_volume/1e6:.2f}M")
    
    # Price chart
    st.markdown("### 📈 Price Chart")
    fig = create_price_chart(df, chart_days)
    st.plotly_chart(fig, use_container_width=True)
    
    # Quick prediction
    st.markdown("### 🔮 Quick Prediction")
    
    models = load_models()
    if models:
        predictions = make_predictions(df, models)
        
        if predictions:
            avg_pred = np.mean(list(predictions.values()))
            avg_change = ((avg_pred - current_price) / current_price) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                if avg_change > 0:
                    st.success(f"### 📈 Bullish Signal\nAverage Prediction: **${avg_pred:.2f}** ({avg_change:+.2f}%)")
                else:
                    st.error(f"### 📉 Bearish Signal\nAverage Prediction: **${avg_pred:.2f}** ({avg_change:+.2f}%)")
            
            with col2:
                fig = create_prediction_gauge(current_price, avg_pred)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No predictions available. Check if models are loaded correctly.")
    else:
        st.warning("No models found. Please train models first.")


def show_predictions_page(df):
    """Show predictions page"""
    st.markdown("## 🔮 AI Predictions")
    
    if df is None:
        st.error("Data not found!")
        return
    
    current_price = df['Close'].iloc[-1]
    last_date = df.index[-1]
    
    st.markdown(f"""
    **Current Price:** ${current_price:.2f}  
    **Last Updated:** {last_date.strftime('%Y-%m-%d')}  
    **Predicting For:** {(last_date + timedelta(days=1)).strftime('%Y-%m-%d')}
    """)
    
    # Load models and make predictions
    with st.spinner("Loading models and making predictions..."):
        models = load_models()
        predictions = make_predictions(df, models)
    
    if not predictions:
        st.error("No predictions available. Please check if models are trained.")
        return
    
    # Display predictions
    st.markdown("### 📊 Model Predictions")
    
    cols = st.columns(len(predictions))
    for i, (model_name, pred_price) in enumerate(predictions.items()):
        change = pred_price - current_price
        change_pct = (change / current_price) * 100
        
        with cols[i]:
            if change > 0:
                st.success(f"""
                **{model_name}**  
                ${pred_price:.2f}  
                {change_pct:+.2f}% 📈
                """)
            else:
                st.error(f"""
                **{model_name}**  
                ${pred_price:.2f}  
                {change_pct:+.2f}% 📉
                """)
    
    # Comparison chart
    st.markdown("### 📈 Prediction Comparison")
    fig = create_model_comparison_chart(predictions, current_price)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary
    st.markdown("### 📋 Summary")
    
    avg_pred = np.mean(list(predictions.values()))
    avg_change = ((avg_pred - current_price) / current_price) * 100
    bullish = sum(1 for p in predictions.values() if p > current_price)
    bearish = len(predictions) - bullish
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Prediction", f"${avg_pred:.2f}", f"{avg_change:+.2f}%")
    
    with col2:
        st.metric("Bullish Models", f"{bullish}", f"of {len(predictions)}")
    
    with col3:
        if avg_change > 2:
            st.success("🟢 STRONG BUY")
        elif avg_change > 0:
            st.success("🟢 BUY")
        elif avg_change > -2:
            st.warning("🟡 HOLD")
        else:
            st.error("🔴 SELL")
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    ⚠️ **DISCLAIMER:** These predictions are for educational purposes only. 
    Do not use for actual trading without professional advice. 
    Past performance does not guarantee future results.
    """)


def show_technical_page(df, chart_days):
    """Show technical analysis page"""
    st.markdown("## 📊 Technical Analysis")
    
    if df is None:
        st.error("Data not found!")
        return
    
    # Technical indicators chart
    fig = create_technical_indicators_chart(df, chart_days)
    st.plotly_chart(fig, use_container_width=True)
    
    # Current indicator values
    st.markdown("### 📋 Current Indicator Values")
    
    latest = df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Price Indicators**")
        st.write(f"Close: ${latest['Close']:.2f}")
        if 'SMA_20' in df.columns:
            st.write(f"SMA 20: ${latest['SMA_20']:.2f}")
        if 'SMA_50' in df.columns:
            st.write(f"SMA 50: ${latest['SMA_50']:.2f}")
    
    with col2:
        st.markdown("**Momentum**")
        if 'RSI' in df.columns:
            rsi = latest['RSI']
            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            st.write(f"RSI: {rsi:.2f} ({rsi_status})")
        if 'MACD' in df.columns:
            st.write(f"MACD: {latest['MACD']:.2f}")
    
    with col3:
        st.markdown("**Volatility**")
        if 'ATR' in df.columns:
            st.write(f"ATR: ${latest['ATR']:.2f}")
        if 'BB_Width' in df.columns:
            st.write(f"BB Width: ${latest['BB_Width']:.2f}")
    
    with col4:
        st.markdown("**Volume**")
        st.write(f"Volume: {latest['Volume']/1e6:.2f}M")
        if 'Volume_Ratio' in df.columns:
            vr = latest['Volume_Ratio']
            vr_status = "High" if vr > 1.5 else "Low" if vr < 0.5 else "Normal"
            st.write(f"Vol Ratio: {vr:.2f} ({vr_status})")
    
    # Correlation heatmap
    st.markdown("### 🔥 Feature Correlation")
    fig = create_correlation_heatmap(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)


def show_performance_page():
    """Show model performance page"""
    st.markdown("## 📈 Model Performance")
    
    comparison = load_model_comparison()
    
    if comparison is None:
        st.warning("No performance data found. Train models to generate metrics.")
        return
    
    # Display metrics table
    st.markdown("### 📊 Performance Metrics")
    
    # Format the dataframe for display
    display_df = comparison.copy()
    display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"${x:.2f}")
    display_df['MAE'] = display_df['MAE'].apply(lambda x: f"${x:.2f}")
    display_df['MAPE'] = display_df['MAPE'].apply(lambda x: f"{x:.2f}%")
    display_df['R2'] = display_df['R2'].apply(lambda x: f"{x:.4f}")
    display_df['Directional_Accuracy'] = display_df['Directional_Accuracy'].apply(lambda x: f"{x:.2f}%")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Bar charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(comparison, x=comparison.index, y='RMSE', 
                    title='RMSE by Model (Lower is Better)',
                    color='RMSE', color_continuous_scale='Reds_r')
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(comparison, x=comparison.index, y='Directional_Accuracy',
                    title='Directional Accuracy by Model (Higher is Better)',
                    color='Directional_Accuracy', color_continuous_scale='Greens')
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Best model
    st.markdown("### 🏆 Best Performing Model")
    best_model = comparison['RMSE'].idxmin()
    best_accuracy = comparison['Directional_Accuracy'].idxmax()
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Lowest RMSE:** {best_model}")
    with col2:
        st.success(f"**Highest Accuracy:** {best_accuracy}")


def show_data_page(df):
    """Show data explorer page"""
    st.markdown("## 📋 Data Explorer")
    
    if df is None:
        st.error("Data not found!")
        return
    
    # Data summary
    st.markdown("### 📊 Dataset Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Total Features", len(df.columns))
    with col3:
        st.metric("Date Range", f"{(df.index[-1] - df.index[0]).days} days")
    
    # Date range
    st.markdown(f"**From:** {df.index[0].strftime('%Y-%m-%d')} **To:** {df.index[-1].strftime('%Y-%m-%d')}")
    
    # Data preview
    st.markdown("### 📄 Data Preview")
    
    view_option = st.radio("View", ["Latest Data", "Earliest Data", "Custom Range"], horizontal=True)
    
    if view_option == "Latest Data":
        st.dataframe(df.tail(20), use_container_width=True)
    elif view_option == "Earliest Data":
        st.dataframe(df.head(20), use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input("Start Date", df.index[0])
        with col2:
            end = st.date_input("End Date", df.index[-1])
        
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        st.dataframe(df.loc[mask], use_container_width=True)
    
    # Statistics
    st.markdown("### 📈 Price Statistics")
    
    stats = df['Close'].describe()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"${stats['mean']:.2f}")
    with col2:
        st.metric("Std Dev", f"${stats['std']:.2f}")
    with col3:
        st.metric("Min", f"${stats['min']:.2f}")
    with col4:
        st.metric("Max", f"${stats['max']:.2f}")
    
    # Download data
    st.markdown("### 📥 Download Data")
    csv = df.to_csv()
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="tesla_stock_data.csv",
        mime="text/csv"
    )


def show_about_page():
    """Show about page"""
    st.markdown("## ℹ️ About")
    
    st.markdown("""
    ### 🚀 Tesla Stock Predictor
    
    This application uses **5 advanced AI models** to predict Tesla (TSLA) stock prices:
    
    #### 🤖 Models Used:
    
    1. **LSTM (Long Short-Term Memory)**
       - Deep learning model for sequential data
       - Captures long-term dependencies
       - Best for trend prediction
    
    2. **GRU (Gated Recurrent Unit)**
       - Simplified version of LSTM
       - Faster training, similar performance
       - Good for real-time predictions
    
    3. **Transformer**
       - Attention-based architecture
       - State-of-the-art for sequence modeling
       - Captures complex patterns
    
    4. **XGBoost**
       - Gradient boosting algorithm
       - Fast and interpretable
       - Excellent for feature importance
    
    5. **Moving Average Baseline**
       - Simple baseline for comparison
       - Uses 20-day moving average
    
    #### 📊 Technical Indicators:
    
    - **RSI** (Relative Strength Index)
    - **MACD** (Moving Average Convergence Divergence)
    - **Bollinger Bands**
    - **ATR** (Average True Range)
    - **50+ additional features**
    
    #### ⚠️ Disclaimer:
    
    This application is for **EDUCATIONAL PURPOSES ONLY**. 
    
    - Do NOT use for actual trading without professional advice
    - Stock predictions are inherently uncertain
    - Past performance does not guarantee future results
    - Always consult a financial advisor before investing
    
    ---
    
    **Version:** 1.0  
    **Last Updated:** November 2024
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 System Status")
    
    df = load_data()
    models = load_models()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Data Status:**")
        if df is not None:
            st.success(f"✓ Loaded ({len(df)} records)")
        else:
            st.error("✗ Not found")
    
    with col2:
        st.markdown("**Models Status:**")
        available = [k for k in models.keys() if k in ['LSTM', 'GRU', 'Transformer', 'XGBoost']]
        if available:
            st.success(f"✓ {', '.join(available)}")
        else:
            st.error("✗ No models found")


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    main()
