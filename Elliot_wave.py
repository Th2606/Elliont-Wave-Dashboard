"""
Elliott Wave Interactive Manual Labeling Dashboard (Dash + Plotly + SQLite)
Features:
- Symbol selection (yfinance)
- Candlestick chart with automatically detected peaks/troughs
- Click on pivot points to assign wave labels (1-5, A-C)
- Save and load labeled points to/from SQLite database
- Review and delete labels in a DataTable
- Basic Wave-3 backtester using saved labels
- Short-trade support with automatic direction detection
- Execution realism: slippage and commission
- Batch backtesting across a watchlist with CSV export
- Enhanced direction detection (neighbor pivots, slope regression, RSI confirmation) with configurable weights
- Entry-rule options and long-only/short-only filters
- Price caching for faster batch runs and per-symbol progress summary
- Asynchronous batch runs with a polling interval and live progress table
- Additional entry rules and performance visualizations (trades per symbol)

Run:
pip install dash yfinance plotly pandas numpy scipy sqlalchemy
python Elliott_Wave_Dashboard_Dash.py
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go
import sqlite3
from datetime import datetime, timedelta
import io
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# --------------------------- Helper functions ---------------------------
def fetch_data(symbol: str, period: str, interval: str):
    df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[['Open','High','Low','Close','Volume']]
    df.index = pd.to_datetime(df.index)
    return df

# Cached fetch to speed batch runs
PRICE_CACHE = {}

def fetch_data_cached(symbol: str, period: str, interval: str):
    key = f"{symbol}|{period}|{interval}"
    if key in PRICE_CACHE:
        return PRICE_CACHE[key]
    df = fetch_data(symbol, period, interval)
    PRICE_CACHE[key] = df
    return df


def detect_pivots(df: pd.DataFrame, distance: int = 5, prominence: float = None):
    if df.empty:
        return np.array([]), np.array([])
    close = df['Close'].values
    peaks, _ = find_peaks(close, distance=distance, prominence=prominence)
    troughs, _ = find_peaks(-close, distance=distance, prominence=prominence)
    return peaks, troughs

# Simple RSI implementation
def compute_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0).fillna(0)
    down = -1 * delta.clip(upper=0).fillna(0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# --------------------------- Database functions ---------------------------
DB_FILE = 'elliott_wave.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS wave_labels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        timestamp TEXT,
        price REAL,
        wave_label TEXT,
        created_at TEXT
    )''')
    conn.commit()
    conn.close()

def save_labels_to_db(df, symbol):
    if df is None or df.empty:
        return
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute('''INSERT INTO wave_labels (symbol, timestamp, price, wave_label, created_at)
                          VALUES (?, ?, ?, ?, ?)''',
                       (symbol, str(row['time']), float(row['price']), row['label'], datetime.now().isoformat()))
    conn.commit()
    conn.close()

def load_labels_from_db(symbol):
    conn = sqlite3.connect(DB_FILE)
    query = 'SELECT * FROM wave_labels WHERE symbol = ? ORDER BY timestamp'
    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()
    if df.empty:
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def delete_label_from_db(label_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM wave_labels WHERE id = ?', (label_id,))
    conn.commit()
    conn.close()

# --------------------------- Direction detection enhancements (weighted) ---------------------------
def get_neighbor_labels(all_labels_df, label_row):
    """Return previous and next labeled pivots for same symbol if available"""
    if all_labels_df is None or all_labels_df.empty:
        return None, None
    df = all_labels_df.sort_values('timestamp')
    # filter only same symbol
    df = df[df['symbol'] == label_row['symbol']]
    if df.empty:
        return None, None
    # find position
    matches = df.index[df['id'] == label_row['id']]
    if len(matches) == 0:
        matches = df.index[df['timestamp'] == label_row['timestamp']]
        if len(matches) == 0:
            return None, None
    pos = matches[0]
    idx_list = df.index.tolist()
    i = idx_list.index(pos)
    prev_row = df.loc[idx_list[i-1]] if i-1 >= 0 else None
    next_row = df.loc[idx_list[i+1]] if i+1 < len(idx_list) else None
    return prev_row, next_row


def detect_direction_enhanced_weighted(all_labels_df, df_full, label_row, weights=None, lookahead=5):
    """
    Weighted combination of signals to detect direction.
    weights: dict with keys 'neighbor','slope','rsi' mapping to numeric weights.
    Returns 'long' or 'short' and details.
    """
    if weights is None:
        weights = {'neighbor': 1.0, 'slope': 1.0, 'rsi': 1.0}
    scores = {'long': 0.0, 'short': 0.0}
    reasons = []

    label_price = float(label_row['price'])
    label_time = label_row['timestamp']

    # Neighbor labels
    prev_row, next_row = get_neighbor_labels(all_labels_df, label_row)
    if prev_row is not None:
        try:
            if float(prev_row['price']) < label_price:
                scores['long'] += weights.get('neighbor',1.0)
                reasons.append('prev_lower')
            elif float(prev_row['price']) > label_price:
                scores['short'] += weights.get('neighbor',1.0)
                reasons.append('prev_higher')
        except Exception:
            pass
    if next_row is not None:
        try:
            if float(next_row['price']) > label_price:
                scores['long'] += weights.get('neighbor',1.0)
                reasons.append('next_higher')
            elif float(next_row['price']) < label_price:
                scores['short'] += weights.get('neighbor',1.0)
                reasons.append('next_lower')
        except Exception:
            pass

    # post-label series
    post = df_full[df_full.index > label_time]
    if not post.empty:
        n = min(len(post), lookahead)
        closes = post['Close'].iloc[:n]
        # slope
        if len(closes) >= 2:
            x = np.arange(len(closes))
            slope = np.polyfit(x, closes.values, 1)[0]
            if slope > 0:
                scores['long'] += weights.get('slope',1.0)
                reasons.append('slope_pos')
            elif slope < 0:
                scores['short'] += weights.get('slope',1.0)
                reasons.append('slope_neg')
        # rsi
        rsi = compute_rsi(closes, period=14)
        recent_rsi = float(rsi.iloc[-1]) if not rsi.empty else 50.0
        if recent_rsi < 40:
            scores['long'] += weights.get('rsi',1.0)
            reasons.append('rsi_low')
        elif recent_rsi > 60:
            scores['short'] += weights.get('rsi',1.0)
            reasons.append('rsi_high')

    # Decide
    if scores['long'] >= scores['short']:
        final = 'long'
    else:
        final = 'short'
    return final, {'scores': scores, 'reasons': reasons}

# --------------------------- Backtesting functions (enhanced entry rules) ---------------------------
def is_bullish_engulfing(df, idx):
    if idx <= 0 or idx >= len(df):
        return False
    prev = df.iloc[idx-1]
    cur = df.iloc[idx]
    return (cur['Close'] > cur['Open']) and (prev['Close'] < prev['Open']) and (cur['Close'] > prev['Open']) and (cur['Open'] < prev['Close'])


def is_bearish_engulfing(df, idx):
    if idx <= 0 or idx >= len(df):
        return False
    prev = df.iloc[idx-1]
    cur = df.iloc[idx]
    return (cur['Close'] < cur['Open']) and (prev['Close'] > prev['Open']) and (cur['Close'] < prev['Open']) and (cur['Open'] > prev['Close'])


def run_wave3_backtest_for_symbol(symbol, lookahead_bars=200, risk_buffer=0.01, tp_multiplier=2.0, interval='1d',
                                  slippage_pct=0.0, commission_pct=0.0, all_labels_df=None, entry_rule='next_close',
                                  direction_override=None, weights=None):
    labels = load_labels_from_db(symbol)
    if labels.empty:
        return pd.DataFrame(), {}

    entries = labels[labels['wave_label'].isin(['2','C'])]
    if entries.empty:
        return pd.DataFrame(), {}

    trades = []
    df_full = fetch_data_cached(symbol, period='max', interval=interval)
    if df_full.empty:
        return pd.DataFrame(), {}

    for _, row in entries.iterrows():
        label_time = row['timestamp']
        labeled_price = float(row['price'])
        post = df_full[df_full.index > label_time]
        if post.empty:
            continue

        # direction detection
        if direction_override in ['long','short']:
            direction = direction_override
            direction_meta = {'forced': True}
        else:
            direction, direction_meta = detect_direction_enhanced_weighted(all_labels_df, df_full, row, weights=weights, lookahead=5)

        # entry selection
        entry_time = None
        entry_price = None
        if entry_rule == 'next_close':
            entry_time = post.index[0]
            entry_price = float(post['Close'].iloc[0])
        elif entry_rule == 'breakout':
            lookback = 5
            prior = df_full[df_full.index <= label_time].iloc[-lookback:]
            if prior.empty:
                continue
            if direction == 'long':
                level = prior['High'].max()
                hits = post[post['Close'] > level]
                if hits.empty:
                    continue
                entry_time = hits.index[0]
                entry_price = float(hits['Close'].iloc[0])
            else:
                level = prior['Low'].min()
                hits = post[post['Close'] < level]
                if hits.empty:
                    continue
                entry_time = hits.index[0]
                entry_price = float(hits['Close'].iloc[0])
        elif entry_rule == 'bullish_engulfing' or entry_rule == 'bearish_engulfing':
            idx_in_full = df_full.index.get_indexer([post.index[0]])[0]
            if entry_rule == 'bullish_engulfing' and not is_bullish_engulfing(df_full, idx_in_full):
                continue
            if entry_rule == 'bearish_engulfing' and not is_bearish_engulfing(df_full, idx_in_full):
                continue
            entry_time = post.index[0]
            entry_price = float(post['Close'].iloc[0])
        elif entry_rule == 'ma_cross':
            ma_fast = 5
            ma_slow = 20
            prior = df_full[df_full.index <= label_time]
            if len(prior) < ma_slow:
                continue
            ma_fast_prev = prior['Close'].rolling(ma_fast).mean().iloc[-1]
            ma_slow_prev = prior['Close'].rolling(ma_slow).mean().iloc[-1]
            # check next bar for cross
            if len(post) < 1:
                continue
            ma_fast_next = prior['Close'].append(post['Close'].iloc[:1]).rolling(ma_fast).mean().iloc[-1]
            ma_slow_next = prior['Close'].append(post['Close'].iloc[:1]).rolling(ma_slow).mean().iloc[-1]
            if direction == 'long' and ma_fast_prev <= ma_slow_prev and ma_fast_next > ma_slow_next:
                entry_time = post.index[0]
                entry_price = float(post['Close'].iloc[0])
            elif direction == 'short' and ma_fast_prev >= ma_slow_prev and ma_fast_next < ma_slow_next:
                entry_time = post.index[0]
                entry_price = float(post['Close'].iloc[0])
            else:
                continue
        else:
            entry_time = post.index[0]
            entry_price = float(post['Close'].iloc[0])

        # apply slippage
        if direction == 'long':
            entry_price_ex = entry_price * (1.0 + slippage_pct)
            stop_price = labeled_price * (1.0 - risk_buffer)
            if stop_price >= entry_price_ex:
                continue
            risk = entry_price_ex - stop_price
            target_price = entry_price_ex + risk * tp_multiplier
        else:
            entry_price_ex = entry_price * (1.0 - slippage_pct)
            stop_price = labeled_price * (1.0 + risk_buffer)
            if stop_price <= entry_price_ex:
                continue
            risk = stop_price - entry_price_ex
            target_price = entry_price_ex - risk * tp_multiplier

        window = post.loc[entry_time:entry_time + pd.Timedelta(days=3650)].iloc[:lookahead_bars]
        if window.empty:
            continue

        if direction == 'long':
            hit_target = (window['High'] >= target_price).any()
            hit_stop = (window['Low'] <= stop_price).any()
            if not hit_target and not hit_stop:
                exit_price = float(window['Close'].iloc[-1]) * (1.0 - slippage_pct)
                exit_time = window.index[-1]
                pnl = exit_price - entry_price_ex
                outcome = 'no_hit'
                duration = len(window)
            else:
                target_idx = window[window['High'] >= target_price].index
                stop_idx = window[window['Low'] <= stop_price].index
                first_target_idx = target_idx[0] if len(target_idx) > 0 else None
                first_stop_idx = stop_idx[0] if len(stop_idx) > 0 else None
                if first_target_idx is not None and (first_stop_idx is None or first_target_idx <= first_stop_idx):
                    exit_time = first_target_idx
                    exit_price = target_price * (1.0 - slippage_pct)
                    outcome = 'win'
                    duration = (window.index.get_loc(exit_time) + 1)
                else:
                    exit_time = first_stop_idx
                    exit_price = stop_price * (1.0 - slippage_pct)
                    outcome = 'loss'
                    duration = (window.index.get_loc(exit_time) + 1)
            pnl = exit_price - entry_price_ex
        else:
            hit_target = (window['Low'] <= target_price).any()
            hit_stop = (window['High'] >= stop_price).any()
            if not hit_target and not hit_stop:
                exit_price = float(window['Close'].iloc[-1]) * (1.0 + slippage_pct)
                exit_time = window.index[-1]
                pnl = entry_price_ex - exit_price
                outcome = 'no_hit'
                duration = len(window)
            else:
                target_idx = window[window['Low'] <= target_price].index
                stop_idx = window[window['High'] >= stop_price].index
                first_target_idx = target_idx[0] if len(target_idx) > 0 else None
                first_stop_idx = stop_idx[0] if len(stop_idx) > 0 else None
                if first_target_idx is not None and (first_stop_idx is None or first_target_idx <= first_stop_idx):
                    exit_time = first_target_idx
                    exit_price = target_price * (1.0 + slippage_pct)
                    outcome = 'win'
                    duration = (window.index.get_loc(exit_time) + 1)
                else:
                    exit_time = first_stop_idx
                    exit_price = stop_price * (1.0 + slippage_pct)
                    outcome = 'loss'
                    duration = (window.index.get_loc(exit_time) + 1)
            pnl = entry_price_ex - exit_price

        commission_cost = commission_pct * (entry_price_ex + (exit_price if exit_price is not None else 0.0))
        pnl_net = pnl - commission_cost
        r_multiple = pnl_net / risk if risk != 0 else np.nan

        trades.append({
            'symbol': symbol,
            'label_id': int(row['id']),
            'label_time': label_time,
            'direction': direction,
            'entry_time': entry_time,
            'entry_price': entry_price_ex,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'pnl': pnl_net,
            'r_multiple': r_multiple,
            'outcome': outcome,
            'duration_bars': duration
        })

    if not trades:
        return pd.DataFrame(), {}

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df['outcome']=='win']
    losses = trades_df[trades_df['outcome']=='loss']
    no_hits = trades_df[trades_df['outcome']=='no_hit']

    total = len(trades_df)
    win_rate = len(wins) / total if total>0 else 0.0
    avg_r = trades_df['r_multiple'].mean()
    expectancy = ((wins['r_multiple'].mean() if len(wins)>0 else 0.0) * len(wins) +
                 (losses['r_multiple'].mean() if len(losses)>0 else 0.0) * len(losses) +
                 (no_hits['r_multiple'].mean() if len(no_hits)>0 else 0.0) * len(no_hits)) / total

    equity = trades_df['r_multiple'].cumsum().fillna(method='ffill').fillna(0)

    summary = {
        'trades': total,
        'wins': len(wins),
        'losses': len(losses),
        'no_hit': len(no_hits),
        'win_rate': win_rate,
        'avg_r': avg_r,
        'expectancy_r': expectancy,
        'equity_curve': equity.tolist()
    }

    return trades_df, summary

# Batch runner with background thread and caching
BACKTEST_THREAD = None
BACKTEST_PROGRESS = []
BACKTEST_RESULT_DF = pd.DataFrame()
BACKTEST_RUNNING = False

def background_batch_runner(watchlist, params):
    global BACKTEST_THREAD, BACKTEST_PROGRESS, BACKTEST_RESULT_DF, BACKTEST_RUNNING
    BACKTEST_RUNNING = True
    BACKTEST_PROGRESS = []
    BACKTEST_RESULT_DF = pd.DataFrame()
    t0 = time.time()
    results = []

    # process symbols in parallel to speed up; but update progress per symbol
    def process_symbol(sym):
        t1 = time.time()
        trades_df, summary = run_wave3_backtest_for_symbol(sym, lookahead_bars=params['lookahead_bars'],
                                                           risk_buffer=params['risk_buffer'], tp_multiplier=params['tp_multiplier'],
                                                           interval=params['interval'], slippage_pct=params['slippage_pct'],
                                                           commission_pct=params['commission_pct'], all_labels_df=params['all_labels_df'],
                                                           entry_rule=params['entry_rule'], direction_override=params['direction_override'],
                                                           weights=params['weights'])
        t2 = time.time()
        duration_ms = int((t2 - t1)*1000)
        BACKTEST_PROGRESS.append({'symbol': sym, 'trades': len(trades_df), 'time_ms': duration_ms})
        return trades_df

    with ThreadPoolExecutor(max_workers=min(6, max(1, len(watchlist)))) as ex:
        futures = {ex.submit(process_symbol, s): s for s in watchlist}
        for fut in futures:
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    results.append(df)
            except Exception as e:
                BACKTEST_PROGRESS.append({'symbol': futures[fut], 'trades': 0, 'time_ms': 0, 'error': str(e)})

    total_time_ms = int((time.time() - t0)*1000)
    BACKTEST_RESULT_DF = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    BACKTEST_RUNNING = False
    return BACKTEST_RESULT_DF, BACKTEST_PROGRESS, total_time_ms

# Initialize DB
init_db()

# --------------------------- Initialize Dash App ---------------------------
app = dash.Dash(__name__)
app.title = "Elliott Wave Manual Labeling + Backtester"

# Temporary storage of labels
labels_df = pd.DataFrame(columns=['idx','time','price','label'])
LAST_BACKTEST_DF = pd.DataFrame()

app.layout = html.Div([
    html.H1("Elliott Wave Interactive Manual Labeling Dashboard"),

    html.Div([
        html.Label("Symbol:"),
        dcc.Input(id='symbol-input', type='text', value='AAPL'),
        html.Label("Period:"),
        dcc.Dropdown(id='period-dropdown', options=[
            {'label':'3mo','value':'3mo'},
            {'label':'6mo','value':'6mo'},
            {'label':'1y','value':'1y'},
            {'label':'2y','value':'2y'},
            {'label':'5y','value':'5y'},
            {'label':'max','value':'max'}], value='6mo'),
        html.Label("Interval:"),
        dcc.Dropdown(id='interval-dropdown', options=[
            {'label':'1d','value':'1d'},
            {'label':'1wk','value':'1wk'},
            {'label':'1h','value':'1h'}], value='1d'),
        html.Button("Load Data", id='load-btn', n_clicks=0),
        html.Button("Save Labels", id='save-btn', n_clicks=0, style={'marginLeft':'10px'}),
        html.Button("Load Saved Labels", id='load-saved-btn', n_clicks=0, style={'marginLeft':'10px'})
    ], style={'width':'40%','display':'inline-block','verticalAlign':'top'}),

    html.Div([
        html.Label("Select Wave Label:"),
        dcc.Dropdown(id='wave-label-dropdown', options=[
            {'label': '1', 'value':'1'},
            {'label': '2', 'value':'2'},
            {'label': '3', 'value':'3'},
            {'label': '4', 'value':'4'},
            {'label': '5', 'value':'5'},
            {'label': 'A', 'value':'A'},
            {'label': 'B', 'value':'B'},
            {'label': 'C', 'value':'C'}], value='1'
        )
    ], style={'width':'20%','display':'inline-block','verticalAlign':'top','marginLeft':'20px'}),

    dcc.Graph(id='candlestick-graph', config={'displayModeBar':True}),

    html.Div(id='info', style={'marginTop':'20px'}),

    html.H3("Review Saved Labels"),
    dash_table.DataTable(id='saved-labels-table', columns=[
        {'name':'id','id':'id'},
        {'name':'symbol','id':'symbol'},
        {'name':'timestamp','id':'timestamp'},
        {'name':'price','id':'price'},
        {'name':'wave_label','id':'wave_label'},
        {'name':'created_at','id':'created_at'}
    ], row_selectable='single', page_size=5, style_table={'overflowX':'auto'}),
    html.Button("Delete Selected Label", id='delete-btn', n_clicks=0, style={'marginTop':'10px'}),

    html.H2("Wave-3 Backtester"),
    html.Div([
        html.Label('Backtest Symbol (or comma-separated watchlist):'),
        dcc.Input(id='bt-symbol-input', type='text', value='AAPL'),
        html.Label('Interval for price series:'),
        dcc.Dropdown(id='bt-interval', options=[{'label':'1d','value':'1d'},{'label':'1wk','value':'1wk'},{'label':'1h','value':'1h'}], value='1d'),
        html.Label('Risk buffer (% below/above labeled low):'),
        dcc.Input(id='bt-risk-buffer', type='number', value=1.0, step=0.1),
        html.Label('TP multiplier (e.g., 2.0):'),
        dcc.Input(id='bt-tp-mult', type='number', value=2.0, step=0.1),
        html.Label('Slippage (% of price):'),
        dcc.Input(id='bt-slippage', type='number', value=0.0, step=0.01),
        html.Label('Commission (% of trade value):'),
        dcc.Input(id='bt-commission', type='number', value=0.0, step=0.01),
        html.Label('Entry rule:'),
        dcc.Dropdown(id='bt-entry-rule', options=[{'label':'Next close','value':'next_close'},{'label':'Breakout','value':'breakout'},{'label':'Engulfing','value':'bullish_engulfing'},{'label':'MA cross','value':'ma_cross'},{'label':'Bearish Engulfing','value':'bearish_engulfing'}], value='next_close'),
        html.Label('Direction filter:'),
        dcc.Dropdown(id='bt-direction-filter', options=[{'label':'Both','value':'both'},{'label':'Long only','value':'long'},{'label':'Short only','value':'short'}], value='both'),
        html.Button('Start Backtest (async)', id='run-backtest-btn', n_clicks=0, style={'marginLeft':'10px'}),
        html.Button('Export CSV (last run)', id='export-csv-btn', n_clicks=0, style={'marginLeft':'10px'})
    ], style={'display':'flex','flexWrap':'wrap','gap':'10px','alignItems':'center'}),

    html.Div([
        html.Label('Weights for direction detection (neighbor, slope, rsi):'),
        dcc.Input(id='w-neighbor', type='number', value=1.0, step=0.1),
        dcc.Input(id='w-slope', type='number', value=1.0, step=0.1),
        dcc.Input(id='w-rsi', type='number', value=1.0, step=0.1)
    ], style={'marginTop':'10px','display':'flex','gap':'10px','alignItems':'center'}),

    html.Div(id='backtest-summary', style={'marginTop':'20px'}),
    dash_table.DataTable(id='backtest-table', columns=[
        {'name':'symbol','id':'symbol'},{'name':'label_id','id':'label_id'},{'name':'label_time','id':'label_time'},{'name':'direction','id':'direction'},
        {'name':'entry_time','id':'entry_time'},{'name':'entry_price','id':'entry_price'},{'name':'exit_time','id':'exit_time'},{'name':'exit_price','id':'exit_price'},
        {'name':'pnl','id':'pnl'},{'name':'r_multiple','id':'r_multiple'},{'name':'outcome','id':'outcome'},{'name':'duration_bars','id':'duration_bars'}
    ], page_size=10, style_table={'overflowX':'auto'}),

    dcc.Graph(id='equity-curve'),
    dcc.Download(id='download-trades'),

    html.H4('Batch progress (per symbol)'),
    dash_table.DataTable(id='batch-progress-table', columns=[{'name':'symbol','id':'symbol'},{'name':'trades','id':'trades'},{'name':'time_ms','id':'time_ms'},{'name':'error','id':'error'}], page_size=10),
    html.Div(id='batch-total-time'),
    dcc.Interval(id='batch-interval', interval=2000, n_intervals=0, disabled=True),

    html.H3('Performance by symbol'),
    dcc.Graph(id='trades-per-symbol'),
    dash_table.DataTable(id='per-symbol-table', columns=[{'name':'symbol','id':'symbol'},{'name':'trades','id':'trades'},{'name':'wins','id':'wins'},{'name':'losses','id':'losses'},{'name':'win_rate','id':'win_rate'}], page_size=10)
])

# --------------------------- Callbacks ---------------------------
@app.callback(
    Output('candlestick-graph', 'figure'),
    Input('load-btn', 'n_clicks'),
    State('symbol-input', 'value'),
    State('period-dropdown', 'value'),
    State('interval-dropdown', 'value')
)
def update_chart(n_clicks, symbol, period, interval):
    if n_clicks == 0:
        return go.Figure()
    df = fetch_data(symbol, period, interval)
    peaks, troughs = detect_pivots(df, distance=5, prominence=None)

    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price')])
    if len(peaks)>0:
        fig.add_trace(go.Scatter(x=df.index[peaks], y=df['Close'].iloc[peaks], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Peaks'))
    if len(troughs)>0:
        fig.add_trace(go.Scatter(x=df.index[troughs], y=df['Close'].iloc[troughs], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Troughs'))

    if not labels_df.empty:
        fig.add_trace(go.Scatter(x=labels_df['time'], y=labels_df['price'], mode='text', text=labels_df['label'], textposition='top center', name='Labels'))

    fig.update_layout(height=700, margin=dict(l=10,r=10,t=30,b=10))
    return fig

@app.callback(
    Output('info', 'children'),
    Input('candlestick-graph', 'clickData'),
    State('wave-label-dropdown', 'value'),
    State('symbol-input', 'value'),
    State('period-dropdown', 'value'),
    State('interval-dropdown', 'value')
)
def label_wave(clickData, wave_label, symbol, period, interval):
    if clickData is None:
        return "Click a pivot point to assign a wave label."

    point = clickData['points'][0]
    clicked_time = point['x']
    df = fetch_data(symbol, period, interval)
    if df.empty:
        return "No data available to label."
    try:
        clicked_idx = df.index.get_loc(pd.to_datetime(clicked_time))
    except KeyError:
        clicked_idx = df.index.get_indexer([pd.to_datetime(clicked_time)], method='nearest')[0]
    clicked_price = df['Close'].iloc[clicked_idx]

    global labels_df
    if clicked_idx in labels_df['idx'].values:
        labels_df.loc[labels_df['idx']==clicked_idx,'label'] = wave_label
    else:
        labels_df = pd.concat([labels_df, pd.DataFrame([{
            'idx': clicked_idx,
            'time': df.index[clicked_idx],
            'price': clicked_price,
            'label': wave_label
        }])], ignore_index=True)

    return f"Labeled point at {df.index[clicked_idx]} with Wave {wave_label}. Total labeled points: {len(labels_df)}"

@app.callback(
    Output('saved-labels-table', 'data'),
    Input('save-btn', 'n_clicks'),
    State('symbol-input', 'value')
)
def save_labels(n_clicks, symbol):
    if n_clicks > 0 and not labels_df.empty:
        save_labels_to_db(labels_df, symbol)
        df_saved = load_labels_from_db(symbol)
        return df_saved.to_dict('records')
    return []

@app.callback(
    Output('saved-labels-table', 'data', allow_duplicate=True),
    Input('load-saved-btn', 'n_clicks'),
    State('symbol-input', 'value'),
    prevent_initial_call=True
)
def load_saved_labels(n_clicks, symbol):
    if n_clicks > 0:
        df_saved = load_labels_from_db(symbol)
        return df_saved.to_dict('records')
    return []

@app.callback(
    Output('saved-labels-table', 'data', allow_duplicate=True),
    Input('delete-btn', 'n_clicks'),
    State('saved-labels-table', 'selected_rows'),
    State('saved-labels-table', 'data'),
    prevent_initial_call=True
)
def delete_label(n_clicks, selected_rows, table_data):
    if n_clicks > 0 and selected_rows:
        row_id = table_data[selected_rows[0]]['id']
        delete_label_from_db(row_id)
        df_saved = pd.DataFrame(table_data)
        df_saved = df_saved[df_saved['id'] != row_id]
        return df_saved.to_dict('records')
    return table_data

# Start backtest (spawns background thread)
@app.callback(
    Output('batch-interval', 'disabled'),
    Output('backtest-summary', 'children'),
    Input('run-backtest-btn', 'n_clicks'),
    State('bt-symbol-input', 'value'),
    State('bt-interval', 'value'),
    State('bt-risk-buffer', 'value'),
    State('bt-tp-mult', 'value'),
    State('bt-slippage', 'value'),
    State('bt-commission', 'value'),
    State('bt-entry-rule', 'value'),
    State('bt-direction-filter', 'value'),
    State('w-neighbor', 'value'),
    State('w-slope', 'value'),
    State('w-rsi', 'value')
)
def start_backtest(n_clicks, symbol_or_list, interval, risk_buf_perc, tp_mult, slippage_perc, commission_perc, entry_rule, direction_filter, w_neighbor, w_slope, w_rsi):
    global BACKTEST_THREAD, BACKTEST_RUNNING
    if n_clicks == 0:
        return True, ""

    watchlist = [s.strip() for s in symbol_or_list.split(',') if s.strip()]
    if not watchlist:
        return True, html.Div("No symbols provided.")

    try:
        risk_buffer = float(risk_buf_perc) / 100.0
    except Exception:
        risk_buffer = 0.01
    try:
        tp_multiplier = float(tp_mult)
    except Exception:
        tp_multiplier = 2.0
    try:
        slippage_pct = float(slippage_perc) / 100.0
    except Exception:
        slippage_pct = 0.0
    try:
        commission_pct = float(commission_perc) / 100.0
    except Exception:
        commission_pct = 0.0

    weights = {'neighbor': float(w_neighbor or 1.0), 'slope': float(w_slope or 1.0), 'rsi': float(w_rsi or 1.0)}

    all_labels = pd.concat([load_labels_from_db(s) for s in watchlist]) if watchlist else pd.DataFrame()

    params = {
        'lookahead_bars': 200,
        'risk_buffer': risk_buffer,
        'tp_multiplier': tp_multiplier,
        'interval': interval,
        'slippage_pct': slippage_pct,
        'commission_pct': commission_pct,
        'all_labels_df': all_labels,
        'entry_rule': entry_rule,
        'direction_override': (direction_filter if direction_filter in ['long','short'] else None),
        'weights': weights
    }

    # start background thread
    if BACKTEST_RUNNING:
        return False, html.Div('Backtest already running. Poll progress...')

    BACKTEST_THREAD = threading.Thread(target=background_batch_runner, args=(watchlist, params), daemon=True)
    BACKTEST_THREAD.start()
    return False, html.Div(f"Started backtest for {len(watchlist)} symbols. Polling progress...")

# Interval polling callback to update progress and show results when ready
@app.callback(
    Output('backtest-table', 'data'),
    Output('backtest-summary', 'children'),
    Output('equity-curve', 'figure'),
    Output('batch-progress-table', 'data'),
    Output('batch-total-time', 'children'),
    Output('trades-per-symbol', 'figure'),
    Output('per-symbol-table', 'data'),
    Output('batch-interval', 'disabled'),
    Input('batch-interval', 'n_intervals')
)
def poll_backtest(n_intervals):
    global BACKTEST_RUNNING, BACKTEST_PROGRESS, BACKTEST_RESULT_DF, LAST_BACKTEST_DF
    # Always show current progress
    progress = BACKTEST_PROGRESS.copy()
    if BACKTEST_RUNNING:
        # still running: return progress only
        return dash.no_update, dash.no_update, dash.no_update, progress, dash.no_update, dash.no_update, dash.no_update, False

    # finished: prepare results
    if BACKTEST_RESULT_DF is None or BACKTEST_RESULT_DF.empty:
        return [], html.Div('No trades found.'), go.Figure(), progress, 'Total time: 0 ms', go.Figure(), [], True

    trades_out = BACKTEST_RESULT_DF.copy()
    trades_out['label_time'] = trades_out['label_time'].astype(str)
    trades_out['entry_time'] = trades_out['entry_time'].astype(str)
    trades_out['exit_time'] = trades_out['exit_time'].astype(str)

    # summary
    total = len(trades_out)
    wins = trades_out[trades_out['outcome']=='win']
    losses = trades_out[trades_out['outcome']=='loss']
    no_hits = trades_out[trades_out['outcome']=='no_hit']
    win_rate = len(wins)/total if total>0 else 0.0
    avg_r = trades_out['r_multiple'].mean()
    expectancy = ((wins['r_multiple'].mean() if len(wins)>0 else 0.0) * len(wins) +
                 (losses['r_multiple'].mean() if len(losses)>0 else 0.0) * len(losses) +
                 (no_hits['r_multiple'].mean() if len(no_hits)>0 else 0.0) * len(no_hits)) / total
    summary_div = html.Div([
        html.P(f"Trades: {total}"),
        html.P(f"Wins: {len(wins)}"),
        html.P(f"Losses: {len(losses)}"),
        html.P(f"No hits: {len(no_hits)}"),
        html.P(f"Win rate: {win_rate:.2%}"),
        html.P(f"Average R: {avg_r:.3f}"),
        html.P(f"Expectancy (R): {expectancy:.3f}")
    ])

    equity = trades_out['r_multiple'].cumsum().fillna(method='ffill').fillna(0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity.tolist(), mode='lines+markers', name='Equity (R)'))
    fig.update_layout(title='Equity Curve (R-units)', xaxis_title='Trade #', yaxis_title='Cumulative R')

    LAST_BACKTEST_DF = trades_out

    # per-symbol breakdown
    per_symbol = trades_out.groupby('symbol').agg(trades=('label_id','count'), wins=('outcome', lambda x: (x=='win').sum()), losses=('outcome', lambda x: (x=='loss').sum()))
    per_symbol['win_rate'] = per_symbol['wins'] / per_symbol['trades']
    per_symbol = per_symbol.reset_index()

    bar = go.Figure()
    bar.add_trace(go.Bar(x=per_symbol['symbol'], y=per_symbol['trades'], name='Trades'))
    bar.update_layout(title='Trades per symbol', xaxis_title='Symbol', yaxis_title='Number of trades')

    return trades_out.to_dict('records'), summary_div, fig, BACKTEST_PROGRESS, f"Total time: {len(BACKTEST_PROGRESS)} symbols processed", bar, per_symbol.to_dict('records'), True

@app.callback(
    Output('download-trades', 'data'),
    Input('export-csv-btn', 'n_clicks')
)
def export_csv(n_clicks):
    if n_clicks == 0:
        return dash.no_update
    if 'LAST_BACKTEST_DF' not in globals() or LAST_BACKTEST_DF is None or LAST_BACKTEST_DF.empty:
        return dash.no_update
    buf = io.StringIO()
    LAST_BACKTEST_DF.to_csv(buf, index=False)
    buf.seek(0)
    return dcc.send_string(buf.getvalue(), filename=f'backtest_trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

# --------------------------- Run App ---------------------------

