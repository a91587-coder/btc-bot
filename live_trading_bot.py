# -*- coding: utf-8 -*-

import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException

# =========================================================
# 설정
# =========================================================

API_KEY = "irl3m2s1C2oOuRH134QbOr9aHg37GfBB4IDQDJ7uLw1mrBTcZca6bkUXW0fqBB5j"
API_SECRET = "LP8DSvLNy0AJanwftLTETt6YwOCpY6CxevNun7ZYDwfBVkKJuRofwyunJc5dRU1V"

TELEGRAM_BOT_TOKEN = " 8779527463:AAG3wnJ6kkchWO4cEe7mM5yQ1XXl6d-_H1U"   # 없으면 빈칸
TELEGRAM_CHAT_ID = "6929667030"     # 없으면 빈칸

SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1HOUR

# ===== 모드 =====
DRY_RUN = True            # 처음엔 무조건 True
USE_TESTNET = False
LEVERAGE = 3
MARGIN_TYPE = "ISOLATED"

# ===== 백테스트에서 좋았던 파라미터 =====
RISK_PCT = 0.03

TP1_PCT = 0.011
TP2_PCT = 0.018
SL_PCT = 0.0045

TIME_STOP_HOURS = 24
COOLDOWN_BARS = 4

ADX_MIN = 12
BODY_RATIO_MIN = 0.48
VOL_MULT = 1.0
RSI_MAX = 52
EMA_TOUCH_BUFFER = 0.001

PARTIAL_TP1_RATIO = 0.5   # TP1에서 절반 익절

# ===== 루프 =====
LOOP_SECONDS = 20

# ===== 파일 =====
STATE_FILE = "live_bot_state.json"
LOG_FILE = "live_bot_log.txt"

# =========================================================
# 유틸
# =========================================================

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    text = f"[{now_str()}] {msg}"
    print(text)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=10
        )
    except Exception as e:
        log(f"텔레그램 실패: {e}")

def round_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step

# =========================================================
# 상태 저장
# =========================================================

def default_state():
    return {
        "position": None,
        "last_entry_bar_time": None,
        "last_exit_bar_time": None,
    }

def load_state():
    if not Path(STATE_FILE).exists():
        return default_state()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        base = default_state()
        base.update(data)
        return base
    except Exception:
        return default_state()

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

# =========================================================
# 바이낸스
# =========================================================

def make_client():
    client = Client(API_KEY, API_SECRET, testnet=USE_TESTNET)
    if USE_TESTNET:
        client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
    return client

def ensure_margin_and_leverage(client: Client):
    try:
        client.futures_change_margin_type(symbol=SYMBOL, marginType=MARGIN_TYPE)
    except BinanceAPIException as e:
        if "No need to change margin type" not in str(e):
            log(f"마진타입 설정: {e}")

    try:
        client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
    except Exception as e:
        log(f"레버리지 설정 실패: {e}")

def get_symbol_filters(client: Client, symbol: str):
    info = client.futures_exchange_info()
    for s in info["symbols"]:
        if s["symbol"] == symbol:
            filters = {f["filterType"]: f for f in s["filters"]}
            return {
                "step_size": float(filters["LOT_SIZE"]["stepSize"]),
                "min_qty": float(filters["LOT_SIZE"]["minQty"]),
                "tick_size": float(filters["PRICE_FILTER"]["tickSize"]),
            }
    raise ValueError(f"{symbol} 필터 못 찾음")

def get_usdt_balance(client: Client) -> float:
    balances = client.futures_account_balance()
    for b in balances:
        if b["asset"] == "USDT":
            return float(b["balance"])
    return 0.0

def get_mark_price(client: Client) -> float:
    return float(client.futures_mark_price(symbol=SYMBOL)["markPrice"])

def get_open_position(client: Client):
    positions = client.futures_position_information(symbol=SYMBOL)
    for p in positions:
        amt = float(p["positionAmt"])
        if abs(amt) > 0:
            return {
                "amount": amt,
                "entry_price": float(p["entryPrice"]),
                "unrealized_pnl": float(p["unRealizedProfit"]),
            }
    return None

def market_buy(client: Client, qty: float):
    if DRY_RUN:
        log(f"[DRY_RUN] BUY MARKET qty={qty}")
        return
    client.futures_create_order(
        symbol=SYMBOL,
        side="BUY",
        type="MARKET",
        quantity=qty
    )

def market_sell_reduce(client: Client, qty: float):
    if DRY_RUN:
        log(f"[DRY_RUN] SELL reduce qty={qty}")
        return
    client.futures_create_order(
        symbol=SYMBOL,
        side="SELL",
        type="MARKET",
        quantity=qty,
        reduceOnly=True
    )

# =========================================================
# 지표
# =========================================================

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def adx(df, n=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(n).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(n).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(n).mean() / atr

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
    return dx.rolling(n).mean()

def get_klines_df(client: Client, limit=300):
    raw = client.futures_klines(symbol=SYMBOL, interval=INTERVAL, limit=limit)

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    df["ema20"] = ema(df["close"], 20)
    df["ema200"] = ema(df["close"], 200)
    df["rsi"] = rsi(df["close"])
    df["adx"] = adx(df)
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = df["body"] / df["range"]

    return df

# =========================================================
# 시그널
# =========================================================

def get_latest_closed_bar_signal(df: pd.DataFrame):
    row = df.iloc[-2]   # 직전 완성봉

    cond = (
        (row["close"] > row["ema200"]) and
        (row["low"] <= row["ema20"] * (1 + EMA_TOUCH_BUFFER)) and
        (row["close"] > row["ema20"]) and
        (row["body_ratio"] > BODY_RATIO_MIN) and
        (row["volume"] > row["vol_ma20"] * VOL_MULT) and
        (row["rsi"] < RSI_MAX) and
        (row["adx"] > ADX_MIN)
    )

    return {
        "signal": bool(cond),
        "bar_time": str(row["open_time"]),
        "close": float(row["close"]),
        "ema20": float(row["ema20"]),
        "ema200": float(row["ema200"]),
        "rsi": float(row["rsi"]) if pd.notna(row["rsi"]) else None,
        "adx": float(row["adx"]) if pd.notna(row["adx"]) else None,
        "body_ratio": float(row["body_ratio"]) if pd.notna(row["body_ratio"]) else None,
        "volume": float(row["volume"]),
        "vol_ma20": float(row["vol_ma20"]) if pd.notna(row["vol_ma20"]) else None,
    }

# =========================================================
# 수량 계산
# =========================================================

def calc_order_qty(balance: float, entry_price: float, filters: dict) -> float:
    risk_usdt = balance * RISK_PCT
    sl_price = entry_price * (1 - SL_PCT)
    risk_per_unit = entry_price - sl_price

    if risk_per_unit <= 0:
        return 0.0

    qty = risk_usdt / risk_per_unit

    max_notional = balance * LEVERAGE * 0.95
    max_qty = max_notional / entry_price
    qty = min(qty, max_qty)

    qty = round_step(qty, filters["step_size"])

    if qty < filters["min_qty"]:
        return 0.0

    return qty

# =========================================================
# 메인
# =========================================================

def main():
    state = load_state()
    client = make_client()
    filters = get_symbol_filters(client, SYMBOL)

    ensure_margin_and_leverage(client)

    log("===== 실전형 BTC 자동매매 시작 =====")
    log(f"SYMBOL={SYMBOL}, DRY_RUN={DRY_RUN}, LEVERAGE={LEVERAGE}")
    send_telegram(
        f"🚀 BTC 실전형 자동매매 시작\n"
        f"SYMBOL={SYMBOL}\n"
        f"DRY_RUN={DRY_RUN}\n"
        f"LEVERAGE={LEVERAGE}\n"
        f"RISK={RISK_PCT}\n"
        f"TP1={TP1_PCT}, TP2={TP2_PCT}, SL={SL_PCT}"
    )

    while True:
        try:
            df = get_klines_df(client, limit=300)
            sig = get_latest_closed_bar_signal(df)
            pos_exchange = get_open_position(client)

            # 거래소 포지션 있는데 상태 없으면 복구
            if pos_exchange is not None and state["position"] is None:
                entry_price = pos_exchange["entry_price"]
                qty = abs(pos_exchange["amount"])
                state["position"] = {
                    "entry_price": entry_price,
                    "qty_total": qty,
                    "qty_remaining": qty,
                    "entry_time": int(time.time()),
                    "tp1_price": entry_price * (1 + TP1_PCT),
                    "tp2_price": entry_price * (1 + TP2_PCT),
                    "sl_price": entry_price * (1 - SL_PCT),
                    "tp1_done": False,
                    "side": "LONG"
                }
                save_state(state)
                log("상태 복구 완료")

            # 거래소 포지션 없는데 상태만 있으면 정리
            if pos_exchange is None and state["position"] is not None:
                state["position"] = None
                state["last_exit_bar_time"] = sig["bar_time"]
                save_state(state)

            # =================================================
            # 포지션 관리
            # =================================================
            if state["position"] is not None and pos_exchange is not None:
                p = state["position"]
                mark_price = get_mark_price(client)
                held_hours = (int(time.time()) - p["entry_time"]) / 3600

                # TP1
                if (not p["tp1_done"]) and mark_price >= p["tp1_price"]:
                    close_qty = round_step(p["qty_remaining"] * PARTIAL_TP1_RATIO, filters["step_size"])
                    if close_qty >= filters["min_qty"]:
                        log(f"TP1 도달: {mark_price}, 부분익절 qty={close_qty}")
                        send_telegram(
                            f"✅ TP1 부분익절\n"
                            f"진입가={p['entry_price']:.2f}\n"
                            f"현재가={mark_price:.2f}\n"
                            f"청산수량={close_qty}"
                        )
                        market_sell_reduce(client, close_qty)
                        p["qty_remaining"] -= close_qty
                        p["tp1_done"] = True
                        p["sl_price"] = p["entry_price"]   # 브레이크이븐
                        save_state(state)
                        time.sleep(2)

                # TP2
                if mark_price >= p["tp2_price"]:
                    qty_now = abs(pos_exchange["amount"])
                    log(f"TP2 도달: {mark_price}, 전체청산 qty={qty_now}")
                    send_telegram(
                        f"🎯 TP2 최종익절\n"
                        f"진입가={p['entry_price']:.2f}\n"
                        f"현재가={mark_price:.2f}"
                    )
                    close_position_qty = round_step(qty_now, filters["step_size"])
                    if close_position_qty >= filters["min_qty"]:
                        market_sell_reduce(client, close_position_qty)
                    state["position"] = None
                    state["last_exit_bar_time"] = sig["bar_time"]
                    save_state(state)
                    time.sleep(LOOP_SECONDS)
                    continue

                # SL / BE
                if mark_price <= p["sl_price"]:
                    qty_now = abs(pos_exchange["amount"])
                    log(f"SL/BE 도달: {mark_price}, 전체청산 qty={qty_now}")
                    send_telegram(
                        f"🛑 손절/BE 청산\n"
                        f"진입가={p['entry_price']:.2f}\n"
                        f"현재가={mark_price:.2f}"
                    )
                    close_position_qty = round_step(qty_now, filters["step_size"])
                    if close_position_qty >= filters["min_qty"]:
                        market_sell_reduce(client, close_position_qty)
                    state["position"] = None
                    state["last_exit_bar_time"] = sig["bar_time"]
                    save_state(state)
                    time.sleep(LOOP_SECONDS)
                    continue

                # 시간청산
                if held_hours >= TIME_STOP_HOURS:
                    qty_now = abs(pos_exchange["amount"])
                    log(f"시간청산: {held_hours:.2f}h, qty={qty_now}")
                    send_telegram(
                        f"⏰ 시간청산\n"
                        f"보유시간={held_hours:.2f}h\n"
                        f"현재가={mark_price:.2f}"
                    )
                    close_position_qty = round_step(qty_now, filters["step_size"])
                    if close_position_qty >= filters["min_qty"]:
                        market_sell_reduce(client, close_position_qty)
                    state["position"] = None
                    state["last_exit_bar_time"] = sig["bar_time"]
                    save_state(state)
                    time.sleep(LOOP_SECONDS)
                    continue

            # =================================================
            # 신규 진입
            # =================================================
            if state["position"] is None and pos_exchange is None:
                # 같은 봉 중복 진입 방지
                if state["last_entry_bar_time"] == sig["bar_time"]:
                    time.sleep(LOOP_SECONDS)
                    continue

                # 쿨다운
                if state["last_exit_bar_time"] is not None:
                    last_exit_time = pd.to_datetime(state["last_exit_bar_time"])
                    current_bar_time = pd.to_datetime(sig["bar_time"])
                    bars_diff = (current_bar_time - last_exit_time).total_seconds() / 3600
                    if bars_diff < COOLDOWN_BARS:
                        time.sleep(LOOP_SECONDS)
                        continue

                if sig["signal"]:
                    balance = get_usdt_balance(client)
                    entry_price = sig["close"]
                    qty = calc_order_qty(balance, entry_price, filters)

                    if qty > 0:
                        log(
                            f"LONG 신호 | bar={sig['bar_time']} | close={sig['close']:.2f} | "
                            f"rsi={sig['rsi']} | adx={sig['adx']} | body={sig['body_ratio']} | qty={qty}"
                        )
                        send_telegram(
                            f"📍 LONG 진입 신호\n"
                            f"시간={sig['bar_time']}\n"
                            f"가격={sig['close']:.2f}\n"
                            f"RSI={sig['rsi']}\n"
                            f"ADX={sig['adx']}\n"
                            f"BodyRatio={sig['body_ratio']}\n"
                            f"수량={qty}"
                        )

                        market_buy(client, qty)

                        state["position"] = {
                            "entry_price": entry_price,
                            "qty_total": qty,
                            "qty_remaining": qty,
                            "entry_time": int(time.time()),
                            "tp1_price": entry_price * (1 + TP1_PCT),
                            "tp2_price": entry_price * (1 + TP2_PCT),
                            "sl_price": entry_price * (1 - SL_PCT),
                            "tp1_done": False,
                            "side": "LONG"
                        }
                        state["last_entry_bar_time"] = sig["bar_time"]
                        save_state(state)

            time.sleep(LOOP_SECONDS)

        except BinanceAPIException as e:
            log(f"바이낸스 에러: {e}")
            send_telegram(f"❌ 바이낸스 에러\n{e}")
            time.sleep(LOOP_SECONDS)

        except Exception as e:
            log(f"일반 에러: {e}")
            send_telegram(f"❌ 일반 에러\n{e}")
            time.sleep(LOOP_SECONDS)

if __name__ == "__main__":
    main()