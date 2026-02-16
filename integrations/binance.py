#!/usr/bin/env python3
"""
Binance REST API 클라이언트
필요 권한: Read Only (읽기 전용, IP 제한 권장)
설정 파일: integrations/.env

엔드포인트:
  GET /api/v3/account                          → Spot 잔고
  GET /sapi/v1/simple-earn/flexible/position   → Flexible Earn
  GET /sapi/v1/simple-earn/locked/position     → Locked Earn
"""

import hmac
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

BASE_URL = "https://api.binance.com"
ENV_PATH  = Path(__file__).parent / ".env"


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
def _load_keys() -> tuple:
    """integrations/.env 에서 API 키 로드"""
    if not ENV_PATH.exists():
        raise FileNotFoundError(
            f"API 키 파일 없음: {ENV_PATH}\n"
            "integrations/.env.example 을 복사하고 키를 입력하세요."
        )
    keys = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            keys[k.strip()] = v.strip()

    api_key = keys.get("BINANCE_API_KEY", "")
    secret  = keys.get("BINANCE_SECRET_KEY", "")
    if not api_key or not secret or api_key.startswith("your_"):
        raise ValueError(
            "BINANCE_API_KEY / BINANCE_SECRET_KEY 를 .env 파일에 입력하세요.\n"
            f"파일 경로: {ENV_PATH}"
        )
    return api_key, secret


def _request(endpoint: str, api_key: str, secret: str, params: dict = None) -> dict:
    """HMAC-SHA256 서명 GET 요청"""
    params = dict(params or {})
    params["timestamp"] = int(time.time() * 1000)
    query = urllib.parse.urlencode(params)
    sig   = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    url   = f"{BASE_URL}{endpoint}?{query}&signature={sig}"

    req = urllib.request.Request(url, headers={"X-MBX-APIKEY": api_key})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


# ─────────────────────────────────────────────────────────────────────────────
# 공개 함수
# ─────────────────────────────────────────────────────────────────────────────
def fetch_balances() -> dict:
    """
    Spot + Simple Earn 잔고 조회

    Returns:
        spot       : {asset: qty}  — Spot 지갑
        earn       : {asset: qty}  — Simple Earn (Flexible + Locked 합산)
        fetched_at : ISO 시각 문자열
    """
    api_key, secret = _load_keys()

    # Spot
    account = _request("/api/v3/account", api_key, secret)
    spot = {
        b["asset"]: float(b["free"]) + float(b["locked"])
        for b in account.get("balances", [])
        if float(b["free"]) + float(b["locked"]) > 1e-8
    }

    earn: dict = {}

    # Flexible Earn
    try:
        flex = _request("/sapi/v1/simple-earn/flexible/position",
                        api_key, secret, {"size": 100})
        for row in flex.get("rows", []):
            a = row["asset"]
            earn[a] = earn.get(a, 0.0) + float(row.get("totalAmount", 0))
    except Exception:
        pass

    # Locked Earn
    try:
        locked = _request("/sapi/v1/simple-earn/locked/position",
                          api_key, secret, {"size": 100})
        for row in locked.get("rows", []):
            a = row["asset"]
            earn[a] = earn.get(a, 0.0) + float(row.get("amount", 0))
    except Exception:
        pass

    return {
        "spot": spot,
        "earn": earn,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


def get_crypto_summary(btc_price_usd: float) -> dict:
    """
    크립토 포트폴리오 USD 환산 요약

    Args:
        btc_price_usd: 현재 BTC/USD 가격 (yfinance 조회값 사용)

    Returns:
        btc_usd_value : BTC 평가액 (USD)
        usdc_usd      : USDC 잔고 (USD)
        usdt_usd      : USDT 잔고 (USD)
        detail        : 수량 및 원시 잔고 상세
        fetched_at    : 조회 시각
        source        : "Binance API"
    """
    data    = fetch_balances()
    spot    = data["spot"]
    earn    = data["earn"]

    def total(asset: str) -> float:
        return spot.get(asset, 0.0) + earn.get(asset, 0.0)

    btc_qty  = total("BTC")
    usdc_qty = total("USDC")
    usdt_qty = total("USDT")

    return {
        "btc_usd_value": round(btc_qty * btc_price_usd, 2),
        "usdc_usd":      round(usdc_qty, 2),
        "usdt_usd":      round(usdt_qty, 2),
        "detail": {
            "btc_qty":  btc_qty,
            "usdc_qty": usdc_qty,
            "usdt_qty": usdt_qty,
            "spot":     spot,
            "earn":     earn,
        },
        "fetched_at": data["fetched_at"],
        "source": "Binance API",
    }


if __name__ == "__main__":
    import sys
    print("Binance API 연결 테스트...")
    try:
        balances = fetch_balances()
        print(f"✅ 연결 성공  ({balances['fetched_at']})")
        print(f"  Spot 잔고: {list(balances['spot'].keys())}")
        print(f"  Earn 잔고: {list(balances['earn'].keys())}")
    except Exception as e:
        print(f"❌ 연결 실패: {e}", file=sys.stderr)
        sys.exit(1)
