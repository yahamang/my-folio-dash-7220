#!/usr/bin/env python3
"""
한국투자증권 Open API 클라이언트
"""

import os
import requests
from datetime import datetime, timedelta
from pathlib import Path

# 환경변수 로드
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass

# API 설정
KIS_APP_KEY = os.getenv("KIS_APP_KEY")
KIS_APP_SECRET = os.getenv("KIS_APP_SECRET")
KIS_ENV = os.getenv("KIS_ENV", "real")  # real or virtual

# 디버그: KIS_ENV 값 확인
print(f"[KIS DEBUG] KIS_ENV={KIS_ENV}, APP_KEY={KIS_APP_KEY[:10] if KIS_APP_KEY else None}...")

# API URL - 명시적으로 real 환경 사용
# Vercel 환경변수 문제로 인해 임시로 하드코딩
BASE_URL = "https://openapi.koreainvestment.com:9443"  # real
# if KIS_ENV == "real":
#     BASE_URL = "https://openapi.koreainvestment.com:9443"
# else:
#     BASE_URL = "https://openapivts.koreainvestment.com:29443"

# 토큰 캐시 (메모리)
_token_cache = {"access_token": None, "expires_at": None}


def get_access_token() -> str:
    """OAuth 토큰 발급 (캐시 사용)"""
    global _token_cache

    # 캐시된 토큰이 유효하면 재사용
    if _token_cache["access_token"] and _token_cache["expires_at"]:
        if datetime.now() < _token_cache["expires_at"]:
            return _token_cache["access_token"]

    # 새 토큰 발급
    url = f"{BASE_URL}/oauth2/tokenP"
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET
    }

    response = requests.post(url, headers=headers, json=body, timeout=10)
    response.raise_for_status()

    data = response.json()
    access_token = data["access_token"]
    expires_in = int(data["expires_in"])  # 초 단위 (보통 86400 = 24시간)

    # 캐시 저장 (만료 10분 전으로 설정)
    _token_cache["access_token"] = access_token
    _token_cache["expires_at"] = datetime.now() + timedelta(seconds=expires_in - 600)

    return access_token


def get_stock_price(ticker: str) -> dict:
    """
    국내 주식 현재가 조회

    Args:
        ticker: 종목코드 (6자리, 예: "005930")

    Returns:
        {
            "price": 현재가,
            "change_pct": 등락률,
            "volume": 거래량,
            "open": 시가,
            "high": 고가,
            "low": 저가
        }
    """
    # 티커에서 .KS 또는 .KQ 제거
    ticker_code = ticker.replace(".KS", "").replace(".KQ", "").zfill(6)

    token = get_access_token()
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price"

    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
        "tr_id": "FHKST01010100"  # 국내주식 현재가 시세
    }

    params = {
        "FID_COND_MRKT_DIV_CODE": "J",  # 주식
        "FID_INPUT_ISCD": ticker_code
    }

    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    if data.get("rt_cd") != "0":
        raise Exception(f"KIS API Error: {data.get('msg1', 'Unknown error')}")

    output = data["output"]

    return {
        "price": float(output["stck_prpr"]),  # 현재가
        "change_pct": float(output["prdy_ctrt"]),  # 전일대비율
        "volume": int(output["acml_vol"]),  # 누적거래량
        "open": float(output["stck_oprc"]),  # 시가
        "high": float(output["stck_hgpr"]),  # 고가
        "low": float(output["stck_lwpr"])  # 저가
    }


def get_foreign_institution_data(ticker: str) -> dict:
    """
    외국인/기관 매매 동향 조회

    Args:
        ticker: 종목코드 (6자리)

    Returns:
        {
            "foreign_buy": 외국인 매수량,
            "foreign_sell": 외국인 매도량,
            "foreign_net": 외국인 순매수,
            "institution_net": 기관 순매수
        }
    """
    ticker_code = ticker.replace(".KS", "").replace(".KQ", "").zfill(6)

    token = get_access_token()
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-investor"

    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
        "tr_id": "FHKST01010900"  # 국내주식 투자자별 매매동향
    }

    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": ticker_code
    }

    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    if data.get("rt_cd") != "0":
        # 데이터 없으면 0으로 반환
        return {
            "foreign_buy": 0,
            "foreign_sell": 0,
            "foreign_net": 0,
            "institution_net": 0
        }

    output = data["output"]

    return {
        "foreign_buy": int(output.get("frgn_ntby_qty", 0)),  # 외국인 순매수
        "foreign_sell": 0,  # API에서 별도 제공 안 함
        "foreign_net": int(output.get("frgn_ntby_qty", 0)),
        "institution_net": int(output.get("orgn_ntby_qty", 0))  # 기관 순매수
    }


def fetch_korean_stocks(tickers: list) -> dict:
    """
    여러 한국 주식 현재가 일괄 조회

    Args:
        tickers: 종목코드 리스트 (예: ["055550.KS", "267260.KS"])

    Returns:
        {
            "055550.KS": {"price": 100000, "change_pct": 2.5, ...},
            "267260.KS": {"price": 95000, "change_pct": -1.2, ...},
            ...
        }
    """
    results = {}
    errors = []

    for ticker in tickers:
        try:
            data = get_stock_price(ticker)
            results[ticker] = data
        except Exception as e:
            errors.append(f"{ticker}: {str(e)}")
            results[ticker] = None

    return {"prices": results, "errors": errors}


if __name__ == "__main__":
    # 테스트
    if not KIS_APP_KEY or not KIS_APP_SECRET:
        print("❌ KIS API Key가 설정되지 않았습니다.")
        print("integrations/.env 파일에 KIS_APP_KEY와 KIS_APP_SECRET을 설정하세요.")
    else:
        print("🔑 API Key 확인 완료")
        print(f"📍 환경: {KIS_ENV}")

        # 신한지주 테스트
        try:
            print("\n📊 신한지주(055550) 현재가 조회 중...")
            data = get_stock_price("055550.KS")
            print(f"✅ 현재가: ₩{data['price']:,.0f}")
            print(f"   등락률: {data['change_pct']:+.2f}%")
            print(f"   거래량: {data['volume']:,}")
        except Exception as e:
            print(f"❌ 오류: {e}")
