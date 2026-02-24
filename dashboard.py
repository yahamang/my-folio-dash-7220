#!/usr/bin/env python3
"""
투자 대시보드 생성기
사용법:
  python3 ~/investment/dashboard.py          # 대시보드 생성 + 스냅샷 저장
  python3 ~/investment/dashboard.py --no-save # 스냅샷 저장 없이 HTML만 생성
"""

import json
import sys
import warnings
from datetime import datetime, date
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("경고: yfinance 미설치. pip3 install yfinance 실행 후 재시도")

BASE_DIR        = Path(__file__).parent
CONFIG_PATH     = BASE_DIR / "config.json"
DATA_DIR        = BASE_DIR / "data"
OUTPUT_PATH     = BASE_DIR / "dashboard.html"
INTEGRATIONS_DIR = BASE_DIR / "integrations"

# Binance 연동 시도
sys.path.insert(0, str(BASE_DIR))
try:
    from integrations.binance import get_crypto_summary as _binance_summary
    HAS_BINANCE = True
except ImportError:
    HAS_BINANCE = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. 설정 로더
# ─────────────────────────────────────────────────────────────────────────────
def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 데이터 수집 (yfinance 배치 조회)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_all_prices(config: dict) -> dict:
    if not HAS_YFINANCE:
        return {"prices": {}, "errors": ["yfinance not installed"], "fetched_at": datetime.now().isoformat()}

    needed = set()
    for acc in config["portfolio"]["accounts"].values():
        for h in acc["holdings"]:
            needed.add(h["ticker"])
    for group in ["indices", "macro", "commodities", "crypto"]:
        for item in config["market_indicators"].get(group, []):
            needed.add(item["ticker"])

    prices, errors = {}, []
    ticker_list = list(needed)

    try:
        raw = yf.download(ticker_list, period="2d", interval="1d",
                          progress=False, auto_adjust=True)
        close = raw["Close"]

        for t in ticker_list:
            try:
                series = close[t] if len(ticker_list) > 1 else close
                series = series.dropna()
                if len(series) == 0:
                    raise ValueError("no data")
                cur  = float(series.iloc[-1])
                prev = float(series.iloc[-2]) if len(series) >= 2 else cur
                chg  = (cur - prev) / prev * 100 if prev else 0
                prices[t] = {"price": cur, "prev_close": prev, "change_pct": round(chg, 2)}
            except Exception as e:
                errors.append(f"{t}: {e}")
                prices[t] = {"price": None, "prev_close": None, "change_pct": 0}
    except Exception as e:
        errors.append(f"batch download: {e}")

    return {
        "prices": prices,
        "errors": errors,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "source": "Yahoo Finance",
        "binance": None,   # main()에서 채워짐
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. 포트폴리오 계산
# ─────────────────────────────────────────────────────────────────────────────
def calculate_portfolio(config: dict, price_data: dict) -> dict:
    prices   = price_data.get("prices", {})
    fx_info  = prices.get("KRW=X", {})
    usd_krw  = fx_info.get("price") or config["meta"]["usd_to_krw_fallback"]

    ac_values    = {}   # asset_class → KRW
    account_details = {}
    total_krw    = 0

    def price_to_krw(ticker: str, shares: float) -> tuple:
        info  = prices.get(ticker, {})
        raw   = info.get("price")
        if raw is None:
            return 0, "—", info.get("change_pct", 0)
        if ticker.endswith(".KS") or ticker.endswith(".KQ"):
            val   = raw * shares
            disp  = f"₩{raw:,.0f}"
        else:
            val   = raw * shares * usd_krw
            disp  = f"${raw:.2f}"
        return val, disp, info.get("change_pct", 0)

    # 계좌별 집계
    total_cost_krw   = 0   # 전체 매입금액 합산
    usd_val_krw      = 0   # USD 자산 평가액 합산 (환율 노출)
    annual_div_krw   = 0   # 연간 배당 예측 합산
    div_details      = []  # 배당 상세

    for acc_id, acc in config["portfolio"]["accounts"].items():
        acc_total = 0
        acc_cost  = 0
        holdings_detail = []
        is_pension = (acc["type"] == "pension")

        for h in acc["holdings"]:
            val, price_disp, chg = price_to_krw(h["ticker"], h["shares"])
            # 연금계좌는 종목 asset_class 무시하고 "pension" 으로 강제 매핑
            ac = "pension" if is_pension else h["asset_class"]
            ac_values[ac] = ac_values.get(ac, 0) + val
            acc_total     += val

            # USD 자산 노출 집계
            is_kr_ticker = h["ticker"].endswith(".KS") or h["ticker"].endswith(".KQ")
            if not is_kr_ticker:
                usd_val_krw += val

            # 수익률 계산 (avg_cost_krw 우선, 없으면 avg_price 사용)
            avg_cost_krw_h = h.get("avg_cost_krw", 0) or 0
            avg_p  = h.get("avg_price", 0) or 0
            info   = prices.get(h["ticker"], {})
            cur_p  = info.get("price")
            if avg_cost_krw_h > 0 and val > 0:
                pnl_krw = val - avg_cost_krw_h
                pnl_pct = pnl_krw / avg_cost_krw_h * 100
                acc_cost += avg_cost_krw_h
            elif avg_p > 0 and cur_p:
                pnl_pct = (cur_p - avg_p) / avg_p * 100
                cost_krw_h = avg_p * h["shares"] * (1 if is_kr_ticker else usd_krw)
                pnl_krw  = val - cost_krw_h
                acc_cost += cost_krw_h
            else:
                pnl_pct, pnl_krw = None, None

            # 배당 예측
            div_yield = h.get("dividend_yield", 0) or 0
            if div_yield > 0 and val > 0:
                div_krw = val * div_yield / 100
                annual_div_krw += div_krw
                div_details.append({
                    "name": h["name"], "yield_pct": div_yield,
                    "value_krw": val, "div_krw": div_krw
                })

            holdings_detail.append({
                "name": h["name"], "ticker": h["ticker"], "shares": h["shares"],
                "price": cur_p, "price_disp": price_disp,
                "change_pct": chg, "value_krw": val, "asset_class": ac,
                "avg_price": avg_p, "pnl_pct": pnl_pct, "pnl_krw": pnl_krw,
            })

        acc_pnl_krw = (acc_total - acc_cost) if acc_cost > 0 else None
        acc_pnl_pct = (acc_pnl_krw / acc_cost * 100) if acc_cost > 0 else None
        account_details[acc_id] = {
            "name": acc["name"], "type": acc["type"],
            "holdings": holdings_detail, "total_krw": acc_total,
            "cost_krw": acc_cost, "pnl_krw": acc_pnl_krw, "pnl_pct": acc_pnl_pct,
        }
        total_cost_krw += acc_cost
        total_krw += acc_total

    # 현금
    cash_cfg   = config["portfolio"]["cash"]
    cash_krw   = cash_cfg["deposit_krw"] + cash_cfg["securities_usd"] * usd_krw
    ac_values["cash"] = ac_values.get("cash", 0) + cash_krw
    total_krw += cash_krw

    # 크립토 (Binance API 우선, 없으면 config.json 값)
    binance_live = price_data.get("binance")
    if binance_live:
        c = binance_live
    else:
        c = config["portfolio"]["crypto"]["binance"]
    crypto_usd = c["btc_usd_value"] + c["usdc_usd"] + c["usdt_usd"]
    crypto_krw = crypto_usd * usd_krw
    crypto_source = (f"Binance API · {binance_live['fetched_at'][11:16]}"
                     if binance_live else "config.json (수동)")
    ac_values["crypto"] = ac_values.get("crypto", 0) + crypto_krw
    total_krw += crypto_krw

    # 배분 분석
    allocation = []
    threshold  = config["rebalancing"]["threshold_pct"]
    for ac_id, info in config["targets"]["allocations"].items():
        cur_krw  = ac_values.get(ac_id, 0)
        cur_pct  = cur_krw / total_krw * 100 if total_krw else 0
        tgt_pct  = info["target_pct"]
        gap_pct  = cur_pct - tgt_pct
        gap_krw  = gap_pct / 100 * total_krw
        status   = "over" if gap_pct > threshold else ("under" if gap_pct < -threshold else "ok")
        allocation.append({
            "id": ac_id, "label": info["label"], "color": info["color"],
            "current_krw": cur_krw, "current_pct": round(cur_pct, 1),
            "target_pct": tgt_pct, "gap_pct": round(gap_pct, 1),
            "gap_krw": round(gap_krw), "status": status
        })

    # 전체 수익성 요약
    total_pnl_krw = (total_krw - total_cost_krw - cash_krw - crypto_krw) if total_cost_krw > 0 else None
    total_pnl_pct = (total_pnl_krw / total_cost_krw * 100) if (total_cost_krw > 0 and total_pnl_krw is not None) else None

    # 리스크 지표
    usd_exposure_pct = usd_val_krw / total_krw * 100 if total_krw else 0
    top_ac = max(ac_values.items(), key=lambda x: x[1]) if ac_values else ("—", 0)
    top_ac_pct = top_ac[1] / total_krw * 100 if total_krw else 0
    top_ac_label = next(
        (v["label"] for k, v in config["targets"]["allocations"].items() if k == top_ac[0]),
        top_ac[0]
    )
    cash_pct = cash_krw / total_krw * 100 if total_krw else 0
    vix = (price_data.get("prices", {}).get("^VIX", {}).get("price") or 0)

    return {
        "total_krw": total_krw, "usd_krw": usd_krw,
        "accounts": account_details, "ac_values": ac_values,
        "allocation": allocation, "cash_krw": cash_krw,
        "crypto_krw": crypto_krw, "crypto_source": crypto_source,
        # 수익성
        "total_cost_krw": total_cost_krw,
        "total_pnl_krw": total_pnl_krw,
        "total_pnl_pct": total_pnl_pct,
        "annual_div_krw": annual_div_krw,
        "div_details": div_details,
        # 리스크
        "usd_exposure_pct": round(usd_exposure_pct, 1),
        "top_ac_label": top_ac_label,
        "top_ac_pct": round(top_ac_pct, 1),
        "cash_pct": round(cash_pct, 1),
        "vix": vix,
    }


def calculate_strategy_signals(config: dict, price_data: dict, portfolio: dict) -> dict:
    """
    전략 신호 계산: 매수/매도/대기 판단을 위한 지표 생성

    Returns:
        market_momentum: 시장 모멘텀 분석 (KOSPI/KOSDAQ/VIX)
        phase_signals: Phase 2/3 실행 신호
        rebalancing_urgency: 리밸런싱 긴급도
        recommended_action: 오늘의 권장 행동
    """
    prices = price_data.get("prices", {})
    today = date.today()

    # ═══════════════════════════════════════════════════════════════
    # 1. 시장 모멘텀 분석
    # ═══════════════════════════════════════════════════════════════
    kospi = prices.get("^KS11", {})
    kosdaq = prices.get("^KQ11", {})
    vix = prices.get("^VIX", {})

    kospi_chg = kospi.get("change_pct", 0)
    kosdaq_chg = kosdaq.get("change_pct", 0)
    vix_val = vix.get("price", 0)

    # 모멘텀 스코어 (-3 ~ +3)
    momentum_score = 0

    # 한국 지수 추세
    if kospi_chg > 1 and kosdaq_chg > 1:
        momentum_score += 2
        kr_trend = "강한 상승"
    elif kospi_chg > 0 and kosdaq_chg > 0:
        momentum_score += 1
        kr_trend = "완만한 상승"
    elif kospi_chg < -1 and kosdaq_chg < -1:
        momentum_score -= 2
        kr_trend = "강한 하락"
    elif kospi_chg < 0 and kosdaq_chg < 0:
        momentum_score -= 1
        kr_trend = "완만한 하락"
    else:
        kr_trend = "혼조"

    # VIX 공포지수
    if vix_val >= 30:
        momentum_score -= 2
        vix_level = "위험 (패닉)"
    elif vix_val >= 25:
        momentum_score -= 1
        vix_level = "주의 (불안)"
    elif vix_val >= 20:
        vix_level = "중립"
    else:
        momentum_score += 1
        vix_level = "안정 (낙관)"

    # 외국인/기관 순매수 (데이터 없으므로 placeholder)
    foreign_inst_flow = "데이터 없음"

    market_momentum = {
        "score": momentum_score,
        "kr_index_trend": kr_trend,
        "kospi_chg": kospi_chg,
        "kosdaq_chg": kosdaq_chg,
        "vix_level": vix_level,
        "vix_value": vix_val,
        "foreign_institutional": foreign_inst_flow,
        "interpretation": (
            "강한 매수 환경" if momentum_score >= 2 else
            "매수 선호" if momentum_score == 1 else
            "중립 관망" if momentum_score == 0 else
            "조정 대기" if momentum_score == -1 else
            "강한 조정"
        )
    }

    # ═══════════════════════════════════════════════════════════════
    # 2. Phase 2/3 실행 신호
    # ═══════════════════════════════════════════════════════════════

    # 1%+ 조정 발생 여부
    correction_1pct = (kospi_chg < -1 or kosdaq_chg < -1)

    # FOMC 근접도 계산
    fomc_events = [e for e in config["events"] if "FOMC" in e["title"]]
    nearest_fomc = None
    fomc_days_away = 999

    if fomc_events:
        future_fomc = [e for e in fomc_events if date.fromisoformat(e["date"]) >= today]
        if future_fomc:
            nearest = min(future_fomc, key=lambda e: date.fromisoformat(e["date"]))
            nearest_fomc = nearest["date"]
            fomc_days_away = (date.fromisoformat(nearest_fomc) - today).days

    # Phase 2 신호: 1%+ 조정
    phase2_triggered = correction_1pct

    # Phase 3 신호: FOMC 7일 이내
    phase3_triggered = (0 <= fomc_days_away <= 7)

    # 매수 타이밍 스코어 (0-10)
    buy_timing_score = 5  # 기본값
    if correction_1pct:
        buy_timing_score += 2
    if vix_val >= 25:
        buy_timing_score += 2
    if momentum_score >= 2:
        buy_timing_score -= 2
    buy_timing_score = max(0, min(10, buy_timing_score))

    phase_signals = {
        "correction_detected": correction_1pct,
        "correction_pct": min(kospi_chg, kosdaq_chg) if correction_1pct else 0,
        "nearest_fomc": nearest_fomc or "없음",
        "fomc_days_away": fomc_days_away if fomc_days_away < 999 else None,
        "phase2_ready": phase2_triggered,
        "phase3_ready": phase3_triggered,
        "buy_timing_score": buy_timing_score,
        "buy_timing_label": (
            "매우 좋음" if buy_timing_score >= 8 else
            "좋음" if buy_timing_score >= 6 else
            "보통" if buy_timing_score >= 4 else
            "나쁨"
        )
    }

    # ═══════════════════════════════════════════════════════════════
    # 3. 리밸런싱 긴급도
    # ═══════════════════════════════════════════════════════════════

    allocation = portfolio["allocation"]
    over = [a for a in allocation if a["status"] == "over"]
    under = [a for a in allocation if a["status"] == "under"]

    # 가장 큰 이탈 자산군
    max_over = max(over, key=lambda a: a["gap_pct"]) if over else None
    max_under = max(under, key=lambda a: abs(a["gap_pct"])) if under else None

    # 긴급도 점수 (0-10)
    urgency_score = 0
    if max_over and abs(max_over["gap_pct"]) > 10:
        urgency_score += 5
    if max_under and abs(max_under["gap_pct"]) > 10:
        urgency_score += 5

    rebalancing_urgency = {
        "urgency_score": urgency_score,
        "urgency_level": (
            "매우 높음" if urgency_score >= 8 else
            "높음" if urgency_score >= 5 else
            "보통" if urgency_score >= 3 else
            "낮음"
        ),
        "top_over": {
            "label": max_over["label"],
            "gap_pct": max_over["gap_pct"],
            "gap_krw": max_over["gap_krw"],
            "action": "매도 또는 추가 매수 중단"
        } if max_over else None,
        "top_under": {
            "label": max_under["label"],
            "gap_pct": max_under["gap_pct"],
            "gap_krw": max_under["gap_krw"],
            "action": "추가 매수 우선"
        } if max_under else None,
        "total_imbalanced": len(over) + len(under)
    }

    # ═══════════════════════════════════════════════════════════════
    # 4. 오늘의 권장 액션
    # ═══════════════════════════════════════════════════════════════

    action = "관망"
    rationale = []
    metrics_to_watch = []

    # Phase 2 실행 여부
    if phase2_triggered and max_under:
        action = "Phase 2 매수"
        rationale.append(f"1%+ 조정 발생 ({min(kospi_chg, kosdaq_chg):.1f}%)")
        rationale.append(f"부족 자산군: {max_under['label']} ({max_under['gap_pct']:+.1f}%p)")
        metrics_to_watch.extend(["KOSPI/KOSDAQ 추가 하락", "VIX 30 돌파 여부"])

    # Phase 3 실행 여부
    elif phase3_triggered and max_under:
        action = "Phase 3 매수 준비"
        rationale.append(f"FOMC {fomc_days_away}일 전")
        rationale.append(f"부족 자산군: {max_under['label']} ({max_under['gap_pct']:+.1f}%p)")
        metrics_to_watch.extend(["FOMC 성명 톤", "연준 점도표"])

    # 리밸런싱 매도 필요
    elif max_over and abs(max_over["gap_pct"]) > 8:
        action = "부분 매도 검토"
        rationale.append(f"초과 자산군: {max_over['label']} ({max_over['gap_pct']:+.1f}%p)")
        rationale.append("목표 비중으로 복귀 필요")
        metrics_to_watch.extend(["추가 상승 시 매도 타이밍"])

    # 과열 경계
    elif momentum_score >= 2 and vix_val < 15:
        action = "관망 (과열 주의)"
        rationale.append("시장 과열 신호 (VIX 낮음 + 강한 상승)")
        rationale.append("추격 매수 자제")
        metrics_to_watch.extend(["VIX 상승 반전", "조정 진입 여부"])

    # 일반 매수
    elif max_under and buy_timing_score >= 6:
        action = "분할 매수"
        rationale.append(f"부족 자산군: {max_under['label']} ({max_under['gap_pct']:+.1f}%p)")
        rationale.append(f"매수 타이밍 양호 ({buy_timing_score}/10점)")
        metrics_to_watch.extend(["목표가 도달 여부", "추가 조정 대기"])

    # 기본 관망
    else:
        rationale.append("큰 이탈 없음, 시장 중립")
        rationale.append("정기 리밸런싱 일정 대기")
        metrics_to_watch.extend(["FOMC 일정", "지수 추세 변화"])

    # 7일 내 이벤트
    upcoming_events = [
        e for e in config["events"]
        if 0 <= (date.fromisoformat(e["date"]) - today).days <= 7
    ]

    recommended_action = {
        "action": action,
        "rationale": rationale,
        "metrics_to_watch": metrics_to_watch,
        "upcoming_events": [
            f"{e['date']} {e['title']}" for e in upcoming_events
        ] if upcoming_events else ["없음"]
    }

    return {
        "market_momentum": market_momentum,
        "phase_signals": phase_signals,
        "rebalancing_urgency": rebalancing_urgency,
        "recommended_action": recommended_action
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. HTML 섹션 빌더들
# ─────────────────────────────────────────────────────────────────────────────
def _chg_cls(chg):
    return "up" if chg > 0 else ("down" if chg < 0 else "neutral")

def _fmt_chg(chg):
    sign = "+" if chg >= 0 else ""
    return f"{sign}{chg:.2f}%"


def build_market(config: dict, price_data: dict) -> str:
    prices = price_data.get("prices", {})
    groups = [
        ("한국 지수",  config["market_indicators"]["indices"]),
        ("매크로",     config["market_indicators"]["macro"]),
        ("원자재",     config["market_indicators"]["commodities"]),
        ("크립토",     config["market_indicators"]["crypto"]),
    ]
    parts = []
    for group_name, items in groups:
        parts.append(f'<div class="ind-group"><div class="ind-group-label">{group_name}</div><div class="ind-row">')
        for item in items:
            t     = item["ticker"]
            info  = prices.get(t, {})
            raw   = info.get("price")
            chg   = info.get("change_pct", 0)
            label = item["label"]

            if raw is None:
                val_str, chg_str, cls = "—", "", "neutral"
            else:
                cls = _chg_cls(chg)
                chg_str = _fmt_chg(chg)
                if t == "KRW=X":      val_str = f"₩{raw:,.1f}"
                elif t in ("^KS11","^KQ11","^GSPC","^IXIC"): val_str = f"{raw:,.2f}"
                elif t == "^VIX":     val_str = f"{raw:.2f}"
                elif t == "^TNX":     val_str = f"{raw:.3f}%"
                elif t == "BTC-USD":  val_str = f"${raw:,.0f}"
                elif t in ("GC=F","HG=F"): val_str = f"${raw:,.2f}"
                else:                 val_str = f"${raw:,.2f}"

            parts.append(f"""
            <div class="ind-card">
              <div class="ind-label">{label}</div>
              <div class="ind-value">{val_str}</div>
              <div class="ind-chg {cls}">{chg_str}</div>
            </div>""")
        parts.append("</div></div>")
    return "\n".join(parts)


def build_allocation(portfolio: dict) -> str:
    parts = []
    for item in portfolio["allocation"]:
        color   = item["color"]
        cur     = item["current_pct"]
        tgt     = item["target_pct"]
        gap     = item["gap_pct"]
        status  = item["status"]
        bar_cur = min(max(cur, 0), 50) * 2   # 50% = 100px 기준
        bar_tgt = min(max(tgt, 0), 50) * 2
        gap_sign = "+" if gap >= 0 else ""
        badge_cls = {"over": "badge-over", "under": "badge-under", "ok": "badge-ok"}.get(status, "")
        badge_txt = {"over": "초과", "under": "부족", "ok": "정상"}.get(status, "")
        gap_krw_str = f"₩{abs(item['gap_krw']):,.0f}"

        parts.append(f"""
        <div class="alloc-row">
          <div class="alloc-meta">
            <span class="alloc-name">{item['label']}</span>
            <span class="alloc-pcts">
              <b>{cur:.1f}%</b> <span class="muted">/ 목표 {tgt}%</span>
              <span class="badge {badge_cls}">{badge_txt}</span>
            </span>
          </div>
          <div class="bar-wrap">
            <div class="bar-bg">
              <div class="bar-tgt" style="width:{bar_tgt}%;background:{color}"></div>
              <div class="bar-cur" style="width:{bar_cur}%;background:{color}"></div>
            </div>
            <span class="bar-gap {'red' if status=='over' else ('blue' if status=='under' else 'muted')}">
              {gap_sign}{gap:.1f}%p &nbsp; {gap_krw_str} {'초과' if gap>0 else '부족'}
            </span>
          </div>
        </div>""")
    return "\n".join(parts)


def build_accounts(portfolio: dict) -> str:
    parts = []
    for acc_id, acc in portfolio["accounts"].items():
        total_str = f"₩{acc['total_krw']:,.0f}"
        parts.append(f"""
        <div class="acc-block">
          <div class="acc-title">{acc['name']} <span class="muted">{total_str}</span></div>
          <table>
            <thead><tr>
              <th>종목</th><th class="tr">현재가</th>
              <th class="tr">등락</th><th class="tr">평가금액</th>
              <th class="tr">수익률</th>
            </tr></thead><tbody>""")

        for h in acc["holdings"]:
            chg     = h.get("change_pct", 0)
            cls     = _chg_cls(chg)
            chg_str = _fmt_chg(chg)
            price_display = h["price_disp"] if h["price_disp"] != "—" else "—"
            val_str = f"₩{h['value_krw']:,.0f}" if h["value_krw"] else "—"

            # 수익률 셀
            pnl_pct = h.get("pnl_pct")
            pnl_krw = h.get("pnl_krw")
            if pnl_pct is not None:
                pnl_cls  = _chg_cls(pnl_pct)
                sign     = "+" if pnl_pct >= 0 else ""
                krw_sign = "+" if pnl_krw >= 0 else ""
                pnl_str  = (f'<span class="{pnl_cls}">{sign}{pnl_pct:.1f}%</span>'
                            f'<br><span class="muted" style="font-size:10px">'
                            f'{krw_sign}₩{pnl_krw:,.0f}</span>')
            elif h.get("avg_price", 0) == 0:
                pnl_str = '<span class="muted" style="font-size:10px">평단가 미입력</span>'
            else:
                pnl_str = "—"

            parts.append(f"""
              <tr>
                <td>{h['name']} <span class="muted">({h['shares']}주)</span></td>
                <td class="tr">{price_display}</td>
                <td class="tr {cls}">{chg_str}</td>
                <td class="tr">{val_str}</td>
                <td class="tr">{pnl_str}</td>
              </tr>""")

        parts.append("</tbody></table></div>")

    # 현금/크립토 요약
    cash         = portfolio["cash_krw"]
    crypto       = portfolio["crypto_krw"]
    crypto_src   = portfolio.get("crypto_source", "config.json")
    parts.append(f"""
    <div class="cash-summary">
      <div class="cash-row"><span class="muted">현금 (증권+예금)</span><span>₩{cash:,.0f}</span></div>
      <div class="cash-row">
        <span class="muted">크립토 (바이낸스) <span style="font-size:10px">· {crypto_src}</span></span>
        <span>₩{crypto:,.0f}</span>
      </div>
    </div>""")
    return "\n".join(parts)


def build_events(config: dict) -> str:
    today  = date.today()
    events = sorted(config["events"], key=lambda e: e["date"])
    parts  = []
    for ev in events:
        ev_date   = date.fromisoformat(ev["date"])
        if ev_date < today:
            continue
        days = (ev_date - today).days
        type_color = {"portfolio": "#4ade80", "macro": "#60a5fa", "policy": "#fbbf24"}.get(ev["type"], "#94a3b8")
        urgency    = "ev-urgent" if days <= 30 else ("ev-soon" if days <= 60 else "")
        days_str   = f"D-{days}"
        parts.append(f"""
        <div class="ev-row {urgency}">
          <div class="ev-dot" style="background:{type_color}"></div>
          <div class="ev-body">
            <div class="ev-head">
              <span class="ev-title">{ev['title']}</span>
              <span class="ev-date muted">{ev['date']} <b>({days_str})</b></span>
            </div>
            <div class="ev-desc muted">{ev['description']}</div>
          </div>
        </div>""")
    return "\n".join(parts) if parts else "<p class='muted'>예정된 이벤트 없음</p>"


def build_rebalancing(portfolio: dict, config: dict) -> str:
    threshold   = config["rebalancing"]["threshold_pct"]
    next_review = config["rebalancing"]["next_review"]
    needs       = [x for x in portfolio["allocation"] if x["status"] != "ok"]
    parts = [f"""
    <div class="rebal-meta muted">
      기준: ±{threshold}%p 이탈 검토 &nbsp;|&nbsp; 다음 정기 리뷰: {next_review}
    </div>"""]

    if not needs:
        parts.append('<div class="rebal-ok">✅ 모든 자산군이 목표 비중 내에 있습니다.</div>')
    else:
        for item in sorted(needs, key=lambda x: abs(x["gap_pct"]), reverse=True):
            is_over = item["status"] == "over"
            badge_c = "badge-over" if is_over else "badge-under"
            action  = "매도 또는 비중 축소 검토" if is_over else "추가 매수 우선 고려"
            gap_c   = "red" if is_over else "blue"
            gap_sign = "+" if item["gap_pct"] >= 0 else ""
            parts.append(f"""
            <div class="rebal-row">
              <div>
                <div class="rebal-name">{item['label']}</div>
                <div class="rebal-action muted">{action}</div>
              </div>
              <div class="rebal-right">
                <span class="badge {badge_c}">{'초과' if is_over else '부족'}</span>
                <div class="{gap_c}" style="font-size:12px;margin-top:4px">
                  {item['current_pct']}% → {item['target_pct']}%
                  ({gap_sign}{item['gap_pct']}%p)
                </div>
              </div>
            </div>""")

    parts.append('<div class="rebal-rules">')
    for rule in config["rebalancing"]["rules"]:
        parts.append(f'<div class="rebal-rule muted">• {rule}</div>')
    parts.append("</div>")
    return "\n".join(parts)


def build_summary(portfolio: dict) -> str:
    """포트폴리오 수익성 요약 카드"""
    total     = portfolio["total_krw"]
    cost      = portfolio["total_cost_krw"]
    pnl_krw   = portfolio["total_pnl_krw"]
    pnl_pct   = portfolio["total_pnl_pct"]
    div_annual = portfolio["annual_div_krw"]
    usd_krw   = portfolio["usd_krw"]

    # 전체 수익률
    if pnl_pct is not None:
        sign = "+" if pnl_pct >= 0 else ""
        cls  = _chg_cls(pnl_pct)
        pnl_str = f'<span class="{cls}">{sign}{pnl_pct:.1f}%</span>'
        pnl_sub = f'<span class="{cls}">{sign}₩{pnl_krw:,.0f}</span>'
    else:
        pnl_str = '<span class="muted">—</span>'
        pnl_sub = '<span class="muted">매입금액 미입력</span>'

    # 배당 수익률 (총자산 대비)
    div_yield_total = div_annual / cost * 100 if cost > 0 else 0
    div_monthly = div_annual / 12

    parts = [f"""
    <div class="summary-grid">
      <div class="summary-item">
        <div class="summary-label">총 평가액</div>
        <div class="summary-value blue">₩{total:,.0f}</div>
        <div class="summary-sub muted">≈ ${total/usd_krw:,.0f}</div>
      </div>
      <div class="summary-item">
        <div class="summary-label">총 매입금액</div>
        <div class="summary-value">₩{cost:,.0f}</div>
        <div class="summary-sub muted">주식 계좌 합산</div>
      </div>
      <div class="summary-item">
        <div class="summary-label">총 수익률</div>
        <div class="summary-value">{pnl_str}</div>
        <div class="summary-sub">{pnl_sub}</div>
      </div>
      <div class="summary-item">
        <div class="summary-label">연간 배당 예측</div>
        <div class="summary-value green">₩{div_annual:,.0f}</div>
        <div class="summary-sub muted">월 ≈ ₩{div_monthly:,.0f} · {div_yield_total:.1f}%</div>
      </div>
    </div>"""]

    # 계좌별 수익률
    parts.append('<div class="summary-divider"></div>')
    for acc_id, acc in portfolio["accounts"].items():
        p_pct = acc.get("pnl_pct")
        p_krw = acc.get("pnl_krw")
        if p_pct is not None:
            sign  = "+" if p_pct >= 0 else ""
            cls   = _chg_cls(p_pct)
            right = f'<span class="{cls}">{sign}{p_pct:.1f}%</span> <span class="muted" style="font-size:11px">({sign}₩{p_krw:,.0f})</span>'
        else:
            right = '<span class="muted">—</span>'
        parts.append(f"""
    <div class="acc-pnl-row">
      <span>{acc['name']}</span>
      <span>₩{acc['total_krw']:,.0f} &nbsp; {right}</span>
    </div>""")

    return "\n".join(parts)


def build_risk(portfolio: dict, price_data: dict) -> str:
    """리스크 지표 카드"""
    vix       = portfolio["vix"]
    usd_exp   = portfolio["usd_exposure_pct"]
    top_label = portfolio["top_ac_label"]
    top_pct   = portfolio["top_ac_pct"]
    cash_pct  = portfolio["cash_pct"]

    # VIX 경보
    if vix >= 30:
        vix_cls = "risk-danger"
        vix_note = "위험 — 변동성 급등"
    elif vix >= 25:
        vix_cls = "risk-warn"
        vix_note = "주의 — 변동성 상승"
    else:
        vix_cls = "risk-ok"
        vix_note = "안정 구간"
    vix_str = f"{vix:.1f}" if vix else "—"

    # 현금 비율
    if cash_pct < 5:
        cash_cls  = "risk-danger"
        cash_note = "위험 — 유동성 부족"
    elif cash_pct < 10:
        cash_cls  = "risk-warn"
        cash_note = "주의 — 여유 확보 권장"
    else:
        cash_cls  = "risk-ok"
        cash_note = "양호"

    # USD 노출
    if usd_exp > 60:
        usd_cls  = "risk-warn"
        usd_note = "달러 집중 — 환율 주의"
    elif usd_exp > 40:
        usd_cls  = "risk-ok"
        usd_note = "적정 수준"
    else:
        usd_cls  = "risk-ok"
        usd_note = "낮은 달러 노출"

    # 집중도
    if top_pct > 25:
        conc_cls  = "risk-warn"
        conc_note = "집중 위험"
    else:
        conc_cls  = "risk-ok"
        conc_note = "분산 양호"

    return f"""
    <div class="risk-grid">
      <div class="risk-item">
        <div class="risk-label">VIX (시장 공포)</div>
        <div class="risk-value {vix_cls}">{vix_str}</div>
        <div class="risk-sub">{vix_note}</div>
      </div>
      <div class="risk-item">
        <div class="risk-label">현금 비율</div>
        <div class="risk-value {cash_cls}">{cash_pct:.1f}%</div>
        <div class="risk-sub">{cash_note}</div>
      </div>
      <div class="risk-item">
        <div class="risk-label">USD 노출 비중</div>
        <div class="risk-value {usd_cls}">{usd_exp:.1f}%</div>
        <div class="risk-sub">{usd_note}</div>
      </div>
      <div class="risk-item">
        <div class="risk-label">최대 자산군 비중</div>
        <div class="risk-value {conc_cls}">{top_pct:.1f}%</div>
        <div class="risk-sub">{top_label}</div>
      </div>
    </div>"""


def build_dividends(portfolio: dict) -> str:
    """배당 수익 예측 카드"""
    details   = portfolio["div_details"]
    annual    = portfolio["annual_div_krw"]
    monthly   = annual / 12

    if not details:
        return "<p class='muted'>배당 수익 데이터 없음 (dividend_yield 미입력)</p>"

    parts = [f'<div class="div-total green">연간 예측 배당: ₩{annual:,.0f} &nbsp;<span class="muted" style="font-size:12px">/ 월 ≈ ₩{monthly:,.0f}</span></div>']
    for d in sorted(details, key=lambda x: x["div_krw"], reverse=True):
        parts.append(f"""
    <div class="div-row">
      <span>{d['name']} <span class="muted">({d['yield_pct']}%)</span></span>
      <span class="green">₩{d['div_krw']:,.0f}/년</span>
    </div>""")
    return "\n".join(parts)


def build_watchlist(config: dict, price_data: dict) -> str:
    prices  = price_data.get("prices", {})
    wl      = config["portfolio"]["watchlist"]
    labels  = {
        "korea_electric": "전기/전력 인프라",
        "korea_low_pbr":  "저PBR/금융 (정책 수혜)",
        "korea_defense":  "방산",
        "commodities":    "원자재"
    }
    parts = []
    for key, items in wl.items():
        parts.append(f'<div class="wl-group"><div class="wl-label">{labels.get(key, key)}</div>')
        for item in items:
            t    = item["ticker"]
            info = prices.get(t, {})
            raw  = info.get("price")
            chg  = info.get("change_pct", 0)
            if raw:
                is_kr = t.endswith(".KS") or t.endswith(".KQ")
                val_str = f"₩{raw:,.0f}" if is_kr else f"${raw:,.2f}"
                chg_str = _fmt_chg(chg)
                cls = _chg_cls(chg)
            else:
                val_str, chg_str, cls = "—", "—", "neutral"
            parts.append(f"""
            <div class="wl-row">
              <div><span class="wl-name">{item['name']}</span>
                <span class="wl-note muted">{item.get('note','')}</span></div>
              <div class="wl-right">
                <span>{val_str}</span>
                <span class="{cls}" style="margin-left:8px">{chg_str}</span>
              </div>
            </div>""")
        parts.append("</div>")
    return "\n".join(parts)


def build_buy_plan(config: dict, price_data: dict) -> str:
    """매수 계획 카드"""
    buy_plan = config["portfolio"].get("buy_plan")
    if not buy_plan:
        return "<p class='muted'>매수 계획 없음</p>"

    prices = price_data.get("prices", {})
    parts = []

    for phase_id in ["phase1", "phase2", "phase3"]:
        phase = buy_plan.get(phase_id)
        if not phase:
            continue

        label = phase.get("label", phase_id)
        trigger = phase.get("trigger", "")
        items = phase.get("items", [])

        parts.append(f'<div class="buy-phase">')
        parts.append(f'<div class="buy-phase-label">{label}')
        if trigger:
            parts.append(f' <span class="muted" style="font-size:11px">— {trigger}</span>')
        parts.append('</div>')

        for item in items:
            ticker = item["ticker"]
            name = item["name"]
            target_shares = item["target_shares"]
            status = item.get("status", "pending")
            note = item.get("note", "")

            # 현재가 조회
            info = prices.get(ticker, {})
            price = info.get("price")
            chg = info.get("change_pct", 0)

            if price:
                is_kr = ticker.endswith(".KS") or ticker.endswith(".KQ")
                price_str = f"₩{price:,.0f}" if is_kr else f"${price:.2f}"
                amount = price * target_shares * (1 if is_kr else config["meta"]["usd_to_krw_fallback"])
                amount_str = f"₩{amount:,.0f}"
                chg_str = _fmt_chg(chg)
                chg_cls = _chg_cls(chg)
            else:
                price_str = "—"
                amount_str = "—"
                chg_str = "—"
                chg_cls = "neutral"

            status_badge = {
                "pending": '<span class="buy-status-pending">대기</span>',
                "waiting": '<span class="buy-status-waiting">조건대기</span>',
                "done": '<span class="buy-status-done">완료</span>',
            }.get(status, "")

            parts.append(f"""
            <div class="buy-row">
              <div class="buy-left">
                <span class="buy-name">{name}</span>
                <span class="muted" style="font-size:11px"> × {target_shares}주</span>
                {status_badge}
              </div>
              <div class="buy-right">
                <span>{price_str}</span>
                <span class="{chg_cls}" style="margin-left:6px">{chg_str}</span>
                <span class="muted" style="margin-left:8px">{amount_str}</span>
              </div>
            </div>""")

            if note:
                parts.append(f'<div class="buy-note muted">{note}</div>')

        parts.append('</div>')

    return "\n".join(parts)


def build_strategy_recommendations(config: dict, price_data: dict, portfolio: dict, signals: dict) -> str:
    """전략 추천 섹션 HTML 생성 (4개 컴포넌트)"""

    mm = signals["market_momentum"]
    ps = signals["phase_signals"]
    ru = signals["rebalancing_urgency"]
    ra = signals["recommended_action"]

    # 1. 시장 모멘텀 분석
    mm_score_cls = "green" if mm["score"] >= 2 else ("red" if mm["score"] <= -2 else "muted")

    momentum_html = f"""
    <div class="strategy-component">
      <div class="strategy-comp-title">1. 시장 모멘텀 분석</div>
      <div class="strategy-grid">
        <div class="strategy-metric">
          <span class="muted">한국 지수</span>
          <span class="{_chg_cls(mm['kospi_chg'])}">{mm['kr_index_trend']}</span>
          <span class="muted" style="font-size:10px">
            KOSPI {_fmt_chg(mm['kospi_chg'])} · KOSDAQ {_fmt_chg(mm['kosdaq_chg'])}
          </span>
        </div>
        <div class="strategy-metric">
          <span class="muted">VIX 공포지수</span>
          <span class="{_chg_cls(-mm['vix_value'] + 20)}">{mm['vix_level']}</span>
          <span class="muted" style="font-size:10px">{mm['vix_value']:.1f}</span>
        </div>
        <div class="strategy-metric">
          <span class="muted">외국인/기관</span>
          <span class="muted">{mm['foreign_institutional']}</span>
        </div>
        <div class="strategy-metric">
          <span class="muted">종합 판단</span>
          <span class="{mm_score_cls}" style="font-weight:700">{mm['interpretation']}</span>
          <span class="muted" style="font-size:10px">점수: {mm['score']:+d}/3</span>
        </div>
      </div>
    </div>"""

    # 2. Phase 2/3 실행 신호
    p2_cls = "green" if ps["phase2_ready"] else "muted"
    p3_cls = "green" if ps["phase3_ready"] else "muted"
    timing_cls = (
        "green" if ps["buy_timing_score"] >= 7 else
        "blue" if ps["buy_timing_score"] >= 5 else
        "muted"
    )

    phase_html = f"""
    <div class="strategy-component">
      <div class="strategy-comp-title">2. Phase 2/3 실행 신호</div>
      <div class="strategy-grid">
        <div class="strategy-metric">
          <span class="muted">1%+ 조정 발생</span>
          <span class="{p2_cls}" style="font-weight:700">
            {'✓ 발생' if ps['correction_detected'] else '✗ 미발생'}
          </span>
          <span class="muted" style="font-size:10px">
            {f"최대 {ps['correction_pct']:.1f}%" if ps['correction_detected'] else "Phase 2 대기"}
          </span>
        </div>
        <div class="strategy-metric">
          <span class="muted">FOMC 근접도</span>
          <span class="{p3_cls}" style="font-weight:700">
            {f"D-{ps['fomc_days_away']}" if ps['fomc_days_away'] else "예정 없음"}
          </span>
          <span class="muted" style="font-size:10px">
            {ps['nearest_fomc'] if ps['nearest_fomc'] != '없음' else 'Phase 3 대기'}
          </span>
        </div>
        <div class="strategy-metric">
          <span class="muted">매수 타이밍</span>
          <span class="{timing_cls}" style="font-weight:700">{ps['buy_timing_label']}</span>
          <span class="muted" style="font-size:10px">{ps['buy_timing_score']}/10점</span>
        </div>
        <div class="strategy-metric">
          <span class="muted">실행 단계</span>
          <span class="{'green' if ps['phase2_ready'] or ps['phase3_ready'] else 'muted'}" style="font-weight:700">
            {'Phase 2 실행' if ps['phase2_ready'] else ('Phase 3 준비' if ps['phase3_ready'] else 'Phase 1 대기')}
          </span>
        </div>
      </div>
    </div>"""

    # 3. 리밸런싱 긴급도
    urgency_cls = (
        "red" if ru["urgency_score"] >= 7 else
        "yellow" if ru["urgency_score"] >= 4 else
        "green"
    )

    priority_rows = []
    if ru["top_over"]:
        over = ru["top_over"]
        priority_rows.append(f"""
        <div class="rebal-priority-row">
          <span class="badge badge-over">초과</span>
          <span>{over['label']}</span>
          <span class="red">{over['gap_pct']:+.1f}%p (₩{abs(over['gap_krw']):,.0f})</span>
          <span class="muted" style="font-size:11px">{over['action']}</span>
        </div>""")

    if ru["top_under"]:
        under = ru["top_under"]
        priority_rows.append(f"""
        <div class="rebal-priority-row">
          <span class="badge badge-under">부족</span>
          <span>{under['label']}</span>
          <span class="blue">{under['gap_pct']:+.1f}%p (₩{abs(under['gap_krw']):,.0f})</span>
          <span class="muted" style="font-size:11px">{under['action']}</span>
        </div>""")

    if not priority_rows:
        priority_rows.append('<div class="muted" style="text-align:center;padding:10px">✅ 모든 자산군 정상 범위</div>')

    rebal_html = f"""
    <div class="strategy-component">
      <div class="strategy-comp-title">3. 리밸런싱 긴급도</div>
      <div class="strategy-urgency">
        <div class="urgency-badge">
          <span class="muted">긴급도</span>
          <span class="{urgency_cls}" style="font-size:18px;font-weight:700">{ru['urgency_level']}</span>
          <span class="muted" style="font-size:10px">{ru['urgency_score']}/10점 · 이탈 {ru['total_imbalanced']}개</span>
        </div>
        <div class="priority-actions">
          {''.join(priority_rows)}
        </div>
      </div>
    </div>"""

    # 4. 오늘의 액션 플랜
    action_cls = (
        "green" if "매수" in ra["action"] else
        ("red" if "매도" in ra["action"] else "blue")
    )

    action_html = f"""
    <div class="strategy-component strategy-action">
      <div class="strategy-comp-title">4. 오늘의 액션 플랜</div>
      <div class="action-box {action_cls}">
        <div class="action-main">{ra['action']}</div>
        <div class="action-rationale">
          {'<br>'.join(['• ' + r for r in ra['rationale']])}
        </div>
      </div>
      <div class="action-metrics">
        <div class="action-section">
          <div class="action-section-title">주목할 지표</div>
          {'<br>'.join(['• ' + m for m in ra['metrics_to_watch']])}
        </div>
        <div class="action-section">
          <div class="action-section-title">7일 내 이벤트</div>
          {'<br>'.join(['• ' + e for e in ra['upcoming_events']])}
        </div>
      </div>
    </div>"""

    return momentum_html + phase_html + rebal_html + action_html


# ─────────────────────────────────────────────────────────────────────────────
# 5. HTML 조합
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
:root{--bg:#0f1117;--card:#1a1d27;--border:#2a2d3a;--text:#e2e8f0;
      --muted:#94a3b8;--green:#4ade80;--red:#f87171;--yellow:#fbbf24;
      --blue:#60a5fa;--font:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--font);font-size:14px;line-height:1.6}
.wrap{max-width:1400px;margin:0 auto;padding:20px}
.header{display:flex;justify-content:space-between;align-items:center;
        margin-bottom:24px;padding-bottom:16px;border-bottom:1px solid var(--border)}
.header h1{font-size:20px;font-weight:700;color:var(--blue)}
.header .sub{font-size:12px;color:var(--muted);margin-top:2px}
.total-val{font-size:28px;font-weight:800;color:var(--blue);text-align:right}
.total-usd{font-size:13px;color:var(--muted);text-align:right}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px}
.full{margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px}
.card-title{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;
            letter-spacing:.8px;margin-bottom:16px}
.muted{color:var(--muted)}
.up{color:var(--green)}.down{color:var(--red)}.neutral{color:var(--muted)}
.red{color:var(--red)}.blue{color:var(--blue)}

/* 시장 지표 */
.ind-group{margin-bottom:16px}
.ind-group-label{font-size:11px;color:var(--muted);font-weight:600;margin-bottom:8px}
.ind-row{display:flex;flex-wrap:wrap;gap:8px}
.ind-card{background:var(--bg);border-radius:8px;padding:10px 14px;min-width:120px}
.ind-label{font-size:11px;color:var(--muted);margin-bottom:2px}
.ind-value{font-size:16px;font-weight:700}
.ind-chg{font-size:12px;margin-top:1px}

/* 자산 배분 */
.alloc-row{margin-bottom:14px}
.alloc-meta{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;font-size:12px}
.alloc-name{font-weight:600}
.alloc-pcts{display:flex;align-items:center;gap:6px}
.bar-wrap{display:flex;align-items:center;gap:10px}
.bar-bg{position:relative;height:8px;background:var(--border);border-radius:4px;flex:1;min-width:60px}
.bar-tgt{position:absolute;height:100%;border-radius:4px;opacity:.25;top:0;left:0}
.bar-cur{position:absolute;height:100%;border-radius:4px;top:0;left:0}
.bar-gap{font-size:11px;white-space:nowrap}

/* 배지 */
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:10px;font-weight:700}
.badge-over{background:rgba(248,113,113,.2);color:var(--red)}
.badge-under{background:rgba(96,165,250,.2);color:var(--blue)}
.badge-ok{background:rgba(74,222,128,.2);color:var(--green)}

/* 계좌 */
.acc-block{margin-bottom:18px}
.acc-title{font-size:12px;font-weight:700;color:var(--muted);margin-bottom:8px}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;padding:6px 8px;font-size:10px;color:var(--muted);
   font-weight:600;border-bottom:1px solid var(--border)}
td{padding:6px 8px;border-bottom:1px solid var(--border)}
tr:last-child td{border-bottom:none}
.tr{text-align:right}
.cash-summary{border-top:1px solid var(--border);padding-top:10px;margin-top:4px}
.cash-row{display:flex;justify-content:space-between;font-size:12px;padding:3px 0}

/* 이벤트 */
.ev-row{display:flex;gap:10px;padding:10px 0;border-bottom:1px solid var(--border)}
.ev-row:last-child{border-bottom:none}
.ev-dot{width:8px;height:8px;border-radius:50%;margin-top:6px;flex-shrink:0}
.ev-body{flex:1}
.ev-head{display:flex;justify-content:space-between;align-items:baseline}
.ev-title{font-size:13px;font-weight:600}
.ev-date{font-size:11px}
.ev-desc{font-size:12px;margin-top:2px}
.ev-urgent .ev-title{color:var(--red)}
.ev-soon .ev-title{color:var(--yellow)}

/* 리밸런싱 */
.rebal-meta{font-size:12px;margin-bottom:12px}
.rebal-ok{color:var(--green);padding:8px 0;font-size:13px}
.rebal-row{display:flex;justify-content:space-between;align-items:center;
           padding:10px 0;border-bottom:1px solid var(--border)}
.rebal-row:last-of-type{border-bottom:none}
.rebal-name{font-size:13px;font-weight:600}
.rebal-action{font-size:11px;margin-top:2px}
.rebal-right{text-align:right}
.rebal-rules{margin-top:14px;padding-top:12px;border-top:1px solid var(--border)}
.rebal-rule{font-size:12px;padding:2px 0}

/* 관심종목 */
.wl-group{margin-bottom:18px}
.wl-label{font-size:11px;font-weight:700;color:var(--muted);margin-bottom:8px}
.wl-row{display:flex;justify-content:space-between;align-items:center;
        padding:7px 0;border-bottom:1px solid var(--border);font-size:12px}
.wl-row:last-child{border-bottom:none}
.wl-name{font-weight:600;margin-right:6px}
.wl-note{font-size:11px}
.wl-right{text-align:right;white-space:nowrap}

/* 데이터 출처 */
.data-source{font-size:10px;color:var(--muted);margin-top:12px;padding-top:8px;
             border-top:1px solid var(--border);text-align:right;letter-spacing:.2px}

/* 수익성 요약 */
.summary-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:4px}
.summary-item{background:var(--bg);border-radius:8px;padding:10px 14px}
.summary-label{font-size:11px;color:var(--muted);margin-bottom:3px}
.summary-value{font-size:18px;font-weight:700}
.summary-sub{font-size:11px;margin-top:2px}
.summary-divider{border-top:1px solid var(--border);margin:12px 0}
.acc-pnl-row{display:flex;justify-content:space-between;align-items:center;
             padding:5px 0;border-bottom:1px solid var(--border);font-size:12px}
.acc-pnl-row:last-child{border-bottom:none}

/* 리스크 지표 */
.risk-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:4px}
.risk-item{background:var(--bg);border-radius:8px;padding:10px 14px}
.risk-label{font-size:11px;color:var(--muted);margin-bottom:3px}
.risk-value{font-size:16px;font-weight:700}
.risk-sub{font-size:11px;color:var(--muted);margin-top:2px}
.risk-warn{color:var(--yellow)}
.risk-danger{color:var(--red)}
.risk-ok{color:var(--green)}

/* 배당 예측 */
.div-total{font-size:14px;font-weight:700;margin-bottom:10px}
.div-row{display:flex;justify-content:space-between;align-items:center;
         padding:5px 0;border-bottom:1px solid var(--border);font-size:12px}
.div-row:last-child{border-bottom:none}

/* 매수 계획 */
.buy-phase{margin-bottom:16px;padding-bottom:16px;border-bottom:1px solid var(--border)}
.buy-phase:last-child{border-bottom:none}
.buy-phase-label{font-size:13px;font-weight:700;margin-bottom:10px;color:var(--blue)}
.buy-row{display:flex;justify-content:space-between;align-items:center;
         padding:6px 0;border-bottom:1px solid var(--border);font-size:12px}
.buy-row:last-child{border-bottom:none}
.buy-left{display:flex;align-items:center;gap:8px}
.buy-name{font-weight:600}
.buy-right{display:flex;align-items:center;gap:4px;text-align:right}
.buy-note{font-size:11px;padding-left:8px;margin-top:2px;color:var(--muted)}
.buy-status-pending{display:inline-block;padding:2px 8px;border-radius:10px;
                    font-size:10px;font-weight:700;background:rgba(251,191,36,.2);color:var(--yellow)}
.buy-status-waiting{display:inline-block;padding:2px 8px;border-radius:10px;
                    font-size:10px;font-weight:700;background:rgba(148,163,184,.2);color:var(--muted)}
.buy-status-done{display:inline-block;padding:2px 8px;border-radius:10px;
                 font-size:10px;font-weight:700;background:rgba(74,222,128,.2);color:var(--green)}

/* 전략 추천 */
.strategy-component{margin-bottom:24px;padding-bottom:24px;
  border-bottom:1px solid var(--border)}
.strategy-component:last-child{border-bottom:none}
.strategy-comp-title{font-size:13px;font-weight:700;color:var(--blue);
  margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid var(--border)}
.strategy-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.strategy-metric{background:var(--bg);border-radius:8px;padding:10px 12px;
  display:flex;flex-direction:column;gap:4px}
.strategy-metric>span:nth-child(2){font-size:15px;font-weight:700}

.strategy-urgency{display:grid;grid-template-columns:200px 1fr;gap:16px}
.urgency-badge{background:var(--bg);border-radius:8px;padding:16px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px}
.priority-actions{display:flex;flex-direction:column;gap:8px}
.rebal-priority-row{display:grid;grid-template-columns:60px 1fr 150px 1fr;
  gap:10px;align-items:center;padding:10px 12px;background:var(--bg);
  border-radius:8px;font-size:12px}

.strategy-action .action-box{padding:16px;border-radius:10px;margin-bottom:16px}
.strategy-action .action-box.green{background:rgba(74,222,128,.1);border:1px solid var(--green)}
.strategy-action .action-box.red{background:rgba(248,113,113,.1);border:1px solid var(--red)}
.strategy-action .action-box.blue{background:rgba(96,165,250,.1);border:1px solid var(--blue)}
.action-main{font-size:18px;font-weight:700;margin-bottom:10px}
.action-rationale{font-size:13px;line-height:1.8;color:var(--muted)}
.action-metrics{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.action-section{background:var(--bg);border-radius:8px;padding:12px;font-size:12px;
  line-height:1.7}
.action-section-title{font-weight:700;margin-bottom:8px;color:var(--blue)}

/* Refresh Button */
.header-right{display:flex;align-items:center;gap:20px}
.refresh-btn{display:flex;align-items:center;gap:6px;padding:8px 16px;
  background:var(--card);border:1px solid var(--border);border-radius:8px;
  color:var(--text);font-size:12px;font-weight:600;cursor:pointer;
  transition:all .2s;font-family:var(--font)}
.refresh-btn:hover{background:var(--border);border-color:var(--blue)}
.refresh-btn:disabled{opacity:.5;cursor:not-allowed}
.refresh-btn svg{transition:transform .5s}
.refresh-btn.spinning svg{animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

@media(max-width:900px){
  .grid2,.grid3{grid-template-columns:1fr}
  .strategy-grid{grid-template-columns:1fr 1fr}
  .strategy-urgency{grid-template-columns:1fr}
  .action-metrics{grid-template-columns:1fr}
}

@media(max-width:480px){
  .wrap{padding:10px 8px}
  .card{padding:14px 12px;border-radius:10px}
  .header{flex-direction:column;align-items:flex-start;gap:6px}
  .header-right{flex-direction:column-reverse;align-items:flex-start;gap:10px}
  .refresh-btn{padding:6px 12px;font-size:11px}
  .total-val{font-size:22px;text-align:left}
  .total-usd{text-align:left}
  .header .sub{font-size:11px}
  .ind-card{min-width:90px;padding:8px 10px}
  .ind-value{font-size:14px}
  .bar-wrap{gap:6px}
  .bar-gap{font-size:10px}
  .alloc-meta{flex-wrap:wrap;gap:2px}
  .acc-block>table{display:block;overflow-x:auto;-webkit-overflow-scrolling:touch;white-space:nowrap}
  .ev-desc{font-size:11px}
  .rebal-rules{display:none}
  .summary-grid{grid-template-columns:1fr 1fr}
  .risk-grid{grid-template-columns:1fr 1fr}
  .summary-value{font-size:15px}
  .risk-value{font-size:14px}
  .strategy-grid{grid-template-columns:1fr}
  .rebal-priority-row{grid-template-columns:1fr;gap:6px}
  .action-main{font-size:16px}
}
"""


def generate_html(config: dict, price_data: dict, portfolio: dict, strategy_signals: dict) -> str:
    now      = datetime.now().strftime("%Y-%m-%d %H:%M")
    total    = portfolio["total_krw"]
    usd_krw  = portfolio["usd_krw"]
    errors   = price_data.get("errors", [])
    err_html = ""
    if errors:
        err_html = f'<div style="color:var(--yellow);font-size:12px;margin-bottom:12px">⚠ 조회 실패: {", ".join(errors[:5])}</div>'

    # 출처 메타
    yf_time      = price_data.get("fetched_at", "")[:16].replace("T", " ")
    binance_data = price_data.get("binance")
    binance_src  = (f"Binance API · {binance_data['fetched_at'][11:16]}"
                    if binance_data else "config.json (수동, API 미연결)")
    crypto_src   = portfolio.get("crypto_source", "config.json")

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>투자 대시보드 {now}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

  <!-- 헤더 -->
  <div class="header">
    <div>
      <h1>📊 투자 대시보드</h1>
      <div class="sub">업데이트: {now} &nbsp;|&nbsp; 환율: ₩{usd_krw:,.0f}/USD</div>
    </div>
    <div class="header-right">
      <button id="refreshBtn" class="refresh-btn" onclick="triggerRefresh()">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/>
        </svg>
        <span id="refreshText">새로고침</span>
      </button>
      <div>
        <div class="total-val">₩{total:,.0f}</div>
        <div class="total-usd">≈ ${total/usd_krw:,.0f} USD</div>
      </div>
    </div>
  </div>

  {err_html}

  <!-- 수익성 요약 -->
  <div class="full">
    <div class="card">
      <div class="card-title">포트폴리오 수익성 요약</div>
      {build_summary(portfolio)}
    </div>
  </div>

  <!-- 리스크 지표 -->
  <div class="full">
    <div class="card">
      <div class="card-title">리스크 지표</div>
      {build_risk(portfolio, price_data)}
    </div>
  </div>

  <!-- 전략 추천 -->
  <div class="full">
    <div class="card">
      <div class="card-title">💡 전략 추천 (Strategy Recommendations)</div>
      {build_strategy_recommendations(config, price_data, portfolio, strategy_signals)}
    </div>
  </div>

  <!-- 시장 지표 -->
  <div class="full">
    <div class="card">
      <div class="card-title">핵심 시장 지표</div>
      {build_market(config, price_data)}
      <div class="data-source">출처: Yahoo Finance &nbsp;|&nbsp; 조회: {yf_time}</div>
    </div>
  </div>

  <!-- 자산 배분 + 계좌 현황 -->
  <div class="grid2">
    <div class="card">
      <div class="card-title">자산군별 배분 현황</div>
      {build_allocation(portfolio)}
      <div class="data-source">종목가: Yahoo Finance · 크립토: {crypto_src}</div>
    </div>
    <div class="card">
      <div class="card-title">계좌별 보유 현황</div>
      {build_accounts(portfolio)}
      <div class="data-source">주식: Yahoo Finance &nbsp;|&nbsp; 크립토: {binance_src}</div>
    </div>
  </div>

  <!-- 리밸런싱 + 이벤트 -->
  <div class="grid2">
    <div class="card">
      <div class="card-title">리밸런싱 가이드</div>
      {build_rebalancing(portfolio, config)}
    </div>
    <div class="card">
      <div class="card-title">이벤트 캘린더</div>
      {build_events(config)}
    </div>
  </div>

  <!-- 배당 예측 + 관심종목 -->
  <div class="grid2">
    <div class="card">
      <div class="card-title">배당 수익 예측 (연간)</div>
      {build_dividends(portfolio)}
    </div>
    <div class="card">
      <div class="card-title">관심 종목 (미편입 watchlist)</div>
      {build_watchlist(config, price_data)}
      <div class="data-source">출처: Yahoo Finance &nbsp;|&nbsp; 조회: {yf_time}</div>
    </div>
  </div>

  <!-- 매수 계획 -->
  <div class="full">
    <div class="card">
      <div class="card-title">📋 매수 계획 (3단계 분할 매수)</div>
      {build_buy_plan(config, price_data)}
    </div>
  </div>

</div>

<script>
async function triggerRefresh() {{
  const btn = document.getElementById('refreshBtn');
  const text = document.getElementById('refreshText');

  btn.disabled = true;
  btn.classList.add('spinning');
  text.textContent = '업데이트 중...';

  try {{
    const response = await fetch('https://YOUR_VERCEL_URL/api/trigger-refresh', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }}
    }});

    const data = await response.json();

    if (data.success) {{
      text.textContent = '✓ 시작됨';
      setTimeout(() => {{
        text.textContent = '새로고침';
        btn.disabled = false;
        btn.classList.remove('spinning');
      }}, 3000);
    }} else {{
      throw new Error('Failed to trigger workflow');
    }}
  }} catch (error) {{
    text.textContent = '✗ 실패';
    btn.classList.remove('spinning');
    setTimeout(() => {{
      text.textContent = '새로고침';
      btn.disabled = false;
    }}, 3000);
  }}
}}
</script>

</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# 6. 스냅샷 저장 + 메인
# ─────────────────────────────────────────────────────────────────────────────
def save_snapshot(price_data: dict, portfolio: dict) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    snap  = {
        "date": today,
        "total_krw": portfolio["total_krw"],
        "usd_krw": portfolio["usd_krw"],
        "allocation": portfolio["allocation"],
        "fetched_at": price_data.get("fetched_at")
    }
    path = DATA_DIR / f"{today}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)
    return path


def main():
    no_save = "--no-save" in sys.argv

    print("📂 설정 로드 중...")
    config = load_config()

    print("📡 시장 데이터 조회 중 (yfinance)...")
    price_data = fetch_all_prices(config)
    if price_data["errors"]:
        print(f"  ⚠ {len(price_data['errors'])}개 조회 실패: {price_data['errors'][:3]}")

    # Binance API 잔고 조회
    if HAS_BINANCE:
        btc_price = price_data["prices"].get("BTC-USD", {}).get("price", 0) or 0
        try:
            binance_live = _binance_summary(btc_price)
            price_data["binance"] = binance_live
            d = binance_live["detail"]
            print(f"  ✅ 바이낸스 연동: BTC {d['btc_qty']:.6f} (${binance_live['btc_usd_value']:,.0f})"
                  f" + 스테이블 ${binance_live['usdc_usd']+binance_live['usdt_usd']:,.0f}")
        except Exception as e:
            price_data["binance"] = None
            print(f"  ⚠ 바이낸스 API 실패: {e}")
            print(f"     config.json 기록 값으로 대체")
    else:
        env_path = INTEGRATIONS_DIR / ".env"
        if env_path.exists():
            print("  ℹ 바이낸스: integrations/.env 있음 (모듈 로드 실패)")
        else:
            print("  ℹ 바이낸스: API 미연결 (integrations/.env 설정 필요)")

    print("🔢 포트폴리오 계산 중...")
    portfolio = calculate_portfolio(config, price_data)

    print("🧠 전략 신호 분석 중...")
    strategy_signals = calculate_strategy_signals(config, price_data, portfolio)

    print("🎨 HTML 생성 중...")
    html = generate_html(config, price_data, portfolio, strategy_signals)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✅ 대시보드 생성: {OUTPUT_PATH}")

    if not no_save:
        snap = save_snapshot(price_data, portfolio)
        print(f"  ✅ 스냅샷 저장: {snap}")

    total   = portfolio["total_krw"]
    usd_krw = portfolio["usd_krw"]
    print(f"\n{'─'*50}")
    print(f"💰 총 자산: ₩{total:,.0f}  (${total/usd_krw:,.0f})")
    print(f"💱 환율:    ₩{usd_krw:,.0f}/USD")
    print(f"📊 오늘의 추천: {strategy_signals['recommended_action']['action']}")
    print(f"{'─'*50}")

    needs = [x for x in portfolio["allocation"] if x["status"] != "ok"]
    if needs:
        print("⚠ 리밸런싱 필요:")
        for item in sorted(needs, key=lambda x: abs(x["gap_pct"]), reverse=True):
            sign = "+" if item["gap_pct"] >= 0 else ""
            act  = "▼ 축소" if item["status"] == "over" else "▲ 매수"
            print(f"  {act} {item['label']:20s} {item['current_pct']}% → {item['target_pct']}% ({sign}{item['gap_pct']}%p)")
    else:
        print("✅ 모든 자산군 목표 비중 내")

    # D-30 이내 이벤트
    today  = date.today()
    urgent = [e for e in config["events"] if (date.fromisoformat(e["date"]) - today).days <= 30
              and date.fromisoformat(e["date"]) >= today]
    if urgent:
        print("\n🔔 30일 내 이벤트:")
        for e in sorted(urgent, key=lambda x: x["date"]):
            days = (date.fromisoformat(e["date"]) - today).days
            print(f"  D-{days:2d} {e['date']}  {e['title']}")


if __name__ == "__main__":
    main()
