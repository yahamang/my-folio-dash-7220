#!/usr/bin/env python3
"""
시장 뉴스 모니터링 모듈 (수동 큐레이션 방식)
Claude가 WebSearch로 뉴스를 수집하고 캐시에 저장하면,
Dashboard는 캐시에서 읽어서 표시
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

BASE_DIR = Path(__file__).parent.parent
CACHE_FILE = BASE_DIR / "data" / "news_cache.json"

# 카테고리별 한글 라벨
CATEGORY_LABELS = {
    "geopolitical": "지정학",
    "economic_us": "미국경제",
    "economic_korea": "한국경제",
    "central_banks": "중앙은행",
    "sector_electric": "전력",
    "sector_defense": "방산",
    "sector_financial": "금융"
}

# 카테고리별 색상
CATEGORY_COLORS = {
    "geopolitical": "#ef4444",      # red
    "economic_us": "#3b82f6",       # blue
    "economic_korea": "#3b82f6",    # blue
    "central_banks": "#eab308",     # yellow
    "sector_electric": "#22c55e",   # green
    "sector_defense": "#22c55e",    # green
    "sector_financial": "#22c55e"   # green
}


def load_news_cache() -> Optional[Dict]:
    """
    뉴스 캐시 파일 읽기

    Returns:
        {"headlines": [...], "fetched_at": "...", "curated_by": "Claude"} 또는 None
    """
    try:
        if not CACHE_FILE.exists():
            return None

        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 유효성 검사
        if not isinstance(data, dict) or "headlines" not in data:
            return None

        return data
    except Exception as e:
        print(f"  ⚠️  뉴스 캐시 읽기 실패: {e}")
        return None


def get_cache_age_hours() -> Optional[float]:
    """
    캐시 나이 계산 (시간 단위)

    Returns:
        캐시가 생성된 후 경과 시간 (hours) 또는 None
    """
    cache = load_news_cache()
    if not cache or "fetched_at" not in cache:
        return None

    try:
        fetched_at = datetime.fromisoformat(cache["fetched_at"].replace("Z", ""))
        age = (datetime.now() - fetched_at).total_seconds() / 3600
        return age
    except:
        return None


def format_cache_timestamp(cache_data: Dict) -> str:
    """
    캐시 타임스탬프를 한국어로 포맷

    Args:
        cache_data: 캐시 데이터 딕셔너리

    Returns:
        "3시간 전 업데이트" 형식의 문자열
    """
    if not cache_data or "fetched_at" not in cache_data:
        return "업데이트 정보 없음"

    try:
        fetched_at = datetime.fromisoformat(cache_data["fetched_at"].replace("Z", ""))
        age_hours = (datetime.now() - fetched_at).total_seconds() / 3600

        if age_hours < 1:
            minutes = int(age_hours * 60)
            return f"{minutes}분 전 업데이트"
        elif age_hours < 24:
            hours = int(age_hours)
            return f"{hours}시간 전 업데이트"
        else:
            days = int(age_hours / 24)
            return f"{days}일 전 업데이트"
    except:
        return fetched_at[:10]  # YYYY-MM-DD만 표시
