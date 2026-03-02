#!/usr/bin/env python3
"""
RSS 뉴스 자동 수집 모듈 (Hybrid 방식)
- 매일 자동: RSS 피드에서 뉴스 수집 및 스코어링
- 수동 우선: Claude WebSearch가 캐시를 덮어쓸 수 있음
"""

import json
import feedparser
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import re

# 경로 설정
BASE_DIR = Path(__file__).parent.parent
CONFIG_FILE = BASE_DIR / "integrations" / "rss_config.json"
CACHE_FILE = BASE_DIR / "data" / "news_cache.json"

# 카테고리별 점수 가중치
CATEGORY_WEIGHTS = {
    "geopolitical": 1.0,
    "economic_us": 1.0,
    "economic_korea": 1.0,
    "central_banks": 1.0,
    "sector_electric": 0.9,
    "sector_defense": 0.9,
    "sector_financial": 0.9
}


def load_rss_config() -> Dict:
    """RSS 피드 설정 파일 로드"""
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠️  RSS 설정 파일 읽기 실패: {e}")
        return {"feeds": [], "keywords": []}


def parse_date(date_str: str) -> Optional[datetime]:
    """RSS 날짜 문자열을 datetime 객체로 변환"""
    if not date_str:
        return None

    try:
        # feedparser는 parsed 형식으로 제공
        return datetime(*date_str[:6]) if hasattr(date_str, '__iter__') else None
    except:
        try:
            # 일반 문자열 파싱 시도
            from dateutil import parser
            return parser.parse(date_str)
        except:
            return None


def fetch_all_feeds(config: Dict) -> List[Dict]:
    """모든 RSS 피드 수집"""
    all_entries = []
    feeds = config.get("feeds", [])

    print(f"  📡 RSS 피드 수집 시작 ({len(feeds)}개 소스)")

    for feed_info in feeds:
        url = feed_info["url"]
        category = feed_info["category"]
        source_name = feed_info["name"]
        weight = feed_info.get("weight", 1.0)

        try:
            print(f"    - {source_name} ({category})...", end=" ")
            feed = feedparser.parse(url)

            if feed.bozo:
                print(f"⚠️  파싱 오류")
                continue

            # 엔트리 추출
            for entry in feed.entries[:10]:  # 최근 10개만
                title = entry.get("title", "").strip()
                link = entry.get("link", "")
                summary = entry.get("summary", entry.get("description", ""))

                # 날짜 파싱
                published = None
                if hasattr(entry, "published_parsed"):
                    published = parse_date(entry.published_parsed)
                elif hasattr(entry, "updated_parsed"):
                    published = parse_date(entry.updated_parsed)

                if not published:
                    published = datetime.now()

                all_entries.append({
                    "title": title,
                    "link": link,
                    "summary": summary[:200] if summary else "",
                    "category": category,
                    "source": source_name,
                    "source_weight": weight,
                    "published": published
                })

            print(f"✅ {len(feed.entries)} 건")

        except Exception as e:
            print(f"❌ 실패: {e}")
            continue

    print(f"  ✅ 총 {len(all_entries)}개 엔트리 수집 완료\n")
    return all_entries


def calculate_recency_score(published: datetime) -> int:
    """최신성 점수 계산 (0-20점)"""
    age_hours = (datetime.now() - published).total_seconds() / 3600

    if age_hours < 24:
        return 20
    elif age_hours < 48:
        return 15
    elif age_hours < 168:  # 7 days
        return 10
    else:
        return 5


def calculate_keyword_score(title: str, summary: str, keywords: List[str]) -> int:
    """키워드 매칭 점수 (0-10점)"""
    text = (title + " " + summary).lower()

    for keyword in keywords:
        if keyword.lower() in text:
            return 10

    return 0


def score_entry(entry: Dict, keywords: List[str]) -> int:
    """RSS 엔트리 점수 계산 (0-100점)

    - 카테고리 가중치: 40점
    - 출처 신뢰도: 30점
    - 최신성: 20점
    - 키워드 매칭: 10점
    """
    score = 0

    # 1. 카테고리 가중치 (40점)
    category = entry["category"]
    category_weight = CATEGORY_WEIGHTS.get(category, 0.5)
    score += int(40 * category_weight)

    # 2. 출처 신뢰도 (30점)
    source_weight = entry["source_weight"]
    score += int(30 * source_weight)

    # 3. 최신성 (20점)
    score += calculate_recency_score(entry["published"])

    # 4. 키워드 매칭 (10점)
    score += calculate_keyword_score(entry["title"], entry["summary"], keywords)

    return min(score, 100)


def calculate_similarity(title1: str, title2: str) -> float:
    """두 제목의 유사도 계산 (0.0-1.0)"""
    # 간단한 단어 기반 유사도
    words1 = set(re.findall(r'\w+', title1.lower()))
    words2 = set(re.findall(r'\w+', title2.lower()))

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0


def deduplicate_entries(entries: List[Dict]) -> List[Dict]:
    """중복 제거 (제목 유사도 기반)"""
    unique = []

    for entry in entries:
        is_duplicate = False

        for existing in unique:
            similarity = calculate_similarity(entry["title"], existing["title"])

            # 유사도 70% 이상이면 중복으로 간주
            if similarity > 0.7:
                is_duplicate = True
                # 점수가 더 높은 것을 유지
                if entry.get("score", 0) > existing.get("score", 0):
                    unique.remove(existing)
                    unique.append(entry)
                break

        if not is_duplicate:
            unique.append(entry)

    return unique


def filter_top_headlines(entries: List[Dict], max_count: int = 5) -> List[Dict]:
    """상위 헤드라인 선택 (카테고리 균형 유지)"""
    # 점수 순으로 정렬
    sorted_entries = sorted(entries, key=lambda x: x.get("score", 0), reverse=True)

    selected = []
    category_count = {}

    for entry in sorted_entries:
        category = entry["category"]

        # 같은 카테고리 최대 2개까지
        if category_count.get(category, 0) >= 2:
            continue

        selected.append(entry)
        category_count[category] = category_count.get(category, 0) + 1

        if len(selected) >= max_count:
            break

    return selected


def format_headline(entry: Dict) -> Dict:
    """헤드라인 포맷 (dashboard.py 호환)"""
    return {
        "title": entry["title"],
        "url": entry["link"],
        "category": entry["category"],
        "source": entry["source"],
        "score": entry.get("score", 0)
    }


def save_to_cache(headlines: List[Dict]):
    """뉴스 캐시 저장"""
    CACHE_FILE.parent.mkdir(exist_ok=True)

    cache_data = {
        "headlines": headlines,
        "fetched_at": datetime.now().isoformat(),
        "curated_by": "RSS"
    }

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)

    print(f"  💾 캐시 저장: {CACHE_FILE}")


def main():
    """메인 실행 (GitHub Actions용)"""
    print("=" * 60)
    print("📰 RSS 뉴스 자동 수집 시작")
    print("=" * 60)

    # 1. 설정 로드
    config = load_rss_config()
    keywords = config.get("keywords", [])

    if not config.get("feeds"):
        print("  ⚠️  RSS 피드 설정이 없습니다.")
        return

    # 2. RSS 피드 수집
    entries = fetch_all_feeds(config)

    if not entries:
        print("  ⚠️  수집된 엔트리가 없습니다.")
        return

    # 3. 점수 계산
    print(f"  🎯 엔트리 점수 계산 중...")
    for entry in entries:
        entry["score"] = score_entry(entry, keywords)

    # 4. 중복 제거
    print(f"  🔍 중복 제거 중...")
    unique_entries = deduplicate_entries(entries)
    print(f"  ✅ 중복 제거 후: {len(unique_entries)}개")

    # 5. 상위 헤드라인 선택
    print(f"  📌 상위 헤드라인 선택 중...")
    top_headlines = filter_top_headlines(unique_entries, max_count=5)

    # 6. 포맷 변환
    formatted = [format_headline(h) for h in top_headlines]

    # 7. 캐시 저장
    save_to_cache(formatted)

    # 8. 결과 출력
    print("\n" + "=" * 60)
    print(f"✅ RSS 뉴스 업데이트 완료: {len(formatted)}개 헤드라인")
    print("=" * 60)

    for i, headline in enumerate(formatted, 1):
        print(f"{i}. [{headline['category']}] {headline['title'][:60]}...")
        print(f"   출처: {headline['source']} | 점수: {headline['score']}")

    print("\n")


if __name__ == "__main__":
    main()
