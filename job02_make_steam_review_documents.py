# job02_make_steam_review_documents_user_rule_no_mean_no_votes_penalty.py
# 원본 기준: job02_make_steam_review_documents_user_rule_no_mean.py
# 수정 내용: votes_up은 통계로만 남기고 review_quality_multiplier 감점 계산에서는 제외
# Steam 게임 리뷰 기반 추천 시스템 - job02
#
# 목적:
#   job01에서 만든 두 CSV 파일을 사용해서
#   "게임 1개 = 추천 모델용 문서 1개" 형태의 CSV를 만든다.
#
# 입력:
#   ./datasets/steam_reviews_raw_v2.csv
#   ./datasets/steam_games_detail_v2.csv
#
# 출력:
#   ./datasets/steam_game_review_documents.csv
#
# ------------------------------------------------------------
# 이번 버전의 핵심 수정
# ------------------------------------------------------------
#
# 이전 코드에서는 avg_playtime_at_review, avg_votes_up,
# avg_weighted_vote_score처럼 "평균값"을 만든 뒤 기준과 비교했다.
#
# 하지만 이번 버전에서는 평균을 구하지 않는다.
#
# 대신 리뷰 하나하나에 대해 먼저 기준 이상 / 기준 미만을 나눈다.
#
#   playtime_at_review >= 120       -> 해당 리뷰는 플레이타임 기준 통과
#   votes_up >= 1                   -> 해당 리뷰는 도움됨 수 기준 통과, 단 감점 계산에는 사용하지 않음
#   weighted_vote_score >= 0.5      -> 해당 리뷰는 weighted score 기준 통과
#
# 그 다음 게임별로:
#   - 기준을 통과한 긍정 리뷰 수
#   - 기준을 통과한 긍정 리뷰 비율
# 을 계산한다.
#
# 즉, 평균값 하나로 판단하지 않고,
# "이 게임의 긍정 리뷰들 중 몇 개가 기준을 통과했는가?"를 본다.
#
# ------------------------------------------------------------
# 사용자가 정한 컬럼 사용 기준
# ------------------------------------------------------------
#
# [학습에 사용하는 텍스트]
#   positive_reviews + tags + categories
#
# [학습에 사용하지 않고 UI 표시 / 필터용으로만 보존]
#   genres
#   short_description
#   release_year
#   required_age
#   is_free
#   header_image
#   platform_windows
#   platform_mac
#   platform_linux
#
# [점수 보정에 사용하는 컬럼]
#   review_score
#   playtime_at_review
#   weighted_vote_score
#
# [분석용으로만 남기는 컬럼]
#   votes_up
#
# [중복 제거 / 식별]
#   appid
#   game_title
#   review_id
#
# ------------------------------------------------------------
# 게임별 패널티 기준
# ------------------------------------------------------------
#
# 리뷰 하나하나를 기준 이상/미만으로 나눈 뒤,
# 게임별 통과 비율을 계산한다.
#
# 기본 기준:
#   해당 기준을 통과한 긍정 리뷰 비율이 0.5 미만이면 패널티
#
# 예:
#   positive_review_count = 10
#   playtime_pass_count = 3
#   playtime_pass_ratio = 0.3
#
#   0.3 < 0.5 이므로 플레이타임 기준 패널티 적용
#
# 이 기준을 쓰는 이유:
#   - 평균값은 극단값에 흔들릴 수 있다.
#   - 예를 들어 리뷰 하나의 플레이타임이 너무 크면 평균이 과하게 올라간다.
#   - 그래서 평균 대신 "기준을 넘긴 리뷰가 충분히 많은가?"를 본다.


import os
import pandas as pd


# ============================================================
# 0. 경로 설정
# ============================================================

DATA_DIR = "./datasets"

REVIEWS_PATH = os.path.join(DATA_DIR, "steam_reviews_raw_v2.csv")
GAMES_PATH = os.path.join(DATA_DIR, "steam_games_detail_v2.csv")

OUTPUT_PATH = os.path.join(DATA_DIR, "steam_game_review_documents.csv")


# ============================================================
# 1. 리뷰 단위 기준값 설정
# ============================================================

# playtime_at_review는 분 단위로 판단한다.
# 120분 = 2시간
PLAYTIME_THRESHOLD = 120

# votes_up은 1 이상이면 기준 통과
VOTES_UP_THRESHOLD = 1

# weighted_vote_score는 0.5 이상이면 기준 통과
WEIGHTED_VOTE_SCORE_THRESHOLD = 0.5


# ============================================================
# 2. 게임 단위 패널티 기준 설정
# ============================================================

# 한 게임의 긍정 리뷰 중 기준을 통과한 리뷰 비율이
# 50% 미만이면 해당 항목에 패널티를 준다.
#
# 이 값을 0.5보다 높이면 더 엄격해지고,
# 0.5보다 낮추면 더 느슨해진다.
PASS_RATIO_THRESHOLD = 0.5


# 기준 미달 시 곱할 패널티 값
PLAYTIME_LOW_MULTIPLIER = 0.90
# votes_up은 이제 review_quality_multiplier 감점 계산에 사용하지 않는다.
# 단, votes_up_pass_count / votes_up_pass_ratio는 분석용으로 유지한다.
VOTES_UP_LOW_MULTIPLIER = 1.00
WEIGHTED_SCORE_LOW_MULTIPLIER = 0.90


# ============================================================
# 3. 유틸 함수
# ============================================================

def check_required_columns(df, required_columns, file_label):
    """
    필요한 컬럼이 CSV에 있는지 확인한다.

    예를 들어 review 컬럼이 없는데 뒤에서 df["review"]를 사용하면
    pandas 에러가 발생한다.

    그래서 코드 초반에 어떤 파일에 어떤 컬럼이 없는지
    명확하게 확인한다.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"{file_label}에 필수 컬럼이 없습니다: {missing_columns}")


def ensure_columns(df, columns, default_value=""):
    """
    선택적으로 사용하는 컬럼이 없을 때 기본값으로 만들어준다.

    예:
    - header_image 컬럼이 없어도 추천 모델 생성 자체는 가능하다.
    - 이런 UI 표시용 컬럼은 없으면 빈 문자열로 채운다.
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            df[col] = default_value

    return df


def normalize_bool_series(series):
    """
    voted_up 컬럼을 True/False로 안전하게 변환한다.

    CSV를 읽으면 True/False가 실제 bool 타입이 아니라
    문자열로 들어오는 경우가 있다.

    처리 예:
    - "True", "true", "1", "yes"   -> True
    - "False", "false", "0", "no"  -> False
    """
    text = series.fillna("").astype(str).str.strip().str.lower()

    return text.map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    })


def to_number(series, fill_value=0):
    """
    컬럼을 숫자형으로 변환한다.

    errors="coerce":
    - 숫자로 바꿀 수 없는 값은 NaN으로 만든다.

    fillna(fill_value):
    - NaN을 기본값으로 채운다.
    """
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def join_reviews(series):
    """
    여러 리뷰를 하나의 긴 문자열로 합친다.

    줄바꿈으로 합치는 이유:
    - 리뷰와 리뷰 사이의 경계를 보존하기 위해
    - 다음 단계 job03에서 줄 단위로 다시 나누기 쉽게 하기 위해
    """
    cleaned = series.fillna("").astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]

    return "\n".join(cleaned)


def make_review_score_norm(series):
    """
    review_score를 0~1 사이 값으로 변환한다.

    사용 기준:
    - review_score가 0~9 범위라고 가정
    - 0은 가장 낮은 평가
    - 9는 가장 높은 평가

    따라서 9로 나눈다.
    - 0 / 9 = 0
    - 9 / 9 = 1

    값이 비어 있으면 중립값인 4.5를 넣어 0.5로 처리한다.
    """
    score = pd.to_numeric(series, errors="coerce")
    score = score.fillna(4.5)
    score = score.clip(lower=0, upper=9)

    return score / 9


def add_review_level_pass_columns(df):
    """
    리뷰 1개마다 기준 통과 여부를 계산한다.

    여기서 평균은 구하지 않는다.
    각 리뷰가 기준을 통과했는지만 True/False로 표시한다.

    예:
    playtime_at_review = 150이면
    playtime_threshold_pass = True

    playtime_at_review = 30이면
    playtime_threshold_pass = False
    """
    df = df.copy()

    df["playtime_threshold_pass"] = df["playtime_at_review"] >= PLAYTIME_THRESHOLD
    df["votes_up_threshold_pass"] = df["votes_up"] >= VOTES_UP_THRESHOLD
    df["weighted_score_threshold_pass"] = df["weighted_vote_score"] >= WEIGHTED_VOTE_SCORE_THRESHOLD

    return df


def add_game_level_quality_multiplier(df):
    """
    게임별 통과 비율을 기준으로 패널티 배수를 만든다.

    이번 수정 버전의 핵심:
    - playtime_pass_ratio는 감점 계산에 사용한다.
    - weighted_score_pass_ratio는 감점 계산에 사용한다.
    - votes_up_pass_ratio는 분석용으로만 남기고 감점 계산에는 사용하지 않는다.

    이유:
    - votes_up은 리뷰 노출량과 게임 인지도에 영향을 많이 받는다.
    - 마이너/인디 게임은 좋은 리뷰여도 votes_up이 0일 수 있다.
    - weighted_vote_score가 리뷰 유용성 판단에 더 적합하므로,
      votes_up은 review_quality_multiplier에서 제외한다.
    """
    df = df.copy()

    # 게임 단위 기준 통과 여부
    df["playtime_game_pass"] = df["playtime_pass_ratio"] >= PASS_RATIO_THRESHOLD
    df["votes_up_game_pass"] = df["votes_up_pass_ratio"] >= PASS_RATIO_THRESHOLD
    df["weighted_score_game_pass"] = df["weighted_score_pass_ratio"] >= PASS_RATIO_THRESHOLD

    # votes_up_game_pass는 분석용 컬럼이다.
    # 아래 multiplier 계산에는 playtime과 weighted score만 사용한다.
    df["review_quality_multiplier"] = 1.0

    df.loc[~df["playtime_game_pass"], "review_quality_multiplier"] *= PLAYTIME_LOW_MULTIPLIER
    df.loc[~df["weighted_score_game_pass"], "review_quality_multiplier"] *= WEIGHTED_SCORE_LOW_MULTIPLIER

    df["review_quality_multiplier"] = df["review_quality_multiplier"].round(4)

    return df


# ============================================================
# 4. 메인 동작 코드
# ============================================================

def main():
    print("=" * 80)
    print("job02: Steam 게임별 긍정 리뷰 문서 생성 시작 - 평균 미사용 버전")
    print("=" * 80)

    # --------------------------------------------------------
    # 1) 입력 파일 확인
    # --------------------------------------------------------
    print("[1/8] 입력 CSV 파일 확인 중...")

    if not os.path.exists(REVIEWS_PATH):
        raise FileNotFoundError(f"리뷰 파일을 찾을 수 없습니다: {REVIEWS_PATH}")

    if not os.path.exists(GAMES_PATH):
        raise FileNotFoundError(f"게임 상세 파일을 찾을 수 없습니다: {GAMES_PATH}")

    # --------------------------------------------------------
    # 2) CSV 읽기
    # --------------------------------------------------------
    print("[2/8] CSV 파일 읽는 중...")

    df_reviews = pd.read_csv(REVIEWS_PATH, low_memory=False)
    df_games = pd.read_csv(GAMES_PATH, low_memory=False)

    print("리뷰 원본 행 수:", len(df_reviews))
    print("리뷰 원본 게임 수:", df_reviews["appid"].nunique() if "appid" in df_reviews.columns else "appid 없음")
    print("게임 상세 행 수:", len(df_games))
    print("게임 상세 게임 수:", df_games["appid"].nunique() if "appid" in df_games.columns else "appid 없음")

    # --------------------------------------------------------
    # 3) 필수 컬럼 확인
    # --------------------------------------------------------
    print("[3/8] 필수 컬럼 확인 중...")

    required_review_cols = [
        "appid",
        "game_title",
        "review_id",
        "review",
        "voted_up",
        "playtime_at_review",
        "votes_up",
        "weighted_vote_score",
    ]

    required_game_cols = [
        "appid",
        "game_title",
    ]

    check_required_columns(df_reviews, required_review_cols, "리뷰 파일")
    check_required_columns(df_games, required_game_cols, "게임 상세 파일")

    # --------------------------------------------------------
    # 4) 리뷰 데이터 정리
    # --------------------------------------------------------
    print("[4/8] 리뷰 데이터 정리 중...")

    df_reviews["appid"] = to_number(df_reviews["appid"], fill_value=pd.NA)
    before = len(df_reviews)
    df_reviews = df_reviews[df_reviews["appid"].notna()].copy()
    df_reviews["appid"] = df_reviews["appid"].astype(int)
    print("appid 없는 리뷰 제거:", before - len(df_reviews), "행")

    before = len(df_reviews)
    df_reviews["review_id"] = df_reviews["review_id"].fillna("").astype(str).str.strip()
    df_reviews = df_reviews[df_reviews["review_id"] != ""].copy()
    df_reviews = df_reviews.drop_duplicates(subset=["appid", "review_id"]).copy()
    print("review_id 없음 또는 중복 리뷰 제거:", before - len(df_reviews), "행")

    before = len(df_reviews)
    df_reviews["review"] = df_reviews["review"].fillna("").astype(str).str.strip()
    df_reviews = df_reviews[df_reviews["review"] != ""].copy()
    print("빈 리뷰 제거:", before - len(df_reviews), "행")

    df_reviews["voted_up"] = normalize_bool_series(df_reviews["voted_up"])

    before = len(df_reviews)
    df_reviews = df_reviews[df_reviews["voted_up"].notna()].copy()
    print("voted_up 판별 불가 리뷰 제거:", before - len(df_reviews), "행")

    # 기준 비교를 위해 숫자형으로 변환한다.
    # 평균을 구하려는 목적이 아니라, 각 리뷰가 기준 이상인지 비교하기 위한 목적이다.
    df_reviews["playtime_at_review"] = to_number(df_reviews["playtime_at_review"], fill_value=0)
    df_reviews["votes_up"] = to_number(df_reviews["votes_up"], fill_value=0)
    df_reviews["weighted_vote_score"] = to_number(df_reviews["weighted_vote_score"], fill_value=0)

    # --------------------------------------------------------
    # 5) voted_up=True 리뷰만 선택
    # --------------------------------------------------------
    print("[5/8] 긍정 리뷰만 선택 중...")

    df_positive = df_reviews[df_reviews["voted_up"] == True].copy()

    print("전체 정리된 리뷰 수:", len(df_reviews))
    print("학습에 사용할 긍정 리뷰 수:", len(df_positive))
    print("긍정 리뷰가 있는 게임 수:", df_positive["appid"].nunique())

    # 리뷰 단위 기준 통과 여부를 만든다.
    # 이 단계에서도 평균은 구하지 않는다.
    df_positive = add_review_level_pass_columns(df_positive)

    # --------------------------------------------------------
    # 6) appid 기준으로 게임별 문서 생성
    # --------------------------------------------------------
    print("[6/8] appid 기준으로 긍정 리뷰 문서 생성 중...")

    # bool 값의 sum():
    # True는 1, False는 0처럼 계산된다.
    # 따라서 sum()을 하면 기준을 통과한 리뷰 개수가 된다.
    #
    # 예:
    # [True, False, True].sum() -> 2
    df_documents = (
        df_positive
        .groupby("appid", as_index=False)
        .agg(
            titles=("game_title", "first"),
            positive_reviews=("review", join_reviews),
            positive_review_count=("review", "count"),

            playtime_pass_count=("playtime_threshold_pass", "sum"),
            votes_up_pass_count=("votes_up_threshold_pass", "sum"),
            weighted_score_pass_count=("weighted_score_threshold_pass", "sum"),
        )
    )

    # 통과 비율 계산
    # 예:
    # playtime_pass_count = 7
    # positive_review_count = 10
    # playtime_pass_ratio = 0.7
    df_documents["playtime_pass_ratio"] = (
        df_documents["playtime_pass_count"] / df_documents["positive_review_count"]
    ).round(4)

    df_documents["votes_up_pass_ratio"] = (
        df_documents["votes_up_pass_count"] / df_documents["positive_review_count"]
    ).round(4)

    df_documents["weighted_score_pass_ratio"] = (
        df_documents["weighted_score_pass_count"] / df_documents["positive_review_count"]
    ).round(4)

    # 게임 단위 패널티 배수 생성
    df_documents = add_game_level_quality_multiplier(df_documents)

    print("문서화된 게임 수:", len(df_documents))

    # --------------------------------------------------------
    # 7) 게임 상세 정보 merge
    # --------------------------------------------------------
    print("[7/8] 게임 상세 정보 정리 및 merge 중...")

    df_games["appid"] = to_number(df_games["appid"], fill_value=pd.NA)
    before = len(df_games)
    df_games = df_games[df_games["appid"].notna()].copy()
    df_games["appid"] = df_games["appid"].astype(int)
    print("appid 없는 게임 상세 제거:", before - len(df_games), "행")

    game_detail_cols = [
        "appid",
        "game_title",
        "release_year",
        "genres",
        "tags",
        "categories",
        "review_score",
        "required_age",
        "is_free",
        "short_description",
        "header_image",
        "platform_windows",
        "platform_mac",
        "platform_linux",
    ]

    df_games = ensure_columns(df_games, game_detail_cols, default_value="")

    df_games_detail = df_games[game_detail_cols].drop_duplicates(subset=["appid"]).copy()
    df_games_detail = df_games_detail.rename(columns={"game_title": "detail_game_title"})

    df_result = pd.merge(
        df_documents,
        df_games_detail,
        on="appid",
        how="left",
    )

    df_result["titles"] = df_result["titles"].fillna(df_result["detail_game_title"])

    # review_score는 0~1 점수로 변환한다.
    df_result["review_score_norm"] = make_review_score_norm(df_result["review_score"]).round(4)

    # review_score에 리뷰 신뢰도 패널티 배수를 곱한다.
    df_result["review_score_adjusted"] = (
        df_result["review_score_norm"] * df_result["review_quality_multiplier"]
    ).round(4)

    # 다음 단계 job03에서 사용할 학습 후보 텍스트
    #
    # 사용:
    #   positive_reviews
    #   tags
    #   categories
    #
    # 미사용:
    #   genres
    #   short_description
    #   release_year
    #   required_age
    #   is_free
    #   header_image
    #   platform_windows/mac/linux
    df_result["tags"] = df_result["tags"].fillna("").astype(str)
    df_result["categories"] = df_result["categories"].fillna("").astype(str)

    df_result["model_source_text"] = (
        df_result["positive_reviews"].fillna("").astype(str) + "\n" +
        df_result["tags"] + "\n" +
        df_result["categories"]
    )

    # --------------------------------------------------------
    # 8) 컬럼 순서 정리 및 저장
    # --------------------------------------------------------
    print("[8/8] 결과 저장 중...")

    first_cols = [
        # 식별 정보
        "appid",
        "titles",
        "detail_game_title",

        # 다음 단계 학습용 텍스트
        "model_source_text",
        "positive_reviews",
        "positive_review_count",

        # 리뷰 기준 통과 개수
        "playtime_pass_count",
        "votes_up_pass_count",
        "weighted_score_pass_count",

        # 리뷰 기준 통과 비율
        "playtime_pass_ratio",
        "votes_up_pass_ratio",
        "weighted_score_pass_ratio",

        # 게임 단위 기준 통과 여부
        "playtime_game_pass",
        "votes_up_game_pass",
        "weighted_score_game_pass",

        # 점수 보정용
        "review_score",
        "review_score_norm",
        "review_quality_multiplier",
        "review_score_adjusted",

        # 학습 보조 텍스트
        "tags",
        "categories",

        # UI 표시 / 필터용
        "release_year",
        "genres",
        "required_age",
        "is_free",
        "short_description",
        "header_image",
        "platform_windows",
        "platform_mac",
        "platform_linux",
    ]

    first_cols = [col for col in first_cols if col in df_result.columns]
    other_cols = [col for col in df_result.columns if col not in first_cols]

    df_result = df_result[first_cols + other_cols]

    os.makedirs(DATA_DIR, exist_ok=True)
    df_result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print()
    print("=" * 80)
    print("job02 완료 - 평균 미사용 + votes_up 감점 제외 버전")
    print("=" * 80)
    print("저장 파일:", OUTPUT_PATH)
    print("최종 게임 수:", len(df_result))
    print("최종 컬럼 수:", len(df_result.columns))

    print()
    print("리뷰 단위 기준")
    print("- playtime_at_review >=", PLAYTIME_THRESHOLD)
    print("- votes_up >=", VOTES_UP_THRESHOLD, "(분석용, 감점 계산 제외)")
    print("- weighted_vote_score >=", WEIGHTED_VOTE_SCORE_THRESHOLD)

    print()
    print("게임 단위 패널티 기준")
    print("- 기준 통과 리뷰 비율 <", PASS_RATIO_THRESHOLD, "이면 패널티")

    print()
    print("패널티 배수")
    print("- 플레이타임 패널티:", PLAYTIME_LOW_MULTIPLIER)
    print("- 도움됨 수(votes_up): 감점 계산에서 제외, 통계 컬럼으로만 유지")
    print("- weighted score 패널티:", WEIGHTED_SCORE_LOW_MULTIPLIER)

    print()
    print("결과 예시")
    preview_cols = [
        "appid",
        "titles",
        "positive_review_count",
        "playtime_pass_ratio",
        "votes_up_pass_ratio",
        "weighted_score_pass_ratio",
        "review_quality_multiplier",
        "review_score_adjusted",
    ]
    preview_cols = [col for col in preview_cols if col in df_result.columns]
    print(df_result[preview_cols].head())


if __name__ == "__main__":
    main()
