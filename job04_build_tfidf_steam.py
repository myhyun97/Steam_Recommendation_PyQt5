# job04_build_tfidf_steam_split_user_rule.py
# Steam 게임 리뷰 기반 추천 시스템 - job04 split TF-IDF 버전
#
# 목적:
#   job03에서 만든 전처리 결과 CSV를 읽어서
#   추천 방식별로 가중치를 다르게 줄 수 있도록 TF-IDF 모델을 컬럼별로 따로 만든다.
#
# 왜 split TF-IDF가 필요한가?
#   이전 job04는 model_text 하나만 TF-IDF로 만들었다.
#   model_text는 아래 3개 텍스트를 합친 통합 텍스트다.
#
#       model_text = positive_reviews_processed + tags_processed + categories_processed
#
#   그런데 추천 방식이 2개라면, 각 방식에서 중요하게 볼 텍스트가 다르다.
#
#   방식 1. 키워드 기반 추천
#       사용자가 "스토리 좋은 로그라이크 보스전"처럼 직접 검색어를 입력한다.
#       이때는 실제 유저들이 긍정 리뷰에서 많이 말한 내용이 중요하다.
#       따라서 positive_reviews_processed 비중을 높게 줄 수 있다.
#
#   방식 2. 특정 게임과 비슷한 게임 추천
#       사용자가 "Hades와 비슷한 게임"처럼 기준 게임을 고른다.
#       이때는 리뷰 감상도 중요하지만, 장르/플레이 방식이 더 중요하다.
#       따라서 tags_processed 비중을 높게 줄 수 있다.
#
#   그래서 job04에서 TF-IDF를 아래처럼 따로 만든다.
#
#       1. positive_reviews_processed TF-IDF
#       2. tags_processed TF-IDF
#       3. categories_processed TF-IDF
#       4. model_text TF-IDF
#
#   job06에서는 추천 방식에 따라 아래처럼 다르게 섞을 수 있다.
#
#       키워드 기반 추천:
#           0.55 * positive_reviews_similarity
#         + 0.35 * tags_similarity
#         + 0.10 * categories_similarity
#
#       특정 게임 기반 추천:
#           0.35 * positive_reviews_similarity
#         + 0.50 * tags_similarity
#         + 0.15 * categories_similarity
#
# 입력 파일:
#   ./datasets/steam_game_reviews_preprocessed.csv
#
# 입력 주요 컬럼:
#   appid
#   titles
#   positive_reviews_processed
#   tags_processed
#   categories_processed
#   model_text
#
# 출력 파일:
#   ./datasets/steam_recommendation_index.csv
#       - 모든 TF-IDF matrix의 row 순서와 일치하는 게임 인덱스
#
#   ./models/tfidf/tfidf_positive_reviews.pkl
#   ./models/tfidf/tfidf_positive_reviews.mtx
#
#   ./models/tfidf/tfidf_tags.pkl
#   ./models/tfidf/tfidf_tags.mtx
#
#   ./models/tfidf/tfidf_categories.pkl
#   ./models/tfidf/tfidf_categories.mtx
#
#   ./models/tfidf/tfidf_model_text.pkl
#   ./models/tfidf/tfidf_model_text.mtx
#
#   ./models/tfidf/tfidf_manifest.json
#       - job06에서 어떤 파일을 불러와야 하는지 기록한 파일
#
# ------------------------------------------------------------
# 중요한 설계 원칙
# ------------------------------------------------------------
#
# 1. job04에서는 추가 전처리를 하지 않는다.
#    형태소 분석, 불용어 제거, 토큰 정리는 job03에서 끝난 것으로 본다.
#
# 2. TF-IDF는 컬럼별로 따로 만든다.
#    그래야 job06에서 추천 방식별 가중치를 다르게 줄 수 있다.
#
# 3. 모든 matrix는 같은 행 순서를 가진다.
#    예를 들어 matrix_positive[10], matrix_tags[10], matrix_categories[10]은
#    모두 index_df.iloc[10]에 해당하는 같은 게임이어야 한다.
#
# 4. review_score_adjusted, release_year, platform 정보는 TF-IDF에 넣지 않는다.
#    이 값들은 job06에서 점수 보정과 필터로 사용한다.
#
# 5. max_df, min_df는 사용하지 않는다.
#    불용어 제거는 job03에서 직접 관리한 stopwords 기준으로 끝났다고 본다.


import os
import json
import pickle
import time
from pathlib import Path

import pandas as pd
from scipy.io import mmwrite
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# 0. 경로 설정
# ============================================================

DATA_DIR = Path("./datasets")
MODEL_DIR = Path("./models/tfidf")

INPUT_FILE = DATA_DIR / "steam_game_reviews_preprocessed.csv"

# 추천 결과 표시와 필터에 사용할 인덱스 파일
INDEX_FILE = DATA_DIR / "steam_recommendation_tfidf_index.csv"

# job06에서 파일 경로를 쉽게 불러오기 위한 manifest 파일
MANIFEST_FILE = MODEL_DIR / "tfidf_manifest.json"

# 실행 설정과 결과를 사람이 읽기 좋게 저장하는 파일
CONFIG_FILE = MODEL_DIR / "job04_tfidf_config.txt"


# ============================================================
# 1. TF-IDF 대상 컬럼 설정
# ============================================================

# key:
#   저장 파일 이름에 들어갈 짧은 이름
#
# column:
#   실제 CSV 컬럼 이름
#
# description:
#   이 TF-IDF가 어떤 의미인지 설명
TEXT_SOURCES = {
    "positive_reviews": {
        "column": "positive_reviews_processed",
        "description": "긍정 리뷰 전처리 텍스트",
    },
    "tags": {
        "column": "tags_processed",
        "description": "Steam tags 전처리 텍스트",
    },
    "categories": {
        "column": "categories_processed",
        "description": "Steam categories 전처리 텍스트",
    },
    "model_text": {
        "column": "model_text",
        "description": "positive_reviews + tags + categories 통합 전처리 텍스트",
    },
}


# ============================================================
# 2. TF-IDF 설정
# ============================================================

# sublinear_tf=True:
#   단어 빈도를 그대로 쓰지 않고 1 + log(tf)로 완만하게 줄인다.
#   특정 단어가 한 문서에 너무 많이 반복되어도 영향이 과하게 커지는 것을 막는다.
#
# norm="l2":
#   각 문서 벡터의 길이를 정규화한다.
#   코사인 유사도 계산에 적합하다.
#
# dtype="float32":
#   float64보다 메모리를 덜 사용한다.
#   추천 시스템 점수 계산에서는 보통 충분하다.
TFIDF_PARAMS = {
    "sublinear_tf": True,
    "norm": "l2",
    "dtype": "float32",
}


# 추천 방식별 기본 가중치도 config에 같이 기록해 둔다.
# 실제 계산은 job06에서 한다.
RECOMMENDATION_WEIGHTS_EXAMPLE = {
    "keyword_based": {
        "positive_reviews": 0.55,
        "tags": 0.35,
        "categories": 0.10,
    },
    "similar_game_based": {
        "positive_reviews": 0.35,
        "tags": 0.50,
        "categories": 0.15,
    },
}


# 확인용 핵심 단어들이다.
# 이 단어들이 TF-IDF 사전에 살아 있는지 출력만 한다.
# 여기서 단어를 추가하거나 제거하지는 않는다.
CORE_WORDS_TO_CHECK = [
    "좋다", "나쁘다", "재미", "재밌다", "재미있다", "재미없다",
    "추천", "비추천", "갓겜", "망겜",
    "스토리", "그래픽", "난이도", "타격감", "몰입",
    "캐릭터", "보스", "전투", "퍼즐", "탐험",
    "파밍", "성장", "퀘스트", "아이템", "엔딩",
    "세계관", "사운드", "조작", "밸런스",
    "공포", "멀티", "싱글", "협동", "힐링", "농사",
    "생존", "전략", "건설", "로그라이크", "오픈월드",
    "액션", "어드벤처", "시뮬레이션", "턴제",
    "버그", "최적화", "프레임", "패치", "번역", "한글",
    "모드", "dlc", "가격", "할인", "가성비",
]


# ============================================================
# 3. 유틸 함수
# ============================================================

def check_required_columns(df, required_columns):
    """
    필수 컬럼이 존재하는지 확인한다.

    컬럼이 없는데 뒤 단계로 넘어가면 오류 원인을 찾기 어려워진다.
    그래서 초반에 명확하게 에러를 낸다.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")


def clean_text_columns(df):
    """
    TF-IDF 대상 텍스트 컬럼을 정리한다.

    job03에서 이미 전처리는 끝났으므로 여기서는 최소한만 처리한다.

    처리 내용:
    - NaN을 빈 문자열로 변경
    - 문자열 타입으로 변경
    - 앞뒤 공백 제거
    """
    df = df.copy()

    for source_info in TEXT_SOURCES.values():
        col = source_info["column"]
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df


def remove_empty_model_text_rows(df):
    """
    model_text가 비어 있는 게임은 제거한다.

    이유:
    - model_text는 positive_reviews + tags + categories를 합친 최종 텍스트다.
    - 이 값이 비어 있으면 TF-IDF와 Word2Vec 추천에 쓸 텍스트가 없다는 뜻이다.
    """
    before_count = len(df)

    df = df[df["model_text"] != ""].reset_index(drop=True)

    removed_count = before_count - len(df)

    return df, removed_count


def make_recommendation_index(df):
    """
    job06/job07에서 사용할 추천 인덱스 파일을 만든다.

    매우 중요한 점:
    - 이 index_df의 row 순서가 모든 TF-IDF matrix의 row 순서다.
    - index_df.iloc[0]은 각 matrix의 0번 row와 같은 게임이다.

    이 파일에는 TF-IDF 학습에 직접 쓰지 않는 정보도 저장한다.
    이유:
    - 추천 결과 화면 표시
    - 출시연도 필터
    - 플랫폼 필터
    - 연령 필터
    - 평가 점수 보정
    """
    preferred_columns = [
        # 식별 정보
        "appid",
        "titles",
        "detail_game_title",

        # 텍스트 정보 확인용
        "positive_review_count",
        "model_token_count",
        "positive_reviews_processed",
        "tags_processed",
        "categories_processed",
        "model_text",

        # 리뷰 신뢰도/점수 보정용
        "playtime_pass_count",
        "votes_up_pass_count",
        "weighted_score_pass_count",
        "playtime_pass_ratio",
        "votes_up_pass_ratio",
        "weighted_score_pass_ratio",
        "playtime_game_pass",
        "votes_up_game_pass",
        "weighted_score_game_pass",
        "review_score",
        "review_score_norm",
        "review_quality_multiplier",
        "review_score_adjusted",

        # UI 표시 / 필터용
        "release_year",
        "genres",
        "tags",
        "categories",
        "required_age",
        "is_free",
        "short_description",
        "header_image",
        "platform_windows",
        "platform_mac",
        "platform_linux",
    ]

    existing_columns = [col for col in preferred_columns if col in df.columns]

    return df[existing_columns].copy()


def make_output_paths(source_name):
    """
    source_name에 맞는 출력 파일 경로를 만든다.

    예:
    source_name = "positive_reviews"

    결과:
    tfidf_positive_reviews_user_rule.pkl
    tfidf_positive_reviews_user_rule.mtx
    tfidf_positive_reviews_features_user_rule.csv
    """
    model_file = MODEL_DIR / f"tfidf_{source_name}.pkl"
    matrix_file = MODEL_DIR / f"tfidf_{source_name}.mtx"
    features_file = MODEL_DIR / f"tfidf_{source_name}_features.csv"

    return model_file, matrix_file, features_file


def train_tfidf_for_source(df, source_name, column_name):
    """
    특정 컬럼 하나에 대해 TF-IDF vectorizer와 matrix를 만든다.

    반환:
    - vectorizer
    - matrix

    주의:
    - df의 행 순서를 바꾸지 않는다.
    - 그래야 모든 matrix의 row 순서가 같게 유지된다.
    """
    texts = df[column_name].fillna("").astype(str).str.strip()

    non_empty_count = int((texts != "").sum())

    if non_empty_count == 0:
        raise ValueError(
            f"{source_name}({column_name}) 컬럼에 TF-IDF 학습 가능한 텍스트가 없습니다."
        )

    print(f"\n[{source_name}] TF-IDF 학습 시작")
    print(f"- 대상 컬럼: {column_name}")
    print(f"- 전체 게임 수: {len(texts)}")
    print(f"- 비어 있지 않은 텍스트 수: {non_empty_count}")

    print("- 텍스트 예시:")
    for text in texts.head(3):
        preview = text[:120]
        if len(text) > 120:
            preview += "..."
        print("  ", preview)

    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)

    # fit_transform:
    #   fit      -> 전체 텍스트를 보고 단어 사전을 만든다.
    #   transform -> 각 게임 텍스트를 TF-IDF 벡터로 바꾼다.
    matrix = vectorizer.fit_transform(texts)

    print(f"- matrix shape: {matrix.shape}")
    print(f"- 단어 사전 크기: {len(vectorizer.vocabulary_)}")

    return vectorizer, matrix


def save_tfidf_outputs(source_name, vectorizer, matrix):
    """
    특정 source의 TF-IDF 결과를 저장한다.

    저장 내용:
    - vectorizer pkl
    - matrix mtx
    - feature names csv
    """
    model_file, matrix_file, features_file = make_output_paths(source_name)

    with open(model_file, "wb") as f:
        pickle.dump(vectorizer, f)

    mmwrite(matrix_file, matrix)

    feature_names = vectorizer.get_feature_names_out()

    df_features = pd.DataFrame({
        "feature_index": range(len(feature_names)),
        "feature_name": feature_names,
    })
    df_features.to_csv(features_file, index=False, encoding="utf-8-sig")

    return {
        "model_file": str(model_file),
        "matrix_file": str(matrix_file),
        "features_file": str(features_file),
        "matrix_shape": list(matrix.shape),
        "vocabulary_size": len(vectorizer.vocabulary_),
    }


def print_core_word_check(source_name, vectorizer):
    """
    핵심 단어들이 특정 TF-IDF 사전에 들어갔는지 확인한다.

    예를 들어 tags_processed 사전에는 "로그라이크"가 있고,
    positive_reviews_processed 사전에는 없을 수도 있다.

    이것은 오류가 아니다.
    컬럼별로 담고 있는 정보가 다르기 때문이다.
    """
    vocab = set(vectorizer.get_feature_names_out())

    print(f"\n[{source_name}] 핵심 단어 사전 포함 여부:")

    for word in CORE_WORDS_TO_CHECK:
        status = "있음" if word in vocab else "없음"
        print(f"- {word}: {status}")


def save_manifest_and_config(results, elapsed_seconds, removed_empty_rows, final_game_count):
    """
    manifest JSON과 config TXT를 저장한다.

    manifest:
    - job06에서 모델 파일을 자동으로 불러오기 좋게 만든 파일

    config:
    - 사람이 읽기 좋게 이번 실행 결과를 정리한 파일
    """
    manifest = {
        "job": "job04_split_tfidf",
        "input_file": str(INPUT_FILE),
        "index_file": str(INDEX_FILE),
        "model_dir": str(MODEL_DIR),
        "tfidf_params": TFIDF_PARAMS,
        "text_sources": TEXT_SOURCES,
        "recommendation_weights_example": RECOMMENDATION_WEIGHTS_EXAMPLE,
        "removed_empty_model_text_rows": removed_empty_rows,
        "final_game_count": final_game_count,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "sources": results,
        "note": (
            "모든 TF-IDF matrix는 같은 row 순서를 가진다. "
            "row 순서는 index_file의 row 순서와 같다. "
            "job06에서는 추천 방식별로 positive_reviews/tags/categories 유사도 가중치를 다르게 적용한다."
        ),
    }

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("job04_build_tfidf_steam_split_user_rule.py")
    lines.append("Steam 게임 리뷰 기반 추천 시스템 - split TF-IDF")
    lines.append("")
    lines.append("[입력 파일]")
    lines.append(f"- {INPUT_FILE}")
    lines.append("")
    lines.append("[출력 인덱스]")
    lines.append(f"- {INDEX_FILE}")
    lines.append("")
    lines.append("[TF-IDF 설정]")
    for key, value in TFIDF_PARAMS.items():
        lines.append(f"- {key}: {value}")
    lines.append("- max_df: 사용 안 함")
    lines.append("- min_df: 사용 안 함")
    lines.append("- stopwords: job03에서 처리")
    lines.append("")
    lines.append("[추천 방식별 가중치 예시]")
    for mode, weights in RECOMMENDATION_WEIGHTS_EXAMPLE.items():
        lines.append(f"- {mode}: {weights}")
    lines.append("")
    lines.append("[결과]")
    lines.append(f"- removed_empty_model_text_rows: {removed_empty_rows}")
    lines.append(f"- final_game_count: {final_game_count}")
    lines.append(f"- elapsed_seconds: {elapsed_seconds:.1f}")
    lines.append("")
    lines.append("[source별 결과]")
    for source_name, result in results.items():
        lines.append(f"- {source_name}")
        lines.append(f"  - column: {result['column']}")
        lines.append(f"  - matrix_shape: {result['matrix_shape']}")
        lines.append(f"  - vocabulary_size: {result['vocabulary_size']}")
        lines.append(f"  - model_file: {result['model_file']}")
        lines.append(f"  - matrix_file: {result['matrix_file']}")
        lines.append(f"  - features_file: {result['features_file']}")

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 4. 메인 함수
# ============================================================

def main():
    start_time = time.time()

    print("=" * 80)
    print("job04: Steam split TF-IDF 모델 생성 시작")
    print("=" * 80)

    # --------------------------------------------------------
    # 1) 입력 CSV 읽기
    # --------------------------------------------------------
    print("[1/7] 입력 CSV 읽는 중...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print("입력 행 수:", len(df))
    print("입력 컬럼 수:", len(df.columns))

    # --------------------------------------------------------
    # 2) 필수 컬럼 확인
    # --------------------------------------------------------
    print("[2/7] 필수 컬럼 확인 중...")

    required_columns = ["appid", "titles"]
    required_columns += [info["column"] for info in TEXT_SOURCES.values()]

    check_required_columns(df, required_columns)

    # --------------------------------------------------------
    # 3) 텍스트 컬럼 정리
    # --------------------------------------------------------
    print("[3/7] 텍스트 컬럼 정리 중...")

    df = clean_text_columns(df)
    df, removed_empty_rows = remove_empty_model_text_rows(df)

    print("빈 model_text 제거:", removed_empty_rows, "행")
    print("TF-IDF 대상 게임 수:", len(df))

    if len(df) == 0:
        raise ValueError("TF-IDF에 사용할 게임 데이터가 없습니다.")

    # --------------------------------------------------------
    # 4) 추천 인덱스 저장
    # --------------------------------------------------------
    print("[4/7] 추천 인덱스 저장 중...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    index_df = make_recommendation_index(df)
    index_df.to_csv(INDEX_FILE, index=False, encoding="utf-8-sig")

    print("추천 인덱스 저장:", INDEX_FILE)
    print("추천 인덱스 행 수:", len(index_df))
    print("추천 인덱스 컬럼 수:", len(index_df.columns))

    # --------------------------------------------------------
    # 5) source별 TF-IDF 학습 및 저장
    # --------------------------------------------------------
    print("[5/7] source별 TF-IDF 학습 및 저장 중...")

    results = {}

    for source_name, source_info in TEXT_SOURCES.items():
        column_name = source_info["column"]

        vectorizer, matrix = train_tfidf_for_source(
            df=df,
            source_name=source_name,
            column_name=column_name,
        )

        saved_info = save_tfidf_outputs(source_name, vectorizer, matrix)

        saved_info["column"] = column_name
        saved_info["description"] = source_info["description"]

        results[source_name] = saved_info

        print_core_word_check(source_name, vectorizer)

    # --------------------------------------------------------
    # 6) manifest/config 저장
    # --------------------------------------------------------
    print("[6/7] manifest/config 저장 중...")

    elapsed = time.time() - start_time

    save_manifest_and_config(
        results=results,
        elapsed_seconds=elapsed,
        removed_empty_rows=removed_empty_rows,
        final_game_count=len(df),
    )

    # --------------------------------------------------------
    # 7) 최종 요약 출력
    # --------------------------------------------------------
    print("[7/7] 최종 요약 출력 중...")

    print()
    print("=" * 80)
    print("job04 split TF-IDF 완료")
    print("=" * 80)
    print("추천 인덱스:", INDEX_FILE)
    print("manifest:", MANIFEST_FILE)
    print("config:", CONFIG_FILE)

    print()
    print("source별 저장 결과:")
    for source_name, result in results.items():
        print(f"\n[{source_name}]")
        print("- column:", result["column"])
        print("- matrix shape:", result["matrix_shape"])
        print("- vocabulary size:", result["vocabulary_size"])
        print("- model:", result["model_file"])
        print("- matrix:", result["matrix_file"])
        print("- features:", result["features_file"])

    print()
    print("추천 방식별 TF-IDF 가중치 예시:")
    print("- 키워드 기반 추천:", RECOMMENDATION_WEIGHTS_EXAMPLE["keyword_based"])
    print("- 특정 게임 기반 추천:", RECOMMENDATION_WEIGHTS_EXAMPLE["similar_game_based"])

    print()
    print(f"총 경과 시간: {elapsed:.1f}초")


if __name__ == "__main__":
    main()
