# job05_train_word2vec_steam_split_user_rule.py
# Steam 게임 리뷰 기반 추천 시스템 - job05 split Word2Vec 버전
#
# 목적:
#   job03에서 만든 전처리 결과 CSV를 읽어서 Word2Vec 모델을 학습한다.
#   그리고 job04 split TF-IDF 구조와 맞게 게임별 평균 Word2Vec 벡터를 컬럼별로 따로 만든다.
#
# 왜 split Word2Vec이 필요한가?
#   job04 split 버전에서는 TF-IDF를 아래 컬럼별로 따로 만들었다.
#
#       1. positive_reviews_processed
#       2. tags_processed
#       3. categories_processed
#       4. model_text
#
#   추천 방식이 2개라면 Word2Vec도 같은 방식으로 나눠 두는 것이 좋다.
#
#   방식 1. 키워드 기반 추천
#       사용자가 "스토리 좋은 로그라이크 보스전"처럼 직접 입력한다.
#       이때는 긍정 리뷰 문맥이 중요하므로 positive_reviews 벡터 비중을 높게 줄 수 있다.
#
#   방식 2. 특정 게임과 비슷한 게임 추천
#       사용자가 특정 게임을 기준으로 비슷한 게임을 찾는다.
#       이때는 장르/플레이 방식이 중요하므로 tags 벡터 비중을 높게 줄 수 있다.
#
#   따라서 job05에서는 Word2Vec 모델은 하나만 학습하되,
#   게임별 평균 벡터는 source별로 따로 저장한다.
#
#       1. positive_reviews 평균 벡터
#       2. tags 평균 벡터
#       3. categories 평균 벡터
#       4. model_text 평균 벡터
#
# ------------------------------------------------------------
# 중요한 설계 원칙
# ------------------------------------------------------------
#
# 1. Word2Vec 모델은 하나만 만든다.
#    tags나 categories만 따로 Word2Vec을 학습하면 데이터 양이 적어서 품질이 낮을 수 있다.
#    그래서 positive_reviews + tags + categories 전체 문맥을 하나의 Word2Vec 모델로 학습한다.
#
# 2. 평균 벡터는 컬럼별로 따로 만든다.
#    같은 Word2Vec 모델을 사용하되, 어떤 텍스트 컬럼의 토큰을 평균내느냐만 다르게 한다.
#
# 3. 모든 벡터 npy 파일은 같은 row 순서를 가진다.
#    예를 들어:
#       word2vec_positive_reviews_vectors[10]
#       word2vec_tags_vectors[10]
#       word2vec_categories_vectors[10]
#    위 3개는 모두 index_df.iloc[10]에 해당하는 같은 게임이어야 한다.
#
# 4. model_text는 통합 fallback 용도다.
#    positive_reviews/tags/categories를 분리해서 쓰는 것이 기본이지만,
#    필요하면 model_text 벡터 하나로 단순 추천도 만들 수 있다.
#
# 5. job05에서는 형태소 분석이나 불용어 제거를 하지 않는다.
#    이 작업은 job03에서 이미 끝난 것으로 본다.
#
# ------------------------------------------------------------
# 입력 파일
# ------------------------------------------------------------
#
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
# ------------------------------------------------------------
# 출력 파일
# ------------------------------------------------------------
#
#   ./models/word2vec/word2vec_steam.model
#       - 학습된 Word2Vec 모델
#
#   ./models/word2vec/word2vec_positive_reviews_vectors.npy
#   ./models/word2vec/word2vec_tags_vectors.npy
#   ./models/word2vec/word2vec_categories_vectors.npy
#   ./models/word2vec/word2vec_model_text_vectors.npy
#       - source별 게임 평균 벡터
#
#   ./datasets/word2vec_game_vector_index.csv
#       - 모든 Word2Vec 벡터 npy 파일의 row 순서와 일치하는 게임 인덱스
#
#   ./datasets/word2vec_check.csv
#       - 대표 단어의 유사어 확인용 CSV
#
#   ./models/word2vec/word2vec_manifest.json
#       - job06에서 어떤 파일을 불러와야 하는지 기록한 파일
#
#   ./models/word2vec/job05_word2vec_config.json
#       - 이번 학습 설정과 결과 요약


import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


# ============================================================
# 0. 경로 설정
# ============================================================

DATA_DIR = Path("./datasets")
MODEL_DIR = Path("./models/word2vec")

INPUT_CSV = DATA_DIR / "steam_game_reviews_preprocessed.csv"

OUTPUT_MODEL = MODEL_DIR / "word2vec_steam.model"
OUTPUT_INDEX_CSV = DATA_DIR / "word2vec_game_vector_index.csv"
OUTPUT_CHECK_CSV = DATA_DIR / "word2vec_check.csv"
OUTPUT_MANIFEST = MODEL_DIR / "word2vec_manifest.json"
OUTPUT_CONFIG = MODEL_DIR / "job05_word2vec_config.json"


# ============================================================
# 1. Word2Vec 벡터 생성 대상 컬럼 설정
# ============================================================

# job04 split TF-IDF와 같은 이름 구조를 사용한다.
#
# key:
#   저장 파일 이름과 job06에서 사용할 source 이름
#
# column:
#   실제 CSV 컬럼 이름
#
# description:
#   해당 source가 의미하는 내용
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
# 2. Word2Vec 학습 설정
# ============================================================

# vector_size:
#   단어 하나를 몇 차원 벡터로 표현할지 정한다.
#   100이면 단어 하나가 길이 100짜리 숫자 배열이 된다.
VECTOR_SIZE = 100

# window:
#   기준 단어 앞뒤 몇 개 단어까지 문맥으로 볼지 정한다.
#   window=4이면 기준 단어 주변 4개 정도의 단어를 문맥으로 본다.
WINDOW = 4

# min_count:
#   전체 학습 문장 안에서 이 횟수보다 적게 나온 단어는 Word2Vec 단어장에서 제외한다.
#
# 예전 v4는 min_count=20이었다.
# 하지만 Steam 태그/장르 단어는 등장 횟수가 적어도 중요할 수 있다.
# 그래서 이번 split 버전에서는 3으로 낮춰서 희귀하지만 중요한 단어가 최대한 살아남게 한다.
MIN_COUNT = 3

# workers:
#   CPU 작업자 수다.
#   너무 크게 잡으면 오히려 시스템이 느려질 수 있으므로 최대 4개로 제한한다.
WORKERS = min(4, max(1, multiprocessing.cpu_count() - 1))

# epochs:
#   전체 학습 데이터를 몇 번 반복해서 학습할지 정한다.
EPOCHS = 50

# sg:
#   1 = Skip-gram
#   0 = CBOW
#
# Skip-gram은 상대적으로 희귀한 단어의 의미를 잡는 데 유리한 편이다.
# 게임 추천에서는 로그라이크, 소울라이크, 턴제 같은 단어가 중요할 수 있으므로 sg=1을 사용한다.
SG = 1

# seed:
#   재현성을 위한 난수 고정값이다.
SEED = 42


# ============================================================
# 3. 학습 문장 생성 설정
# ============================================================

# Word2Vec은 문장 단위의 토큰 리스트를 입력으로 받는다.
# 하지만 현재 job03 결과는 리뷰별 원문 경계를 완벽히 보존하지 않을 수 있다.
# 그래서 긴 토큰 리스트를 일정 길이로 잘라 여러 문장처럼 사용한다.
SENTENCE_TOKEN_CHUNK_SIZE = 80

# 너무 짧은 문장은 문맥 정보가 부족하므로 제외한다.
MIN_SENTENCE_TOKENS = 2


# ============================================================
# 4. 추천 방식별 Word2Vec 가중치 예시
# ============================================================

# 실제 계산은 job06에서 한다.
# 여기서는 manifest/config에 기록해 두기 위한 값이다.
RECOMMENDATION_WEIGHTS_EXAMPLE = {
    "keyword_based": {
        "positive_reviews": 0.65,
        "tags": 0.25,
        "categories": 0.10,
    },
    "similar_game_based": {
        "positive_reviews": 0.40,
        "tags": 0.45,
        "categories": 0.15,
    },
}


# ============================================================
# 5. 대표 유사어 확인용 단어
# ============================================================

CHECK_WORDS = [
    # 평가 / 감성
    "좋다", "나쁘다", "재미", "재밌다", "재미있다", "재미없다",
    "추천", "비추천", "갓겜", "망겜",

    # 게임 특징
    "스토리", "그래픽", "난이도", "타격감", "몰입",
    "캐릭터", "보스", "전투", "퍼즐", "탐험",
    "파밍", "성장", "퀘스트", "아이템", "엔딩",
    "세계관", "사운드", "조작", "밸런스",

    # 장르 / 플레이 방식
    "공포", "멀티", "싱글", "협동", "힐링", "농사",
    "생존", "전략", "건설", "로그라이크", "오픈월드",
    "액션", "어드벤처", "시뮬레이션", "턴제",

    # 품질 / 구매 판단
    "버그", "최적화", "프레임", "패치", "번역", "한글",
    "모드", "dlc", "가격", "할인", "가성비",

    # 영어 태그/장르
    "rpg", "fps", "roguelike", "survival", "horror", "sandbox",
]


# ============================================================
# 6. Word2Vec 진행 상황 출력 callback
# ============================================================

class EpochLogger(CallbackAny2Vec):
    """
    Word2Vec 학습 epoch 진행 상황을 출력하는 callback 클래스다.

    gensim Word2Vec은 epochs 수만큼 학습 데이터를 반복해서 본다.
    이 callback을 넣으면 각 epoch의 시작/종료 시간을 확인할 수 있다.
    """

    def __init__(self, total_epochs):
        self.epoch = 0
        self.total_epochs = total_epochs
        self.train_start_time = None
        self.epoch_start_time = None

    def on_train_begin(self, model):
        self.train_start_time = time.time()
        print("Word2Vec 학습 루프 시작")

    def on_epoch_begin(self, model):
        self.epoch += 1
        self.epoch_start_time = time.time()
        print(f"epoch {self.epoch}/{self.total_epochs} 시작")

    def on_epoch_end(self, model):
        epoch_elapsed = time.time() - self.epoch_start_time
        total_elapsed = time.time() - self.train_start_time

        print(
            f"epoch {self.epoch}/{self.total_epochs} 완료 | "
            f"이번 epoch: {epoch_elapsed:.1f}초 | "
            f"누적 학습: {total_elapsed:.1f}초"
        )


# ============================================================
# 7. 유틸 함수
# ============================================================

def check_required_columns(df, required_columns):
    """
    필수 컬럼이 존재하는지 확인한다.

    컬럼이 없는데 뒤 단계로 넘어가면 오류 원인을 찾기 어렵다.
    그래서 초반에 명확하게 에러를 낸다.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")


def clean_text_columns(df):
    """
    Word2Vec 대상 텍스트 컬럼을 정리한다.

    job03에서 이미 형태소 분석과 불용어 제거가 끝났으므로
    여기서는 최소한의 정리만 한다.

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
    - 이 값이 비어 있으면 Word2Vec 추천에 사용할 텍스트가 없다는 뜻이다.

    job04 split TF-IDF도 같은 기준으로 row를 제거한다.
    그래야 job04 matrix와 job05 vector의 row 순서가 맞을 가능성이 높다.
    그래도 job06에서는 appid 기준으로 한 번 더 맞추는 것이 안전하다.
    """
    before_count = len(df)

    df = df[df["model_text"] != ""].reset_index(drop=True)

    removed_count = before_count - len(df)

    return df, removed_count


def split_tokens(text):
    """
    공백으로 구분된 전처리 텍스트를 토큰 리스트로 바꾼다.

    job03에서 이미 토큰화 후 공백으로 합쳐 저장했으므로
    여기서는 split()만 사용한다.
    """
    if pd.isna(text):
        return []

    text = str(text).strip()

    if not text:
        return []

    return text.split()


def split_tokens_to_chunks(tokens, chunk_size=SENTENCE_TOKEN_CHUNK_SIZE):
    """
    긴 토큰 리스트를 일정 길이의 여러 문장으로 나눈다.

    예:
        tokens 길이 = 200
        chunk_size = 80

    결과:
        0~79번 토큰
        80~159번 토큰
        160~199번 토큰

    이렇게 나누는 이유:
    - 게임 하나의 거대한 문서를 통째로 한 문장으로 넣으면 문맥 범위가 너무 넓어진다.
    - Word2Vec은 주변 단어 관계를 학습하므로, 너무 긴 문장은 의미가 섞일 수 있다.
    """
    chunks = []

    for start in range(0, len(tokens), chunk_size):
        chunk = tokens[start:start + chunk_size]

        if len(chunk) >= MIN_SENTENCE_TOKENS:
            chunks.append(chunk)

    return chunks


def build_training_sentences(df):
    """
    Word2Vec 학습에 사용할 문장 리스트를 만든다.

    반환 형태:
        sentences = [
            ["스토리", "좋다", "캐릭터", "매력"],
            ["전투", "타격감", "좋다"],
            ["로그라이크", "액션", "싱글"],
        ]

    학습 문장 구성 방식:
    - positive_reviews_processed:
        길 수 있으므로 chunk로 잘라 여러 문장처럼 사용한다.

    - tags_processed:
        짧지만 게임 장르/특징 단어가 들어 있으므로 하나의 보조 문장으로 추가한다.

    - categories_processed:
        싱글/멀티/도전과제 같은 기능 정보가 있으므로 하나의 보조 문장으로 추가한다.

    주의:
    - model_text는 위 3개가 합쳐진 컬럼이므로 학습 문장에는 따로 넣지 않는다.
    - model_text까지 넣으면 같은 단어가 중복 학습될 수 있다.
    - 대신 model_text는 나중에 게임별 통합 평균 벡터를 만들 때 사용한다.
    """
    sentences = []

    for _, row in df.iterrows():
        positive_tokens = split_tokens(row.get("positive_reviews_processed", ""))
        tags_tokens = split_tokens(row.get("tags_processed", ""))
        categories_tokens = split_tokens(row.get("categories_processed", ""))

        # 긍정 리뷰는 길 수 있으므로 여러 문장으로 나눈다.
        sentences.extend(split_tokens_to_chunks(positive_tokens))

        # tags는 보통 짧으므로 하나의 문장처럼 넣는다.
        if len(tags_tokens) >= MIN_SENTENCE_TOKENS:
            sentences.append(tags_tokens)

        # categories도 하나의 문장처럼 넣는다.
        if len(categories_tokens) >= MIN_SENTENCE_TOKENS:
            sentences.append(categories_tokens)

    return sentences


def make_vector_file_path(source_name):
    """
    source 이름에 맞는 평균 벡터 npy 저장 경로를 만든다.

    예:
        source_name = "positive_reviews"
        -> word2vec_positive_reviews_vectors.npy
    """
    return MODEL_DIR / f"word2vec_{source_name}_vectors.npy"


def make_game_vector(tokens, model):
    """
    게임 하나의 평균 Word2Vec 벡터를 만든다.

    처리 방식:
    1. tokens 중 Word2Vec 단어장에 있는 단어만 사용한다.
    2. 각 단어의 벡터를 가져온다.
    3. 벡터들의 평균을 계산한다.

    반환:
    - game_vector:
        평균 벡터
    - valid_count:
        평균 계산에 실제로 사용된 토큰 수

    만약 사용할 수 있는 토큰이 하나도 없으면:
    - 0 벡터를 반환한다.
    - valid_count는 0이다.
    """
    vectors = []

    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])

    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32), 0

    vector = np.mean(vectors, axis=0).astype(np.float32)

    return vector, len(vectors)


def build_vectors_for_source(df, source_name, column_name, model):
    """
    특정 source 컬럼에 대해 모든 게임의 평균 Word2Vec 벡터를 만든다.

    예:
        source_name = "tags"
        column_name = "tags_processed"

    결과:
        vectors.shape = (게임 수, VECTOR_SIZE)

    row 순서:
        df의 row 순서를 그대로 따른다.
    """
    game_vectors = []
    valid_token_counts = []
    total_token_counts = []

    for _, row in df.iterrows():
        tokens = split_tokens(row.get(column_name, ""))

        vector, valid_count = make_game_vector(tokens, model)

        game_vectors.append(vector)
        valid_token_counts.append(valid_count)
        total_token_counts.append(len(tokens))

    vectors_array = np.vstack(game_vectors).astype(np.float32)

    return vectors_array, valid_token_counts, total_token_counts


def make_game_vector_index(df, source_stats):
    """
    Word2Vec 벡터 npy 파일과 row 순서를 맞추는 인덱스 파일을 만든다.

    source_stats에는 source별 valid_token_count 등이 들어 있다.

    중요한 점:
    - index_df.iloc[0]은 모든 벡터 npy 파일의 0번 row와 같은 게임이어야 한다.
    - 이 순서가 틀어지면 추천 결과가 엉뚱한 게임으로 표시된다.
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

    index_df = df[existing_columns].copy()

    # source별 토큰 통계를 인덱스에 추가한다.
    for source_name, stats in source_stats.items():
        index_df[f"word2vec_{source_name}_total_token_count"] = stats["total_token_counts"]
        index_df[f"word2vec_{source_name}_valid_token_count"] = stats["valid_token_counts"]

    return index_df


def count_word_in_sentences(sentences, word):
    """
    학습 문장 전체에서 특정 단어가 몇 번 등장했는지 센다.
    """
    count = 0

    for sentence in sentences:
        count += sentence.count(word)

    return count


def build_check_rows(model, sentences, check_words, topn=10):
    """
    대표 단어들이 Word2Vec 단어장에 들어갔는지,
    들어갔다면 어떤 단어들과 유사한지 확인할 rows를 만든다.
    """
    rows = []

    for word in check_words:
        original_count = count_word_in_sentences(sentences, word)

        if word not in model.wv:
            rows.append({
                "word": word,
                "count": original_count,
                "in_word2vec": False,
                "similar_words": "",
            })
            continue

        similar = model.wv.most_similar(word, topn=topn)
        similar_text = ", ".join([f"{w}({score:.3f})" for w, score in similar])

        rows.append({
            "word": word,
            "count": original_count,
            "in_word2vec": True,
            "similar_words": similar_text,
        })

    return rows


def print_check_result(check_df):
    """
    대표 단어 유사어 확인 결과를 콘솔에 출력한다.
    """
    print("\n유사어 확인 예시:")

    for _, row in check_df.iterrows():
        word = row["word"]
        count = int(row["count"])
        in_word2vec = bool(row["in_word2vec"])
        similar_words = row["similar_words"]

        if not in_word2vec:
            print(f"- {word} | 출현 {count}회: 단어장에 없음")
        else:
            print(f"- {word} | 출현 {count}회: {similar_words}")


def save_manifest_and_config(
    results,
    removed_empty_rows,
    final_game_count,
    total_sentences,
    total_tokens,
    vocab_size,
    elapsed_seconds,
):
    """
    manifest JSON과 config JSON을 저장한다.

    manifest:
    - job06에서 모델 파일과 source별 벡터 파일을 자동으로 불러오기 좋게 만든 파일

    config:
    - 사람이 설정과 결과를 확인하기 좋게 저장하는 파일
    """
    manifest = {
        "job": "job05_split_word2vec",
        "input_csv": str(INPUT_CSV),
        "index_file": str(OUTPUT_INDEX_CSV),
        "model_dir": str(MODEL_DIR),
        "word2vec_model_file": str(OUTPUT_MODEL),
        "check_csv": str(OUTPUT_CHECK_CSV),
        "text_sources": TEXT_SOURCES,
        "recommendation_weights_example": RECOMMENDATION_WEIGHTS_EXAMPLE,
        "word2vec_params": {
            "vector_size": VECTOR_SIZE,
            "window": WINDOW,
            "min_count": MIN_COUNT,
            "workers": WORKERS,
            "epochs": EPOCHS,
            "sg": SG,
            "seed": SEED,
        },
        "sentence_settings": {
            "sentence_token_chunk_size": SENTENCE_TOKEN_CHUNK_SIZE,
            "min_sentence_tokens": MIN_SENTENCE_TOKENS,
        },
        "removed_empty_model_text_rows": removed_empty_rows,
        "final_game_count": final_game_count,
        "total_sentences": total_sentences,
        "total_tokens": int(total_tokens),
        "vocab_size": int(vocab_size),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "sources": results,
        "note": (
            "Word2Vec 모델은 하나만 학습하고, 게임별 평균 벡터는 source별로 따로 저장한다. "
            "모든 vector npy 파일은 같은 row 순서를 가진다. "
            "row 순서는 index_file의 row 순서와 같다. "
            "job06에서는 추천 방식별로 positive_reviews/tags/categories Word2Vec 유사도 가중치를 다르게 적용한다."
        ),
    }

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    config = {
        "job": "job05_train_word2vec_steam_split_user_rule.py",
        "input_csv": str(INPUT_CSV),
        "outputs": {
            "model": str(OUTPUT_MODEL),
            "index_csv": str(OUTPUT_INDEX_CSV),
            "check_csv": str(OUTPUT_CHECK_CSV),
            "manifest": str(OUTPUT_MANIFEST),
        },
        "word2vec_params": manifest["word2vec_params"],
        "sentence_settings": manifest["sentence_settings"],
        "recommendation_weights_example": RECOMMENDATION_WEIGHTS_EXAMPLE,
        "results": {
            "removed_empty_model_text_rows": removed_empty_rows,
            "final_game_count": final_game_count,
            "total_sentences": total_sentences,
            "total_tokens": int(total_tokens),
            "vocab_size": int(vocab_size),
            "elapsed_seconds": round(elapsed_seconds, 1),
            "sources": results,
        },
    }

    with open(OUTPUT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# ============================================================
# 8. 메인 함수
# ============================================================

def main():
    start_time = time.time()

    print("=" * 80)
    print("job05: Steam split Word2Vec 모델 생성 시작")
    print("=" * 80)

    # --------------------------------------------------------
    # 1) 입력 CSV 읽기
    # --------------------------------------------------------
    print("[1/9] 입력 CSV 읽는 중...")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)

    print("입력 행 수:", len(df))
    print("입력 컬럼 수:", len(df.columns))

    # --------------------------------------------------------
    # 2) 필수 컬럼 확인
    # --------------------------------------------------------
    print("[2/9] 필수 컬럼 확인 중...")

    required_columns = ["appid", "titles"]
    required_columns += [info["column"] for info in TEXT_SOURCES.values()]

    check_required_columns(df, required_columns)

    # --------------------------------------------------------
    # 3) 텍스트 컬럼 정리
    # --------------------------------------------------------
    print("[3/9] 텍스트 컬럼 정리 중...")

    df = clean_text_columns(df)
    df, removed_empty_rows = remove_empty_model_text_rows(df)

    print("빈 model_text 제거:", removed_empty_rows, "행")
    print("Word2Vec 대상 게임 수:", len(df))

    if len(df) == 0:
        raise ValueError("Word2Vec에 사용할 게임 데이터가 없습니다.")

    # --------------------------------------------------------
    # 4) Word2Vec 학습 문장 생성
    # --------------------------------------------------------
    print("[4/9] Word2Vec 학습 문장 생성 중...")

    sentences = build_training_sentences(df)
    total_tokens = sum(len(sentence) for sentence in sentences)

    if not sentences:
        raise ValueError("Word2Vec 학습에 사용할 문장이 없습니다.")

    print("학습 문장 수:", len(sentences))
    print("전체 토큰 수:", total_tokens)
    print("평균 토큰 수/문장:", round(total_tokens / len(sentences), 2))

    print("\n학습 문장 예시:")
    for sentence in sentences[:3]:
        preview = sentence[:30]
        print("-", preview, "..." if len(sentence) > 30 else "")

    # --------------------------------------------------------
    # 5) Word2Vec 단어장 생성 및 학습
    # --------------------------------------------------------
    print("[5/9] Word2Vec 단어장 생성 및 학습 중...")

    print(
        "설정: "
        f"vector_size={VECTOR_SIZE}, window={WINDOW}, min_count={MIN_COUNT}, "
        f"sg={SG}, epochs={EPOCHS}, workers={WORKERS}, seed={SEED}"
    )

    model = Word2Vec(
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=SG,
        seed=SEED,
    )

    # build_vocab:
    #   학습 문장 전체를 보고 Word2Vec 단어장을 만든다.
    #   min_count보다 적게 등장한 단어는 이 단계에서 제외된다.
    model.build_vocab(sentences)

    print("단어장 크기:", len(model.wv.index_to_key))
    print("학습 대상 단어 수(corpus_total_words):", model.corpus_total_words)

    if len(model.wv.index_to_key) == 0:
        raise ValueError(
            "Word2Vec 단어장이 비었습니다. "
            "MIN_COUNT가 너무 높거나 job03 전처리 결과가 비어 있을 수 있습니다."
        )

    # train:
    #   실제 Word2Vec 학습을 수행한다.
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=EPOCHS,
        callbacks=[EpochLogger(EPOCHS)],
    )

    # --------------------------------------------------------
    # 6) source별 게임 평균 벡터 생성
    # --------------------------------------------------------
    print("[6/9] source별 게임 평균 Word2Vec 벡터 생성 중...")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    source_stats = {}

    for source_name, source_info in TEXT_SOURCES.items():
        column_name = source_info["column"]

        print(f"\n[{source_name}] 평균 벡터 생성")
        print("- 대상 컬럼:", column_name)

        vectors, valid_counts, total_counts = build_vectors_for_source(
            df=df,
            source_name=source_name,
            column_name=column_name,
            model=model,
        )

        vector_file = make_vector_file_path(source_name)
        np.save(vector_file, vectors)

        zero_vector_count = int(sum(1 for count in valid_counts if count == 0))

        print("- vector shape:", vectors.shape)
        print("- 0 벡터 게임 수:", zero_vector_count)
        print("- 평균 전체 토큰 수:", round(float(np.mean(total_counts)), 2))
        print("- 평균 유효 토큰 수:", round(float(np.mean(valid_counts)), 2))

        results[source_name] = {
            "column": column_name,
            "description": source_info["description"],
            "vector_file": str(vector_file),
            "vector_shape": list(vectors.shape),
            "zero_vector_count": zero_vector_count,
            "avg_total_token_count": round(float(np.mean(total_counts)), 4),
            "avg_valid_token_count": round(float(np.mean(valid_counts)), 4),
            "min_valid_token_count": int(np.min(valid_counts)),
            "max_valid_token_count": int(np.max(valid_counts)),
        }

        source_stats[source_name] = {
            "valid_token_counts": valid_counts,
            "total_token_counts": total_counts,
        }

    # --------------------------------------------------------
    # 7) 모델과 인덱스 저장
    # --------------------------------------------------------
    print("\n[7/9] Word2Vec 모델과 게임 벡터 인덱스 저장 중...")

    model.save(str(OUTPUT_MODEL))

    index_df = make_game_vector_index(df, source_stats)
    index_df.to_csv(OUTPUT_INDEX_CSV, index=False, encoding="utf-8-sig")

    print("Word2Vec 모델 저장:", OUTPUT_MODEL)
    print("게임 벡터 인덱스 저장:", OUTPUT_INDEX_CSV)
    print("게임 벡터 인덱스 행 수:", len(index_df))
    print("게임 벡터 인덱스 컬럼 수:", len(index_df.columns))

    # --------------------------------------------------------
    # 8) 대표 단어 유사어 점검
    # --------------------------------------------------------
    print("[8/9] 대표 단어 유사어 점검 CSV 저장 중...")

    check_rows = build_check_rows(model, sentences, CHECK_WORDS, topn=10)
    check_df = pd.DataFrame(check_rows)
    check_df.to_csv(OUTPUT_CHECK_CSV, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 9) manifest/config 저장 및 최종 요약
    # --------------------------------------------------------
    print("[9/9] manifest/config 저장 및 최종 요약 출력 중...")

    elapsed = time.time() - start_time

    save_manifest_and_config(
        results=results,
        removed_empty_rows=removed_empty_rows,
        final_game_count=len(df),
        total_sentences=len(sentences),
        total_tokens=total_tokens,
        vocab_size=len(model.wv.index_to_key),
        elapsed_seconds=elapsed,
    )

    print()
    print("=" * 80)
    print("job05 split Word2Vec 완료")
    print("=" * 80)
    print("Word2Vec 모델:", OUTPUT_MODEL)
    print("게임 벡터 인덱스:", OUTPUT_INDEX_CSV)
    print("유사어 점검 CSV:", OUTPUT_CHECK_CSV)
    print("manifest:", OUTPUT_MANIFEST)
    print("config:", OUTPUT_CONFIG)

    print()
    print("source별 벡터 저장 결과:")
    for source_name, result in results.items():
        print(f"\n[{source_name}]")
        print("- column:", result["column"])
        print("- vector shape:", result["vector_shape"])
        print("- vector file:", result["vector_file"])
        print("- 0 vector count:", result["zero_vector_count"])

    print()
    print("추천 방식별 Word2Vec 가중치 예시:")
    print("- 키워드 기반 추천:", RECOMMENDATION_WEIGHTS_EXAMPLE["keyword_based"])
    print("- 특정 게임 기반 추천:", RECOMMENDATION_WEIGHTS_EXAMPLE["similar_game_based"])

    print()
    print("최종 사용 게임 수:", len(df))
    print("최종 학습 문장 수:", len(sentences))
    print("최종 학습 토큰 수:", total_tokens)
    print("최종 단어장 크기:", len(model.wv.index_to_key))

    print_check_result(check_df)

    print()
    print(f"총 경과 시간: {elapsed:.1f}초")


if __name__ == "__main__":
    main()
