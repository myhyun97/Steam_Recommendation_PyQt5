# job06_recommend_steam_user_rule_normalized.py
# 원본 기준: job06_recommend_steam_user_rule.py
# 수정 내용: keyword 기반 model_text 보조 사용 + final_score 계산 전 후보 기준 정규화
# Steam 게임 리뷰 기반 추천 시스템 - job06
#
# 목적:
#   job04 split TF-IDF 결과와 job05 split Word2Vec 결과를 불러와서
#   실제 추천 함수를 만든다.
#
# 지원하는 추천 방식:
#
#   방식 1. 키워드 기반 추천
#       사용자가 직접 문장을 입력한다.
#
#       예:
#           "스토리 좋은 로그라이크 보스전 게임"
#           "힐링 농사 낚시 게임"
#           "공포 생존 멀티 게임"
#
#   방식 2. 특정 게임과 비슷한 게임 추천
#       사용자가 기준 게임명을 입력한다.
#
#       예:
#           "Hades"
#           "Stardew Valley"
#
# ------------------------------------------------------------
# 전체 pipeline에서 job06의 위치
# ------------------------------------------------------------
#
# job01:
#   Steam 데이터 수집
#
# job02:
#   게임별 문서 생성
#   - positive_reviews 생성
#   - all_reviews 생성
#   - negative_reviews 생성
#   - tags, categories, 평가정보 merge
#
# job03:
#   텍스트 전처리
#   - positive_reviews 전처리
#   - tags/categories 전처리
#   - 불용어 후보 추출
#   - 최종 stopwords 적용
#   - model_text 생성
#
# job04:
#   split TF-IDF 모델 생성
#   - positive_reviews TF-IDF
#   - tags TF-IDF
#   - categories TF-IDF
#   - model_text TF-IDF
#
# job05:
#   split Word2Vec 모델 생성
#   - Word2Vec 모델 하나 학습
#   - positive_reviews 평균 벡터 생성
#   - tags 평균 벡터 생성
#   - categories 평균 벡터 생성
#   - model_text 평균 벡터 생성
#
# job06:
#   추천 함수 생성
#   - 입력 텍스트 전처리
#   - TF-IDF 유사도 계산
#   - Word2Vec 유사도 계산
#   - 평가 점수 보정
#   - 출시연도/플랫폼/연령 필터 적용
#
# job07:
#   Streamlit UI 생성
#
# ------------------------------------------------------------
# 이번 job06에서 사용하는 파일
# ------------------------------------------------------------
#
# TF-IDF:
#   ./models/tfidf/tfidf_manifest.json
#
# Word2Vec:
#   ./models/word2vec/word2vec_manifest.json
#
# Stopwords:
#   ./datasets/steam_stopwords.csv
#
# ------------------------------------------------------------
# 추천 점수 구조
# ------------------------------------------------------------
#
# 1. 키워드 기반 추천
#
#   tfidf_score =
#       0.55 * positive_reviews TF-IDF 유사도
#     + 0.35 * tags TF-IDF 유사도
#     + 0.10 * categories TF-IDF 유사도
#
#   word2vec_score =
#       0.65 * positive_reviews Word2Vec 유사도
#     + 0.25 * tags Word2Vec 유사도
#     + 0.10 * categories Word2Vec 유사도
#
#   final_score =
#       0.60 * tfidf_score
#     + 0.20 * word2vec_score
#     + 0.20 * review_score_adjusted
#
#
# 2. 특정 게임 기반 추천
#
#   tfidf_score =
#       0.35 * positive_reviews TF-IDF 유사도
#     + 0.50 * tags TF-IDF 유사도
#     + 0.15 * categories TF-IDF 유사도
#
#   word2vec_score =
#       0.40 * positive_reviews Word2Vec 유사도
#     + 0.45 * tags Word2Vec 유사도
#     + 0.15 * categories Word2Vec 유사도
#
#   final_score =
#       0.45 * tfidf_score
#     + 0.35 * word2vec_score
#     + 0.20 * review_score_adjusted
#
# ------------------------------------------------------------
# 왜 TF-IDF와 Word2Vec을 같이 쓰는가?
# ------------------------------------------------------------
#
# TF-IDF:
#   사용자가 입력한 단어와 게임 문서의 단어가 직접적으로 얼마나 겹치는지 잘 본다.
#
# Word2Vec:
#   단어가 정확히 같지 않아도 문맥상 가까운 단어 관계를 일부 반영할 수 있다.
#
# review_score_adjusted:
#   추천 후보 중 평가가 좋은 게임을 조금 더 위로 올리는 보정 점수다.
#
# release_year / platform / required_age:
#   모델 점수에 넣지 않고, 추천 후보 필터로 사용한다.


import json
import os
import pickle
import re
import html
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse
from gensim.models import Word2Vec


# ============================================================
# 0. Okt 설정
# ============================================================

# job03과 같은 방식으로 사용자 입력도 Okt로 전처리한다.
# Java 메모리 설정은 konlpy import 전에 하는 것이 안전하다.
os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Xmx4g")

from konlpy.tag import Okt


# ============================================================
# 1. 경로 설정
# ============================================================

DATA_DIR = Path("./datasets")

TFIDF_MANIFEST_FILE = Path("./models/tfidf/tfidf_manifest.json")
WORD2VEC_MANIFEST_FILE = Path("./models/word2vec/word2vec_manifest.json")

STOPWORDS_PATH = DATA_DIR / "steam_stopwords.csv"


# ============================================================
# 2. 추천 점수 가중치 설정
# ============================================================

# TF-IDF source별 가중치
TFIDF_SOURCE_WEIGHTS = {
    "keyword": {
        # 키워드 검색은 사용자 입력이 짧기 때문에
        # positive/tags/categories split 점수에 model_text를 보조로 섞는다.
        "positive_reviews": 0.40,
        "tags": 0.35,
        "categories": 0.10,
        "model_text": 0.15,
    },
    "similar_game": {
        # 특정 게임 기반 추천은 기존처럼 split source 중심으로 유지한다.
        "positive_reviews": 0.35,
        "tags": 0.50,
        "categories": 0.15,
    },
}

# Word2Vec source별 가중치
WORD2VEC_SOURCE_WEIGHTS = {
    "keyword": {
        # keyword 기반에서는 model_text 벡터를 보조로 사용한다.
        "positive_reviews": 0.50,
        "tags": 0.30,
        "categories": 0.10,
        "model_text": 0.10,
    },
    "similar_game": {
        # similar_game 기반은 기존 split source 중심 구조 유지
        "positive_reviews": 0.40,
        "tags": 0.45,
        "categories": 0.15,
    },
}

# 최종 점수 가중치
FINAL_SCORE_WEIGHTS = {
    "keyword": {
        "tfidf": 0.60,
        "word2vec": 0.20,
        "review": 0.20,
    },
    "similar_game": {
        "tfidf": 0.45,
        "word2vec": 0.35,
        "review": 0.20,
    },
}


# ============================================================
# 3. 사용자 입력 전처리 설정
# ============================================================

# job03과 같은 품사 기준을 사용한다.
POS_TO_KEEP = {"Noun", "Verb", "Adjective", "Alpha"}

# 1~2글자 영어는 대부분 잡음이지만, 게임 장르 약어는 허용한다.
ALLOWED_SHORT_ENGLISH = {
    "rpg", "fps", "tps", "rts", "mmo", "moba",
    "vr", "ar", "ui", "ux", "ai", "npc",
    "pvp", "pve", "dlc",
}

ALLOWED_SHORT_ALNUM = {
    "2d", "3d", "4x",
}

# job03 코드의 DEFAULT_STOPWORDS와 같은 역할이다.
# 사용자 입력을 전처리할 때도 너무 일반적인 단어는 제거한다.
DEFAULT_STOPWORDS = [
    "그리고", "하지만", "그러나", "그래서", "그런데", "또한", "혹은", "또는",
    "정말", "진짜", "너무", "아주", "매우", "완전", "약간", "조금", "많이", "거의", "계속",
    "그냥", "일단", "뭔가", "어느", "이런", "저런", "그런",
    "이렇게", "저렇게", "그렇게",
    "여기", "저기", "거기", "이거", "저거", "그거", "이것", "저것", "그것",
    "때문", "정도", "느낌", "생각", "사람", "유저", "플레이어", "게이머",
    "경우", "부분", "처음", "마지막", "자체", "하나", "두개", "이번",
    "요즘", "현재", "과거", "다시", "한번", "때", "수", "것", "거",
    "듯", "점", "편", "내", "나", "우리", "저", "제",
    "하다", "되다", "이다", "같다", "보다", "싶다",
    "가다", "오다", "주다", "받다", "만들다", "나오다", "들어가다",
    "해보다", "모르다", "보여주다", "시키다", "버리다", "두다",
    "넣다", "알다", "느끼다",
    "게임", "겜", "스팀", "steam", "플레이", "플레이하다",
]


# ============================================================
# 4. 기본 유틸 함수
# ============================================================

def load_json(path):
    """
    JSON 파일을 읽어 dict로 반환한다.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path):
    """
    pickle 파일을 읽어 Python 객체로 반환한다.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"pickle 파일을 찾을 수 없습니다: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def load_stopwords(path):
    """
    사용자 stopwords CSV와 코드 기본 stopwords를 합쳐 set으로 반환한다.

    job03에서 stopwords를 사용해 model_text를 만들었기 때문에,
    job06에서 사용자 입력을 전처리할 때도 같은 stopwords를 적용하는 것이 좋다.
    """
    stopwords = set(DEFAULT_STOPWORDS)

    path = Path(path)

    if path.exists():
        df_stopwords = pd.read_csv(path)

        if "stopword" not in df_stopwords.columns:
            raise ValueError("stopwords CSV에는 'stopword' 컬럼이 있어야 합니다.")

        file_stopwords = (
            df_stopwords["stopword"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )

        stopwords.update(file_stopwords)

    return {word for word in stopwords if word}


def clean_text(text):
    """
    사용자 입력 문자열을 1차 정리한다.

    job03의 clean_text와 같은 방향이다.

    처리 내용:
    - NaN 방어
    - HTML 특수문자 복원
    - URL 제거
    - 영어 소문자화
    - 한글/영어/숫자만 남김
    - 공백 정리
    """
    if pd.isna(text):
        return ""

    text = str(text)
    text = html.unescape(text)

    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)

    text = text.lower()

    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_noise_token(token):
    """
    토큰 자체가 잡음인지 판단한다.
    """
    if not token:
        return True

    token = str(token).strip().lower()

    if len(token) < 2:
        return True

    # aaa, bbbb 같은 반복 영어 제거
    if re.fullmatch(r"([a-z])\1{2,}", token):
        return True

    if re.search(r"([a-z])\1{3,}", token):
        return True

    # 짧은 영어는 대부분 잡음이지만 게임 약어는 허용
    if re.fullmatch(r"[a-z]{1,2}", token) and token not in ALLOWED_SHORT_ENGLISH:
        return True

    # 2d, 3d, 4x 같은 표현은 허용
    if re.fullmatch(r"[0-9][a-z]", token) and token in ALLOWED_SHORT_ALNUM:
        return False

    return False


def preprocess_user_text(text, okt, stopwords):
    """
    사용자 입력을 job03과 비슷한 방식으로 전처리한다.

    반환:
    - processed_text:
        공백으로 연결된 최종 토큰 문자열

    - tokens:
        최종 토큰 리스트
    """
    cleaned = clean_text(text)

    if not cleaned:
        return "", []

    try:
        pos_result = okt.pos(cleaned, norm=True, stem=True)
    except Exception as e:
        raise RuntimeError(f"Okt 사용자 입력 전처리 실패: {e}")

    tokens = []

    for word, pos in pos_result:
        word = str(word).strip().lower()

        if pos not in POS_TO_KEEP:
            continue

        if is_noise_token(word):
            continue

        if word in stopwords:
            continue

        tokens.append(word)

    return " ".join(tokens), tokens


def normalize_bool_series(series):
    """
    플랫폼 컬럼처럼 True/False가 다양한 형태로 저장된 경우를 bool로 맞춘다.

    처리 예:
    - True, "True", "true", "1", 1 -> True
    - False, "False", "false", "0", 0 -> False
    """
    return (
        series
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def safe_numeric(series, default=np.nan):
    """
    숫자 변환이 안 되는 값을 NaN 또는 default로 처리한다.
    """
    converted = pd.to_numeric(series, errors="coerce")

    if not pd.isna(default):
        converted = converted.fillna(default)

    return converted


def min_max_clip(values, low=0.0, high=1.0):
    """
    점수 배열을 지정 범위로 자른다.
    """
    return np.clip(np.asarray(values, dtype=np.float32), low, high)



def normalize_scores_by_candidates(scores, candidate_mask):
    """
    추천 후보 안에서 점수 배열을 0~1로 재정규화한다.

    왜 필요한가?
    - similar_game 기반은 문서 vs 문서 비교라 TF-IDF 값이 0.4~0.5까지 나올 수 있다.
    - keyword 기반은 짧은 입력 vs 긴 문서 비교라 TF-IDF 값이 0.1 미만으로 나오는 경우가 많다.
    - raw score를 그대로 final_score에 넣으면 keyword 기반에서 text similarity 영향력이 너무 약해진다.

    처리 방식:
    - 현재 후보군(candidate_mask=True) 안의 min/max를 기준으로 min-max 정규화한다.
    - 모든 후보 점수가 같거나 max <= min이면 0으로 처리한다.

    반환:
    - 전체 게임 길이의 numpy array
    - 후보가 아닌 행은 0
    """
    scores = np.asarray(scores, dtype=np.float32)
    mask = np.asarray(candidate_mask, dtype=bool)

    normalized = np.zeros_like(scores, dtype=np.float32)

    if scores.size == 0 or not mask.any():
        return normalized

    candidate_scores = scores[mask]

    min_score = float(np.min(candidate_scores))
    max_score = float(np.max(candidate_scores))

    if max_score <= min_score:
        return normalized

    normalized[mask] = (candidate_scores - min_score) / (max_score - min_score)

    return np.clip(normalized, 0.0, 1.0)


def cosine_dense_matrix(query_vector, matrix):
    """
    dense 벡터 1개와 dense matrix의 코사인 유사도를 계산한다.

    사용 위치:
    - Word2Vec 사용자 입력 벡터 vs 게임별 평균 벡터
    - Word2Vec 기준 게임 벡터 vs 게임별 평균 벡터

    반환:
    - 각 게임에 대한 cosine similarity 배열
    """
    query_vector = np.asarray(query_vector, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)

    query_norm = np.linalg.norm(query_vector)

    if query_norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)

    matrix_norm = np.linalg.norm(matrix, axis=1)

    # 0벡터 게임은 division by zero가 나므로 분모를 안전하게 처리한다.
    safe_matrix_norm = np.where(matrix_norm == 0, 1.0, matrix_norm)

    scores = matrix @ query_vector / (safe_matrix_norm * query_norm)

    # 원래 0벡터였던 게임은 유사도를 0으로 강제한다.
    scores[matrix_norm == 0] = 0.0

    return scores.astype(np.float32)


def convert_word2vec_cosine_to_score(scores):
    """
    Word2Vec cosine similarity는 -1~1 범위가 나올 수 있다.

    최종 점수에서 TF-IDF 점수와 섞기 쉽게 0~1 범위로 변환한다.

    변환:
        -1 -> 0
         0 -> 0.5
         1 -> 1

    단, 0벡터 fallback으로 나온 0점도 0.5가 되면 헷갈릴 수 있다.
    그래서 실제 추천에서 Word2Vec을 계산할 수 없는 경우는 별도로 0 배열을 사용한다.
    """
    scores = np.asarray(scores, dtype=np.float32)
    return np.clip((scores + 1.0) / 2.0, 0.0, 1.0)


def make_word2vec_query_vector(tokens, model):
    """
    사용자 입력 토큰 리스트를 Word2Vec 평균 벡터로 바꾼다.

    Word2Vec 단어장에 있는 토큰만 사용한다.
    사용 가능한 토큰이 하나도 없으면 None을 반환한다.
    """
    vectors = []

    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])

    if not vectors:
        return None

    return np.mean(vectors, axis=0).astype(np.float32)


def sparse_dot_similarity(query_vector, matrix):
    """
    TF-IDF sparse matrix 유사도를 계산한다.

    job04에서 TfidfVectorizer(norm="l2")를 사용했기 때문에
    같은 vectorizer로 변환한 query_vector도 L2 정규화되어 있다.

    따라서 cosine similarity는 단순 dot product와 거의 같다.
    """
    if query_vector.nnz == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)

    scores = matrix @ query_vector.T

    if sparse.issparse(scores):
        scores = scores.toarray().ravel()
    else:
        scores = np.asarray(scores).ravel()

    return scores.astype(np.float32)


# ============================================================
# 5. 추천 엔진 클래스
# ============================================================

class SteamGameRecommender:
    """
    Steam 게임 추천 엔진 클래스.

    이 클래스 하나가 하는 일:
    1. job04 TF-IDF 모델/행렬 로드
    2. job05 Word2Vec 모델/벡터 로드
    3. index row 순서 정렬
    4. 사용자 입력 전처리
    5. 키워드 기반 추천
    6. 특정 게임 기반 추천

    Streamlit UI에서는 이 클래스를 한 번 로드해두고,
    버튼을 누를 때 recommend_by_keyword 또는 recommend_by_game만 호출하면 된다.
    """

    def __init__(
        self,
        tfidf_manifest_file=TFIDF_MANIFEST_FILE,
        word2vec_manifest_file=WORD2VEC_MANIFEST_FILE,
        stopwords_path=STOPWORDS_PATH,
    ):
        self.tfidf_manifest_file = Path(tfidf_manifest_file)
        self.word2vec_manifest_file = Path(word2vec_manifest_file)
        self.stopwords_path = Path(stopwords_path)

        self.tfidf_manifest = None
        self.word2vec_manifest = None

        self.index_df = None

        self.tfidf_vectorizers = {}
        self.tfidf_matrices = {}

        self.word2vec_model = None
        self.word2vec_vectors = {}

        self.stopwords = None
        self.okt = None

    # --------------------------------------------------------
    # 5-1. 전체 로드
    # --------------------------------------------------------
    def load(self):
        """
        추천에 필요한 모든 파일을 로드한다.

        사용 예:
            recommender = SteamGameRecommender()
            recommender.load()
        """
        print("추천 엔진 로드 시작")

        self.stopwords = load_stopwords(self.stopwords_path)
        self.okt = Okt()

        self._load_tfidf_artifacts()
        self._load_word2vec_artifacts()
        self._align_by_appid()

        print("추천 엔진 로드 완료")
        print("최종 게임 수:", len(self.index_df))

        return self

    # --------------------------------------------------------
    # 5-2. TF-IDF 로드
    # --------------------------------------------------------
    def _load_tfidf_artifacts(self):
        """
        job04 split TF-IDF 결과를 로드한다.
        """
        self.tfidf_manifest = load_json(self.tfidf_manifest_file)

        index_file = Path(self.tfidf_manifest["index_file"])

        if not index_file.exists():
            raise FileNotFoundError(f"TF-IDF index 파일을 찾을 수 없습니다: {index_file}")

        self.tfidf_index_df = pd.read_csv(index_file, low_memory=False)

        sources = self.tfidf_manifest["sources"]

        for source_name, info in sources.items():
            model_file = Path(info["model_file"])
            matrix_file = Path(info["matrix_file"])

            vectorizer = load_pickle(model_file)
            matrix = mmread(matrix_file).tocsr()

            self.tfidf_vectorizers[source_name] = vectorizer
            self.tfidf_matrices[source_name] = matrix

        print("TF-IDF 로드 완료")
        print("- source:", list(self.tfidf_vectorizers.keys()))
        print("- index rows:", len(self.tfidf_index_df))

    # --------------------------------------------------------
    # 5-3. Word2Vec 로드
    # --------------------------------------------------------
    def _load_word2vec_artifacts(self):
        """
        job05 split Word2Vec 결과를 로드한다.
        """
        self.word2vec_manifest = load_json(self.word2vec_manifest_file)

        index_file = Path(self.word2vec_manifest["index_file"])
        model_file = Path(self.word2vec_manifest["word2vec_model_file"])

        if not index_file.exists():
            raise FileNotFoundError(f"Word2Vec index 파일을 찾을 수 없습니다: {index_file}")

        if not model_file.exists():
            raise FileNotFoundError(f"Word2Vec 모델 파일을 찾을 수 없습니다: {model_file}")

        self.word2vec_index_df = pd.read_csv(index_file, low_memory=False)
        self.word2vec_model = Word2Vec.load(str(model_file))

        sources = self.word2vec_manifest["sources"]

        for source_name, info in sources.items():
            vector_file = Path(info["vector_file"])

            if not vector_file.exists():
                raise FileNotFoundError(f"Word2Vec vector 파일을 찾을 수 없습니다: {vector_file}")

            self.word2vec_vectors[source_name] = np.load(vector_file)

        print("Word2Vec 로드 완료")
        print("- source:", list(self.word2vec_vectors.keys()))
        print("- index rows:", len(self.word2vec_index_df))

    # --------------------------------------------------------
    # 5-4. TF-IDF / Word2Vec row 정렬
    # --------------------------------------------------------
    def _align_by_appid(self):
        """
        TF-IDF index와 Word2Vec index를 appid 기준으로 맞춘다.

        왜 필요한가?
        - job04와 job05는 같은 입력 파일을 쓰지만,
          중간에 빈 행 제거나 실행 순서 차이로 row 순서가 어긋날 가능성이 있다.
        - 추천 점수를 합치려면 모든 배열의 row가 같은 게임을 가리켜야 한다.

        처리 방식:
        1. TF-IDF index의 appid 순서를 기준으로 삼는다.
        2. Word2Vec index에도 존재하는 appid만 남긴다.
        3. TF-IDF matrix와 Word2Vec vector를 같은 appid 순서로 재정렬한다.
        """
        if "appid" not in self.tfidf_index_df.columns:
            raise ValueError("TF-IDF index에 appid 컬럼이 없습니다.")

        if "appid" not in self.word2vec_index_df.columns:
            raise ValueError("Word2Vec index에 appid 컬럼이 없습니다.")

        tfidf_appids = self.tfidf_index_df["appid"].astype(str).tolist()
        w2v_appids = self.word2vec_index_df["appid"].astype(str).tolist()

        # 이미 완전히 같은 순서라면 그대로 사용한다.
        if tfidf_appids == w2v_appids:
            self.index_df = self.tfidf_index_df.reset_index(drop=True)
            print("TF-IDF / Word2Vec index 순서 일치")
            return

        print("TF-IDF / Word2Vec index 순서가 달라 appid 기준으로 정렬합니다.")

        w2v_pos_by_appid = {appid: i for i, appid in enumerate(w2v_appids)}

        tfidf_keep_indices = []
        w2v_keep_indices = []

        for tfidf_pos, appid in enumerate(tfidf_appids):
            if appid in w2v_pos_by_appid:
                tfidf_keep_indices.append(tfidf_pos)
                w2v_keep_indices.append(w2v_pos_by_appid[appid])

        if not tfidf_keep_indices:
            raise ValueError("TF-IDF와 Word2Vec index 사이에 공통 appid가 없습니다.")

        self.index_df = self.tfidf_index_df.iloc[tfidf_keep_indices].reset_index(drop=True)

        for source_name in list(self.tfidf_matrices.keys()):
            self.tfidf_matrices[source_name] = self.tfidf_matrices[source_name][tfidf_keep_indices]

        for source_name in list(self.word2vec_vectors.keys()):
            self.word2vec_vectors[source_name] = self.word2vec_vectors[source_name][w2v_keep_indices]

        print("appid 기준 정렬 완료")
        print("- 공통 게임 수:", len(self.index_df))

    # --------------------------------------------------------
    # 5-5. 사용자 입력 전처리
    # --------------------------------------------------------
    def preprocess_query(self, user_text):
        """
        사용자 입력을 전처리한다.

        반환:
        - processed_text
        - tokens
        """
        processed_text, tokens = preprocess_user_text(
            text=user_text,
            okt=self.okt,
            stopwords=self.stopwords,
        )

        return processed_text, tokens

    # --------------------------------------------------------
    # 5-6. 필터 적용
    # --------------------------------------------------------
    def _make_filter_mask(
        self,
        min_release_year=None,
        platform=None,
        include_adult=False,
        only_free=None,
        min_positive_review_count=None,
    ):
        """
        추천 후보 필터 mask를 만든다.

        반환:
        - mask: True인 행만 추천 후보로 사용

        필터 종류:
        - min_release_year:
            해당 연도 이상 출시 게임만 추천

        - platform:
            "windows", "mac", "linux" 중 하나 또는 리스트
            예: platform="windows"
            예: platform=["windows", "mac"]

        - include_adult:
            False이면 required_age >= 19 게임 제외

        - only_free:
            True이면 무료 게임만
            False이면 유료 게임만
            None이면 무료/유료 모두 포함

        - min_positive_review_count:
            긍정 리뷰 수가 너무 적은 게임 제외
        """
        df = self.index_df
        mask = pd.Series(True, index=df.index)

        # 출시연도 필터
        if min_release_year is not None and "release_year" in df.columns:
            release_year = safe_numeric(df["release_year"])
            mask &= release_year.fillna(0) >= int(min_release_year)

        # 플랫폼 필터
        if platform is not None:
            if isinstance(platform, str):
                platforms = [platform]
            else:
                platforms = list(platform)

            for p in platforms:
                p = str(p).strip().lower()

                col = f"platform_{p}"

                if col not in df.columns:
                    raise ValueError(f"플랫폼 컬럼이 없습니다: {col}")

                mask &= normalize_bool_series(df[col])

        # 연령 필터
        if not include_adult and "required_age" in df.columns:
            required_age = safe_numeric(df["required_age"])

            # required_age가 비어 있으면 성인 게임이라고 단정할 수 없으므로 통과시킨다.
            mask &= required_age.isna() | (required_age < 19)

        # 무료/유료 필터
        if only_free is not None and "is_free" in df.columns:
            is_free = normalize_bool_series(df["is_free"])

            if only_free:
                mask &= is_free
            else:
                mask &= ~is_free

        # 최소 긍정 리뷰 수 필터
        if min_positive_review_count is not None and "positive_review_count" in df.columns:
            review_count = safe_numeric(df["positive_review_count"], default=0)
            mask &= review_count >= int(min_positive_review_count)

        return mask.to_numpy(dtype=bool)

    # --------------------------------------------------------
    # 5-7. 평가 점수 가져오기
    # --------------------------------------------------------
    def _get_review_score(self):
        """
        최종 추천 점수에 사용할 평가 보정 점수를 가져온다.

        우선순위:
        1. review_score_adjusted
        2. review_score_norm
        3. 없으면 기본값 0.5

        반환:
        - 0~1 범위 numpy array
        """
        df = self.index_df

        if "review_score_adjusted" in df.columns:
            score = safe_numeric(df["review_score_adjusted"], default=0.5)
        elif "review_score_norm" in df.columns:
            score = safe_numeric(df["review_score_norm"], default=0.5)
        else:
            score = pd.Series([0.5] * len(df))

        return min_max_clip(score.fillna(0.5).to_numpy(dtype=np.float32), 0.0, 1.0)

    # --------------------------------------------------------
    # 5-8. source별 점수 합치기
    # --------------------------------------------------------
    def _combine_source_scores(self, source_scores, weights):
        """
        positive_reviews/tags/categories source별 유사도를 가중합한다.

        source_scores 예:
            {
                "positive_reviews": np.array([...]),
                "tags": np.array([...]),
                "categories": np.array([...]),
            }

        weights 예:
            {
                "positive_reviews": 0.55,
                "tags": 0.35,
                "categories": 0.10,
            }
        """
        n = len(self.index_df)
        combined = np.zeros(n, dtype=np.float32)

        for source_name, weight in weights.items():
            if source_name not in source_scores:
                continue

            combined += float(weight) * source_scores[source_name]

        return combined

    # --------------------------------------------------------
    # 5-9. 키워드 기반 TF-IDF 점수
    # --------------------------------------------------------
    def _tfidf_scores_for_query(self, processed_query):
        """
        사용자 입력 문장 기준 TF-IDF 유사도를 source별로 계산한다.

        keyword 기반 추천에서는 model_text도 보조 source로 사용하므로
        여기서는 model_text를 건너뛰지 않는다.
        """
        scores = {}

        for source_name, vectorizer in self.tfidf_vectorizers.items():
            matrix = self.tfidf_matrices[source_name]

            query_vector = vectorizer.transform([processed_query])
            source_score = sparse_dot_similarity(query_vector, matrix)

            scores[source_name] = min_max_clip(source_score, 0.0, 1.0)

        return scores

    # --------------------------------------------------------
    # 5-10. 키워드 기반 Word2Vec 점수
    # --------------------------------------------------------
    def _word2vec_scores_for_query(self, query_tokens):
        """
        사용자 입력 문장 기준 Word2Vec 유사도를 source별로 계산한다.

        keyword 기반 추천에서는 model_text도 보조 source로 사용하므로
        여기서는 model_text를 건너뛰지 않는다.
        """
        scores = {}

        query_vector = make_word2vec_query_vector(query_tokens, self.word2vec_model)

        # 사용자 입력 토큰이 Word2Vec 단어장에 하나도 없으면 Word2Vec 점수는 0으로 둔다.
        if query_vector is None:
            n = len(self.index_df)

            for source_name in self.word2vec_vectors.keys():
                scores[source_name] = np.zeros(n, dtype=np.float32)

            return scores

        for source_name, vectors in self.word2vec_vectors.items():
            cosine_scores = cosine_dense_matrix(query_vector, vectors)

            # Word2Vec cosine은 -1~1이므로 0~1로 변환한다.
            scores[source_name] = convert_word2vec_cosine_to_score(cosine_scores)

        return scores

    # --------------------------------------------------------
    # 5-11. 특정 게임 기반 TF-IDF 점수
    # --------------------------------------------------------
    def _tfidf_scores_for_game(self, base_index):
        """
        기준 게임 index를 기준으로 TF-IDF 유사도를 source별로 계산한다.
        """
        scores = {}

        for source_name, matrix in self.tfidf_matrices.items():
            if source_name == "model_text":
                continue

            base_vector = matrix[base_index]

            source_score = sparse_dot_similarity(base_vector, matrix)

            scores[source_name] = min_max_clip(source_score, 0.0, 1.0)

        return scores

    # --------------------------------------------------------
    # 5-12. 특정 게임 기반 Word2Vec 점수
    # --------------------------------------------------------
    def _word2vec_scores_for_game(self, base_index):
        """
        기준 게임 index를 기준으로 Word2Vec 유사도를 source별로 계산한다.
        """
        scores = {}
        n = len(self.index_df)

        for source_name, vectors in self.word2vec_vectors.items():
            if source_name == "model_text":
                continue

            base_vector = vectors[base_index]

            if np.linalg.norm(base_vector) == 0:
                scores[source_name] = np.zeros(n, dtype=np.float32)
                continue

            cosine_scores = cosine_dense_matrix(base_vector, vectors)
            scores[source_name] = convert_word2vec_cosine_to_score(cosine_scores)

        return scores

    # --------------------------------------------------------
    # 5-13. 최종 점수 계산
    # --------------------------------------------------------
    def _make_final_result(
        self,
        mode,
        tfidf_score,
        word2vec_score,
        filter_mask,
        top_n,
        exclude_indices=None,
        debug_scores=None,
    ):
        """
        TF-IDF 점수, Word2Vec 점수, 평가 점수를 합쳐 최종 추천 결과를 만든다.

        이번 수정 버전의 핵심:
        - raw tfidf_score / raw word2vec_score를 바로 final_score에 넣지 않는다.
        - 필터와 제외 대상이 적용된 후보군 안에서 0~1로 정규화한 뒤 final_score를 계산한다.
        """
        review_score = self._get_review_score()

        candidate_mask = filter_mask.copy()

        if exclude_indices:
            for idx in exclude_indices:
                if 0 <= idx < len(candidate_mask):
                    candidate_mask[idx] = False

        # final_score 계산용 정규화 점수
        tfidf_score_norm = normalize_scores_by_candidates(tfidf_score, candidate_mask)
        word2vec_score_norm = normalize_scores_by_candidates(word2vec_score, candidate_mask)

        weights = FINAL_SCORE_WEIGHTS[mode]

        final_score = (
            weights["tfidf"] * tfidf_score_norm
            + weights["word2vec"] * word2vec_score_norm
            + weights["review"] * review_score
        )

        result_df = self.index_df.copy()

        # UI와 기존 job07 호환을 위해 정규화 점수는 기존 컬럼명에 넣는다.
        result_df["tfidf_score"] = tfidf_score_norm
        result_df["word2vec_score"] = word2vec_score_norm

        # 분석용 raw score도 같이 보존한다.
        result_df["tfidf_score_raw"] = tfidf_score
        result_df["word2vec_score_raw"] = word2vec_score

        result_df["review_score_for_recommend"] = review_score
        result_df["final_score"] = final_score

        if debug_scores:
            for name, values in debug_scores.items():
                result_df[name] = values

        result_df = result_df[candidate_mask].copy()

        result_df = result_df.sort_values(
            by="final_score",
            ascending=False,
        ).head(top_n)

        return self._select_result_columns(result_df)

    # --------------------------------------------------------
    # 5-14. 결과 표시 컬럼 선택
    # --------------------------------------------------------
    def _select_result_columns(self, df):
        """
        추천 결과로 보여줄 컬럼만 추린다.

        UI에서는 이 결과를 카드 형태로 보여주면 된다.
        """
        preferred_columns = [
            "appid",
            "titles",
            "final_score",
            "tfidf_score",
            "word2vec_score",
            "tfidf_score_raw",
            "word2vec_score_raw",
            "review_score_for_recommend",
            "review_score",
            "review_score_adjusted",
            "positive_review_count",
            "release_year",
            "genres",
            "tags",
            "categories",
            "required_age",
            "is_free",
            "platform_windows",
            "platform_mac",
            "platform_linux",
            "short_description",
            "header_image",
        ]

        # debug score 컬럼들도 뒤에 붙인다.
        debug_columns = [
            col for col in df.columns
            if col.startswith("tfidf_") or col.startswith("w2v_")
        ]

        preferred_columns += debug_columns

        # 중복 제거
        seen = set()
        final_columns = []

        for col in preferred_columns:
            if col in df.columns and col not in seen:
                final_columns.append(col)
                seen.add(col)

        return df[final_columns].reset_index(drop=True)

    # --------------------------------------------------------
    # 5-15. 키워드 기반 추천
    # --------------------------------------------------------
    def recommend_by_keyword(
        self,
        user_text,
        top_n=10,
        min_release_year=None,
        platform=None,
        include_adult=False,
        only_free=None,
        min_positive_review_count=None,
        show_debug_scores=True,
    ):
        """
        방식 1. 키워드 기반 추천

        사용 예:
            recommender.recommend_by_keyword(
                "스토리 좋은 로그라이크 보스전 게임",
                platform="windows",
                min_release_year=2015,
                top_n=10,
            )
        """
        processed_query, query_tokens = self.preprocess_query(user_text)

        if not query_tokens:
            raise ValueError(
                "사용자 입력을 전처리한 결과 토큰이 없습니다. "
                "검색어를 조금 더 구체적으로 입력하세요."
            )

        tfidf_source_scores = self._tfidf_scores_for_query(processed_query)
        word2vec_source_scores = self._word2vec_scores_for_query(query_tokens)

        tfidf_score = self._combine_source_scores(
            tfidf_source_scores,
            TFIDF_SOURCE_WEIGHTS["keyword"],
        )

        word2vec_score = self._combine_source_scores(
            word2vec_source_scores,
            WORD2VEC_SOURCE_WEIGHTS["keyword"],
        )

        filter_mask = self._make_filter_mask(
            min_release_year=min_release_year,
            platform=platform,
            include_adult=include_adult,
            only_free=only_free,
            min_positive_review_count=min_positive_review_count,
        )

        debug_scores = None

        if show_debug_scores:
            debug_scores = {}

            for source_name, values in tfidf_source_scores.items():
                debug_scores[f"tfidf_{source_name}_score"] = values

            for source_name, values in word2vec_source_scores.items():
                debug_scores[f"w2v_{source_name}_score"] = values

        result = self._make_final_result(
            mode="keyword",
            tfidf_score=tfidf_score,
            word2vec_score=word2vec_score,
            filter_mask=filter_mask,
            top_n=top_n,
            exclude_indices=None,
            debug_scores=debug_scores,
        )

        # 결과 확인용으로 전처리된 검색어도 속성에 저장해 둔다.
        self.last_processed_query = processed_query
        self.last_query_tokens = query_tokens

        return result

    # --------------------------------------------------------
    # 5-16. 게임 검색
    # --------------------------------------------------------
    def search_games(self, keyword, top_n=20):
        """
        게임명을 검색한다.

        recommend_by_game에서 기준 게임을 찾지 못할 때,
        이 함수를 먼저 사용해서 정확한 제목을 확인하면 좋다.
        """
        if "titles" not in self.index_df.columns:
            raise ValueError("index_df에 titles 컬럼이 없습니다.")

        keyword = str(keyword).strip().lower()

        if not keyword:
            return pd.DataFrame(columns=["appid", "titles"])

        titles = self.index_df["titles"].fillna("").astype(str)
        titles_lower = titles.str.lower()

        if keyword.isdigit() and "appid" in self.index_df.columns:
            appid_match = self.index_df["appid"].astype(str) == keyword
        else:
            appid_match = pd.Series(False, index=self.index_df.index)

        contains_match = titles_lower.str.contains(re.escape(keyword), na=False)

        mask = appid_match | contains_match

        result = self.index_df[mask].copy()

        if result.empty:
            return result[["appid", "titles"]].head(0)

        # 제목이 정확히 일치하는 게임을 위로 올린다.
        result["_exact_title_match"] = titles_lower[mask] == keyword
        result = result.sort_values("_exact_title_match", ascending=False)

        cols = [
            "appid",
            "titles",
            "release_year",
            "genres",
            "tags",
            "platform_windows",
            "platform_mac",
            "platform_linux",
        ]
        cols = [col for col in cols if col in result.columns]

        return result[cols].head(top_n).reset_index(drop=True)

    def _find_game_index(self, game_query):
        """
        게임명 또는 appid로 기준 게임의 index를 찾는다.

        찾는 순서:
        1. appid 정확히 일치
        2. title 정확히 일치
        3. title 부분 포함
        """
        game_query = str(game_query).strip()

        if not game_query:
            raise ValueError("기준 게임명이 비어 있습니다.")

        df = self.index_df

        # 1) appid 정확히 일치
        if "appid" in df.columns and game_query.isdigit():
            appid_mask = df["appid"].astype(str) == game_query

            if appid_mask.any():
                return int(np.where(appid_mask.to_numpy())[0][0])

        if "titles" not in df.columns:
            raise ValueError("index_df에 titles 컬럼이 없습니다.")

        titles = df["titles"].fillna("").astype(str)
        titles_lower = titles.str.lower()
        query_lower = game_query.lower()

        # 2) title 정확히 일치
        exact_mask = titles_lower == query_lower

        if exact_mask.any():
            return int(np.where(exact_mask.to_numpy())[0][0])

        # 3) title 부분 포함
        contains_mask = titles_lower.str.contains(re.escape(query_lower), na=False)

        if contains_mask.any():
            return int(np.where(contains_mask.to_numpy())[0][0])

        candidates = self.search_games(game_query, top_n=10)

        raise ValueError(
            f"기준 게임을 찾지 못했습니다: {game_query}\n"
            f"search_games('{game_query}')로 후보를 먼저 확인하세요.\n"
            f"현재 후보 결과:\n{candidates}"
        )

    # --------------------------------------------------------
    # 5-17. 특정 게임 기반 추천
    # --------------------------------------------------------
    def recommend_by_game(
        self,
        game_query,
        top_n=10,
        min_release_year=None,
        platform=None,
        include_adult=False,
        only_free=None,
        min_positive_review_count=None,
        show_debug_scores=True,
    ):
        """
        방식 2. 특정 게임과 비슷한 게임 추천

        사용 예:
            recommender.recommend_by_game(
                "Hades",
                platform="windows",
                top_n=10,
            )
        """
        base_index = self._find_game_index(game_query)

        base_title = self.index_df.iloc[base_index].get("titles", "")
        print(f"기준 게임: {base_title} | index={base_index}")

        tfidf_source_scores = self._tfidf_scores_for_game(base_index)
        word2vec_source_scores = self._word2vec_scores_for_game(base_index)

        tfidf_score = self._combine_source_scores(
            tfidf_source_scores,
            TFIDF_SOURCE_WEIGHTS["similar_game"],
        )

        word2vec_score = self._combine_source_scores(
            word2vec_source_scores,
            WORD2VEC_SOURCE_WEIGHTS["similar_game"],
        )

        filter_mask = self._make_filter_mask(
            min_release_year=min_release_year,
            platform=platform,
            include_adult=include_adult,
            only_free=only_free,
            min_positive_review_count=min_positive_review_count,
        )

        debug_scores = None

        if show_debug_scores:
            debug_scores = {}

            for source_name, values in tfidf_source_scores.items():
                debug_scores[f"tfidf_{source_name}_score"] = values

            for source_name, values in word2vec_source_scores.items():
                debug_scores[f"w2v_{source_name}_score"] = values

        result = self._make_final_result(
            mode="similar_game",
            tfidf_score=tfidf_score,
            word2vec_score=word2vec_score,
            filter_mask=filter_mask,
            top_n=top_n,
            exclude_indices=[base_index],
            debug_scores=debug_scores,
        )

        return result


# ============================================================
# 6. 실행 예시
# ============================================================

def main():
    """
    job06 단독 테스트용 main 함수다.

    실제 Streamlit UI에서는 이 파일을 import해서 사용하면 된다.

    예:
        from job06_recommend_steam_user_rule import SteamGameRecommender

        recommender = SteamGameRecommender().load()
        result = recommender.recommend_by_keyword("스토리 좋은 로그라이크", top_n=10)
    """
    recommender = SteamGameRecommender().load()

    print("\n" + "=" * 80)
    print("키워드 기반 추천 예시")
    print("=" * 80)

    keyword_result = recommender.recommend_by_keyword(
        user_text="스토리 좋은 로그라이크 보스전 게임",
        top_n=10,
        platform="windows",
        include_adult=False,
        min_positive_review_count=3,
    )

    print("\n전처리된 검색어:", recommender.last_processed_query)
    print("검색 토큰:", recommender.last_query_tokens)
    print(keyword_result)

    print("\n" + "=" * 80)
    print("특정 게임 기반 추천 예시")
    print("=" * 80)

    # 주의:
    #   아래 게임명이 데이터에 없으면 ValueError가 발생할 수 있다.
    #   그럴 경우 recommender.search_games("hades")로 실제 저장된 제목을 먼저 확인한다.
    try:
        similar_result = recommender.recommend_by_game(
            game_query="Hades",
            top_n=10,
            platform="windows",
            include_adult=False,
            min_positive_review_count=3,
        )
        print(similar_result)

    except ValueError as e:
        print("특정 게임 기반 추천 예시 실행 실패:")
        print(e)
        print("\n게임 검색 예시:")
        print(recommender.search_games("hades", top_n=10))


if __name__ == "__main__":
    main()
