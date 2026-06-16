# job03_preprocess_steam_reviews_user_rule.py
# Steam 게임 리뷰 기반 추천 시스템 - job03
#
# 목적:
#   job02에서 만든 "게임 1개 = 문서 1행" CSV를 읽어서
#   TF-IDF / Word2Vec에 사용할 수 있는 전처리 텍스트를 만든다.
#
# 입력 파일:
#   ./datasets/steam_game_review_documents.csv
#
# 출력 파일:
#   ./datasets/steam_game_reviews_preprocessed.csv
#
# 같이 생성되는 파일:
#   ./datasets/steam_stopwords.csv
#       - 최종 불용어 목록
#       - 없으면 기본 파일을 자동 생성한다.
#
#   ./datasets/steam_stopword_candidates.csv
#       - 데이터에서 자주 등장하는 불용어 후보 목록
#       - 자동으로 제거하는 파일이 아니라, 사람이 보고 판단하기 위한 참고 파일이다.
#
# ------------------------------------------------------------
# job03의 역할
# ------------------------------------------------------------
#
# job01:
#   Steam 데이터 수집
#
# job02:
#   게임별 문서 생성
#   - positive_reviews 생성
#   - tags, categories, 평가정보 merge
#
# job03:
#   텍스트 전처리
#   - positive_reviews 전처리
#   - tags 전처리
#   - categories 전처리
#   - 불용어 후보 추출
#   - 최종 stopwords 적용
#
# job04:
#   TF-IDF 모델 생성
#
# job05:
#   Word2Vec 모델 생성
#
# ------------------------------------------------------------
# 이번 코드의 핵심 기준
# ------------------------------------------------------------
#
# 학습에 사용할 원본 텍스트:
#   positive_reviews + tags + categories
#
# 학습에 사용하지 않는 컬럼:
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
# 위 컬럼들은 UI 표시 또는 필터용으로만 보존한다.
#
# ------------------------------------------------------------
# 전처리 방식
# ------------------------------------------------------------
#
# 1. URL 제거
# 2. HTML 특수문자 복원
# 3. 영어 소문자 변환
# 4. 한글, 영어, 숫자만 남김
# 5. Okt.pos(..., stem=True)로 형태소 분석
# 6. 명사 / 동사 / 형용사 / 영어 토큰만 사용
# 7. 불용어 제거
# 8. 너무 짧거나 의미 없는 토큰 제거
#
# stem=True를 쓰는 이유:
#   "재밌었다", "재밌음", "재밌다"처럼 활용된 표현을
#   최대한 기본형에 가깝게 맞춰주기 위해서다.
#
# ------------------------------------------------------------
# 불용어 처리 원칙
# ------------------------------------------------------------
#
# 이 코드는 불용어 후보를 자동으로 stopwords에 넣지 않는다.
#
# 이유:
#   빈도가 높다고 해서 무조건 불용어가 아니기 때문이다.
#
# 예:
#   "스토리", "공포", "멀티", "로그라이크"는 많이 나와도
#   게임 추천에 중요한 단어이므로 제거하면 안 된다.
#
# 따라서:
#   1. steam_stopword_candidates_user_rule.csv를 확인한다.
#   2. 게임 특징과 관련 없는 단어만 steam_stopwords_user_rule.csv에 추가한다.
#   3. job03을 다시 실행한다.


import os
import re
import time
import html
from collections import Counter

import pandas as pd


# ============================================================
# 0. Okt Java 메모리 설정
# ============================================================

# Okt는 내부적으로 Java를 사용한다.
# 긴 리뷰 문서를 처리하면 Java heap memory 오류가 날 수 있으므로
# konlpy import 전에 메모리를 넉넉하게 설정한다.
os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Xmx4g")

from konlpy.tag import Okt


# ============================================================
# 1. 경로 설정
# ============================================================

DATA_DIR = "./datasets"


INPUT_PATH = os.path.join(DATA_DIR, "steam_game_review_documents.csv")

OUTPUT_PATH = os.path.join(DATA_DIR, "steam_game_reviews_preprocessed.csv")

STOPWORDS_PATH = os.path.join(DATA_DIR, "steam_stopwords.csv")

STOPWORD_CANDIDATES_PATH = os.path.join(DATA_DIR, "steam_stopword_candidates.csv")


# ============================================================
# 2. 전처리 설정
# ============================================================

# Okt가 너무 긴 문자열을 한 번에 처리하면 느려지거나 오류가 날 수 있다.
# 그래서 긴 텍스트는 일정 길이 이하의 chunk로 나눠 처리한다.
CHUNK_MAX_LEN = 800

# 몇 개 게임마다 진행 상황을 출력할지 정한다.
PROGRESS_INTERVAL = 50

# Okt 품사 중 추천 모델에 사용할 품사만 남긴다.
#
# Noun:
#   명사. 예: 스토리, 그래픽, 전투, 캐릭터
#
# Verb:
#   동사. 예: 하다, 즐기다, 죽다, 싸우다
#
# Adjective:
#   형용사. 예: 좋다, 어렵다, 쉽다, 재미있다
#
# Alpha:
#   영어. 예: rpg, fps, dlc, roguelike
POS_TO_KEEP = {"Noun", "Verb", "Adjective", "Alpha"}

# 불용어 후보 CSV에 몇 개까지 저장할지 정한다.
# 너무 많이 저장하면 확인하기 어렵기 때문에 상위 500개만 저장한다.
STOPWORD_CANDIDATE_TOP_N = 500

# 문서 등장 비율이 이 값 이상이면 불용어 후보로 강하게 의심할 수 있다.
# 단, 자동 제거하지는 않는다.
HIGH_DOCUMENT_RATIO = 0.30


# ============================================================
# 3. 보호 단어 설정
# ============================================================

# 아래 단어들은 추천에서 중요할 가능성이 높은 단어다.
# stopwords 파일에 들어 있으면 코드가 중단되도록 한다.
#
# 이유:
#   실수로 "스토리", "공포", "멀티" 같은 단어를 불용어로 제거하면
#   추천 품질이 크게 떨어질 수 있기 때문이다.
PROTECTED_WORDS = {
    # 평가 / 감성 관련
    "좋다", "나쁘다", "재미", "재밌다", "재미있다", "재미없다",
    "추천", "비추천", "갓겜", "망겜",

    # 게임 특징 관련
    "스토리", "그래픽", "난이도", "타격감", "몰입",
    "공포", "멀티", "싱글", "협동",
    "힐링", "농사", "생존", "전투", "퍼즐",
    "로그라이크", "오픈월드", "액션", "전략", "건설",
}


# ============================================================
# 4. 기본 불용어 설정
# ============================================================

# 코드 안에서 기본으로 제거할 단어들이다.
# steam_stopwords_user_rule.csv와 합쳐져 최종 stopwords가 된다.
#
# 주의:
#   평가어와 장르/특징 단어는 여기 넣지 않는다.
#   예: 좋다, 재밌다, 스토리, 공포, 멀티 등
DEFAULT_STOPWORDS = [
    # 접속 / 담화 표현
    "그리고", "하지만", "그러나", "그래서", "그런데", "또한", "혹은", "또는",

    # 강조 표현
    "정말", "진짜", "너무", "아주", "매우", "완전", "약간", "조금", "많이", "거의", "계속",

    # 지시 표현
    "그냥", "일단", "뭔가", "어느", "이런", "저런", "그런",
    "이렇게", "저렇게", "그렇게",

    # 대명사 / 지시어
    "여기", "저기", "거기", "이거", "저거", "그거", "이것", "저것", "그것",

    # 너무 일반적인 명사
    "때문", "정도", "느낌", "생각", "사람", "유저", "플레이어", "게이머",
    "경우", "부분", "처음", "마지막", "자체", "하나", "두개", "이번",
    "요즘", "현재", "과거", "다시", "한번", "때", "수", "것", "거",
    "듯", "점", "편", "내", "나", "우리", "저", "제",

    # 너무 일반적인 동사 / 형용사
    # Okt stem=True 후 기본형으로 들어오는 경우가 많다.
    "하다", "되다", "이다", "같다", "보다", "싶다",
    "가다", "오다", "주다", "받다", "만들다", "나오다", "들어가다",
    "해보다", "모르다", "보여주다", "시키다", "버리다", "두다",
    "넣다", "알다", "느끼다",

    # 프로젝트 전체에서 너무 일반적인 단어
    "게임", "겜", "스팀", "steam", "플레이", "플레이하다",
]

# stopwords 파일이 없을 때 자동 생성할 기본 불용어 목록이다.
# 사용자가 이 CSV를 직접 열어 수정할 수 있게 하기 위함이다.
DEFAULT_STOPWORDS_FILE_WORDS = [
    "그리고", "하지만", "그러나", "그래서", "그런데", "또한",
    "정말", "진짜", "너무", "아주", "매우", "완전", "약간", "조금", "많이",
    "그냥", "일단", "뭔가",
    "이거", "저거", "그거", "이것", "저것", "그것",
    "때문", "정도", "느낌", "생각", "사람", "유저", "플레이어", "부분",
    "하다", "되다", "이다", "같다", "보다", "싶다",
    "게임", "겜", "스팀", "steam", "플레이",
]

# 1~2글자 영어는 대부분 잡음이지만,
# 게임 분야에서는 의미 있는 짧은 영어 약어가 있다.
ALLOWED_SHORT_ENGLISH = {
    "rpg", "fps", "tps", "rts", "mmo", "moba",
    "vr", "ar", "ui", "ux", "ai", "npc",
    "pvp", "pve", "dlc",
}

# 숫자와 영어가 섞인 짧은 게임 용어
ALLOWED_SHORT_ALNUM = {
    "2d", "3d", "4x",
}


# ============================================================
# 5. 유틸 함수
# ============================================================

def check_required_columns(df, required_columns):
    """
    필수 컬럼이 있는지 확인한다.

    왜 필요한가?
    - job04에서는 model_text 컬럼을 사용하게 된다.
    - 그런데 job03 입력 파일에 appid, titles, positive_reviews 등이 없으면
      전처리 결과를 제대로 만들 수 없다.
    - 따라서 초반에 명확하게 오류를 내는 것이 좋다.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")


def create_default_stopwords_file(path):
    """
    stopwords CSV 파일이 없을 때 기본 파일을 생성한다.

    생성되는 CSV 형식:
        stopword
        그리고
        하지만
        ...

    사용자는 이 파일을 열어서 불용어를 추가하거나 삭제할 수 있다.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df_stopwords = pd.DataFrame({"stopword": DEFAULT_STOPWORDS_FILE_WORDS})
    df_stopwords.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"기본 불용어 파일을 생성했습니다: {path}")


def load_stopwords(path):
    """
    stopwords CSV와 코드 기본 불용어를 합쳐 set으로 반환한다.

    set을 쓰는 이유:
    - 리스트보다 포함 여부 검사 속도가 빠르다.
    - 중복 단어가 자동으로 하나로 합쳐진다.
    """
    if not os.path.exists(path):
        create_default_stopwords_file(path)

    df_stopwords = pd.read_csv(path)

    if "stopword" not in df_stopwords.columns:
        raise ValueError("불용어 CSV에는 'stopword' 컬럼이 있어야 합니다.")

    file_stopwords = (
        df_stopwords["stopword"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .tolist()
    )

    stopwords = set(file_stopwords + DEFAULT_STOPWORDS)
    stopwords = {word for word in stopwords if word}

    # 중요한 단어가 불용어에 들어갔는지 검사한다.
    # 들어갔다면 조용히 제거하지 않고, 사용자가 직접 고치도록 에러를 낸다.
    wrong_words = PROTECTED_WORDS & stopwords

    if wrong_words:
        raise ValueError(
            "중요 추천 단어가 불용어에 들어가 있습니다. "
            f"steam_stopwords_user_rule.csv 또는 DEFAULT_STOPWORDS에서 제거하세요: {sorted(wrong_words)}"
        )

    return stopwords


def clean_text(text):
    """
    원본 문자열에서 모델 학습에 방해되는 요소를 1차 정리한다.

    처리 내용:
    1. NaN이면 빈 문자열 처리
    2. HTML 특수문자 복원
    3. URL 제거
    4. 영어 소문자 변환
    5. 한글, 영어, 숫자, 공백만 남김
    6. 중복 공백 제거

    숫자를 남기는 이유:
    - 2D, 3D, 4X 같은 게임 용어가 의미 있을 수 있기 때문이다.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # &amp; 같은 HTML 특수문자를 원래 문자로 복원한다.
    text = html.unescape(text)

    # URL 제거
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)

    # 영어는 소문자로 통일한다.
    # RPG, rpg, Rpg를 같은 단어로 보기 위함이다.
    text = text.lower()

    # 한글, 영어, 숫자, 공백만 남긴다.
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)

    # 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_text_by_length(text, max_len=CHUNK_MAX_LEN):
    """
    긴 텍스트를 Okt가 처리 가능한 길이로 나눈다.

    왜 필요한가?
    - 리뷰를 게임별로 합치면 한 게임의 문서가 매우 길어질 수 있다.
    - Okt가 너무 긴 문자열을 한 번에 처리하면 느리거나 오류가 날 수 있다.
    - 그래서 단어 단위로 잘라 chunk 목록을 만든다.
    """
    if not text:
        return []

    words = text.split()
    chunks = []
    current_words = []
    current_len = 0

    for word in words:
        word_len = len(word) + 1

        # 단어 하나가 너무 길면 강제로 잘라서 넣는다.
        if len(word) > max_len:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
                current_len = 0

            for i in range(0, len(word), max_len):
                chunks.append(word[i:i + max_len])

            continue

        # 현재 chunk에 단어를 추가하면 max_len을 넘는 경우
        # 기존 chunk를 저장하고 새 chunk를 시작한다.
        if current_len + word_len > max_len:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_len = word_len
        else:
            current_words.append(word)
            current_len += word_len

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def is_noise_token(token):
    """
    토큰 자체가 의미 없는 잡음인지 판단한다.

    여기서는 stopwords를 보지 않는다.
    순수하게 형식적인 문제만 검사한다.
    """
    if not token:
        return True

    token = str(token).strip().lower()

    # 한 글자 토큰은 대부분 의미가 약하다.
    if len(token) < 2:
        return True

    # aaa, bbbb처럼 같은 영어 글자 반복은 잡음인 경우가 많다.
    if re.fullmatch(r"([a-z])\1{2,}", token):
        return True

    # aaaa가 포함된 긴 영어 반복도 제거한다.
    if re.search(r"([a-z])\1{3,}", token):
        return True

    # 너무 짧은 영어는 대부분 잡음이다.
    # 단, rpg/fps/vr/ui 같은 게임 약어는 허용한다.
    if re.fullmatch(r"[a-z]{1,2}", token) and token not in ALLOWED_SHORT_ENGLISH:
        return True

    # 2d, 3d, 4x 같은 게임 용어는 허용한다.
    if re.fullmatch(r"[0-9][a-z]", token) and token in ALLOWED_SHORT_ALNUM:
        return False

    return False


def tokenize_without_stopwords(text, okt):
    """
    stopwords 적용 전 토큰을 만든다.

    이 함수의 목적:
    - 불용어 후보를 뽑기 위한 원본 토큰 목록 생성
    - 이후 stopwords를 적용해 최종 토큰을 만들기 위한 중간 단계

    여기서는 stopwords를 제거하지 않는다.
    단, 너무 짧거나 명백한 잡음 토큰은 제거한다.
    """
    cleaned = clean_text(text)
    chunks = split_text_by_length(cleaned)

    tokens = []

    for chunk in chunks:
        if not chunk:
            continue

        try:
            # norm=True:
            #   반복 문자나 일부 표현을 정규화한다.
            #
            # stem=True:
            #   동사/형용사를 기본형에 가깝게 변환한다.
            pos_result = okt.pos(chunk, norm=True, stem=True)

        except Exception as e:
            print("[경고] Okt 처리 실패. 해당 chunk는 건너뜁니다:", e)
            continue

        for word, pos in pos_result:
            word = str(word).strip().lower()

            # 사용할 품사만 남긴다.
            if pos not in POS_TO_KEEP:
                continue

            # 형식상 잡음인 토큰은 제거한다.
            if is_noise_token(word):
                continue

            tokens.append(word)

    return tokens


def apply_stopwords(tokens, stopwords):
    """
    stopwords를 적용해 최종 토큰을 만든다.

    이 단계에서 제거하는 것:
    - stopwords에 들어 있는 단어

    이 단계에서 남기는 것:
    - 게임 특징, 장르, 평가에 도움이 되는 단어
    """
    result = []

    for token in tokens:
        token = str(token).strip().lower()

        if not token:
            continue

        if token in stopwords:
            continue

        result.append(token)

    return result


def make_source_text(row):
    """
    job03에서 사용할 원본 학습 텍스트를 만든다.

    우선순위:
    1. positive_reviews
    2. tags
    3. categories

    short_description, genres는 일부러 넣지 않는다.
    - short_description: UI 표시용
    - genres: UI 표시용
    """
    positive_reviews = str(row.get("positive_reviews", "") or "")
    tags = str(row.get("tags", "") or "")
    categories = str(row.get("categories", "") or "")

    return positive_reviews + "\n" + tags + "\n" + categories


def save_stopword_candidates(token_total_counter, token_doc_counter, total_docs, stopwords):
    """
    불용어 후보 CSV를 저장한다.

    저장 기준:
    - 전체 토큰 빈도 기준 상위 STOPWORD_CANDIDATE_TOP_N개

    저장 컬럼:
    - token: 단어
    - total_count: 전체 등장 횟수
    - document_count: 몇 개 게임 문서에서 등장했는지
    - document_ratio: 전체 게임 중 등장 비율
    - already_stopword: 현재 stopwords에 들어 있는지
    - protected_word: 보호 단어인지
    - candidate_reason: 후보로 의심되는 이유
    """
    rows = []

    for token, total_count in token_total_counter.most_common(STOPWORD_CANDIDATE_TOP_N):
        document_count = token_doc_counter.get(token, 0)
        document_ratio = document_count / total_docs if total_docs > 0 else 0

        reasons = []

        if document_ratio >= HIGH_DOCUMENT_RATIO:
            reasons.append("many_documents")

        if token in stopwords:
            reasons.append("already_stopword")

        if token in PROTECTED_WORDS:
            reasons.append("protected_do_not_remove")

        if not reasons:
            reasons.append("high_total_count")

        rows.append({
            "token": token,
            "total_count": total_count,
            "document_count": document_count,
            "document_ratio": round(document_ratio, 4),
            "already_stopword": token in stopwords,
            "protected_word": token in PROTECTED_WORDS,
            "candidate_reason": ", ".join(reasons),
        })

    df_candidates = pd.DataFrame(rows)
    df_candidates.to_csv(STOPWORD_CANDIDATES_PATH, index=False, encoding="utf-8-sig")


# ============================================================
# 6. 메인 함수
# ============================================================

def main():
    start_time = time.time()

    print("=" * 80)
    print("job03: Steam 리뷰 텍스트 전처리 시작")
    print("=" * 80)

    # --------------------------------------------------------
    # 1) 입력 파일 읽기
    # --------------------------------------------------------
    print("[1/7] 입력 CSV 읽는 중...")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    print("입력 행 수:", len(df))
    print("입력 컬럼 수:", len(df.columns))

    # --------------------------------------------------------
    # 2) 필수 컬럼 확인
    # --------------------------------------------------------
    print("[2/7] 필수 컬럼 확인 중...")

    required_columns = [
        "appid",
        "titles",
        "positive_reviews",
        "tags",
        "categories",
    ]

    check_required_columns(df, required_columns)

    # --------------------------------------------------------
    # 3) 불용어 파일 준비
    # --------------------------------------------------------
    print("[3/7] 불용어 읽는 중...")

    stopwords = load_stopwords(STOPWORDS_PATH)

    print("불용어 수:", len(stopwords))
    print("중요 단어 보호 검사: 통과")

    # --------------------------------------------------------
    # 4) Okt 준비
    # --------------------------------------------------------
    print("[4/7] Okt 준비 중...")

    okt = Okt()

    # --------------------------------------------------------
    # 5) 게임별 전처리
    # --------------------------------------------------------
    print("[5/7] 게임별 텍스트 전처리 중...")

    positive_reviews_processed_list = []
    tags_processed_list = []
    categories_processed_list = []
    model_text_list = []
    token_count_list = []

    # 불용어 후보 추출을 위한 카운터
    token_total_counter = Counter()
    token_doc_counter = Counter()

    for idx, row in df.iterrows():
        title = row.get("titles", "")

        # ----------------------------------------------------
        # 5-1) 컬럼별 원본 텍스트 가져오기
        # ----------------------------------------------------
        positive_reviews = str(row.get("positive_reviews", "") or "")
        tags = str(row.get("tags", "") or "")
        categories = str(row.get("categories", "") or "")

        # ----------------------------------------------------
        # 5-2) stopwords 적용 전 토큰화
        # ----------------------------------------------------
        # 불용어 후보를 뽑으려면 stopwords 적용 전 토큰도 필요하다.
        raw_positive_tokens = tokenize_without_stopwords(positive_reviews, okt)
        raw_tags_tokens = tokenize_without_stopwords(tags, okt)
        raw_categories_tokens = tokenize_without_stopwords(categories, okt)

        raw_all_tokens = raw_positive_tokens + raw_tags_tokens + raw_categories_tokens

        # ----------------------------------------------------
        # 5-3) 불용어 후보 통계 누적
        # ----------------------------------------------------
        # 전체 등장 횟수
        token_total_counter.update(raw_all_tokens)

        # 문서 등장 횟수
        # 한 게임 문서 안에서 같은 단어가 여러 번 나와도 document_count는 1번만 증가시킨다.
        unique_tokens_in_doc = set(raw_all_tokens)
        token_doc_counter.update(unique_tokens_in_doc)

        # ----------------------------------------------------
        # 5-4) 최종 stopwords 적용
        # ----------------------------------------------------
        final_positive_tokens = apply_stopwords(raw_positive_tokens, stopwords)
        final_tags_tokens = apply_stopwords(raw_tags_tokens, stopwords)
        final_categories_tokens = apply_stopwords(raw_categories_tokens, stopwords)

        # ----------------------------------------------------
        # 5-5) 컬럼별 전처리 결과 문자열 생성
        # ----------------------------------------------------
        positive_reviews_processed = " ".join(final_positive_tokens)
        tags_processed = " ".join(final_tags_tokens)
        categories_processed = " ".join(final_categories_tokens)

        # job04에서 TF-IDF 입력으로 사용할 최종 텍스트
        # 기준:
        #   positive_reviews + tags + categories
        model_tokens = final_positive_tokens + final_tags_tokens + final_categories_tokens
        model_text = " ".join(model_tokens)

        positive_reviews_processed_list.append(positive_reviews_processed)
        tags_processed_list.append(tags_processed)
        categories_processed_list.append(categories_processed)
        model_text_list.append(model_text)
        token_count_list.append(len(model_tokens))

        # ----------------------------------------------------
        # 5-6) 진행 상황 출력
        # ----------------------------------------------------
        if (idx + 1) % PROGRESS_INTERVAL == 0 or (idx + 1) == len(df):
            elapsed = time.time() - start_time
            print(
                f"진행: {idx + 1}/{len(df)} | "
                f"현재 게임: {title} | "
                f"최종 토큰 수: {len(model_tokens)} | "
                f"경과: {elapsed:.1f}초"
            )

    # --------------------------------------------------------
    # 6) 결과 컬럼 추가 및 토큰 0개 행 제거
    # --------------------------------------------------------
    print("[6/7] 전처리 결과 정리 중...")

    df["positive_reviews_processed"] = positive_reviews_processed_list
    df["tags_processed"] = tags_processed_list
    df["categories_processed"] = categories_processed_list

    # job04에서 사용할 핵심 컬럼
    df["model_text"] = model_text_list
    df["model_token_count"] = token_count_list

    before_count = len(df)

    # model_text가 비어 있는 게임은 TF-IDF/Word2Vec에 사용할 수 없으므로 제거한다.
    df = df[df["model_token_count"] > 0].copy()

    print("토큰 0개 게임 제거:", before_count - len(df), "행")

    # --------------------------------------------------------
    # 7) 결과 저장
    # --------------------------------------------------------
    print("[7/7] 결과 저장 중...")

    os.makedirs(DATA_DIR, exist_ok=True)

    # 전처리 결과 저장
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # 불용어 후보 저장
    save_stopword_candidates(
        token_total_counter=token_total_counter,
        token_doc_counter=token_doc_counter,
        total_docs=before_count,
        stopwords=stopwords,
    )

    elapsed = time.time() - start_time

    print()
    print("=" * 80)
    print("job03 완료")
    print("=" * 80)
    print("전처리 결과 파일:", OUTPUT_PATH)
    print("불용어 파일:", STOPWORDS_PATH)
    print("불용어 후보 파일:", STOPWORD_CANDIDATES_PATH)
    print("최종 행 수:", len(df))
    print("최종 컬럼 수:", len(df.columns))

    print()
    print("토큰 수 요약")
    print("평균 토큰 수:", round(df["model_token_count"].mean(), 2))
    print("최소 토큰 수:", int(df["model_token_count"].min()))
    print("최대 토큰 수:", int(df["model_token_count"].max()))

    print()
    print("전처리 결과 예시")
    preview_cols = [
        "appid",
        "titles",
        "positive_review_count",
        "model_token_count",
        "model_text",
    ]
    preview_cols = [col for col in preview_cols if col in df.columns]
    print(df[preview_cols].head())

    print()
    print(f"총 경과 시간: {elapsed:.1f}초")


if __name__ == "__main__":
    main()
