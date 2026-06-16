# job07_pyqt5_steam_recommender_user_rule_v11.py
# Steam 게임 리뷰 기반 추천 시스템 - job07 PyQt5 버전 v11
#
# v3 변경사항:
#
# 1. 추천 결과 표 컬럼 변경
#    추천 결과 리스트에는 아래 항목만 표시한다.
#       - appid
#       - 게임명
#       - 출시년도
#       - 장르
#       - 무료 여부
#       - 추천점수
#
# 2. 상세 정보 표시 구조 변경
#    추천 결과 표에 표시하지 않은 세부 정보를 아래처럼 명확하게 나눠서 표시한다.
#       - 측정된 점수 나열
#       - 태그
#       - 카테고리
#       - 플랫폼
#       - 게임 설명
#
# 3. 이미지 로딩 방식 개선
#    기존 QNetworkAccessManager 방식 대신 urllib.request를 사용하는 별도 QThread로 이미지를 불러온다.
#    일부 Ubuntu/PyQt 환경에서 QNetworkAccessManager가 HTTPS 이미지를 제대로 못 가져오는 경우가 있어서,
#    Python 표준 라이브러리 방식으로 변경했다.
#
# 4. header_image 보강
#    추천 결과 row에 header_image가 없거나 비어 있어도,
#    아래 CSV들에서 appid 기준으로 다시 찾아서 이미지를 띄운다.
#
#       ./datasets/steam_game_reviews_preprocessed_user_rule.csv
#       ./datasets/steam_game_review_documents_user_rule_no_mean.csv
#       ./datasets/steam_games_detail_v2.csv
#
#    따라서 job04/job06 index에 header_image가 빠져 있어도 어느 정도 복구 가능하다.
#
# v4 변경사항:
#   게임 상세 정보창을 아래 항목만 표시하도록 정리했다.
#   [점수]
#   - 최종 추천점수
#   - TF-IDF 점수
#   - Word2Vec 점수
#   - 평가 보정 점수
#   [태그]
#   [카테고리]
#   [플랫폼]
#   [게임 설명]
#   게임 설명은 short_description 컬럼을 사용한다.
#
# v5 변경사항:
#   1. 연관 검색어 입력창을 한 줄 입력창(QLineEdit)으로 변경했다.
#   2. 태그 필터를 3개로 늘려 최대 3개 태그를 동시에 필터링할 수 있게 했다.
#      선택한 태그들은 모두 포함해야 하는 AND 조건으로 적용된다.
#   3. 태그 선택 드롭다운이 너무 길게 뜨지 않도록 최대 표시 개수를 제한했다.
#      나머지는 드롭다운 내부 스크롤로 볼 수 있다.
#   4. 상세 설명창 글자를 어두운 색으로 바꾸고 배경을 밝게 해서 시인성을 높였다.
#   5. 페이지 바로가기는 website 컬럼을 무시하고 무조건 Steam Store 페이지로 이동한다.
#
# v6 변경사항:
#   1. 에러창 제목을 error로 통일했다.
#   2. 에러창 배경은 밝게, 글자색은 어둡게 지정해서 시인성을 높였다.
#   3. 태그 선택 드롭다운의 팝업 높이를 제한하고 내부 스크롤바가 나오게 했다.
#   4. 추천 결과 표에 연령 제한(required_age) 열을 추가했다.
#      위치: 무료 여부와 추천점수 사이
#
# v7 변경사항:
#   1. 게임 검색 결과에서 하나를 클릭하면 연관 게임 리스트가 바로 사라지게 했다.
#   2. 선택된 게임 표시 라벨을 가운데 정렬하고, 박스 형태로 더 명확하게 보이게 했다.
#   3. 태그 드롭다운은 QListView를 직접 연결해서 팝업 높이를 더 작게 고정했다.
#   4. 태그 드롭다운 내부 스크롤바를 항상 표시하고, 항목 단위 스크롤로 바꿔 흔들림을 줄였다.
#
# v8 변경사항:
#   1. 태그 선택 UI에서 QComboBox 드롭다운을 완전히 제거했다.
#   2. 태그 선택 버튼을 누르면 고정 크기 태그 선택 창이 뜨는 방식으로 변경했다.
#   3. 태그 선택 창은 검색창 + 리스트 + 확인/초기화 버튼으로 구성된다.
#   4. 따라서 태그 드롭다운의 큰 빈 흰색 영역, 화면 흔들림, 위아래 끝에서 덜덜거리는 문제가 생기지 않는다.
#
# v9 변경사항:
#   1. 좋아하는 게임 입력 부분도 태그 선택과 같은 고정 크기 선택 창 방식으로 변경했다.
#   2. 기존 QLineEdit + 아래 추천 리스트 방식은 제거했다.
#   3. 좋아하는 게임 선택 버튼을 누르면 검색창 + 게임 목록 + 선택/취소 버튼이 있는 창이 열린다.
#   4. 게임 선택 창도 독립 QDialog라서 창 흔들림이나 불필요한 빈 영역 문제가 적다.
#
# v10 변경사항:
#   1. 게임 상세 정보의 [점수] 표시를 정규화 점수와 원본 점수로 분리했다.
#   2. TF-IDF 점수(정규화), TF-IDF 원본 점수, Word2Vec 점수(정규화), Word2Vec 원본 점수를 표시한다.
#   3. job06_recommend_steam_user_rule_normalized.py에서 반환하는
#      tfidf_score_raw, word2vec_score_raw 컬럼을 사용한다.
#
# v11 변경사항:
#   1. 상세 정보 [점수] 표시를 한 줄 요약 방식으로 변경했다.
#      예: TF-IDF 점수: 1.0000 (원본: 0.0832)
#   2. 추천 개수 범위를 UI에 표시했다. 추천 개수: 5 ~ 30
#   3. 최소 출시연도 범위를 UI에 표시했다. 최소 출시연도: 1980 ~ 2026
#   4. 추천 개수는 키보드로 직접 수정하지 못하고 ▲▼ 버튼으로만 조절하게 했다.
#   5. 최소 출시연도는 상하 버튼을 숨겨 더 깔끔하게 표시했다.
#
# 실행 방법:
#   python job07_pyqt5_steam_recommender_user_rule_v11.py


import sys
import re
import urllib.request
import pandas as pd

from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QDesktopServices, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QComboBox,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QMessageBox,
    QSplitter,
    QFrame,
    QPlainTextEdit,
    QListWidget,
    QListWidgetItem,
    QDialog,
    QAbstractSpinBox,
)

from job06_recommend_steam import SteamGameRecommender


# ============================================================
# 1. 추가 정보 보강용 CSV 경로
# ============================================================

DATA_DIR = Path("./datasets")

EXTRA_INFO_CSV_CANDIDATES = [
    DATA_DIR / "steam_game_reviews_preprocessed_user_rule.csv",
    DATA_DIR / "steam_game_review_documents_user_rule_no_mean.csv",
    DATA_DIR / "steam_games_detail_v2.csv",
]


# ============================================================
# 2. 태그 처리 함수
# ============================================================

def split_tag_text(text):
    """
    tags 문자열을 개별 태그 목록으로 나눈다.
    """
    if pd.isna(text):
        return []

    text = str(text).strip()

    if not text:
        return []

    parts = re.split(r"[,;|\n]+", text)

    tags = []

    for part in parts:
        part = part.strip()

        if part:
            tags.append(part)

    return tags


def get_available_tags(index_df):
    """
    전체 게임 index에서 선택 가능한 태그 목록을 만든다.
    """
    if "tags" not in index_df.columns:
        return []

    counter = {}

    for text in index_df["tags"].fillna(""):
        for tag in split_tag_text(text):
            counter[tag] = counter.get(tag, 0) + 1

    tags = sorted(counter.keys(), key=lambda x: (-counter[x], x.lower()))

    return tags


def filter_by_tags(df, selected_tags):
    """
    추천 결과 DataFrame에서 선택한 태그들을 모두 포함한 게임만 남긴다.

    selected_tags:
        ["RPG", "Action", "Singleplayer"]처럼 최대 3개까지 들어올 수 있다.

    필터 방식:
        AND 조건이다.
        즉, 선택한 태그가 2개라면 두 태그를 모두 포함한 게임만 남긴다.
    """
    if "tags" not in df.columns:
        return df

    selected_tags = [
        str(tag).strip()
        for tag in selected_tags
        if str(tag).strip() and str(tag).strip() != "전체 태그"
    ]

    if not selected_tags:
        return df

    tag_text = df["tags"].fillna("").astype(str)

    mask = pd.Series(True, index=df.index)

    for tag in selected_tags:
        mask &= tag_text.str.contains(re.escape(tag), case=False, na=False)

    return df[mask].copy()


# ============================================================

# 3. 표시용 유틸 함수
# ============================================================

def value_to_text(value, default=""):
    """
    NaN 값을 빈 문자열로 바꾸고 화면 표시용 문자열로 변환한다.
    """
    if pd.isna(value):
        return default

    text = str(value)

    if text.lower() == "nan":
        return default

    return text


def format_bool(value):
    """
    True/False 값을 화면 표시용 문자열로 변환한다.
    """
    value_str = str(value).strip().lower()

    if value_str in ["true", "1", "yes", "y"]:
        return "무료" if False else "지원"

    if value_str in ["false", "0", "no", "n", "", "nan"]:
        return "미지원"

    return str(value)


def format_free(value):
    """
    is_free 컬럼을 화면 표시용 문자열로 변환한다.
    """
    value_str = str(value).strip().lower()

    if value_str in ["true", "1", "yes", "y"]:
        return "무료"

    if value_str in ["false", "0", "no", "n", "", "nan"]:
        return "유료"

    return str(value)


def format_age_limit(value):
    """
    required_age 컬럼을 화면 표시용 문자열로 변환한다.

    예:
    - 0, 빈 값 -> 전체 이용 가능
    - 12 -> 12세 이상
    - 19 -> 19세 이상
    """
    if pd.isna(value):
        return "전체 이용 가능"

    value_str = str(value).strip()

    if not value_str or value_str.lower() == "nan":
        return "전체 이용 가능"

    try:
        age = int(float(value_str))
    except Exception:
        return value_str

    if age <= 0:
        return "전체 이용 가능"

    return f"{age}세 이상"


def get_game_url(row):
    """
    게임 상세 페이지 URL을 만든다.

    v5에서는 website 컬럼을 사용하지 않고,
    무조건 appid 기반 Steam Store 페이지로 이동한다.
    """
    appid = row.get("appid", "")

    if pd.notna(appid) and str(appid).strip():
        appid = str(appid).strip()
        return f"https://store.steampowered.com/app/{appid}"

    return ""


def get_first_available_value(row, column_names):
    """
    여러 후보 컬럼 중 처음으로 값이 있는 컬럼 값을 반환한다.
    """
    for col in column_names:
        if col in row.index:
            value = row.get(col, "")

            if pd.notna(value) and str(value).strip():
                return value

    return ""


# ============================================================
# 4. 백그라운드 작업 스레드
# ============================================================

class LoadWorker(QThread):
    """
    추천 모델을 로드하는 스레드다.
    """

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def run(self):
        try:
            recommender = SteamGameRecommender().load()
            self.finished.emit(recommender)
        except Exception as e:
            self.failed.emit(str(e))


class RecommendWorker(QThread):
    """
    추천 계산을 수행하는 스레드다.
    """

    finished = pyqtSignal(object, str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        recommender,
        recommend_mode,
        game_query,
        related_keywords,
        selected_tags,
        top_n,
        min_release_year,
    ):
        super().__init__()

        self.recommender = recommender
        self.recommend_mode = recommend_mode
        self.game_query = game_query
        self.related_keywords = related_keywords
        self.selected_tags = selected_tags
        self.top_n = top_n
        self.min_release_year = min_release_year

    def run(self):
        try:
            candidate_n = max(self.top_n * 8, 80)

            common_kwargs = {
                "top_n": candidate_n,
                "min_release_year": self.min_release_year,
                "platform": None,
                "include_adult": True,
                "only_free": None,
                "min_positive_review_count": None,
                "show_debug_scores": True,
            }

            if self.recommend_mode == "game":
                if not self.game_query.strip():
                    raise ValueError("좋아하는 게임을 입력하거나 검색 결과에서 선택하세요.")

                result_df = self.recommender.recommend_by_game(
                    game_query=self.game_query.strip(),
                    **common_kwargs,
                )
                used_mode = "특정 게임 기반 추천"

            else:
                if not self.related_keywords.strip():
                    raise ValueError("연관 검색어를 입력하세요.")

                result_df = self.recommender.recommend_by_keyword(
                    user_text=self.related_keywords.strip(),
                    **common_kwargs,
                )
                used_mode = "키워드 기반 추천"

            result_df = filter_by_tags(result_df, self.selected_tags)
            result_df = result_df.head(self.top_n).reset_index(drop=True)

            self.finished.emit(result_df, used_mode)

        except Exception as e:
            self.failed.emit(str(e))


class ImageLoadWorker(QThread):
    """
    header_image URL 이미지를 다운로드하는 스레드다.

    QNetworkAccessManager 대신 urllib.request를 사용한다.
    이유:
    - 일부 Ubuntu/PyQt 환경에서 Qt HTTPS 이미지 로딩이 실패할 수 있다.
    - urllib.request는 Python 표준 라이브러리라 비교적 단순하게 동작한다.
    """

    finished = pyqtSignal(str, bytes)
    failed = pyqtSignal(str, str)

    def __init__(self, image_url):
        super().__init__()
        self.image_url = image_url

    def run(self):
        try:
            request = urllib.request.Request(
                self.image_url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (X11; Linux x86_64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0 Safari/537.36"
                    )
                },
            )

            with urllib.request.urlopen(request, timeout=10) as response:
                data = response.read()

            self.finished.emit(self.image_url, data)

        except Exception as e:
            self.failed.emit(self.image_url, str(e))



class TagSelectDialog(QDialog):
    """
    태그를 고르는 고정 크기 팝업 창이다.

    기존 QComboBox 드롭다운은 태그 개수가 많을 때
    팝업 크기와 위치가 자동으로 변하면서 흔들림이 생길 수 있다.

    이 클래스는 독립적인 QDialog를 사용하므로:
    - 팝업 크기가 고정된다.
    - 불필요하게 큰 빈 흰색 영역이 생기지 않는다.
    - 위/아래 끝에서 창이 덜덜거리지 않는다.
    - 리스트는 내부 스크롤로만 움직인다.
    """

    def __init__(self, tags, current_tag="전체 태그", parent=None):
        super().__init__(parent)

        self.tags = ["전체 태그"] + [tag for tag in tags if tag != "전체 태그"]
        self.selected_tag = current_tag if current_tag else "전체 태그"

        self.setWindowTitle("태그 선택")
        self.setModal(True)
        self.setFixedSize(360, 360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("태그 검색")
        self.search_input.textChanged.connect(self.update_list)
        layout.addWidget(self.search_input)

        self.list_widget = QListWidget()
        self.list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_widget.itemDoubleClicked.connect(self.accept_selection)
        layout.addWidget(self.list_widget)

        button_layout = QHBoxLayout()

        self.clear_button = QPushButton("전체 태그")
        self.clear_button.clicked.connect(self.clear_selection)

        self.ok_button = QPushButton("선택")
        self.ok_button.clicked.connect(self.accept_selection)

        self.cancel_button = QPushButton("취소")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setStyleSheet("""
            QDialog {
                background-color: #f8fafc;
            }

            QLabel {
                color: #111827;
            }

            QLineEdit {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #94a3b8;
                padding: 6px;
            }

            QListWidget {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #94a3b8;
                outline: 0px;
            }

            QListWidget::item {
                min-height: 24px;
                padding: 4px 6px;
            }

            QListWidget::item:selected {
                background-color: #2563eb;
                color: #ffffff;
            }


            QSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 22px;
                border-left: 1px solid #94a3b8;
                border-bottom: 1px solid #94a3b8;
                background-color: #e5e7eb;
            }

            QSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 22px;
                border-left: 1px solid #94a3b8;
                background-color: #e5e7eb;
            }

            QSpinBox::up-button:hover,
            QSpinBox::down-button:hover {
                background-color: #cbd5e1;
            }

            QPushButton {
                background-color: #e5e7eb;
                color: #111827;
                border: 1px solid #94a3b8;
                padding: 6px 12px;
                border-radius: 3px;
            }

            QPushButton:hover {
                background-color: #cbd5e1;
            }
        """)

        self.update_list()

    def update_list(self):
        keyword = self.search_input.text().strip().lower()

        self.list_widget.clear()

        for tag in self.tags:
            if keyword and keyword not in tag.lower():
                continue

            item = QListWidgetItem(tag)
            self.list_widget.addItem(item)

            if tag == self.selected_tag:
                item.setSelected(True)
                self.list_widget.scrollToItem(item)

    def clear_selection(self):
        self.selected_tag = "전체 태그"
        self.accept()

    def accept_selection(self):
        item = self.list_widget.currentItem()

        if item is not None:
            self.selected_tag = item.text()

        self.accept()


class TagSelectorButton(QPushButton):
    """
    QComboBox 대신 사용하는 태그 선택 버튼이다.

    버튼을 누르면 TagSelectDialog가 열리고,
    선택된 태그 이름이 버튼 텍스트로 표시된다.
    """

    def __init__(self, label, parent=None):
        super().__init__(label, parent)

        self.tags = []
        self.selected_tag = "전체 태그"

        self.setText("전체 태그")
        self.clicked.connect(self.open_dialog)

    def set_tags(self, tags):
        self.tags = list(tags)

    def currentText(self):
        return self.selected_tag

    def setCurrentIndex(self, index):
        self.selected_tag = "전체 태그"
        self.setText("전체 태그")

    def open_dialog(self):
        dialog = TagSelectDialog(
            tags=self.tags,
            current_tag=self.selected_tag,
            parent=self,
        )

        if dialog.exec_() == QDialog.Accepted:
            self.selected_tag = dialog.selected_tag
            self.setText(self.selected_tag)



class GameSelectDialog(QDialog):
    """
    좋아하는 게임을 고르는 고정 크기 팝업 창이다.

    태그 선택 창과 같은 방식으로 동작한다.

    구조:
    - 게임명 검색 입력창
    - 검색 결과 리스트
    - 선택 해제 / 선택 / 취소 버튼

    검색은 job06의 recommender.search_games()를 사용한다.
    """

    def __init__(self, recommender, current_title="", current_appid="", parent=None):
        super().__init__(parent)

        self.recommender = recommender
        self.selected_title = current_title if current_title else ""
        self.selected_appid = current_appid if current_appid else ""

        self.setWindowTitle("좋아하는 게임 선택")
        self.setModal(True)
        self.setFixedSize(520, 420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        guide_label = QLabel("게임 제목 일부를 입력한 뒤 목록에서 선택하세요.")
        layout.addWidget(guide_label)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("예: Hades, Stardew Valley")
        self.search_input.textChanged.connect(self.update_list)
        layout.addWidget(self.search_input)

        self.list_widget = QListWidget()
        self.list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list_widget.itemDoubleClicked.connect(self.accept_selection)
        layout.addWidget(self.list_widget)

        button_layout = QHBoxLayout()

        self.clear_button = QPushButton("선택 해제")
        self.clear_button.clicked.connect(self.clear_selection)

        self.ok_button = QPushButton("선택")
        self.ok_button.clicked.connect(self.accept_selection)

        self.cancel_button = QPushButton("취소")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setStyleSheet("""
            QDialog {
                background-color: #f8fafc;
            }

            QLabel {
                color: #111827;
                font-size: 13px;
            }

            QLineEdit {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #94a3b8;
                padding: 6px;
            }

            QListWidget {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #94a3b8;
                outline: 0px;
            }

            QListWidget::item {
                min-height: 26px;
                padding: 5px 6px;
            }

            QListWidget::item:selected {
                background-color: #2563eb;
                color: #ffffff;
            }

            QPushButton {
                background-color: #e5e7eb;
                color: #111827;
                border: 1px solid #94a3b8;
                padding: 6px 12px;
                border-radius: 3px;
            }

            QPushButton:hover {
                background-color: #cbd5e1;
            }
        """)

        # 기존에 선택된 게임명이 있으면 검색창에 넣어 바로 후보를 보여준다.
        if self.selected_title:
            self.search_input.setText(self.selected_title)
        else:
            self.update_list()

    def update_list(self):
        keyword = self.search_input.text().strip()

        self.list_widget.clear()

        if not keyword:
            item = QListWidgetItem("검색어를 입력하세요.")
            item.setData(Qt.UserRole, None)
            item.setFlags(Qt.NoItemFlags)
            self.list_widget.addItem(item)
            return

        try:
            result = self.recommender.search_games(keyword, top_n=30)

            if result.empty:
                item = QListWidgetItem("검색 결과 없음")
                item.setData(Qt.UserRole, None)
                item.setFlags(Qt.NoItemFlags)
                self.list_widget.addItem(item)
                return

            for _, row in result.iterrows():
                appid = value_to_text(row.get("appid", ""))
                title = value_to_text(row.get("titles", ""))
                year = value_to_text(row.get("release_year", ""))
                genres = value_to_text(row.get("genres", ""))

                display = f"{title} | 출시 {year} | appid {appid}"

                if genres:
                    display += f" | {genres}"

                item = QListWidgetItem(display)
                item.setData(Qt.UserRole, {
                    "appid": appid,
                    "title": title,
                })

                self.list_widget.addItem(item)

                if self.selected_appid and appid == self.selected_appid:
                    item.setSelected(True)
                    self.list_widget.scrollToItem(item)

        except Exception as e:
            item = QListWidgetItem(f"검색 오류: {e}")
            item.setData(Qt.UserRole, None)
            item.setFlags(Qt.NoItemFlags)
            self.list_widget.addItem(item)

    def clear_selection(self):
        self.selected_title = ""
        self.selected_appid = ""
        self.accept()

    def accept_selection(self):
        item = self.list_widget.currentItem()

        if item is not None:
            data = item.data(Qt.UserRole)

            if isinstance(data, dict):
                self.selected_title = data.get("title", "")
                self.selected_appid = data.get("appid", "")

        self.accept()


class GameSelectorButton(QPushButton):
    """
    좋아하는 게임 선택 버튼이다.

    QComboBox처럼 버튼에 현재 선택된 게임이 표시되고,
    클릭하면 GameSelectDialog가 열린다.
    """

    def __init__(self, parent=None):
        super().__init__("좋아하는 게임 선택", parent)

        self.recommender = None
        self.selected_title = ""
        self.selected_appid = ""

        self.clicked.connect(self.open_dialog)

    def set_recommender(self, recommender):
        self.recommender = recommender

    def clear_selection(self):
        self.selected_title = ""
        self.selected_appid = ""
        self.setText("좋아하는 게임 선택")

    def has_selection(self):
        return bool(self.selected_title or self.selected_appid)

    def game_query(self):
        if self.selected_appid:
            return self.selected_appid

        return self.selected_title

    def open_dialog(self):
        if self.recommender is None:
            return

        dialog = GameSelectDialog(
            recommender=self.recommender,
            current_title=self.selected_title,
            current_appid=self.selected_appid,
            parent=self,
        )

        if dialog.exec_() == QDialog.Accepted:
            self.selected_title = dialog.selected_title
            self.selected_appid = dialog.selected_appid

            if self.selected_title:
                self.setText(self.selected_title)
            else:
                self.setText("좋아하는 게임 선택")


# ============================================================
# 5. 메인 윈도우
# ============================================================

class SteamRecommenderWindow(QMainWindow):
    """
    PyQt5 메인 윈도우 클래스다.
    """

    def __init__(self):
        super().__init__()

        self.recommender = None
        self.tags = []
        self.result_df = pd.DataFrame()
        self.current_selected_row = None

        self.selected_game_query = ""
        self.selected_game_title = ""

        self.load_worker = None
        self.recommend_worker = None
        self.image_worker = None

        self.extra_info_df = pd.DataFrame()

        self.current_image_url = ""

        self.init_window()
        self.init_ui()
        self.apply_style()

        self.start_model_loading()

    # --------------------------------------------------------
    # 5-1. 창 기본 설정
    # --------------------------------------------------------

    def init_window(self):
        self.setWindowTitle("Steam Game Recommender")
        self.resize(1180, 700)
        self.setMinimumSize(1000, 620)

    # --------------------------------------------------------
    # 5-2. UI 생성
    # --------------------------------------------------------
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(26, 22, 26, 22)
        root_layout.setSpacing(18)

        # -------------------------
        # 상단 영역
        # -------------------------
        header_layout = QHBoxLayout()

        title_layout = QVBoxLayout()

        self.title_label = QLabel("Steam 게임 추천")
        self.title_label.setObjectName("TitleLabel")

        self.subtitle_label = QLabel("리뷰 기반 TF-IDF 유사도와 Word2Vec 의미 유사도로 비슷한 게임을 찾습니다.")
        self.subtitle_label.setObjectName("SubtitleLabel")

        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.subtitle_label)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        self.game_count_card = self.create_stat_card("0", "수집된 게임")
        self.tag_count_card = self.create_stat_card("0", "선택 가능 태그")

        header_layout.addWidget(self.game_count_card)
        header_layout.addWidget(self.tag_count_card)

        root_layout.addLayout(header_layout)

        # -------------------------
        # 본문 splitter
        # -------------------------
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(12)

        self.left_panel = self.create_left_panel()
        self.right_panel = self.create_right_panel()

        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setSizes([370, 840])

        root_layout.addWidget(splitter)

        # -------------------------
        # 상태 표시
        # -------------------------
        self.status_label = QLabel("추천 모델을 불러오는 중입니다...")
        self.status_label.setObjectName("StatusLabel")
        root_layout.addWidget(self.status_label)

    def create_stat_card(self, number, label):
        frame = QFrame()
        frame.setObjectName("StatCard")
        frame.setFixedSize(120, 64)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setAlignment(Qt.AlignCenter)

        number_label = QLabel(number)
        number_label.setObjectName("StatNumber")
        number_label.setAlignment(Qt.AlignCenter)

        text_label = QLabel(label)
        text_label.setObjectName("StatText")
        text_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(number_label)
        layout.addWidget(text_label)

        frame.number_label = number_label

        return frame

    def create_left_panel(self):
        group = QGroupBox("추천 조건")
        group.setObjectName("PanelGroup")

        layout = QVBoxLayout(group)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(12)

        help_label = QLabel("좋아하는 게임 또는 연관 검색어 중 하나만 입력하세요.")
        help_label.setObjectName("HelpLabel")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        # ----------------------------------------------------
        # 1. 좋아하는 게임
        # ----------------------------------------------------
        label_game = QLabel("1. 좋아하는 게임")
        label_game.setObjectName("InputTitle")
        layout.addWidget(label_game)

        hint_game = QLabel("버튼을 눌러 게임 제목을 검색하고 선택하세요.")
        hint_game.setObjectName("HelpLabel")
        layout.addWidget(hint_game)

        self.favorite_game_button = GameSelectorButton()
        self.favorite_game_button.setEnabled(False)
        layout.addWidget(self.favorite_game_button)

        # OR 표시
        self.or_label = QLabel("OR")
        self.or_label.setObjectName("OrLabel")
        self.or_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.or_label)

        # ----------------------------------------------------
        # 2. 연관 검색어
        # ----------------------------------------------------
        label_keyword = QLabel("2. 연관 검색어")
        label_keyword.setObjectName("InputTitle")
        layout.addWidget(label_keyword)

        hint_keyword = QLabel("예: 자동화, 오픈월드, 협동, 무료, 로그라이크, 힐링")
        hint_keyword.setObjectName("HelpLabel")
        layout.addWidget(hint_keyword)

        # 한 줄 입력창으로 축소
        self.keyword_input = QLineEdit()
        self.keyword_input.setPlaceholderText("예: 스토리 좋은 로그라이크 보스전 게임")
        layout.addWidget(self.keyword_input)

        # ----------------------------------------------------
        # 3. 태그 필터
        # ----------------------------------------------------
        label_tag = QLabel("3. 태그 필터")
        label_tag.setObjectName("InputTitle")
        layout.addWidget(label_tag)

        hint_tag = QLabel("최대 3개까지 선택할 수 있습니다. 선택한 태그는 모두 포함 조건으로 적용됩니다.")
        hint_tag.setObjectName("HelpLabel")
        hint_tag.setWordWrap(True)
        layout.addWidget(hint_tag)

        self.tag_button_1 = TagSelectorButton("태그 필터 1")
        self.tag_button_2 = TagSelectorButton("태그 필터 2")
        self.tag_button_3 = TagSelectorButton("태그 필터 3")

        self.tag_combos = [
            self.tag_button_1,
            self.tag_button_2,
            self.tag_button_3,
        ]

        for button in self.tag_combos:
            button.setEnabled(False)
            layout.addWidget(button)

        # ----------------------------------------------------
        # 4. 추천 옵션
        # ----------------------------------------------------
        option_grid = QGridLayout()
        option_grid.setHorizontalSpacing(10)
        option_grid.setVerticalSpacing(10)

        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(5, 30)
        self.top_n_spin.setSingleStep(5)
        self.top_n_spin.setValue(10)

        # 추천 개수는 키보드 직접 입력을 막고 ▲▼ 버튼으로만 조절한다.
        self.top_n_spin.lineEdit().setReadOnly(True)
        self.top_n_spin.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
        self.top_n_spin.setFocusPolicy(Qt.NoFocus)

        self.min_year_spin = QSpinBox()
        self.min_year_spin.setRange(1980, 2026)
        self.min_year_spin.setValue(1980)

        # 최소 출시연도는 상하 버튼을 숨겨서 입력칸처럼 깔끔하게 보이게 한다.
        self.min_year_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)

        option_grid.addWidget(QLabel("추천 개수 (5 ~ 30)"), 0, 0)
        option_grid.addWidget(self.top_n_spin, 0, 1)

        option_grid.addWidget(QLabel("최소 출시 연도 (1980 ~ 2026)"), 1, 0)
        option_grid.addWidget(self.min_year_spin, 1, 1)

        layout.addLayout(option_grid)

        # ----------------------------------------------------
        # 버튼
        # ----------------------------------------------------
        self.recommend_button = QPushButton("추천 실행")
        self.recommend_button.setObjectName("PrimaryButton")
        self.recommend_button.clicked.connect(self.run_recommendation)
        self.recommend_button.setEnabled(False)

        self.clear_button = QPushButton("입력 초기화")
        self.clear_button.clicked.connect(self.clear_inputs)

        layout.addWidget(self.recommend_button)
        layout.addWidget(self.clear_button)

        layout.addStretch()

        return group

    def create_right_panel(self):
        group = QGroupBox("추천 결과")
        group.setObjectName("PanelGroup")

        layout = QVBoxLayout(group)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(12)

        self.result_help_label = QLabel("추천 결과 표에는 핵심 정보만 표시하고, 상세 정보는 아래에서 확인합니다.")
        self.result_help_label.setObjectName("HelpLabel")
        layout.addWidget(self.result_help_label)

        self.result_label = QLabel("아직 추천 결과가 없습니다.")
        self.result_label.setObjectName("ResultLabel")
        layout.addWidget(self.result_label)

        # 추천 결과 표
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(7)
        self.result_table.setHorizontalHeaderLabels([
            "appid",
            "게임명",
            "출시 연도",
            "장르",
            "무료 여부",
            "연령 제한",
            "추천점수",
        ])
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.cellClicked.connect(self.on_result_row_clicked)

        header = self.result_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)

        layout.addWidget(self.result_table, stretch=3)

        # 상세 설명 영역
        detail_group = QGroupBox("게임 상세 정보")
        detail_group.setObjectName("DetailGroup")

        detail_layout = QHBoxLayout(detail_group)
        detail_layout.setContentsMargins(14, 18, 14, 14)
        detail_layout.setSpacing(16)

        self.image_label = QLabel("이미지 없음")
        self.image_label.setObjectName("ImageLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(292, 136)
        detail_layout.addWidget(self.image_label)

        detail_text_layout = QVBoxLayout()

        self.detail_title_label = QLabel("게임을 선택하세요.")
        self.detail_title_label.setObjectName("DetailTitle")
        self.detail_title_label.setWordWrap(True)
        detail_text_layout.addWidget(self.detail_title_label)

        self.detail_text = QPlainTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setPlainText("추천 결과에서 게임을 선택하면 상세 설명이 표시됩니다.")
        detail_text_layout.addWidget(self.detail_text)

        button_layout = QHBoxLayout()

        self.open_page_button = QPushButton("Steam 페이지 바로가기")
        self.open_page_button.clicked.connect(self.open_selected_game_page)
        self.open_page_button.setEnabled(False)

        button_layout.addWidget(self.open_page_button)
        button_layout.addStretch()

        detail_text_layout.addLayout(button_layout)

        detail_layout.addLayout(detail_text_layout, stretch=1)

        layout.addWidget(detail_group, stretch=2)

        return group

    # --------------------------------------------------------
    # 5-3. 스타일
    # --------------------------------------------------------
    def apply_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a;
            }

            QWidget {
                color: #e5e7eb;
                font-family: "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "Malgun Gothic", Arial;
                font-size: 13px;
            }

            QLabel#TitleLabel {
                font-size: 32px;
                font-weight: 800;
                color: #ffffff;
            }

            QLabel#SubtitleLabel {
                font-size: 14px;
                color: #93a4bd;
            }

            QLabel#StatusLabel {
                color: #93a4bd;
                font-size: 12px;
            }

            QLabel#HelpLabel {
                color: #9ca3af;
                font-size: 12px;
            }

            QLabel#SelectedGameLabel {
                color: #dbeafe;
                background-color: #0b1220;
                border: 1px solid #334155;
                border-radius: 3px;
                font-size: 12px;
                font-weight: 700;
                padding: 6px;
            }

            QLabel#InputTitle {
                color: #ffffff;
                font-weight: 700;
                font-size: 14px;
            }

            QLabel#ResultLabel {
                color: #cbd5e1;
                font-weight: 700;
            }

            QLabel#DetailTitle {
                color: #ffffff;
                font-size: 20px;
                font-weight: 800;
            }

            QLabel#OrLabel {
                color: #60a5fa;
                font-size: 17px;
                font-weight: 900;
                padding: 8px;
            }

            QGroupBox#PanelGroup,
            QGroupBox#DetailGroup {
                background-color: #111827;
                border: 1px solid #334155;
                border-radius: 4px;
                margin-top: 12px;
                font-size: 19px;
                font-weight: 800;
                color: #ffffff;
            }

            QGroupBox#PanelGroup::title,
            QGroupBox#DetailGroup::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                left: 12px;
            }

            QFrame#StatCard {
                background-color: #1f2937;
                border: 1px solid #334155;
                border-radius: 2px;
            }

            QLabel#StatNumber {
                font-size: 16px;
                font-weight: 800;
                color: #ffffff;
            }

            QLabel#StatText {
                font-size: 11px;
                color: #93a4bd;
            }

            QLineEdit,
            QTextEdit,
            QPlainTextEdit,
            QComboBox,
            QSpinBox,
            QListWidget {
                background-color: #f8fafc;
                color: #111827;
                border: 1px solid #475569;
                padding: 6px;
                selection-background-color: #2563eb;
            }


            QPlainTextEdit {
                background-color: #f8fafc;
                color: #111827;
                border: 1px solid #334155;
            }

            QListWidget#SuggestionList {
                background-color: #0b1220;
                color: #e5e7eb;
                border: 1px solid #334155;
            }

            QListWidget#SuggestionList::item {
                padding: 5px;
            }

            QListWidget#SuggestionList::item:selected {
                background-color: #2563eb;
                color: white;
            }

            QPushButton {
                background-color: #1f2937;
                color: #e5e7eb;
                border: 1px solid #334155;
                padding: 8px 10px;
                border-radius: 3px;
            }

            QPushButton:hover {
                background-color: #334155;
            }

            QPushButton:disabled {
                background-color: #111827;
                color: #64748b;
            }

            QPushButton#PrimaryButton {
                background-color: #2563eb;
                color: #ffffff;
                font-weight: 800;
                border: 1px solid #2563eb;
            }

            QPushButton#PrimaryButton:hover {
                background-color: #1d4ed8;
            }

            QTableWidget {
                background-color: #f8fafc;
                color: #111827;
                gridline-color: #cbd5e1;
                selection-background-color: #bfdbfe;
                selection-color: #111827;
                alternate-background-color: #e5e7eb;
            }

            QHeaderView::section {
                background-color: #dbe4ef;
                color: #111827;
                font-weight: 800;
                padding: 6px;
                border: 1px solid #cbd5e1;
            }

            QLabel#ImageLabel {
                background-color: #0b1220;
                border: 1px solid #334155;
                color: #94a3b8;
            }
        """)

    # --------------------------------------------------------
    # 5-4. 모델 로딩
    # --------------------------------------------------------
    def start_model_loading(self):
        self.set_controls_enabled(False)
        self.status_label.setText("추천 모델을 불러오는 중입니다...")

        self.load_worker = LoadWorker()
        self.load_worker.finished.connect(self.on_model_loaded)
        self.load_worker.failed.connect(self.on_model_load_failed)
        self.load_worker.start()

    def on_model_loaded(self, recommender):
        self.recommender = recommender

        self.extra_info_df = self.load_extra_info_df()

        self.favorite_game_button.set_recommender(self.recommender)

        self.tags = get_available_tags(self.recommender.index_df)

        for button in self.tag_combos:
            button.set_tags(self.tags)
            button.setCurrentIndex(0)

        self.game_count_card.number_label.setText(f"{len(self.recommender.index_df):,}")
        self.tag_count_card.number_label.setText(f"{len(self.tags):,}")

        self.set_controls_enabled(True)
        self.status_label.setText("추천 모델 로드 완료")

    def on_model_load_failed(self, message):
        self.status_label.setText("추천 모델 로드 실패")
        QMessageBox.critical(self, "모델 로드 실패", message)

    def set_controls_enabled(self, enabled):
        self.favorite_game_button.setEnabled(enabled)

        for combo in self.tag_combos:
            combo.setEnabled(enabled)

        self.keyword_input.setEnabled(enabled)
        self.top_n_spin.setEnabled(enabled)
        self.min_year_spin.setEnabled(enabled)
        self.recommend_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)

    def load_extra_info_df(self):
        """
        header_image, short_description, website 등이 job06 결과에 없을 때 보강하기 위해
        원본 CSV 후보를 읽어 appid 기준으로 합친다.
        """
        frames = []

        for path in EXTRA_INFO_CSV_CANDIDATES:
            if not path.exists():
                continue

            try:
                df = pd.read_csv(path, low_memory=False)

                if "appid" not in df.columns:
                    continue

                useful_cols = [
                    "appid",
                    "header_image",
                    "website",
                    "short_description",
                    "genres",
                    "tags",
                    "categories",
                    "is_free",
                    "platform_windows",
                    "platform_mac",
                    "platform_linux",
                    "required_age",
                ]

                existing_cols = [col for col in useful_cols if col in df.columns]

                if len(existing_cols) <= 1:
                    continue

                df = df[existing_cols].copy()
                df["appid"] = df["appid"].astype(str)

                frames.append(df)

            except Exception:
                continue

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames, ignore_index=True)

        # 같은 appid가 여러 파일에 있을 수 있다.
        # 먼저 나온 값이 비어 있으면 뒤에 나온 값으로 채우는 식으로 정리한다.
        result_rows = []

        for appid, group in merged.groupby("appid", sort=False):
            row = {"appid": appid}

            for col in merged.columns:
                if col == "appid":
                    continue

                value = ""

                for v in group[col].tolist():
                    if pd.notna(v) and str(v).strip() and str(v).lower() != "nan":
                        value = v
                        break

                row[col] = value

            result_rows.append(row)

        return pd.DataFrame(result_rows)

    # --------------------------------------------------------
    # 5-5. 결과 row 정보 보강
    # --------------------------------------------------------
    def enrich_row(self, row):
        """
        추천 결과 row에서 누락된 정보를 appid 기준으로 보강한다.

        특히 header_image가 비어 있는 문제를 해결하기 위한 함수다.
        """
        row_dict = row.to_dict()

        appid = str(row_dict.get("appid", "")).strip()

        if not appid:
            return pd.Series(row_dict)

        # 1) recommender.index_df에서 보강
        if self.recommender is not None and "appid" in self.recommender.index_df.columns:
            index_match = self.recommender.index_df[
                self.recommender.index_df["appid"].astype(str) == appid
            ]

            if not index_match.empty:
                index_row = index_match.iloc[0].to_dict()

                for key, value in index_row.items():
                    if key not in row_dict or pd.isna(row_dict.get(key)) or str(row_dict.get(key)).strip() == "":
                        row_dict[key] = value

        # 2) extra_info_df에서 보강
        if not self.extra_info_df.empty and "appid" in self.extra_info_df.columns:
            extra_match = self.extra_info_df[
                self.extra_info_df["appid"].astype(str) == appid
            ]

            if not extra_match.empty:
                extra_row = extra_match.iloc[0].to_dict()

                for key, value in extra_row.items():
                    if key not in row_dict or pd.isna(row_dict.get(key)) or str(row_dict.get(key)).strip() == "":
                        row_dict[key] = value

        return pd.Series(row_dict)

    # --------------------------------------------------------
    # 5-7. 버튼 동작
    # --------------------------------------------------------
    def show_error(self, message):
        """
        에러창을 띄운다.

        기본 QMessageBox는 현재 앱의 어두운 스타일을 상속받으면서
        일부 환경에서 글자색이 거의 보이지 않을 수 있다.
        그래서 에러창만 밝은 배경 + 어두운 글자로 별도 스타일을 적용한다.
        """
        msg = QMessageBox(self)
        msg.setWindowTitle("error")
        msg.setIcon(QMessageBox.Warning)
        msg.setText(str(message))
        msg.setStandardButtons(QMessageBox.Ok)

        msg.setStyleSheet("""
            QMessageBox {
                background-color: #f8fafc;
            }

            QMessageBox QLabel {
                color: #111827;
                font-size: 13px;
                font-weight: 600;
                background-color: #f8fafc;
            }

            QMessageBox QPushButton {
                background-color: #e5e7eb;
                color: #111827;
                border: 1px solid #94a3b8;
                padding: 6px 14px;
                border-radius: 3px;
                min-width: 70px;
            }

            QMessageBox QPushButton:hover {
                background-color: #cbd5e1;
            }
        """)

        msg.exec_()

    def clear_inputs(self):
        self.favorite_game_button.clear_selection()
        self.keyword_input.clear()

        for combo in self.tag_combos:
            combo.setCurrentIndex(0)

        self.top_n_spin.setValue(10)
        self.min_year_spin.setValue(1980)
        self.selected_game_query = ""
        self.selected_game_title = ""

        self.result_df = pd.DataFrame()
        self.result_table.setRowCount(0)
        self.result_label.setText("아직 추천 결과가 없습니다.")
        self.detail_title_label.setText("게임을 선택하세요.")
        self.detail_text.setPlainText("추천 결과에서 게임을 선택하면 상세 설명이 표시됩니다.")
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("이미지 없음")
        self.open_page_button.setEnabled(False)
        self.current_selected_row = None
        self.current_image_url = ""

    def run_recommendation(self):
        if self.recommender is None:
            self.show_error("추천 모델이 아직 로드되지 않았습니다.")
            return

        game_query_from_button = self.favorite_game_button.game_query().strip()
        related_keywords = self.keyword_input.text().strip()

        has_game = bool(game_query_from_button)
        has_keyword = bool(related_keywords)

        if has_game and has_keyword:
            QMessageBox.warning(
                self,
                "입력 확인",
                "좋아하는 게임 또는 연관 검색어 중 하나만 입력하세요.\\n둘 중 하나를 비워주세요.",
            )
            return

        if not has_game and not has_keyword:
            self.show_error(
                "좋아하는 게임 또는 연관 검색어 중 하나를 입력하세요."
            )
            return

        if has_game:
            recommend_mode = "game"
            game_query = game_query_from_button
        else:
            recommend_mode = "keyword"
            game_query = ""

        selected_tags = [
            combo.currentText()
            for combo in self.tag_combos
            if combo.currentText() != "전체 태그"
        ]

        # 중복 태그 제거
        selected_tags = list(dict.fromkeys(selected_tags))

        top_n = self.top_n_spin.value()
        min_release_year = self.min_year_spin.value()

        self.set_controls_enabled(False)
        self.status_label.setText("추천 계산 중입니다...")

        self.recommend_worker = RecommendWorker(
            recommender=self.recommender,
            recommend_mode=recommend_mode,
            game_query=game_query,
            related_keywords=related_keywords,
            selected_tags=selected_tags,
            top_n=top_n,
            min_release_year=min_release_year,
        )

        self.recommend_worker.finished.connect(self.on_recommendation_finished)
        self.recommend_worker.failed.connect(self.on_recommendation_failed)
        self.recommend_worker.start()

    def on_recommendation_finished(self, result_df, used_mode):
        self.set_controls_enabled(True)
        self.status_label.setText("추천 계산 완료")

        if not result_df.empty:
            result_df = result_df.apply(self.enrich_row, axis=1)

        self.result_df = result_df

        if result_df.empty:
            self.result_label.setText(f"{used_mode}: 추천 결과가 없습니다.")
            self.result_table.setRowCount(0)
            return

        self.result_label.setText(f"{used_mode}: {len(result_df)}개 결과")
        self.fill_result_table(result_df)

    def on_recommendation_failed(self, message):
        self.set_controls_enabled(True)
        self.status_label.setText("추천 계산 실패")
        self.show_error(message)

    # --------------------------------------------------------
    # 5-8. 추천 결과 표
    # --------------------------------------------------------
    def fill_result_table(self, df):
        self.result_table.setRowCount(len(df))

        for row_idx, (_, row) in enumerate(df.iterrows()):
            values = [
                value_to_text(row.get("appid", "")),
                value_to_text(row.get("titles", "")),
                value_to_text(row.get("release_year", "")),
                value_to_text(row.get("genres", "")),
                format_free(row.get("is_free", "")),
                format_age_limit(row.get("required_age", "")),
                f"{float(row.get('final_score', 0)):.4f}",
            ]

            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(value)

                if col_idx in [0, 2, 4, 5, 6]:
                    item.setTextAlignment(Qt.AlignCenter)
                else:
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

                self.result_table.setItem(row_idx, col_idx, item)

        self.result_table.resizeRowsToContents()

        if len(df) > 0:
            self.result_table.selectRow(0)
            self.show_detail_for_row(0)

    def on_result_row_clicked(self, row, column):
        self.show_detail_for_row(row)

    # --------------------------------------------------------
    # 5-9. 상세 정보
    # --------------------------------------------------------
    def show_detail_for_row(self, row_index):
        """
        추천 결과에서 선택한 게임의 상세 정보를 표시한다.

        v11 점수 표시 방식:
        [점수]
        - 최종 추천점수
        - TF-IDF 점수: 정규화 점수 (원본: raw 점수)
        - Word2Vec 점수: 정규화 점수 (원본: raw 점수)
        - 평가 보정 점수

        주의:
        - tfidf_score / word2vec_score는 final_score 계산에 사용되는 정규화 점수다.
        - tfidf_score_raw / word2vec_score_raw는 원래 유사도 점수다.
        - raw 컬럼은 job06_recommend_steam_user_rule_normalized.py를 사용해야 표시된다.
        """
        if self.result_df.empty:
            return

        if row_index < 0 or row_index >= len(self.result_df):
            return

        # 추천 결과 row를 가져온 뒤, 누락된 header_image / short_description 등을 보강한다.
        row = self.result_df.iloc[row_index]
        row = self.enrich_row(row)

        self.current_selected_row = row

        title = value_to_text(row.get("titles", "제목 없음"), "제목 없음")
        self.detail_title_label.setText(title)

        final_score = row.get("final_score", None)

        # 정규화 점수
        tfidf_score = row.get("tfidf_score", None)
        word2vec_score = row.get("word2vec_score", None)

        # 원본 점수
        tfidf_score_raw = row.get("tfidf_score_raw", None)
        word2vec_score_raw = row.get("word2vec_score_raw", None)

        # 평가 보정 점수
        review_score = row.get("review_score_for_recommend", None)

        def add_score_line(label, value):
            """
            점수 한 줄을 안전하게 추가한다.

            값이 없거나 숫자로 바꿀 수 없으면 빈 값으로 표시한다.
            """
            if value is None or pd.isna(value):
                lines.append(f"- {label}:")
                return

            try:
                lines.append(f"- {label}: {float(value):.4f}")
            except Exception:
                lines.append(f"- {label}: {value}")

        lines = []

        # ----------------------------------------------------
        # [점수]
        # ----------------------------------------------------
        lines.append("[점수]")
        add_score_line("최종 추천점수", final_score)

        # 정규화 점수와 원본 점수를 한 줄에 같이 표시한다.
        # 예: TF-IDF 점수: 1.0000 (원본: 0.0832)
        def add_score_with_raw_line(label, normalized_value, raw_value):
            normalized_text = ""
            raw_text = ""

            if normalized_value is not None and not pd.isna(normalized_value):
                try:
                    normalized_text = f"{float(normalized_value):.4f}"
                except Exception:
                    normalized_text = str(normalized_value)

            if raw_value is not None and not pd.isna(raw_value):
                try:
                    raw_text = f"{float(raw_value):.4f}"
                except Exception:
                    raw_text = str(raw_value)

            if normalized_text and raw_text:
                lines.append(f"- {label}: {normalized_text} (원본: {raw_text})")
            elif normalized_text:
                lines.append(f"- {label}: {normalized_text}")
            elif raw_text:
                lines.append(f"- {label}: (원본: {raw_text})")
            else:
                lines.append(f"- {label}:")

        add_score_with_raw_line("TF-IDF 점수", tfidf_score, tfidf_score_raw)
        add_score_with_raw_line("Word2Vec 점수", word2vec_score, word2vec_score_raw)
        add_score_line("평가 보정 점수", review_score)

        # ----------------------------------------------------
        # [태그]
        # ----------------------------------------------------
        lines.append("")
        lines.append("[태그]")
        tags = value_to_text(row.get("tags", ""))
        lines.append(tags if tags else "태그 정보가 없습니다.")

        # ----------------------------------------------------
        # [카테고리]
        # ----------------------------------------------------
        lines.append("")
        lines.append("[카테고리]")
        categories = value_to_text(row.get("categories", ""))
        lines.append(categories if categories else "카테고리 정보가 없습니다.")

        # ----------------------------------------------------
        # [플랫폼]
        # ----------------------------------------------------
        lines.append("")
        lines.append("[플랫폼]")

        windows = format_bool(row.get("platform_windows", ""))
        mac = format_bool(row.get("platform_mac", ""))
        linux = format_bool(row.get("platform_linux", ""))

        lines.append(f"- Windows: {windows}")
        lines.append(f"- macOS: {mac}")
        lines.append(f"- Linux: {linux}")

        # ----------------------------------------------------
        # [게임 설명]
        # ----------------------------------------------------
        lines.append("")
        lines.append("[게임 설명]")

        short_description = value_to_text(row.get("short_description", ""))
        lines.append(short_description if short_description else "게임 설명이 없습니다.")

        self.detail_text.setPlainText("\n".join(lines))

        self.open_page_button.setEnabled(bool(get_game_url(row)))

        # header_image URL로부터 이미지를 불러온다.
        image_url = get_first_available_value(
            row,
            ["header_image", "image_url", "capsule_image"],
        )

        self.load_header_image(value_to_text(image_url))

    def load_header_image(self, image_url):
        """
        header_image URL을 이용해 이미지를 비동기로 불러온다.
        """
        self.image_label.setPixmap(QPixmap())

        if not image_url:
            self.current_image_url = ""
            self.image_label.setText("이미지 없음\nheader_image 값이 비어 있습니다.")
            return

        if not image_url.startswith("http://") and not image_url.startswith("https://"):
            self.current_image_url = ""
            self.image_label.setText(f"이미지 URL 형식 오류\n{image_url[:80]}")
            return

        self.current_image_url = image_url
        self.image_label.setText("이미지 불러오는 중...")

        # 기존 이미지 스레드가 실행 중이면 새 요청을 시작하기 전에 참조만 교체한다.
        self.image_worker = ImageLoadWorker(image_url)
        self.image_worker.finished.connect(self.on_image_loaded)
        self.image_worker.failed.connect(self.on_image_failed)
        self.image_worker.start()

    def on_image_loaded(self, image_url, image_data):
        """
        이미지 다운로드 성공 시 QLabel에 표시한다.
        """
        if image_url != self.current_image_url:
            return

        pixmap = QPixmap()

        if not pixmap.loadFromData(image_data):
            self.image_label.setText("이미지 표시 실패\n이미지 데이터를 읽을 수 없습니다.")
            return

        scaled = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.image_label.setPixmap(scaled)
        self.image_label.setText("")

    def on_image_failed(self, image_url, error_message):
        """
        이미지 다운로드 실패 시 원인을 화면에 표시한다.
        """
        if image_url != self.current_image_url:
            return

        self.image_label.setText(
            "이미지 로드 실패\n"
            f"{error_message[:120]}"
        )

    def open_selected_game_page(self):
        if self.current_selected_row is None:
            return

        url = get_game_url(self.current_selected_row)

        if not url:
            QMessageBox.information(self, "바로가기", "열 수 있는 URL이 없습니다.")
            return

        QDesktopServices.openUrl(QUrl(url))


# ============================================================
# 6. 실행
# ============================================================

def main():
    app = QApplication(sys.argv)

    window = SteamRecommenderWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
