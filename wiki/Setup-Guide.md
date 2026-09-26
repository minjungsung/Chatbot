# Setup Guide

## 사전 요구사항

| 도구 | 최소 버전 | 설치 방법 |
|------|----------|----------|
| **Python** | 3.8+ | [python.org](https://www.python.org/) |
| **Java** | 8+ | Komoran 의존성 (konlpy 요구) |
| **pip** | 최신 | Python에 포함 |

> ⚠️ **Java 필수**: konlpy의 Komoran 형태소 분석기는 JVM이 필요합니다.

## 1단계: 클론 및 가상환경

```bash
git clone https://github.com/minjungsung/Chatbot.git
cd Chatbot

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

## 2단계: 의존성 설치

```bash
pip install -r requirements.txt
```

### 주요 의존성

```
konlpy          → 한국어 형태소 분석 (Komoran)
tensorflow      → CNN 의도 분류 모델
sentence-transformers → SBERT 문장 임베딩
torch           → PyTorch (SBERT 백엔드)
numpy, pandas   → 데이터 처리
matplotlib      → 시각화
pillow          → 이미지 처리
h5py            → HDF5 모델 저장
```

### konlpy 설치 (운영체제별)

**macOS:**
```bash
brew install java
pip install konlpy
```

**Ubuntu/Debian:**
```bash
sudo apt-get install default-jdk
pip install konlpy
```

**Windows:**
- JDK 설치 → 환경변수 `JAVA_HOME` 설정
- `pip install konlpy`

## 3단계: 학습 데이터 준비

### 3.1 원본 데이터 다운로드

AI Hub에서 한국어 대화 데이터셋을 다운로드합니다:
- [용도별 목적대화 데이터](https://aihub.or.kr/)
- [주제별 일상 대화 데이터](https://aihub.or.kr/)
- [한국어 위키 QA 데이터](https://aihub.or.kr/)
- [네이버 영화 리뷰 (ratings.txt)](https://github.com/e9t/nsmc)

### 3.2 데이터 디렉토리 구조

```
프로젝트 상위 디렉토리/
├── 원본데이터/
│   ├── 용도별 목적대화 데이터/
│   │   └── TL_01.*/ (JSON 파일들)
│   ├── 주제별 일상 대화 데이터/
│   │   └── TL_01.*/ (JSON 파일들)
│   ├── ko_wiki_v1_squad.json
│   └── ratings.txt
└── Chatbot/
```

### 3.3 데이터 변환

```bash
cd util
python data_to_csv.py
```

변환 결과물:
```
변형데이터/
├── 용도별목적대화데이터.csv
├── 주제별일상대화데이터.csv
├── 일반상식.csv
├── 영화리뷰.csv
└── 통합본데이터.csv
```

## 4단계: 모델 학습

### 4.1 학습 데이터 생성

```bash
cd model/intent
python create_train_data.py
```

→ `train_data.csv` 생성 (text, label 컬럼)

### 4.2 의도 분류 모델 학습

```bash
python train_intent.py
```

학습 파라미터:
| 파라미터 | 값 |
|---------|-----|
| Embedding Size | 128 |
| Conv1D Filters | 128 (kernel 3, 4, 5) |
| Dropout | 0.5 |
| Epochs | 3 |
| Batch Size | 100 |
| 분할 비율 | Train 70% / Val 20% / Test 10% |

학습 결과: `intent_model.h5` 생성

### 4.3 패딩 길이 확인

`create_train_data.py` 실행 시 최적 패딩 길이를 분석합니다:
```
토큰 길이 평균: X.XX
토큰 길이 최대: XX
토큰 길이 표준편차: X.XX
전체 샘플 중 길이가 25 이하인 샘플의 비율: 0.XX
```

## 5단계: 실행

### 사전 준비 파일

학습 후 다음 파일이 필요합니다:

```
train_tools/dict/chatbot_dict.bin  → 단어 인덱스 사전
model/intent/intent_model.h5       → 의도 분류 모델
util/user_dic.tsv                  → 사용자 정의 사전
```

### 챗봇 서버 실행

소켓 서버 기반으로 동작합니다 (`BotServer` 클래스 활용).

```python
from util.bot_server import BotServer
from util.preprocess import Preprocess
from model.intent.intent_model import IntentModel
from util.find_answer import FindAnswer

# 전처리기 초기화
p = Preprocess(word2index_dic='train_tools/dict/chatbot_dict.bin',
               userdic='util/user_dic.tsv')

# 의도 분류 모델 로드
intent = IntentModel('model/intent/intent_model.h5', p)

# 답변 검색기 초기화
answer = FindAnswer(p, df, embedding_data)
```

## 트러블슈팅

### Java 관련 오류

```
JPype: No JVM shared library file (libjvm) found
```

→ `JAVA_HOME` 환경변수 설정 확인:
```bash
export JAVA_HOME=$(/usr/libexec/java_home)  # macOS
```

### SBERT 모델 다운로드

첫 실행 시 `snunlp/KR-SBERT-V40K-klueNLI-augSTS` 모델을 자동 다운로드합니다.
→ 인터넷 연결 필요, 약 400MB

### TensorFlow GPU 사용

```bash
# GPU 버전 설치
pip install tensorflow-gpu
```

### 메모리 부족

대용량 데이터 처리 시 메모리 부족 발생 가능:
```python
import gc
gc.collect()  # 명시적 가비지 컬렉션
```
