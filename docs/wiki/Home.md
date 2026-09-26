# Chatbot

한국어 자연어 처리(NLP) 기반 챗봇 시스템입니다. Komoran 형태소 분석기, CNN 기반 의도 분류 모델, SBERT(Sentence-BERT) 기반 유사도 검색을 결합하여 사용자의 질문에 적절한 답변을 제공합니다.

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **언어** | Python 3.x |
| **형태소 분석** | Komoran (konlpy) |
| **의도 분류** | TensorFlow/Keras CNN 모델 |
| **답변 검색** | Sentence-BERT (KR-SBERT) |
| **데이터** | 한국어 대화 데이터셋 (AI Hub) |
| **통신** | 소켓 기반 서버-클라이언트 |

## 주요 기능

1. **의도 분류 (Intent Classification)**
   - CNN 모델로 사용자 입력의 의도를 분류 (번호/장소/시간)
   - 형태소 분석 → 불용어 제거 → 단어 인덱스 변환 → CNN 예측

2. **답변 검색 (Answer Retrieval)**
   - KR-SBERT 모델로 질문 임베딩 생성
   - 코사인 유사도 기반 가장 유사한 질문-답변 쌍 검색
   - 유사도 점수 및 이미지 URL 반환

3. **형태소 전처리 (Preprocessing)**
   - Komoran 형태소 분석기로 품사 태깅
   - 관계언, 기호, 어미, 접미사 등 불용어 자동 제거
   - 사용자 정의 사전(`user_dic.tsv`) 지원

4. **학습 데이터 생성**
   - AI Hub 한국어 대화 데이터셋 변환 (JSON → CSV)
   - 영화 리뷰, 일반상식, 용도별/주제별 대화 데이터 통합

## 기술 스택 상세

```
konlpy (Komoran)               → 한국어 형태소 분석
TensorFlow / Keras             → CNN 의도 분류 모델
sentence-transformers (SBERT)  → 문장 임베딩 & 유사도 검색
torch (PyTorch)                → 텐서 연산 & SBERT 백엔드
numpy                          → 수치 연산
pandas                         → 데이터 처리
scikit-learn                   → 벡터 유사도 (cosine)
h5py                           → 모델 저장 (.h5)
pickle                         → 사전 직렬화
PIL (Pillow)                   → 이미지 전처리
matplotlib                     → 데이터 시각화
```

## 프로젝트 구조

```
Chatbot/
├── model/
│   └── intent/
│       ├── intent_model.py        # 의도 분류 모델 (추론)
│       ├── train_intent.py        # 의도 분류 모델 (학습)
│       └── create_train_data.py   # 학습 데이터 생성
│
├── util/
│   ├── preprocess.py              # 형태소 분석 & 전처리
│   ├── find_answer.py             # SBERT 기반 답변 검색
│   ├── bot_server.py              # 소켓 서버
│   ├── data_to_csv.py             # 원본 데이터 → CSV 변환
│   └── image_resize.py            # 이미지 전처리
│
└── requirements.txt               # Python 의존성
```

## NLP 파이프라인 요약

```
사용자 입력
    │
    ▼
┌─────────────────────────┐
│ 1. 형태소 분석 (Komoran) │
│    품사 태깅 + 불용어 제거 │
└───────────┬─────────────┘
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
┌──────────┐  ┌──────────────┐
│ 2. CNN   │  │ 3. SBERT     │
│ 의도 분류 │  │ 답변 검색     │
│ (번호/    │  │ (코사인 유사도)│
│  장소/시간)│  │              │
└──────────┘  └──────────────┘
    │               │
    └───────┬───────┘
            │
            ▼
    최종 답변 생성
```

## Wiki 페이지 목록

- [[Architecture]] — NLP 모델 구조 및 대화 처리 흐름
- [[Setup Guide]] — 의존성 설치, 모델 학습, 실행
