from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tiktoken

st.set_page_config(page_title='AI Router | 자동 토큰 예측', page_icon='🧠', layout='centered')
BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / 'difficulty_model.joblib'
OUTPUT_FILE = BASE / 'output_tokens_model.joblib'
PRICE_FILE = BASE / 'OpenAI_API_공식_가격표_2026-11-24.xlsx'
MODEL_MAPPING = {'LOW': 'gpt-6-luna', 'MEDIUM': 'gpt-6.1-terra', 'HIGH': 'gpt-6.2-sol'}
CONFIDENCE_THRESHOLD = 0.45  # 개발자 설정. 검증된 최적값이 아님.

st.title('🧠 AI Router')
st.caption('질문 난이도 분석 → 모델 선택 → 토큰 수 자동 계산·예측 → 예상 비용 비교')
st.info('연구용 시뮬레이션입니다. 실제 GPT API 호출 또는 답변 생성은 하지 않습니다.')

@st.cache_resource
def load_classifier():
    return joblib.load(MODEL_FILE)

@st.cache_resource
def load_output_estimator():
    return joblib.load(OUTPUT_FILE)

@st.cache_resource
def load_tokenizer():
    return tiktoken.get_encoding('o200k_base')

@st.cache_data
def load_prices():
    df = pd.read_excel(PRICE_FILE, sheet_name='Standard API 요금')
    df['모델명'] = df['모델명'].astype(str).str.strip()
    return df.set_index('모델명')

missing = [p.name for p in [MODEL_FILE, OUTPUT_FILE, PRICE_FILE] if not p.exists()]
if missing:
    st.error('필요한 파일이 없습니다: ' + ', '.join(missing))
    st.caption('output_tokens_model.joblib은 train_output_estimator.py를 로컬에서 실행한 후 GitHub에 업로드하세요.')
    st.stop()
try:
    classifier = load_classifier()
    output_bundle = load_output_estimator()
    tokenizer = load_tokenizer()
    prices = load_prices()
    for name in MODEL_MAPPING.values():
        if name not in prices.index:
            raise ValueError(f'가격표에 없는 모델: {name}')
except Exception as err:
    st.error(f'모델 또는 가격표를 불러올 수 없습니다: {err}')
    st.stop()

question = st.text_area('질문을 입력하세요', placeholder='예: Explain the process of photosynthesis.', height=130)

def cost(name, input_tokens, output_tokens):
    input_rate = float(prices.loc[name, '입력 가격'])
    output_rate = float(prices.loc[name, '출력 가격'])
    return (input_rate * input_tokens + output_rate * output_tokens) / 1_000_000

if st.button('🔎 난이도 분석 및 모델 선택', type='primary', use_container_width=True):
    query = question.strip()
    if not query:
        st.warning('질문을 입력해 주세요.')
    else:
        levels = classifier.classes_
        probabilities = classifier.predict_proba([query])[0]
        raw_difficulty = str(levels[probabilities.argmax()])
        confidence = float(probabilities.max())
        recognized = int(classifier.named_steps['tfidf'].transform([query]).nnz)
        fallback = (confidence < CONFIDENCE_THRESHOLD or recognized == 0)
        chosen_difficulty = 'HIGH' if fallback else raw_difficulty
        selected = MODEL_MAPPING[chosen_difficulty]

        # Counts *only question text*, not system prompts or API message metadata.
        input_tokens = len(tokenizer.encode(query))

        # Learned response length from PiCo answer examples; not an API measurement.
        predicted_log_tokens = float(output_bundle['model'].predict([query])[0])
        predicted_tokens = np.expm1(np.clip(predicted_log_tokens, 0, 14))
        output_tokens = max(1, int(round(predicted_tokens)))

        chosen_cost = cost(selected, input_tokens, output_tokens)
        baseline_cost = cost(MODEL_MAPPING['HIGH'], input_tokens, output_tokens)
        saving = (1 - chosen_cost / baseline_cost) * 100 if baseline_cost > 0 else 0.0

        a, b, c = st.columns(3)
        a.metric('분류 AI의 예측', raw_difficulty)
        b.metric('최종 배정 등급', chosen_difficulty)
        c.metric('추천 모델', selected)
        st.subheader('자동 토큰 분석')
        t1, t2 = st.columns(2)
        t1.metric('입력 토큰 수 (텍스트)', f'{input_tokens:,}')
        t2.metric('출력 토큰 예측값', f'{output_tokens:,}')
        st.caption('입력은 o200k_base 토크나이저로 계산한 질문 텍스트 토큰 수입니다. 실제 GPT API 총 입력 토큰 수와 차이가 있을 수 있습니다.')
        st.caption('출력은 PiCo 질문·답변 데이터로 학습한 회귀 모델의 예상값입니다. 실제 GPT 답변의 토큰 수가 아닙니다.')

        st.subheader('질문 난이도 분석')
        scores = pd.DataFrame({'난이도': levels, '예측 확률': probabilities}).set_index('난이도')
        st.bar_chart(scores)
        st.caption(f'분류기 최대 예측 확률: {confidence:.1%} · TF-IDF 특징 인식 수: {recognized} · 확신도 처리 기준: {CONFIDENCE_THRESHOLD:.0%}')
        if fallback:
            st.warning('분류 불확실성 또는 미인식 표현 때문에 HIGH 모델로 보수적으로 배정했습니다.')

        st.subheader('예상 API 요금')
        m1, m2 = st.columns(2)
        m1.metric('추천 모델 예상 비용', f'${chosen_cost:.8f}')
        m2.metric('항상 HIGH 모델 사용', f'${baseline_cost:.8f}')
        st.metric('예상 API 요금 절감률', f'{saving:.1f}%')
        st.caption('두 경우에 동일한 입력·출력 토큰 수가 사용된다고 가정한 가격 비교입니다. 실제 모델별 답변 길이, 품질, 호출 비용과 에너지 소비량은 측정되지 않았습니다.')

with st.expander('이 프로젝트의 한계 및 개인정보 안내'):
    st.write('난이도 분류기는 영어 중심 데이터로 학습되었습니다. 한국어 질문은 부정확할 수 있습니다.')
    st.write('출력 토큰 예측은 PiCo 답변 데이터의 분포를 학습합니다. 모델별 실제 답변 길이가 다를 수 있습니다.')
    st.write('현재 모델 배정은 예측 난이도와 고정 규칙에 기반합니다. 답변 성능을 학습해 고른 최적 모델은 아닙니다.')
    st.write('API 요금은 에너지 소비량의 직접 측정치가 아닙니다. 개인정보는 입력하지 마세요.')
