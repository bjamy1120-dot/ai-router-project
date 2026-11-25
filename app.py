from pathlib import Path
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title='AI Router | 질문 난이도 분석', page_icon='🧠', layout='centered')
BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / 'difficulty_model.joblib'
PRICE_FILE = BASE / 'OpenAI_API_공식_가격표_2026-11-24.xlsx'
MODEL_MAPPING = {'LOW': 'gpt-6-luna', 'MEDIUM': 'gpt-6.1-terra', 'HIGH': 'gpt-6.2-sol'}

st.title('🧠 AI Router')
st.caption('질문 난이도 분석 → GPT 모델 자동 배정 → 예상 API 비용 비교')
st.info('연구용 시뮬레이션입니다. 실제 GPT API를 호출하지 않으며 질문에 대한 답변을 생성하지 않습니다.')

@st.cache_resource
def load_classifier():
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_prices():
    df = pd.read_excel(PRICE_FILE, sheet_name='Standard API 요금')
    df['모델명'] = df['모델명'].astype(str).str.strip()
    return df.set_index('모델명')

if not MODEL_FILE.exists():
    st.error('학습된 모델 파일이 없습니다: difficulty_model.joblib')
    st.code("import joblib\njoblib.dump(model, 'difficulty_model.joblib')", language='python')
    st.stop()
if not PRICE_FILE.exists():
    st.error('API 가격표 파일이 없습니다: OpenAI_API_공식_가격표_2026-11-24.xlsx')
    st.stop()

try:
    classifier = load_classifier()
    prices = load_prices()
    for name in MODEL_MAPPING.values():
        if name not in prices.index:
            raise ValueError(f'가격표에서 모델을 찾을 수 없습니다: {name}')
except Exception as err:
    st.error(f'파일 읽기 오류: {err}')
    st.stop()

question = st.text_area('질문을 입력하세요', placeholder='예: Prove that the square root of 2 is irrational.', height=120)
col1, col2 = st.columns(2)
with col1:
    input_tokens = st.number_input('가정 입력 토큰 수', min_value=0, value=200, step=50)
with col2:
    output_tokens = st.number_input('가정 출력 토큰 수', min_value=0, value=400, step=50)
threshold = st.slider('낮은 확신도 처리 기준 (실험용)', min_value=0.34, max_value=0.80, value=0.45, step=0.01)

def cost(name, inp, out):
    return (float(prices.loc[name, '입력 가격']) * inp + float(prices.loc[name, '출력 가격']) * out) / 1_000_000

if st.button('🔎 난이도 분석 및 모델 선택', type='primary', use_container_width=True):
    if not question.strip():
        st.warning('질문을 먼저 입력해 주세요.')
    else:
        levels = classifier.classes_
        probabilities = classifier.predict_proba([question.strip()])[0]
        predicted = str(levels[probabilities.argmax()])
        max_prob = float(probabilities.max())
        recognized = int(classifier.named_steps['tfidf'].transform([question.strip()]).nnz)
        fallback = max_prob < threshold or recognized == 0
        assigned = 'HIGH' if fallback else predicted
        selected = MODEL_MAPPING[assigned]
        selected_cost = cost(selected, input_tokens, output_tokens)
        baseline = cost(MODEL_MAPPING['HIGH'], input_tokens, output_tokens)
        saving = (1 - selected_cost / baseline) * 100 if baseline else 0.0

        a, b, c = st.columns(3)
        a.metric('분류 AI 예측', predicted)
        b.metric('최종 배정 등급', assigned)
        c.metric('배정 GPT 모델', selected)
        st.write('**난이도별 예측 확률**')
        scores = pd.DataFrame({'난이도': levels, '예측 확률': probabilities}).set_index('난이도')
        st.bar_chart(scores)
        st.caption(f'최대 예측 확률: {max_prob:.1%} · 인식된 TF-IDF 특징 수: {recognized}')
        if fallback:
            st.warning('미인식 표현이 있거나 분류 확률이 낮아 HIGH 모델로 배정했어요. 이는 성능 보증이 아닌 보수적 배정 규칙입니다.')
        d, e = st.columns(2)
        d.metric('선택 모델 예상 요금', f'${selected_cost:.6f}')
        e.metric('항상 HIGH 모델 사용 요금', f'${baseline:.6f}')
        st.metric('예상 API 요금 절감률', f'{saving:.1f}%')
        st.caption('단가는 업로드한 가격표를 사용합니다. 토큰 수는 사용자가 설정한 가정값이고, 모델 응답 품질·분류기 연산비용·실제 전력량은 반영하지 않았습니다.')

st.divider()
with st.expander('이 시스템의 제한 및 개인정보 안내'):
    st.write('현재 분류기는 영어 중심 학습 데이터로 학습되어 한국어 질문에 부정확할 수 있습니다. 모든 질문이 정확하게 분류되는 것은 아닙니다.')
    st.write('비용만으로 최적의 답변 모델을 결정할 수 없습니다. 모델별 정답률 등 품질 자료가 확보되면 배정 규칙을 개선해야 합니다.')
    st.write('이 데모에는 실명, 연락처 등 개인정보를 입력하지 마세요. 공유 서버에 배포하면 질문이 서버에서 처리됩니다.')
