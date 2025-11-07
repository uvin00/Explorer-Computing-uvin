# 아래에 코드를 작성해주세요.
import streamlit as st

st.title("나의 소개 페이지")

st.header("자기소개")
st.write("안녕하세요, 제 이름은 배우빈입니다.")
st.write("저는 **디지털 헬스케어**와 **데이터 분석**에 관심이 있습니다.")

st.header("좋아하는 것")
st.write("저는 음악 감상을 좋아합니다.")
st.markdown("가장 자주 방문하는 사이트는 [Streamlit 공식 페이지](https://streamlit.io) 입니다.")

st.header("앞으로의 목표")
st.write("앞으로 다양한 프로젝트를 진행하면서 실력을 키우고 싶습니다.")

st.caption("제가 좋아하는 파이썬 코드 예시")
st.code("for i in range(3):\n    print('Hello, Streamlit!')", language="python")

st.caption("피타고라스 정리")
st.latex(r"a^2 + b^2 = c^2")
