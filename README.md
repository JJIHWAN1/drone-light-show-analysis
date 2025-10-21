# 드론 라이트 쇼 검색 트렌드 분석

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

네이버 데이터랩 기반 지역별 드론 라이트 쇼 검색 패턴 분석 대시보드입니다.

## 라이브 데모

**[https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)**

## 프로젝트 개요

이 프로젝트는 네이버 데이터랩에서 제공하는 검색 데이터를 활용하여 다음 지역의 드론 라이트 쇼 관련 검색 트렌드를 분석합니다:

- 고흥·녹동항: 우주센터 관련 드론쇼
- 당진·삽교호: 호수 위 드론 라이트 쇼  
- 부산·광안리: 해변 드론 페스티벌

## 주요 기능

### 인터랙티브 대시보드
- 실시간 데이터 필터링 (지역, 기간 선택)
- 줌, 호버, 범례 토글 등 인터랙티브 기능
- 반응형 디자인 (모바일 지원)

### 5가지 분석 탭
1. 시계열 트렌드: 시간별 검색량 변화
2. 월별 패턴: 계절성 및 성수기 분석
3. 피크 분석: 주요 검색 급증 시점 탐지
4. 통계 요약: 기본 통계 및 상관관계
5. 주요 이벤트: 상위 검색 기록 및 인사이트

## 로컬 실행 방법

### 1. 저장소 클론
```bash
git clone https://github.com/yourusername/drone-light-show-analysis.git
cd drone-light-show-analysis
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. Streamlit 앱 실행
```bash
streamlit run streamlit_app.py
```

브라우저에서 `http://localhost:8501`로 접속

## 주요 인사이트

### 핵심 발견사항
- 부산·광안리: 압도적 검색량 (최대 100.0 vs 다른 지역 3.4)
- 2024년 1월 1일: 부산 신년 드론쇼로 역대 최고 검색량
- 지역별 성수기: 부산(1월), 당진(10월), 고흥(9월)
- 총 68개 피크: 고흥 36개, 당진 26개, 부산 6개

### 트렌드 패턴
- 부산: 대형 이벤트 중심의 급격한 피크
- 당진: 안정적이고 지속적인 관심도
- 고흥: 우주센터 연계 이벤트 패턴

## 기술 스택

- Frontend: Streamlit
- Data Processing: Pandas, NumPy
- Visualization: Plotly
- Analytics: SciPy (피크 탐지)
- Deployment: Streamlit Community Cloud

## 프로젝트 구조

```
drone-light-show-analysis/
├── streamlit_app.py         # 메인 Streamlit 앱
├── requirements.txt         # 패키지 의존성
├── .streamlit/
│   └── config.toml         # Streamlit 설정
├── data/
│   └── naver_datalab_fixed.csv  # 분석 데이터
└── README.md               # 프로젝트 문서
```

## 배포 정보

- 플랫폼: Streamlit Community Cloud
- URL: https://your-app-name.streamlit.app
- 업데이트: GitHub push 시 자동 배포
- 비용: 완전 무료

## 데이터 정보

- 출처: 네이버 데이터랩
- 기간: 2023년 1월 ~ 2025년 9월
- 총 데이터: 3,012개 레코드
- 업데이트: 일별 검색 비율 데이터

## 기여 방법

1. 이 저장소를 Fork
2. 새 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 Push (`git push origin feature/amazing-feature`)
5. Pull Request 생성

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 연락처

- 이슈 및 문의: [GitHub Issues](https://github.com/yourusername/drone-light-show-analysis/issues)
- 도움이 되었다면 스타를 눌러주세요!

---

<div align="center">

**Made with ❤️ for Drone Light Show Analysis**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)

</div>
