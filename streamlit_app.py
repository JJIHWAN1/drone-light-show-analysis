#!/usr/bin/env python3
"""
드론 라이트 쇼 검색 트렌드 분석 - Streamlit 버전
Streamlit Community Cloud 배포용
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.signal import find_peaks
from datetime import datetime
import os

# 페이지 설정
st.set_page_config(
    page_title="🚁 드론 라이트 쇼 검색 트렌드 분석",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 캐시된 데이터 로드 함수
@st.cache_data
def load_data():
    """데이터 로드 및 전처리"""
    # GitHub에서 데이터를 로드하거나 업로드된 파일 사용
    try:
        # GitHub 저장소의 데이터 파일 로드
        df = pd.read_csv('data/naver_datalab_fixed.csv')
    except:
        # 샘플 데이터 생성 (실제 배포시에는 GitHub에서 로드)
        st.error("데이터 파일을 찾을 수 없습니다. 샘플 데이터를 사용합니다.")
        dates = pd.date_range('2023-01-01', '2025-09-30', freq='D')
        regions = ['고흥·녹동항', '당진·삽교호', '부산·광안리']
        
        data = []
        for region in regions:
            for date in dates:
                if region == '부산·광안리':
                    ratio = np.random.exponential(0.5) if np.random.random() > 0.95 else np.random.exponential(0.1)
                else:
                    ratio = np.random.exponential(0.05)
                data.append({'date': date, 'region': region, 'ratio': ratio})
        
        df = pd.DataFrame(data)
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.day_name()
    df['quarter'] = df['date'].dt.quarter
    
    return df

@st.cache_data
def load_sns_data():
    """SNS 데이터 로드 (블로그 + 유튜브)"""
    try:
        sns_df = pd.read_csv('data/sns_blog_youtube_with_reaction_2023_2025.csv')
        sns_df['date'] = pd.to_datetime(sns_df['date'], errors='coerce')
        sns_df['year'] = sns_df['date'].dt.year
        sns_df['month'] = sns_df['date'].dt.month
        return sns_df
    except:
        st.warning("SNS 데이터를 찾을 수 없습니다.")
        return None

@st.cache_data
def get_basic_statistics(df):
    """기본 통계 계산"""
    stats_dict = {}
    for region in df['region'].unique():
        region_data = df[df['region'] == region]['ratio']
        stats_dict[region] = {
            'count': len(region_data),
            'mean': region_data.mean(),
            'std': region_data.std(),
            'min': region_data.min(),
            'max': region_data.max(),
            'median': region_data.median(),
        }
    return stats_dict

@st.cache_data
def detect_peaks_cached(df):
    """피크 탐지 (캐시됨)"""
    pivot_df = df.pivot(index='date', columns='region', values='ratio').fillna(0)
    peak_results = {}
    
    for region in pivot_df.columns:
        data = pivot_df[region].values
        threshold = np.mean(data) + 2 * np.std(data)
        peaks, _ = find_peaks(data, height=threshold, distance=7)
        
        peak_results[region] = {
            'peak_count': len(peaks),
            'peak_dates': pivot_df.index[peaks],
            'peak_values': data[peaks],
            'threshold': threshold
        }
    
    return peak_results

def main():
    """메인 앱"""
    
    # 데이터 로드
    with st.spinner('데이터를 로드하는 중...'):
        df = load_data()
    
    # 사이드바 - 분석 유형 선택
    st.sidebar.header("📊 분석 옵션")
    
    analysis_type = st.sidebar.radio(
        "분석 유형 선택",
        ["🔍 네이버 검색 분석", "🌐 SNS 흐름 분석"],
        index=0
    )
    
    st.sidebar.divider()
    
    # 지역 선택
    regions = df['region'].unique()
    selected_regions = st.sidebar.multiselect(
        "분석할 지역 선택",
        regions,
        default=regions
    )
    
    # 기간 선택
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    date_range = st.sidebar.date_input(
        "분석 기간 선택",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # 데이터 필터링
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[
            (df['region'].isin(selected_regions)) &
            (df['date'].dt.date >= start_date) &
            (df['date'].dt.date <= end_date)
        ]
    else:
        filtered_df = df[df['region'].isin(selected_regions)]
    
    # 분석 유형에 따른 화면 분리
    if analysis_type == "🔍 네이버 검색 분석":
        # 네이버 검색 분석 화면
        st.title("🚁 드론 라이트 쇼 검색 트렌드 분석")
        st.markdown("### 네이버 데이터랩 기반 지역별 검색 패턴 분석")
        
        # 메인 대시보드
        if len(filtered_df) > 0:
            
            # 주요 지표 카드
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "총 데이터 포인트",
                    f"{len(filtered_df):,}개"
                )
            
            with col2:
                max_ratio = filtered_df['ratio'].max()
                max_region = filtered_df.loc[filtered_df['ratio'].idxmax(), 'region']
                st.metric(
                    "최고 검색비율",
                    f"{max_ratio:.4f}",
                    f"{max_region}"
                )
            
            with col3:
                avg_ratio = filtered_df['ratio'].mean()
                st.metric(
                    "평균 검색비율",
                    f"{avg_ratio:.4f}"
                )
            
            with col4:
                date_range_days = (filtered_df['date'].max() - filtered_df['date'].min()).days
                st.metric(
                    "분석 기간",
                    f"{date_range_days}일"
                )
            
            st.divider()
            
            # 탭으로 구성 (SNS 흐름 분석 탭 제거)
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 시계열 트렌드", 
                "📊 월별 패턴", 
                "🔥 피크 분석", 
                "📋 통계 요약",
                "🎯 주요 이벤트"
            ])
            
            with tab1:
                st.subheader("지역별 검색 트렌드")
                
                # 시계열 차트
                fig_timeseries = px.line(
                    filtered_df, 
                    x='date', 
                    y='ratio', 
                    color='region',
                    title='시간별 검색 비율 변화',
                    labels={'ratio': '검색 비율', 'date': '날짜', 'region': '지역'}
                )
                fig_timeseries.update_layout(height=600)
                st.plotly_chart(fig_timeseries, use_container_width=True)
            
            # 2024년 상세 보기
            df_2024 = filtered_df[filtered_df['year'] == 2024]
            if len(df_2024) > 0:
                st.subheader("2024년 상세 트렌드")
                fig_2024 = px.line(
                    df_2024, 
                    x='date', 
                    y='ratio', 
                    color='region',
                    title='2024년 검색 트렌드',
                    markers=True
                )
                fig_2024.update_layout(height=500)
                st.plotly_chart(fig_2024, use_container_width=True)
        
        with tab2:
            st.subheader("월별 검색 패턴")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 월별 평균
                monthly_avg = filtered_df.groupby(['month', 'region'])['ratio'].mean().reset_index()
                fig_monthly = px.line(
                    monthly_avg, 
                    x='month', 
                    y='ratio', 
                    color='region',
                    title='월별 평균 검색 비율',
                    markers=True
                )
                fig_monthly.update_xaxes(dtick=1)
                st.plotly_chart(fig_monthly, use_container_width=True)
            
            with col2:
                # 히트맵
                monthly_pivot = monthly_avg.pivot(index='month', columns='region', values='ratio')
                fig_heatmap = px.imshow(
                    monthly_pivot,
                    title='월별-지역별 검색 비율 히트맵',
                    labels=dict(x="지역", y="월", color="검색 비율"),
                    aspect="auto"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # 계절별 분석
            season_map = {12: '겨울', 1: '겨울', 2: '겨울',
                         3: '봄', 4: '봄', 5: '봄',
                         6: '여름', 7: '여름', 8: '여름',
                         9: '가을', 10: '가을', 11: '가을'}
            filtered_df_season = filtered_df.copy()
            filtered_df_season['season'] = filtered_df_season['month'].map(season_map)
            
            seasonal_avg = filtered_df_season.groupby(['season', 'region'])['ratio'].mean().reset_index()
            fig_seasonal = px.bar(
                seasonal_avg,
                x='season',
                y='ratio',
                color='region',
                title='계절별 평균 검색 비율',
                barmode='group'
            )
            st.plotly_chart(fig_seasonal, use_container_width=True)
            
            # 네이버 검색 연도별 비교
            st.markdown("### 📊 네이버 검색 연도별 비교")
            yearly_ratio = filtered_df.groupby(['region', 'year'])['ratio'].mean().reset_index()
            yearly_ratio['year'] = yearly_ratio['year'].astype(str)
            
            fig_naver_yearly = px.bar(
                yearly_ratio,
                x='region',
                y='ratio',
                color='year',
                title='2023~2025 연도별 네이버 검색 비율 평균',
                barmode='group',
                labels={'ratio': '평균 검색 비율', 'region': '지역'},
                category_orders={'year': ['2023', '2024', '2025']}
            )
            fig_naver_yearly.update_layout(height=500)
            st.plotly_chart(fig_naver_yearly, use_container_width=True)
            
            # 네이버 검색 월별 트렌드 (지역별)
            st.markdown("### 📈 네이버 검색 월별 트렌드 (지역별)")
            monthly_ratio = filtered_df.groupby(['region', 'year', 'month'])['ratio'].mean().reset_index()
            
            for region in selected_regions:
                region_monthly = monthly_ratio[monthly_ratio['region'] == region]
                if len(region_monthly) > 0:
                    region_monthly = region_monthly.copy()
                    region_monthly['year_str'] = region_monthly['year'].astype(str)
                    
                    fig_naver_region = px.line(
                        region_monthly,
                        x='month',
                        y='ratio',
                        color='year_str',
                        title=f'{region} 월별 네이버 검색 트렌드',
                        markers=True,
                        labels={'ratio': '검색 비율', 'month': '월', 'year_str': '연도'},
                        category_orders={'year_str': ['2023', '2024', '2025']}
                    )
                    fig_naver_region.update_xaxes(dtick=1)
                    fig_naver_region.update_layout(height=400)
                    st.plotly_chart(fig_naver_region, use_container_width=True)
        
        with tab3:
            st.subheader("주요 검색 피크 분석")
            
            # 피크 탐지
            peaks = detect_peaks_cached(filtered_df)
            
            # 피크 요약
            col1, col2, col3 = st.columns(3)
            total_peaks = sum(peak_info['peak_count'] for peak_info in peaks.values())
            
            with col1:
                st.metric("전체 피크 개수", f"{total_peaks}개")
            
            with col2:
                max_peak_region = max(peaks.keys(), key=lambda x: peaks[x]['peak_count'])
                st.metric("최다 피크 지역", max_peak_region, f"{peaks[max_peak_region]['peak_count']}개")
            
            with col3:
                if total_peaks > 0:
                    all_peak_values = []
                    for peak_info in peaks.values():
                        all_peak_values.extend(peak_info['peak_values'])
                    max_peak_value = max(all_peak_values) if all_peak_values else 0
                    st.metric("최고 피크 값", f"{max_peak_value:.4f}")
            
            # 지역별 피크 차트
            for region in selected_regions:
                if region in peaks:
                    peak_info = peaks[region]
                    region_data = filtered_df[filtered_df['region'] == region]
                    
                    fig_peak = go.Figure()
                    
                    # 원본 데이터
                    fig_peak.add_trace(go.Scatter(
                        x=region_data['date'],
                        y=region_data['ratio'],
                        mode='lines',
                        name=f'{region} 트렌드',
                        line=dict(width=2)
                    ))
                    
                    # 피크 포인트
                    if len(peak_info['peak_dates']) > 0:
                        fig_peak.add_trace(go.Scatter(
                            x=peak_info['peak_dates'],
                            y=peak_info['peak_values'],
                            mode='markers',
                            name=f'피크 ({peak_info["peak_count"]}개)',
                            marker=dict(size=10, color='red', symbol='star')
                        ))
                    
                    # 임계값 선
                    fig_peak.add_hline(
                        y=peak_info['threshold'],
                        line_dash="dash",
                        line_color="gray",
                        annotation_text=f"임계값: {peak_info['threshold']:.4f}"
                    )
                    
                    fig_peak.update_layout(
                        title=f'{region} - 주요 검색 피크 분석',
                        xaxis_title='날짜',
                        yaxis_title='검색 비율',
                        height=400
                    )
                    
                    st.plotly_chart(fig_peak, use_container_width=True)
            
            # 네이버 검색 피크 시점 분석
            st.markdown("### 📍 네이버 검색 피크 시점")
            monthly_ratio = filtered_df.groupby(['region', 'year', 'month'])['ratio'].mean().reset_index()
            peak_points = monthly_ratio.loc[monthly_ratio.groupby(['region', 'year'])['ratio'].idxmax()]
            
            fig_naver_peaks = go.Figure()
            
            for region in selected_regions:
                region_monthly = monthly_ratio[monthly_ratio['region'] == region]
                if len(region_monthly) > 0:
                    fig_naver_peaks.add_trace(go.Scatter(
                        x=region_monthly['month'],
                        y=region_monthly['ratio'],
                        mode='lines+markers',
                        name=region,
                        line=dict(width=2)
                    ))
                    
                    # 피크 포인트 표시
                    region_peaks = peak_points[peak_points['region'] == region]
                    for _, row in region_peaks.iterrows():
                        fig_naver_peaks.add_annotation(
                            x=row['month'],
                            y=row['ratio'],
                            text=f"{int(row['year'])}년",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor='red',
                            ax=20,
                            ay=-30
                        )
            
            fig_naver_peaks.update_layout(
                title='2023~2025 지역별 네이버 검색 피크 시점 비교',
                xaxis_title='월',
                yaxis_title='검색 비율',
                height=500,
                xaxis=dict(dtick=1)
            )
            
            st.plotly_chart(fig_naver_peaks, use_container_width=True)
        
        with tab4:
            st.subheader("통계 요약")
            
            # 기본 통계
            stats = get_basic_statistics(filtered_df)
            
            # 통계 테이블
            stats_df = pd.DataFrame(stats).T
            stats_df = stats_df.round(6)
            st.dataframe(stats_df, use_container_width=True)
            
            # 분포 차트
            col1, col2 = st.columns(2)
            
            with col1:
                fig_box = px.box(
                    filtered_df,
                    x='region',
                    y='ratio',
                    title='지역별 검색 비율 분포'
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                fig_violin = px.violin(
                    filtered_df,
                    x='region',
                    y='ratio',
                    title='지역별 검색 비율 분포 (상세)'
                )
                st.plotly_chart(fig_violin, use_container_width=True)
            
        with tab5:
            st.subheader("주요 이벤트 및 인사이트")
            
            # 최고 검색 기록들
            top_records = filtered_df.nlargest(10, 'ratio')
            
            st.write("**🏆 상위 10개 검색 기록**")
            for idx, row in top_records.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"**{row['region']}**")
                    with col2:
                        st.write(f"{row['date'].strftime('%Y년 %m월 %d일')}")
                    with col3:
                        st.write(f"**{row['ratio']:.4f}**")
            
            # 월별 최고 기록
            st.write("**📅 월별 최고 검색 기록**")
            monthly_max = filtered_df.loc[filtered_df.groupby(['region', 'month'])['ratio'].idxmax()]
            monthly_summary = monthly_max.groupby('region')['month'].apply(
                lambda x: x.value_counts().index[0]
            ).reset_index()
            monthly_summary.columns = ['지역', '최고_검색_월']
            
            for _, row in monthly_summary.iterrows():
                st.write(f"• **{row['지역']}**: {row['최고_검색_월']}월이 성수기")
        
    else:
        st.warning("선택한 조건에 맞는 데이터가 없습니다. 필터를 조정해주세요.")
    
    # 푸터
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        🚁 드론 라이트 쇼 검색 트렌드 분석 대시보드<br>
        데이터 출처: 네이버 데이터랩, SNS 분석 | 분석 도구: Python, Streamlit, Plotly
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
