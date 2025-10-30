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
        # 정규화된 데이터 파일 로드 (각 지역별로 최댓값을 100으로 변환)
        df = pd.read_csv('data/naver_datalab_normalized.csv')
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
            '데이터 개수 (count)': len(region_data),
            '평균 (mean)': region_data.mean(),
            '최소값 (min)': region_data.min(),
            '최대값 (max)': region_data.max(),
            '중앙값 (median)': region_data.median(),
        }
    return stats_dict

@st.cache_data
def detect_peaks_cached(df):
    """피크 탐지 (캐시됨)
    피크: 검색량이 평소보다 현저히 높은 시점 (주요 이벤트, 특정 시기 등)
    """
    pivot_df = df.pivot(index='date', columns='region', values='ratio').fillna(0)
    peak_results = {}
    
    for region in pivot_df.columns:
        data = pivot_df[region].values
        threshold = np.mean(data) + 2.5 * np.std(data)  # 더 엄격한 기준
        peaks, _ = find_peaks(data, height=threshold, distance=14)  # 2주 간격
        
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
        st.title("드론 라이트 쇼 검색 트렌드 분석")
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
            
            # 탭으로 구성
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
                
                # 연도별 비교
                st.markdown("### 📊 네이버 검색 연도별 비교")
                yearly_ratio = filtered_df.groupby(['region', 'year'])['ratio'].mean().reset_index()
                yearly_ratio['year'] = yearly_ratio['year'].astype(str)
                
                fig_naver_yearly = px.bar(
                    yearly_ratio,
                    x='region',
                    y='ratio',
                    color='year',
                    title='연도별 네이버 검색 비율 평균',
                    barmode='group',
                    labels={'ratio': '평균 검색 비율', 'region': '지역'},
                    category_orders={'year': ['2023', '2024', '2025']}
                )
                fig_naver_yearly.update_layout(height=500)
                st.plotly_chart(fig_naver_yearly, use_container_width=True)
        
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
                    title='월별 평균 검색 비율 (해당 월 전체 일수의 평균)',
                    markers=True
                )
                fig_monthly.update_xaxes(dtick=1)
                st.plotly_chart(fig_monthly, use_container_width=True)
                st.caption("💡 월별 평균값이므로, 특정 날짜의 최고점(예: 1월 1일=100.0)이 아닌 해당 월 전체의 평균을 나타냅니다.")
            
            with col2:
                # 히트맵
                monthly_pivot = monthly_avg.pivot(index='month', columns='region', values='ratio')
                fig_heatmap = px.imshow(
                    monthly_pivot,
                    title='월별 평균 검색 비율 히트맵',
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
            
            # 검색 비율 계산 방법 설명
            st.info("""
            **📊 검색 비율 계산 방법 (정규화)**
            
            **왜 정규화가 필요한가?**
            
            원본 데이터에서 부산·광안리는 고흥·녹동항보다 약 19배 많은 검색량을 기록합니다. 이 경우 부산 기준으로 표시하면 고흥과 당진의 값은 모두 0에서 5 사이에 몰려 패턴을 확인할 수 없습니다.
            
            **정규화 방식**
            
            각 지역의 월별·계절별 패턴을 비교하기 위해, 각 지역별로 자체 최댓값을 100으로 정규화했습니다.
            
            • 고흥·녹동항: 고흥의 최고 검색 시점 = 100 (원본 1.44)
            
            • 당진·삽교호: 당진의 최고 검색 시점 = 100 (원본 3.45)
            
            • 부산·광안리: 부산의 최고 검색 시점 = 100 (원본 100.0)
            
            **차트 해석 방법**
            
            차트의 높이는 절대적 검색량이 아닌, 각 지역 내에서 해당 시기가 얼마나 성수기인지를 나타냅니다.
            
            예시: 고흥 9월이 9.37 → 고흥의 최고점 대비 약 9% 수준 (고흥의 상대적 성수기)
            
            ⚠️ 주의: 절대적 검색량은 부산이 압도적으로 많습니다. 이 차트는 각 지역의 시간적 변화 패턴을 비교하기 위한 것입니다.
            """)
            
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
            st.markdown("""
            **피크란?** 검색량이 평소보다 현저히 높은 시점을 의미합니다.  
            주요 이벤트, 축제, 관광 시즌 등에서 발생하며, 피크 시점을 분석하면 드론 라이트 쇼에 대한 관심이 언제 급증했는지 파악할 수 있습니다.
            """)
            
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
                        annotation_text=f"임계값 (평균+2.5σ): {peak_info['threshold']:.4f}",
                        annotation_position="right"
                    )
                    
                    fig_peak.update_layout(
                        title=f'{region} - 주요 검색 피크 분석',
                        xaxis_title='날짜',
                        yaxis_title='검색 비율',
                        height=400
                    )
                    
                    st.plotly_chart(fig_peak, use_container_width=True)
        
        with tab4:
            st.subheader("통계 요약")
            
            # 기본 통계
            stats = get_basic_statistics(filtered_df)
            
            # 통계 테이블
            stats_df = pd.DataFrame(stats).T
            stats_df = stats_df.round(6)
            st.dataframe(stats_df, use_container_width=True)
            
            # 통계 항목 설명
            st.markdown("""
            **통계 항목 설명:**
            - **데이터 개수 (count)**: 분석 대상 데이터의 총 개수
            - **평균 (mean)**: 검색 비율의 평균값
            - **최소값 (min)**: 가장 낮은 검색 비율
            - **최대값 (max)**: 가장 높은 검색 비율
            - **중앙값 (median)**: 데이터를 정렬했을 때 중간에 위치한 값
            """)
            
        with tab5:
            st.subheader("주요 이벤트 및 인사이트")
            
            # 최고 검색 기록들 - 지역별로 분리 표시
            if len(selected_regions) >= 2:
                st.write("**🏆 지역별 상위 10개 검색 기록**")
                for region in selected_regions:
                    region_data = filtered_df[filtered_df['region'] == region]
                    top_records = region_data.nlargest(10, 'ratio')
                    
                    if len(top_records) > 0:
                        st.write(f"\n**{region}**")
                        for idx, row in top_records.iterrows():
                            with st.container():
                                col1, col2, col3 = st.columns([2, 2, 1])
                                with col1:
                                    st.write(f"{row['date'].strftime('%Y년 %m월 %d일')}")
                                with col2:
                                    st.write(f"검색 비율: **{row['ratio']:.4f}**")
                        st.divider()
            else:
                # 단일 지역 선택 시 기존 방식
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
            
            # 월별 최고 기록 (수정: 월별 평균 기준)
            st.write("**📅 월별 최고 검색 기록**")
            monthly_avg_by_region = filtered_df.groupby(['region', 'month'])['ratio'].mean().reset_index()
            
            for region in selected_regions:
                region_monthly = monthly_avg_by_region[monthly_avg_by_region['region'] == region]
                if len(region_monthly) > 0:
                    peak_month = region_monthly.loc[region_monthly['ratio'].idxmax()]
                    st.write(f"• **{region}**: {int(peak_month['month'])}월이 성수기 (평균 검색비율: {peak_month['ratio']:.4f})")
    
    else:
        # SNS 흐름 분석 화면
        st.title("드론 라이트 쇼 검색 트렌드 분석")
        st.markdown("### SNS 흐름 분석 (블로그 + 유튜브)")
        
        # SNS 데이터 로드
        sns_df = load_sns_data()
        
        if sns_df is not None:
            # 필터링 적용
            if len(date_range) == 2:
                start_date, end_date = date_range
                sns_filtered = sns_df[
                    (sns_df['region'].isin(selected_regions)) &
                    (sns_df['date'].notna()) &
                    (sns_df['date'].dt.date >= start_date) &
                    (sns_df['date'].dt.date <= end_date)
                ]
            else:
                sns_filtered = sns_df[
                    (sns_df['region'].isin(selected_regions)) &
                    (sns_df['date'].notna())
                ]
            
            # 선택된 지역만 표시되도록 데이터 확인
            if len(sns_filtered) == 0:
                st.warning(f"선택한 지역({', '.join(selected_regions)})에 대한 SNS 데이터가 없습니다.")
            
            # 주요 지표
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_mentions = len(sns_filtered)
                st.metric("총 언급량", f"{total_mentions:,}건")
            
            with col2:
                blog_count = len(sns_filtered[sns_filtered['platform'] == 'blog'])
                st.metric("블로그 언급", f"{blog_count:,}건")
            
            with col3:
                youtube_count = len(sns_filtered[sns_filtered['platform'] == 'youtube'])
                st.metric("유튜브 언급", f"{youtube_count:,}건")
            
            with col4:
                if youtube_count > 0:
                    avg_views = sns_filtered[sns_filtered['platform'] == 'youtube']['views'].mean()
                    st.metric("평균 조회수", f"{avg_views:,.0f}")
            
            st.divider()
            
            # 탭으로 구성
            sns_tab1, sns_tab2, sns_tab3, sns_tab4, sns_tab5 = st.tabs([
                "📈 시계열 트렌드",
                "📊 월별 패턴",
                "🔥 피크 분석",
                "📋 통계 요약",
                "🎯 주요 이벤트"
            ])
            
            with sns_tab1:
                st.subheader("연도별 SNS 언급량 비교")
                yearly_counts = sns_filtered.groupby(['region', 'year']).size().reset_index(name='count')
                yearly_counts['year'] = yearly_counts['year'].astype(str)
                fig_yearly = px.bar(
                    yearly_counts,
                    x='region',
                    y='count',
                    color='year',
                    title='2023~2025 연도별 SNS 언급량',
                    barmode='group',
                    category_orders={'year': ['2023', '2024', '2025']}
                )
                fig_yearly.update_layout(height=500)
                st.plotly_chart(fig_yearly, use_container_width=True)
            
            with sns_tab2:
                st.subheader("월별 SNS 언급 패턴")
                monthly_counts = sns_filtered.groupby(['region', 'year', 'month']).size().reset_index(name='count')
                
                # 전체 월별 트렌드
                fig_monthly_line = px.line(
                    monthly_counts,
                    x='month',
                    y='count',
                    color='region',
                    line_dash='year',
                    title='월별 언급량 추이',
                    markers=True
                )
                fig_monthly_line.update_xaxes(dtick=1)
                fig_monthly_line.update_layout(height=500)
                st.plotly_chart(fig_monthly_line, use_container_width=True)
                
                # 지역별 월별 트렌드 상세
                st.markdown("#### 지역별 월별 트렌드 상세")
                for region in selected_regions:
                    region_monthly = monthly_counts[monthly_counts['region'] == region]
                    if len(region_monthly) > 0:
                        region_monthly = region_monthly.copy()
                        region_monthly['year_str'] = region_monthly['year'].astype(str)
                        
                        # 연도별로 다른 마커 스타일 적용
                        fig_region = go.Figure()
                        
                        marker_symbols = {'2023': 'circle', '2024': 'square', '2025': 'diamond'}
                        dash_styles = {'2023': 'solid', '2024': 'dash', '2025': 'dot'}
                        
                        for year_val in ['2023', '2024', '2025']:
                            year_data = region_monthly[region_monthly['year_str'] == year_val]
                            if len(year_data) > 0:
                                fig_region.add_trace(go.Scatter(
                                    x=year_data['month'],
                                    y=year_data['count'],
                                    mode='lines+markers',
                                    name=f'{year_val}년',
                                    marker=dict(size=10, symbol=marker_symbols[year_val]),
                                    line=dict(dash=dash_styles[year_val], width=2)
                                ))
                        
                        fig_region.update_xaxes(dtick=1, title='월')
                        fig_region.update_yaxes(title='언급량(건)')
                        fig_region.update_layout(
                            title=f'{region} 월별 SNS 트렌드',
                            height=400,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_region, use_container_width=True)
            
            with sns_tab3:
                st.subheader("SNS 언급 피크 분석")
                st.markdown("""
                **피크란?** SNS 언급량이 평소보다 현저히 높은 시점을 의미합니다.  
                주요 이벤트나 화제가 된 시기를 파악할 수 있습니다.
                """)
                
                # 피크 시점 표시
                peak_months = monthly_counts.loc[monthly_counts.groupby(['region', 'year'])['count'].idxmax()]
                fig_peak = px.scatter(
                    peak_months,
                    x='month',
                    y='count',
                    color='region',
                    size='count',
                    title='월별 최고 언급 시점',
                    text='year'
                )
                fig_peak.update_traces(textposition='top center')
                fig_peak.update_layout(height=500)
                st.plotly_chart(fig_peak, use_container_width=True)
                
                # 지역별 피크 정보
                st.markdown("#### 지역별 최고 언급 시점")
                for region in selected_regions:
                    region_peaks = peak_months[peak_months['region'] == region]
                    if len(region_peaks) > 0:
                        st.write(f"**{region}**")
                        for _, peak in region_peaks.iterrows():
                            st.write(f"  • {int(peak['year'])}년 {int(peak['month'])}월: {int(peak['count'])}건")
            
            with sns_tab4:
                st.subheader("SNS 기본 통계")
                
                # 지역별 통계 - 표 형식으로 변경
                sns_stats_dict = {}
                for region in selected_regions:
                    region_data = sns_filtered[sns_filtered['region'] == region]
                    if len(region_data) > 0:
                        blog_count = len(region_data[region_data['platform'] == 'blog'])
                        youtube_count = len(region_data[region_data['platform'] == 'youtube'])
                        monthly_avg = region_data.groupby('month').size().mean()
                        
                        sns_stats_dict[region] = {
                            '데이터 개수 (count)': len(region_data),
                            '블로그 언급 수': blog_count,
                            '유튜브 언급 수': youtube_count,
                            '블로그 비율 (%)': round(blog_count / len(region_data) * 100, 1),
                            '유튜브 비율 (%)': round(youtube_count / len(region_data) * 100, 1),
                            '월평균 언급량': round(monthly_avg, 1)
                        }
                
                # 통계 테이블 표시
                if sns_stats_dict:
                    sns_stats_df = pd.DataFrame(sns_stats_dict).T
                    st.dataframe(sns_stats_df, use_container_width=True)
                    
                    st.markdown("""
                    **통계 항목 설명:**
                    - **데이터 개수 (count)**: 총 SNS 언급 수
                    - **블로그/유튜브 언급 수**: 플랫폼별 언급 건수
                    - **블로그/유튜브 비율**: 전체 언급 중 각 플랫폼이 차지하는 비율
                    - **월평균 언급량**: 한 달 평균 SNS 언급 건수
                    """)
            
            with sns_tab5:
                st.subheader("플랫폼별 비중 및 유튜브 분석")
                
                # 플랫폼별 비중
                st.markdown("### 💬 플랫폼별 언급 비중")
                platform_counts = sns_filtered.groupby(['region', 'platform']).size().reset_index(name='count')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_platform = px.bar(
                        platform_counts,
                        x='region',
                        y='count',
                        color='platform',
                        title='플랫폼별 언급량 (블로그 vs 유튜브)',
                        barmode='stack',
                        color_discrete_map={'blog': '#1f77b4', 'youtube': '#ff7f0e'}
                    )
                    st.plotly_chart(fig_platform, use_container_width=True)
                
                with col2:
                    platform_total = sns_filtered.groupby('platform').size().reset_index(name='count')
                    fig_pie = px.pie(
                        platform_total,
                        values='count',
                        names='platform',
                        title='전체 플랫폼 비율',
                        color='platform',
                        color_discrete_map={'blog': '#1f77b4', 'youtube': '#ff7f0e'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # 유튜브 상세 분석
                youtube_data = sns_filtered[sns_filtered['platform'] == 'youtube'].copy()
                
                if len(youtube_data) > 0 and youtube_data['views'].sum() > 0:
                    st.markdown("### 🎥 유튜브 상세 분석")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        fig_views = px.box(
                            youtube_data[youtube_data['views'] > 0],
                            x='region',
                            y='views',
                            title='지역별 조회수 분포',
                            log_y=True
                        )
                        st.plotly_chart(fig_views, use_container_width=True)
                    
                    with col2:
                        fig_likes = px.box(
                            youtube_data[youtube_data['likes'] > 0],
                            x='region',
                            y='likes',
                            title='지역별 좋아요 분포',
                            log_y=True
                        )
                        st.plotly_chart(fig_likes, use_container_width=True)
                    
                    with col3:
                        fig_comments = px.box(
                            youtube_data[youtube_data['comments'] > 0],
                            x='region',
                            y='comments',
                            title='지역별 댓글 수 분포',
                            log_y=True
                        )
                        st.plotly_chart(fig_comments, use_container_width=True)
        
        else:
            st.info("SNS 데이터를 로드할 수 없습니다. data 폴더에 'sns_blog_youtube_with_reaction_2023_2025.csv' 파일을 추가해주세요.")
    
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
