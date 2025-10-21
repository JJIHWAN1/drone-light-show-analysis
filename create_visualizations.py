#!/usr/bin/env python3
"""
드론 라이트 쇼 검색 트렌드 시각화 생성 스크립트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_data():
    """데이터 로드 및 전처리"""
    df = pd.read_csv('./data/naver_datalab_fixed.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.day_name()
    return df

def create_time_series_plot(df):
    """시계열 트렌드 차트 생성"""
    plt.figure(figsize=(16, 10))
    
    # 전체 트렌드
    plt.subplot(2, 2, 1)
    for region in df['region'].unique():
        region_data = df[df['region'] == region]
        plt.plot(region_data['date'], region_data['ratio'], 
                label=region, linewidth=1.5, alpha=0.8)
    plt.title('지역별 검색 트렌드 (전체 기간)', fontsize=14, fontweight='bold')
    plt.xlabel('날짜')
    plt.ylabel('검색 비율')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2024년 상세
    plt.subplot(2, 2, 2)
    df_2024 = df[df['year'] == 2024]
    for region in df['region'].unique():
        region_data = df_2024[df_2024['region'] == region]
        plt.plot(region_data['date'], region_data['ratio'], 
                label=region, linewidth=2, marker='o', markersize=2)
    plt.title('2024년 검색 트렌드 상세', fontsize=14, fontweight='bold')
    plt.xlabel('날짜')
    plt.ylabel('검색 비율')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 월별 평균
    plt.subplot(2, 2, 3)
    monthly_avg = df.groupby(['month', 'region'])['ratio'].mean().reset_index()
    sns.lineplot(data=monthly_avg, x='month', y='ratio', hue='region', 
                marker='o', linewidth=3, markersize=8)
    plt.title('월별 평균 검색 비율', fontsize=14, fontweight='bold')
    plt.xlabel('월')
    plt.ylabel('평균 검색 비율')
    plt.grid(True, alpha=0.3)
    
    # 지역별 박스플롯
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='region', y='ratio', palette='Set2')
    plt.title('지역별 검색 비율 분포', fontsize=14, fontweight='bold')
    plt.xlabel('지역')
    plt.ylabel('검색 비율')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('./results/figures/comprehensive_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_heatmap_analysis(df):
    """히트맵 분석 생성"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 월별-지역별 히트맵
    monthly_pivot = df.groupby(['month', 'region'])['ratio'].mean().reset_index()
    monthly_matrix = monthly_pivot.pivot(index='month', columns='region', values='ratio')
    
    sns.heatmap(monthly_matrix, annot=True, fmt='.3f', cmap='YlOrRd', 
                ax=axes[0,0], cbar_kws={'shrink': 0.8})
    axes[0,0].set_title('월별 평균 검색 비율', fontsize=14, fontweight='bold')
    
    # 상관관계 매트릭스
    pivot_df = df.pivot(index='date', columns='region', values='ratio')
    correlation_matrix = pivot_df.corr()
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, ax=axes[0,1], cbar_kws={'shrink': 0.8})
    axes[0,1].set_title('지역간 상관관계', fontsize=14, fontweight='bold')
    
    # 연도별 트렌드
    yearly_data = df.groupby(['year', 'region'])['ratio'].agg(['mean', 'max']).reset_index()
    yearly_mean = yearly_data.pivot(index='year', columns='region', values='mean')
    
    sns.heatmap(yearly_mean, annot=True, fmt='.3f', cmap='viridis', 
                ax=axes[1,0], cbar_kws={'shrink': 0.8})
    axes[1,0].set_title('연도별 평균 검색 비율', fontsize=14, fontweight='bold')
    
    # 요일별 패턴
    df['day_num'] = df['date'].dt.dayofweek
    weekday_data = df.groupby(['day_num', 'region'])['ratio'].mean().reset_index()
    weekday_matrix = weekday_data.pivot(index='day_num', columns='region', values='ratio')
    weekday_matrix.index = ['월', '화', '수', '목', '금', '토', '일']
    
    sns.heatmap(weekday_matrix, annot=True, fmt='.3f', cmap='plasma', 
                ax=axes[1,1], cbar_kws={'shrink': 0.8})
    axes[1,1].set_title('요일별 평균 검색 비율', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./results/figures/heatmap_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_statistical_plots(df):
    """통계 분석 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 히스토그램
    for i, region in enumerate(df['region'].unique()):
        region_data = df[df['region'] == region]['ratio']
        axes[0, i].hist(region_data, bins=50, alpha=0.7, color=f'C{i}', edgecolor='black')
        axes[0, i].set_title(f'{region}\n검색 비율 분포', fontweight='bold')
        axes[0, i].set_xlabel('검색 비율')
        axes[0, i].set_ylabel('빈도')
        axes[0, i].grid(True, alpha=0.3)
        
        # 통계 정보 추가
        mean_val = region_data.mean()
        std_val = region_data.std()
        axes[0, i].axvline(mean_val, color='red', linestyle='--', 
                          label=f'평균: {mean_val:.3f}')
        axes[0, i].legend()
    
    # 시계열 분해 (트렌드)
    for i, region in enumerate(df['region'].unique()):
        region_data = df[df['region'] == region].set_index('date')['ratio']
        
        # 30일 이동평균
        ma_30 = region_data.rolling(window=30).mean()
        
        axes[1, i].plot(region_data.index, region_data.values, 
                       alpha=0.3, color='gray', label='원본')
        axes[1, i].plot(ma_30.index, ma_30.values, 
                       color=f'C{i}', linewidth=2, label='30일 이동평균')
        axes[1, i].set_title(f'{region}\n트렌드 분석', fontweight='bold')
        axes[1, i].set_xlabel('날짜')
        axes[1, i].set_ylabel('검색 비율')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results/figures/statistical_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_interactive_plotly(df):
    """Plotly 인터랙티브 차트 생성"""
    
    # 1. 시계열 차트
    fig1 = px.line(df, x='date', y='ratio', color='region',
                   title='드론 라이트 쇼 검색 트렌드 (인터랙티브)',
                   labels={'ratio': '검색 비율', 'date': '날짜'})
    fig1.update_layout(width=1200, height=600)
    fig1.write_html('./results/figures/interactive_timeseries.html')
    
    # 2. 3D 서피스 플롯
    pivot_df = df.pivot(index='date', columns='region', values='ratio')
    
    fig2 = go.Figure()
    
    for region in pivot_df.columns:
        fig2.add_trace(go.Scatter3d(
            x=pivot_df.index,
            y=[region] * len(pivot_df),
            z=pivot_df[region],
            mode='lines',
            name=region,
            line=dict(width=4)
        ))
    
    fig2.update_layout(
        title='3D 검색 트렌드 시각화',
        scene=dict(
            xaxis_title='날짜',
            yaxis_title='지역',
            zaxis_title='검색 비율'
        ),
        width=1000, height=700
    )
    fig2.write_html('./results/figures/3d_visualization.html')
    
    # 3. 애니메이션 차트
    df_monthly = df.groupby(['year', 'month', 'region'])['ratio'].mean().reset_index()
    df_monthly['date_str'] = df_monthly['year'].astype(str) + '-' + df_monthly['month'].astype(str).str.zfill(2)
    
    fig3 = px.bar(df_monthly, x='region', y='ratio', color='region',
                  animation_frame='date_str',
                  title='월별 검색 트렌드 애니메이션',
                  labels={'ratio': '평균 검색 비율'})
    fig3.write_html('./results/figures/animated_trends.html')
    
    print("✅ 인터랙티브 차트가 생성되었습니다:")
    print("   - interactive_timeseries.html")
    print("   - 3d_visualization.html") 
    print("   - animated_trends.html")

def create_summary_report(df):
    """요약 리포트 생성"""
    print("\n" + "="*60)
    print("🎯 드론 라이트 쇼 검색 트렌드 분석 요약 리포트")
    print("="*60)
    
    # 기본 통계
    print("\n📊 기본 통계:")
    for region in df['region'].unique():
        region_data = df[df['region'] == region]['ratio']
        print(f"📍 {region}:")
        print(f"   평균: {region_data.mean():.4f}")
        print(f"   최대: {region_data.max():.4f}")
        print(f"   표준편차: {region_data.std():.4f}")
    
    # 최고 검색 기록
    max_idx = df['ratio'].idxmax()
    max_region = df.loc[max_idx, 'region']
    max_date = df.loc[max_idx, 'date']
    max_value = df.loc[max_idx, 'ratio']
    
    print(f"\n🏆 최고 검색 기록:")
    print(f"   지역: {max_region}")
    print(f"   날짜: {max_date.strftime('%Y-%m-%d')}")
    print(f"   검색비율: {max_value:.4f}")
    
    # 월별 피크
    print(f"\n📅 월별 최고 검색 시기:")
    monthly_peak = df.groupby(['region', 'month'])['ratio'].mean().reset_index()
    for region in df['region'].unique():
        region_monthly = monthly_peak[monthly_peak['region'] == region]
        peak_month = region_monthly.loc[region_monthly['ratio'].idxmax(), 'month']
        print(f"   {region}: {int(peak_month)}월")
    
    print(f"\n✨ 시각화 파일이 ./results/figures/ 에 저장되었습니다!")

def main():
    """메인 실행 함수"""
    print("🎨 드론 라이트 쇼 검색 트렌드 시각화를 생성합니다...")
    
    # 데이터 로드
    df = load_data()
    print(f"✅ 데이터 로드 완료: {len(df)} 레코드")
    
    # 시각화 생성
    print("\n📈 1. 종합 시계열 분석 차트 생성...")
    create_time_series_plot(df)
    
    print("🔥 2. 히트맵 분석 차트 생성...")
    create_heatmap_analysis(df)
    
    print("📊 3. 통계 분석 차트 생성...")
    create_statistical_plots(df)
    
    print("🌐 4. 인터랙티브 차트 생성...")
    create_interactive_plotly(df)
    
    print("📋 5. 요약 리포트 생성...")
    create_summary_report(df)

if __name__ == "__main__":
    main()
