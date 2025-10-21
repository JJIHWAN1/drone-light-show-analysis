#!/usr/bin/env python3
"""
드론 라이트 쇼 검색 트렌드 분석 실행 스크립트

이 스크립트는 전체 분석 파이프라인을 실행합니다:
1. 데이터 로드 및 전처리
2. 탐색적 데이터 분석
3. 트렌드 분석
4. 시각화 생성
5. 리포트 작성
"""

import sys
import os

# 현재 스크립트의 디렉토리를 기준으로 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from data_processing import DroneSearchAnalyzer, save_analysis_results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def main():
    """메인 분석 실행 함수"""
    
    print("🚁 드론 라이트 쇼 검색 트렌드 분석을 시작합니다...")
    print("=" * 60)
    
    # 1. 데이터 로드 및 분석기 초기화
    print("📊 1. 데이터 로드 및 전처리...")
    try:
        analyzer = DroneSearchAnalyzer("./data/naver_datalab_fixed.csv")
        print(f"   ✅ 데이터 로드 완료: {len(analyzer.df)} 레코드")
        print(f"   📅 분석 기간: {analyzer.df['date'].min()} ~ {analyzer.df['date'].max()}")
        print(f"   🏢 분석 지역: {', '.join(analyzer.df['region'].unique())}")
    except Exception as e:
        print(f"   ❌ 데이터 로드 실패: {e}")
        return
    
    # 2. 기본 통계 분석
    print("\n📈 2. 기본 통계 분석...")
    basic_stats = analyzer.get_basic_statistics()
    
    print("   지역별 기본 통계:")
    for region, stats in basic_stats.items():
        print(f"   📍 {region}:")
        print(f"      - 평균 검색비율: {stats['mean']:.4f}")
        print(f"      - 최대 검색비율: {stats['max']:.4f}")
        print(f"      - 표준편차: {stats['std']:.4f}")
        print(f"      - 변동계수: {stats['cv']:.4f}")
    
    # 3. 피크 분석
    print("\n🔥 3. 주요 검색 피크 분석...")
    peaks = analyzer.detect_peaks(height_multiplier=2)
    
    for region, peak_info in peaks.items():
        print(f"   📍 {region}: {peak_info['peak_count']}개 주요 피크 발견")
        if len(peak_info['peak_dates']) > 0:
            max_peak_idx = np.argmax(peak_info['peak_values'])
            max_peak_date = peak_info['peak_dates'][max_peak_idx]
            max_peak_value = peak_info['peak_values'][max_peak_idx]
            print(f"      🏆 최고 피크: {max_peak_date.strftime('%Y-%m-%d')} ({max_peak_value:.4f})")
    
    # 4. 상관관계 분석
    print("\n🔗 4. 지역간 상관관계 분석...")
    correlation_matrix = analyzer.calculate_correlation_matrix()
    
    print("   지역간 상관계수:")
    regions = list(correlation_matrix.columns)
    for i in range(len(regions)):
        for j in range(i+1, len(regions)):
            corr_value = correlation_matrix.iloc[i, j]
            print(f"   📊 {regions[i]} ↔ {regions[j]}: {corr_value:.3f}")
    
    # 5. 시간적 패턴 분석
    print("\n📅 5. 시간적 패턴 분석...")
    patterns = analyzer.analyze_temporal_patterns()
    
    # 월별 패턴
    monthly_peak = patterns['monthly'].groupby('region')['mean'].idxmax()
    print("   월별 패턴:")
    for region in analyzer.df['region'].unique():
        region_monthly = patterns['monthly'][patterns['monthly']['region'] == region]
        peak_month_idx = region_monthly['mean'].idxmax()
        peak_month = region_monthly.loc[peak_month_idx, 'month']
        peak_value = region_monthly.loc[peak_month_idx, 'mean']
        print(f"   📍 {region}: {int(peak_month)}월이 최고 ({peak_value:.4f})")
    
    # 6. 트렌드 분석
    print("\n📈 6. 장기 트렌드 분석...")
    trend_results = analyzer.perform_trend_analysis()
    
    for region, trend in trend_results.items():
        direction = "상승" if trend['slope'] > 0 else "하락"
        significance = "유의함" if trend['p_value'] < 0.05 else "유의하지 않음"
        print(f"   📍 {region}:")
        print(f"      - 트렌드: {direction} (기울기: {trend['slope']:.6f})")
        print(f"      - R²: {trend['r_squared']:.4f}")
        print(f"      - 통계적 유의성: {significance} (p={trend['p_value']:.4f})")
    
    # 7. 이상치 탐지
    print("\n⚠️  7. 이상치 탐지...")
    anomalies = analyzer.detect_anomalies(method='zscore', threshold=3)
    
    for region, anomaly_info in anomalies.items():
        print(f"   📍 {region}: {anomaly_info['count']}개 이상치 발견")
        if anomaly_info['count'] > 0:
            max_anomaly_idx = np.argmax(anomaly_info['values'])
            max_anomaly_date = anomaly_info['dates'][max_anomaly_idx]
            max_anomaly_value = anomaly_info['values'][max_anomaly_idx]
            print(f"      🚨 최대 이상치: {max_anomaly_date.strftime('%Y-%m-%d')} ({max_anomaly_value:.4f})")
    
    # 8. 결과 저장
    print("\n💾 8. 분석 결과 저장...")
    try:
        save_analysis_results(analyzer, "./results")
        print("   ✅ 분석 결과 저장 완료")
        print("   📁 저장 위치:")
        print("      - 전처리된 데이터: ./results/data/")
        print("      - 분석 리포트: ./results/reports/")
        print("      - 시각화 결과: ./results/figures/")
    except Exception as e:
        print(f"   ❌ 결과 저장 실패: {e}")
    
    # 9. 주요 인사이트 요약
    print("\n🎯 9. 주요 분석 인사이트:")
    print("=" * 60)
    
    # 가장 활발한 지역
    total_searches = analyzer.df.groupby('region')['ratio'].sum()
    most_active_region = total_searches.idxmax()
    print(f"🏆 가장 활발한 검색 지역: {most_active_region}")
    
    # 전체 최고 피크
    max_search_idx = analyzer.df['ratio'].idxmax()
    max_search_region = analyzer.df.loc[max_search_idx, 'region']
    max_search_date = analyzer.df.loc[max_search_idx, 'date']
    max_search_value = analyzer.df.loc[max_search_idx, 'ratio']
    print(f"📈 전체 최고 검색 기록: {max_search_region} ({max_search_date.strftime('%Y-%m-%d')}, {max_search_value:.4f})")
    
    # 가장 상관관계가 높은 지역 쌍
    corr_values = []
    region_pairs = []
    regions = list(correlation_matrix.columns)
    for i in range(len(regions)):
        for j in range(i+1, len(regions)):
            corr_values.append(correlation_matrix.iloc[i, j])
            region_pairs.append((regions[i], regions[j]))
    
    max_corr_idx = np.argmax(corr_values)
    max_corr_pair = region_pairs[max_corr_idx]
    max_corr_value = corr_values[max_corr_idx]
    print(f"🔗 가장 유사한 검색 패턴: {max_corr_pair[0]} ↔ {max_corr_pair[1]} (상관계수: {max_corr_value:.3f})")
    
    # 계절성 패턴
    seasonal_avg = analyzer.df.groupby(['season', 'region'])['ratio'].mean().reset_index()
    peak_season = seasonal_avg.groupby('season')['ratio'].mean().idxmax()
    print(f"🌸 검색이 가장 활발한 계절: {peak_season}")
    
    print("\n✨ 분석 완료! 대시보드를 실행하려면 다음 명령어를 사용하세요:")
    print("   cd src && python dashboard.py")
    print("   브라우저에서 http://localhost:8050 접속")

if __name__ == "__main__":
    main()
