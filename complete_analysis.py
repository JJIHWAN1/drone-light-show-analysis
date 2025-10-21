#!/usr/bin/env python3
"""
드론 라이트 쇼 검색 트렌드 완전 분석 스크립트
모든 분석과 시각화를 한 번에 실행
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from scipy import stats
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class DroneSearchAnalyzer:
    """드론 라이트 쇼 검색 데이터 완전 분석 클래스"""
    
    def __init__(self, data_path):
        self.df = self.load_and_preprocess(data_path)
        self.pivot_df = self.create_pivot_table()
        
    def load_and_preprocess(self, data_path):
        """데이터 로드 및 전처리"""
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.day_name()
        df['day_of_week_num'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        return df
    
    def create_pivot_table(self):
        """피벗 테이블 생성"""
        pivot_df = self.df.pivot(index='date', columns='region', values='ratio')
        pivot_df.fillna(0, inplace=True)
        return pivot_df
    
    def get_basic_statistics(self):
        """기본 통계 계산"""
        stats_dict = {}
        for region in self.df['region'].unique():
            region_data = self.df[self.df['region'] == region]['ratio']
            stats_dict[region] = {
                'count': len(region_data),
                'mean': region_data.mean(),
                'std': region_data.std(),
                'min': region_data.min(),
                'max': region_data.max(),
                'median': region_data.median(),
                'cv': region_data.std() / region_data.mean() if region_data.mean() > 0 else 0
            }
        return stats_dict
    
    def detect_peaks(self, height_multiplier=2, distance=7):
        """피크 탐지"""
        peak_results = {}
        for region in self.pivot_df.columns:
            data = self.pivot_df[region].values
            threshold = np.mean(data) + height_multiplier * np.std(data)
            peaks, properties = find_peaks(data, height=threshold, distance=distance)
            
            peak_results[region] = {
                'peak_indices': peaks,
                'peak_dates': self.pivot_df.index[peaks],
                'peak_values': data[peaks],
                'threshold': threshold,
                'peak_count': len(peaks)
            }
        return peak_results
    
    def calculate_correlation_matrix(self):
        """상관관계 계산"""
        return self.pivot_df.corr()
    
    def create_comprehensive_plots(self):
        """종합 시각화 생성"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        
        # 1. 전체 시계열 트렌드
        for region in self.df['region'].unique():
            region_data = self.df[self.df['region'] == region]
            axes[0,0].plot(region_data['date'], region_data['ratio'], 
                          label=region, linewidth=2, alpha=0.8)
        axes[0,0].set_title('지역별 검색 트렌드 (전체)', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('날짜')
        axes[0,0].set_ylabel('검색 비율')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 2024년 상세 트렌드
        df_2024 = self.df[self.df['year'] == 2024]
        for region in self.df['region'].unique():
            region_data = df_2024[df_2024['region'] == region]
            axes[0,1].plot(region_data['date'], region_data['ratio'], 
                          label=region, linewidth=2, marker='o', markersize=3)
        axes[0,1].set_title('2024년 검색 트렌드', fontsize=14, fontweight='bold')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 월별 평균
        monthly_avg = self.df.groupby(['month', 'region'])['ratio'].mean().reset_index()
        sns.lineplot(data=monthly_avg, x='month', y='ratio', hue='region', 
                    marker='o', linewidth=3, markersize=8, ax=axes[0,2])
        axes[0,2].set_title('월별 평균 검색 비율', fontsize=14, fontweight='bold')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. 지역별 분포
        sns.boxplot(data=self.df, x='region', y='ratio', ax=axes[1,0])
        axes[1,0].set_title('지역별 검색 비율 분포', fontsize=14, fontweight='bold')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 5. 상관관계 히트맵
        correlation_matrix = self.calculate_correlation_matrix()
        sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, ax=axes[1,1])
        axes[1,1].set_title('지역간 상관관계', fontsize=14, fontweight='bold')
        
        # 6. 월별 히트맵
        monthly_pivot = self.df.groupby(['month', 'region'])['ratio'].mean().reset_index()
        monthly_matrix = monthly_pivot.pivot(index='month', columns='region', values='ratio')
        sns.heatmap(monthly_matrix, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1,2])
        axes[1,2].set_title('월별 검색 패턴', fontsize=14, fontweight='bold')
        
        # 7-9. 지역별 히스토그램
        for i, region in enumerate(self.df['region'].unique()):
            region_data = self.df[self.df['region'] == region]['ratio']
            axes[2,i].hist(region_data, bins=50, alpha=0.7, color=f'C{i}', edgecolor='black')
            axes[2,i].set_title(f'{region} 분포', fontweight='bold')
            axes[2,i].set_xlabel('검색 비율')
            axes[2,i].set_ylabel('빈도')
            axes[2,i].grid(True, alpha=0.3)
            
            # 평균선 추가
            mean_val = region_data.mean()
            axes[2,i].axvline(mean_val, color='red', linestyle='--', 
                             label=f'평균: {mean_val:.3f}')
            axes[2,i].legend()
        
        plt.tight_layout()
        plt.savefig('./results/figures/complete_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_peak_analysis_plot(self):
        """피크 분석 시각화"""
        peaks = self.detect_peaks()
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        for i, region in enumerate(self.pivot_df.columns):
            # 원본 데이터
            axes[i].plot(self.pivot_df.index, self.pivot_df[region], 
                        label=f'{region} 트렌드', linewidth=1.5, alpha=0.8)
            
            # 피크 포인트
            peak_info = peaks[region]
            axes[i].scatter(peak_info['peak_dates'], peak_info['peak_values'], 
                           color='red', s=50, zorder=5, label=f'피크 ({len(peak_info["peak_dates"])}개)')
            
            # 임계값 선
            axes[i].axhline(y=peak_info['threshold'], color='gray', 
                           linestyle='--', alpha=0.7, label=f'임계값: {peak_info["threshold"]:.3f}')
            
            axes[i].set_title(f'{region} - 주요 검색 피크 분석', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('검색 비율')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('날짜')
        plt.tight_layout()
        plt.savefig('./results/figures/peak_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_charts(self):
        """인터랙티브 차트 생성"""
        
        # 1. 시계열 차트
        fig1 = px.line(self.df, x='date', y='ratio', color='region',
                      title='드론 라이트 쇼 검색 트렌드 (인터랙티브)',
                      labels={'ratio': '검색 비율', 'date': '날짜'})
        fig1.update_layout(width=1200, height=600)
        fig1.write_html('./results/figures/interactive_timeseries.html')
        
        # 2. 3D 시각화
        fig2 = go.Figure()
        for region in self.pivot_df.columns:
            fig2.add_trace(go.Scatter3d(
                x=self.pivot_df.index,
                y=[region] * len(self.pivot_df),
                z=self.pivot_df[region],
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
        df_monthly = self.df.groupby(['year', 'month', 'region'])['ratio'].mean().reset_index()
        df_monthly['date_str'] = df_monthly['year'].astype(str) + '-' + df_monthly['month'].astype(str).str.zfill(2)
        
        fig3 = px.bar(df_monthly, x='region', y='ratio', color='region',
                     animation_frame='date_str',
                     title='월별 검색 트렌드 애니메이션')
        fig3.write_html('./results/figures/animated_trends.html')
        
        return fig1, fig2, fig3
    
    def generate_report(self):
        """종합 리포트 생성"""
        print("🚁 드론 라이트 쇼 검색 트렌드 분석 리포트")
        print("=" * 60)
        
        # 기본 정보
        print(f"\n📊 데이터 개요:")
        print(f"   총 레코드: {len(self.df):,}개")
        print(f"   분석 기간: {self.df['date'].min()} ~ {self.df['date'].max()}")
        print(f"   분석 지역: {', '.join(self.df['region'].unique())}")
        
        # 기본 통계
        stats = self.get_basic_statistics()
        print(f"\n📈 지역별 기본 통계:")
        for region, stat in stats.items():
            print(f"   📍 {region}:")
            print(f"      평균: {stat['mean']:.4f}")
            print(f"      최대: {stat['max']:.4f}")
            print(f"      표준편차: {stat['std']:.4f}")
            print(f"      변동계수: {stat['cv']:.4f}")
        
        # 피크 분석
        peaks = self.detect_peaks()
        print(f"\n🔥 주요 검색 피크:")
        for region, peak_info in peaks.items():
            print(f"   📍 {region}: {peak_info['peak_count']}개 피크")
            if len(peak_info['peak_dates']) > 0:
                max_peak_idx = np.argmax(peak_info['peak_values'])
                max_peak_date = peak_info['peak_dates'][max_peak_idx]
                max_peak_value = peak_info['peak_values'][max_peak_idx]
                print(f"      🏆 최고: {max_peak_date.strftime('%Y-%m-%d')} ({max_peak_value:.4f})")
        
        # 상관관계
        corr_matrix = self.calculate_correlation_matrix()
        print(f"\n🔗 지역간 상관관계:")
        regions = list(corr_matrix.columns)
        for i in range(len(regions)):
            for j in range(i+1, len(regions)):
                corr_value = corr_matrix.iloc[i, j]
                print(f"   {regions[i]} ↔ {regions[j]}: {corr_value:.3f}")
        
        # 최고 기록
        max_idx = self.df['ratio'].idxmax()
        max_region = self.df.loc[max_idx, 'region']
        max_date = self.df.loc[max_idx, 'date']
        max_value = self.df.loc[max_idx, 'ratio']
        
        print(f"\n🏆 전체 최고 검색 기록:")
        print(f"   지역: {max_region}")
        print(f"   날짜: {max_date.strftime('%Y-%m-%d')}")
        print(f"   검색비율: {max_value:.4f}")
        
        # 월별 패턴
        monthly_peak = self.df.groupby(['region', 'month'])['ratio'].mean().reset_index()
        print(f"\n📅 지역별 최고 검색 월:")
        for region in self.df['region'].unique():
            region_monthly = monthly_peak[monthly_peak['region'] == region]
            peak_month = region_monthly.loc[region_monthly['ratio'].idxmax(), 'month']
            print(f"   {region}: {int(peak_month)}월")

def main():
    """메인 실행 함수"""
    print("🎨 드론 라이트 쇼 검색 트렌드 완전 분석을 시작합니다...")
    
    # 결과 디렉토리 생성
    os.makedirs('./results/figures', exist_ok=True)
    os.makedirs('./results/reports', exist_ok=True)
    
    # 분석기 초기화
    analyzer = DroneSearchAnalyzer('./data/naver_datalab_fixed.csv')
    
    print("\n📊 1. 종합 시각화 생성...")
    analyzer.create_comprehensive_plots()
    
    print("🔥 2. 피크 분석 시각화...")
    analyzer.create_peak_analysis_plot()
    
    print("🌐 3. 인터랙티브 차트 생성...")
    analyzer.create_interactive_charts()
    
    print("📋 4. 종합 리포트 생성...")
    analyzer.generate_report()
    
    print(f"\n✨ 분석 완료!")
    print(f"📁 결과 파일:")
    print(f"   - 정적 차트: ./results/figures/*.png")
    print(f"   - 인터랙티브: ./results/figures/*.html")

if __name__ == "__main__":
    main()
