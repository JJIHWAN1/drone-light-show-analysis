"""
데이터 처리 및 분석을 위한 유틸리티 함수들
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DroneSearchAnalyzer:
    """드론 라이트 쇼 검색 데이터 분석 클래스"""
    
    def __init__(self, data_path):
        """
        초기화
        
        Args:
            data_path (str): CSV 데이터 파일 경로
        """
        self.df = self.load_and_preprocess(data_path)
        self.pivot_df = self.create_pivot_table()
        
    def load_and_preprocess(self, data_path):
        """
        데이터 로드 및 전처리
        
        Args:
            data_path (str): CSV 파일 경로
            
        Returns:
            pd.DataFrame: 전처리된 데이터프레임
        """
        df = pd.read_csv(data_path)
        
        # 날짜 컬럼 변환
        df['date'] = pd.to_datetime(df['date'])
        
        # 추가 시간 관련 컬럼 생성
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.day_name()
        df['day_of_week_num'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # 계절 정보 추가
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        
        return df
    
    def create_pivot_table(self):
        """
        지역별 피벗 테이블 생성
        
        Returns:
            pd.DataFrame: 날짜를 인덱스로 하고 지역을 컬럼으로 하는 피벗 테이블
        """
        pivot_df = self.df.pivot(index='date', columns='region', values='ratio')
        pivot_df.fillna(0, inplace=True)
        return pivot_df
    
    def get_basic_statistics(self):
        """
        기본 통계 정보 계산
        
        Returns:
            dict: 지역별 기본 통계 정보
        """
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
                'q25': region_data.quantile(0.25),
                'q75': region_data.quantile(0.75),
                'skewness': stats.skew(region_data),
                'kurtosis': stats.kurtosis(region_data),
                'cv': region_data.std() / region_data.mean() if region_data.mean() > 0 else 0
            }
            
        return stats_dict
    
    def detect_peaks(self, region=None, height_multiplier=2, distance=7):
        """
        피크 탐지
        
        Args:
            region (str): 분석할 지역 (None이면 모든 지역)
            height_multiplier (float): 임계값 계산을 위한 표준편차 배수
            distance (int): 피크 간 최소 거리 (일)
            
        Returns:
            dict: 지역별 피크 정보
        """
        peak_results = {}
        
        regions = [region] if region else self.pivot_df.columns
        
        for reg in regions:
            data = self.pivot_df[reg].values
            
            # 임계값 설정 (평균 + n*표준편차)
            threshold = np.mean(data) + height_multiplier * np.std(data)
            
            # 피크 탐지
            peaks, properties = find_peaks(data, height=threshold, distance=distance)
            
            peak_dates = self.pivot_df.index[peaks]
            peak_values = data[peaks]
            
            peak_results[reg] = {
                'peak_indices': peaks,
                'peak_dates': peak_dates,
                'peak_values': peak_values,
                'threshold': threshold,
                'peak_count': len(peaks)
            }
            
        return peak_results
    
    def calculate_correlation_matrix(self):
        """
        지역간 상관관계 계산
        
        Returns:
            pd.DataFrame: 상관관계 매트릭스
        """
        return self.pivot_df.corr()
    
    def analyze_temporal_patterns(self):
        """
        시간적 패턴 분석
        
        Returns:
            dict: 다양한 시간적 패턴 분석 결과
        """
        patterns = {}
        
        # 월별 패턴
        patterns['monthly'] = self.df.groupby(['month', 'region'])['ratio'].agg(['mean', 'std', 'max']).reset_index()
        
        # 요일별 패턴
        patterns['weekday'] = self.df.groupby(['day_of_week_num', 'region'])['ratio'].agg(['mean', 'std', 'max']).reset_index()
        
        # 계절별 패턴
        patterns['seasonal'] = self.df.groupby(['season', 'region'])['ratio'].agg(['mean', 'std', 'max']).reset_index()
        
        # 연도별 패턴
        patterns['yearly'] = self.df.groupby(['year', 'region'])['ratio'].agg(['mean', 'std', 'max']).reset_index()
        
        # 분기별 패턴
        patterns['quarterly'] = self.df.groupby(['quarter', 'region'])['ratio'].agg(['mean', 'std', 'max']).reset_index()
        
        return patterns
    
    def calculate_moving_averages(self, windows=[7, 14, 30]):
        """
        이동평균 계산
        
        Args:
            windows (list): 이동평균 윈도우 크기 리스트
            
        Returns:
            pd.DataFrame: 이동평균이 추가된 피벗 테이블
        """
        ma_df = self.pivot_df.copy()
        
        for window in windows:
            for region in self.pivot_df.columns:
                ma_df[f'{region}_MA{window}'] = self.pivot_df[region].rolling(window=window).mean()
                
        return ma_df
    
    def detect_anomalies(self, method='zscore', threshold=3):
        """
        이상치 탐지
        
        Args:
            method (str): 탐지 방법 ('zscore', 'iqr')
            threshold (float): 임계값
            
        Returns:
            dict: 지역별 이상치 정보
        """
        anomalies = {}
        
        for region in self.pivot_df.columns:
            data = self.pivot_df[region]
            
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                anomaly_mask = z_scores > threshold
                
            elif method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                anomaly_mask = (data < lower_bound) | (data > upper_bound)
            
            anomaly_dates = data[anomaly_mask].index
            anomaly_values = data[anomaly_mask].values
            
            anomalies[region] = {
                'dates': anomaly_dates,
                'values': anomaly_values,
                'count': len(anomaly_dates)
            }
            
        return anomalies
    
    def perform_trend_analysis(self):
        """
        트렌드 분석 수행
        
        Returns:
            dict: 트렌드 분석 결과
        """
        trend_results = {}
        
        for region in self.pivot_df.columns:
            data = self.pivot_df[region]
            
            # 선형 회귀를 통한 트렌드 분석
            x = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            
            # 변화율 계산
            pct_change = data.pct_change().dropna()
            
            trend_results[region] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                'avg_pct_change': pct_change.mean(),
                'volatility': pct_change.std()
            }
            
        return trend_results
    
    def compare_regions(self):
        """
        지역간 비교 분석
        
        Returns:
            dict: 지역간 비교 결과
        """
        comparison = {}
        
        # 기본 통계 비교
        stats_df = pd.DataFrame()
        for region in self.pivot_df.columns:
            stats_df[region] = self.pivot_df[region].describe()
        
        comparison['descriptive_stats'] = stats_df
        
        # ANOVA 테스트
        region_data = [self.pivot_df[region].values for region in self.pivot_df.columns]
        f_stat, p_value = stats.f_oneway(*region_data)
        
        comparison['anova'] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # 상관관계
        comparison['correlation'] = self.calculate_correlation_matrix()
        
        return comparison
    
    def generate_summary_report(self):
        """
        종합 분석 리포트 생성
        
        Returns:
            dict: 종합 분석 결과
        """
        report = {}
        
        # 기본 정보
        report['data_info'] = {
            'total_records': len(self.df),
            'date_range': f"{self.df['date'].min()} ~ {self.df['date'].max()}",
            'regions': list(self.df['region'].unique()),
            'total_days': (self.df['date'].max() - self.df['date'].min()).days + 1
        }
        
        # 기본 통계
        report['basic_statistics'] = self.get_basic_statistics()
        
        # 피크 분석
        report['peak_analysis'] = self.detect_peaks()
        
        # 시간적 패턴
        report['temporal_patterns'] = self.analyze_temporal_patterns()
        
        # 트렌드 분석
        report['trend_analysis'] = self.perform_trend_analysis()
        
        # 지역간 비교
        report['region_comparison'] = self.compare_regions()
        
        # 이상치 탐지
        report['anomalies'] = self.detect_anomalies()
        
        return report

def save_analysis_results(analyzer, output_dir):
    """
    분석 결과를 파일로 저장
    
    Args:
        analyzer (DroneSearchAnalyzer): 분석기 객체
        output_dir (str): 출력 디렉토리 경로
    """
    import os
    
    # 디렉토리 생성
    os.makedirs(f"{output_dir}/reports", exist_ok=True)
    os.makedirs(f"{output_dir}/data", exist_ok=True)
    
    # 종합 리포트 생성
    report = analyzer.generate_summary_report()
    
    # 전처리된 데이터 저장
    analyzer.df.to_csv(f"{output_dir}/data/processed_data.csv", index=False)
    analyzer.pivot_df.to_csv(f"{output_dir}/data/pivot_data.csv")
    
    # 기본 통계 저장
    basic_stats = pd.DataFrame(report['basic_statistics']).T
    basic_stats.to_csv(f"{output_dir}/reports/basic_statistics.csv")
    
    # 상관관계 매트릭스 저장
    correlation_matrix = analyzer.calculate_correlation_matrix()
    correlation_matrix.to_csv(f"{output_dir}/reports/correlation_matrix.csv")
    
    # 시간적 패턴 저장
    patterns = report['temporal_patterns']
    for pattern_name, pattern_data in patterns.items():
        pattern_data.to_csv(f"{output_dir}/reports/{pattern_name}_patterns.csv", index=False)
    
    print(f"분석 결과가 {output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    # 사용 예시
    analyzer = DroneSearchAnalyzer("../data/naver_datalab_fixed.csv")
    
    # 기본 통계 출력
    stats = analyzer.get_basic_statistics()
    print("기본 통계:")
    for region, stat in stats.items():
        print(f"{region}: 평균={stat['mean']:.4f}, 최대={stat['max']:.4f}")
    
    # 피크 탐지
    peaks = analyzer.detect_peaks()
    print("\n피크 분석:")
    for region, peak_info in peaks.items():
        print(f"{region}: {peak_info['peak_count']}개 피크 탐지")
    
    # 결과 저장
    save_analysis_results(analyzer, "../results")
