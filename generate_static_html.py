#!/usr/bin/env python3
"""
드론 라이트 쇼 검색 트렌드 정적 HTML 생성기
- 인터랙티브 차트들을 HTML 파일로 저장
- 파일 공유로 누구나 브라우저에서 볼 수 있음
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime
import os

def load_data():
    """데이터 로드"""
    data_path = "/Users/kimjihwan/Downloads/naver_datalab_fixed.csv"
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    return df

def create_main_dashboard():
    """메인 대시보드 HTML 생성"""
    df = load_data()
    
    # 서브플롯 생성 (2x2 레이아웃)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('지역별 검색 트렌드', '월별 평균 패턴', '연도별 비교', '지역별 분포'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 1. 시계열 트렌드
    for i, region in enumerate(df['region'].unique()):
        region_data = df[df['region'] == region]
        fig.add_trace(
            go.Scatter(x=region_data['date'], y=region_data['ratio'],
                      name=region, line=dict(color=colors[i], width=3)),
            row=1, col=1
        )
    
    # 2. 월별 평균
    monthly_avg = df.groupby(['month', 'region'])['ratio'].mean().reset_index()
    for i, region in enumerate(df['region'].unique()):
        region_data = monthly_avg[monthly_avg['region'] == region]
        fig.add_trace(
            go.Scatter(x=region_data['month'], y=region_data['ratio'],
                      name=f"{region} (월별)", line=dict(color=colors[i], width=3),
                      marker=dict(size=8), showlegend=False),
            row=1, col=2
        )
    
    # 3. 연도별 비교
    yearly_avg = df.groupby(['year', 'region'])['ratio'].mean().reset_index()
    for i, region in enumerate(df['region'].unique()):
        region_data = yearly_avg[yearly_avg['region'] == region]
        fig.add_trace(
            go.Bar(x=region_data['year'], y=region_data['ratio'],
                   name=f"{region} (연도별)", marker_color=colors[i],
                   showlegend=False),
            row=2, col=1
        )
    
    # 4. 박스플롯
    for i, region in enumerate(df['region'].unique()):
        region_data = df[df['region'] == region]['ratio']
        fig.add_trace(
            go.Box(y=region_data, name=region, marker_color=colors[i],
                   showlegend=False),
            row=2, col=2
        )
    
    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text="🚁 드론 라이트 쇼 검색 트렌드 종합 분석",
            x=0.5,
            font=dict(size=24, family="Arial Black")
        ),
        height=800,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        template="plotly_white"
    )
    
    # 축 레이블 설정
    fig.update_xaxes(title_text="날짜", row=1, col=1)
    fig.update_xaxes(title_text="월", row=1, col=2)
    fig.update_xaxes(title_text="연도", row=2, col=1)
    fig.update_xaxes(title_text="지역", row=2, col=2)
    
    fig.update_yaxes(title_text="검색 비율", row=1, col=1)
    fig.update_yaxes(title_text="평균 검색 비율", row=1, col=2)
    fig.update_yaxes(title_text="평균 검색 비율", row=2, col=1)
    fig.update_yaxes(title_text="검색 비율 분포", row=2, col=2)
    
    return fig

def create_individual_charts():
    """개별 차트들 생성"""
    df = load_data()
    charts = {}
    
    # 1. 기본 시계열 차트
    fig1 = px.line(df, x='date', y='ratio', color='region',
                   title='지역별 드론 라이트 쇼 검색 트렌드',
                   labels={'ratio': '검색 비율', 'date': '날짜', 'region': '지역'})
    fig1.update_layout(height=600, template="plotly_white")
    charts['timeseries'] = fig1
    
    # 2. 월별 패턴
    monthly_avg = df.groupby(['month', 'region'])['ratio'].mean().reset_index()
    fig2 = px.line(monthly_avg, x='month', y='ratio', color='region',
                   title='월별 평균 검색 패턴', markers=True,
                   labels={'ratio': '평균 검색 비율', 'month': '월', 'region': '지역'})
    fig2.update_layout(height=500, template="plotly_white")
    charts['monthly'] = fig2
    
    # 3. 히트맵
    monthly_pivot = monthly_avg.pivot(index='month', columns='region', values='ratio')
    fig3 = px.imshow(monthly_pivot, 
                     title='월별-지역별 검색 비율 히트맵',
                     labels=dict(x="지역", y="월", color="검색 비율"),
                     aspect="auto")
    fig3.update_layout(height=500, template="plotly_white")
    charts['heatmap'] = fig3
    
    # 4. 박스플롯
    fig4 = px.box(df, x='region', y='ratio',
                  title='지역별 검색 비율 분포',
                  labels={'ratio': '검색 비율', 'region': '지역'})
    fig4.update_layout(height=500, template="plotly_white")
    charts['boxplot'] = fig4
    
    return charts

def generate_html_files():
    """HTML 파일들 생성"""
    print("🎨 정적 HTML 파일 생성을 시작합니다...")
    
    # 결과 디렉토리 생성
    output_dir = "/Users/kimjihwan/CascadeProjects/drone-light-show-analysis/html_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 메인 대시보드
    print("📊 1. 메인 대시보드 생성 중...")
    main_fig = create_main_dashboard()
    main_html = os.path.join(output_dir, "drone_analysis_dashboard.html")
    pyo.plot(main_fig, filename=main_html, auto_open=False)
    print(f"✅ 저장됨: {main_html}")
    
    # 2. 개별 차트들
    print("📈 2. 개별 차트들 생성 중...")
    charts = create_individual_charts()
    
    for chart_name, fig in charts.items():
        filename = os.path.join(output_dir, f"drone_analysis_{chart_name}.html")
        pyo.plot(fig, filename=filename, auto_open=False)
        print(f"✅ 저장됨: {filename}")
    
    # 3. 통합 페이지 생성
    print("🔗 3. 통합 인덱스 페이지 생성 중...")
    create_index_page(output_dir)
    
    print(f"\n🎉 모든 HTML 파일이 생성되었습니다!")
    print(f"📁 저장 위치: {output_dir}")
    print(f"\n📋 생성된 파일들:")
    print(f"   • drone_analysis_dashboard.html (메인 대시보드)")
    print(f"   • drone_analysis_timeseries.html (시계열 차트)")
    print(f"   • drone_analysis_monthly.html (월별 패턴)")
    print(f"   • drone_analysis_heatmap.html (히트맵)")
    print(f"   • drone_analysis_boxplot.html (분포 차트)")
    print(f"   • index.html (통합 페이지)")
    
    return output_dir

def create_index_page(output_dir):
    """통합 인덱스 페이지 생성"""
    html_content = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚁 드론 라이트 쇼 검색 트렌드 분석</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .card h3 {
            color: #333;
            margin-top: 0;
        }
        .card p {
            color: #666;
            line-height: 1.6;
        }
        .btn {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 5px;
            transition: background 0.3s ease;
        }
        .btn:hover {
            background: #5a67d8;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚁 드론 라이트 쇼 검색 트렌드 분석</h1>
        <p>네이버 데이터랩 기반 지역별 검색 패턴 분석 결과</p>
        <p>생성일: """ + datetime.now().strftime('%Y년 %m월 %d일 %H:%M') + """</p>
    </div>
    
    <div class="grid">
        <div class="card">
            <h3>📊 메인 대시보드</h3>
            <p>모든 분석 결과를 한 눈에 볼 수 있는 종합 대시보드입니다. 4개의 차트가 하나의 페이지에 통합되어 있습니다.</p>
            <a href="drone_analysis_dashboard.html" class="btn">대시보드 열기</a>
        </div>
        
        <div class="card">
            <h3>📈 시계열 트렌드</h3>
            <p>2023년부터 2025년까지 지역별 검색 트렌드를 시간 순서로 보여줍니다. 줌인/줌아웃 가능합니다.</p>
            <a href="drone_analysis_timeseries.html" class="btn">차트 열기</a>
        </div>
        
        <div class="card">
            <h3>📅 월별 패턴</h3>
            <p>각 지역의 월별 평균 검색 패턴을 분석합니다. 계절성과 성수기를 파악할 수 있습니다.</p>
            <a href="drone_analysis_monthly.html" class="btn">차트 열기</a>
        </div>
        
        <div class="card">
            <h3>🔥 히트맵 분석</h3>
            <p>월별-지역별 검색 비율을 색상으로 표현한 히트맵입니다. 패턴을 직관적으로 파악할 수 있습니다.</p>
            <a href="drone_analysis_heatmap.html" class="btn">차트 열기</a>
        </div>
        
        <div class="card">
            <h3>📦 분포 분석</h3>
            <p>지역별 검색 비율의 분포를 박스플롯으로 보여줍니다. 평균, 중앙값, 이상치를 확인할 수 있습니다.</p>
            <a href="drone_analysis_boxplot.html" class="btn">차트 열기</a>
        </div>
        
        <div class="card">
            <h3>💡 사용 방법</h3>
            <p>• 각 차트는 인터랙티브합니다 (줌, 필터링 가능)<br>
               • 범례를 클릭하여 지역별 표시/숨김 가능<br>
               • 마우스 호버로 상세 정보 확인<br>
               • 오프라인에서도 사용 가능</p>
        </div>
    </div>
</body>
</html>
"""
    
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 인덱스 페이지 저장됨: {index_path}")

def main():
    """메인 실행 함수"""
    try:
        output_dir = generate_html_files()
        
        print(f"\n🎯 사용 방법:")
        print(f"1. {output_dir} 폴더를 통째로 공유")
        print(f"2. index.html 파일을 더블클릭하여 시작")
        print(f"3. 또는 개별 HTML 파일들을 직접 열기")
        
        print(f"\n📤 공유 방법:")
        print(f"• 카카오톡: 폴더를 압축해서 전송")
        print(f"• 이메일: HTML 파일들을 첨부")
        print(f"• USB: 폴더를 복사")
        print(f"• 클라우드: 구글드라이브, 드롭박스 등에 업로드")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
