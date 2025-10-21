import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc

# 데이터 로드
def load_data():
    df = pd.read_csv('../data/naver_datalab_fixed.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.day_name()
    df['quarter'] = df['date'].dt.quarter
    return df

# 앱 초기화
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "드론 라이트 쇼 검색 트렌드 대시보드"

# 데이터 로드
df = load_data()
regions = df['region'].unique()
years = sorted(df['year'].unique())

# 레이아웃 정의
app.layout = dbc.Container([
    # 헤더
    dbc.Row([
        dbc.Col([
            html.H1("🚁 드론 라이트 쇼 검색 트렌드 분석 대시보드", 
                   className="text-center mb-4", 
                   style={'color': '#2c3e50', 'font-weight': 'bold'})
        ])
    ]),
    
    # 컨트롤 패널
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📊 분석 옵션", className="card-title"),
                    
                    html.Label("지역 선택:", className="fw-bold"),
                    dcc.Dropdown(
                        id='region-dropdown',
                        options=[{'label': '전체', 'value': 'all'}] + 
                               [{'label': region, 'value': region} for region in regions],
                        value='all',
                        multi=True,
                        className="mb-3"
                    ),
                    
                    html.Label("기간 선택:", className="fw-bold"),
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        start_date=df['date'].min(),
                        end_date=df['date'].max(),
                        display_format='YYYY-MM-DD',
                        className="mb-3"
                    ),
                    
                    html.Label("분석 유형:", className="fw-bold"),
                    dcc.RadioItems(
                        id='analysis-type',
                        options=[
                            {'label': ' 시계열 트렌드', 'value': 'timeseries'},
                            {'label': ' 월별 패턴', 'value': 'monthly'},
                            {'label': ' 요일별 패턴', 'value': 'weekday'},
                            {'label': ' 상관관계', 'value': 'correlation'}
                        ],
                        value='timeseries',
                        className="mb-3"
                    )
                ])
            ])
        ], width=3),
        
        # 메인 차트 영역
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='main-chart', style={'height': '500px'})
                ])
            ])
        ], width=9)
    ], className="mb-4"),
    
    # 통계 카드들
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='total-searches', className="text-primary"),
                    html.P("총 검색량", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='peak-region', className="text-success"),
                    html.P("최고 검색 지역", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='peak-date', className="text-warning"),
                    html.P("최고 검색 날짜", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='avg-correlation', className="text-info"),
                    html.P("평균 지역간 상관계수", className="card-text")
                ])
            ])
        ], width=3)
    ], className="mb-4"),
    
    # 하단 차트들
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📈 지역별 비교", className="card-title"),
                    dcc.Graph(id='comparison-chart', style={'height': '400px'})
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🔥 상위 검색 이벤트", className="card-title"),
                    html.Div(id='top-events')
                ])
            ])
        ], width=6)
    ])
], fluid=True)

# 콜백 함수들
@app.callback(
    [Output('main-chart', 'figure'),
     Output('comparison-chart', 'figure'),
     Output('total-searches', 'children'),
     Output('peak-region', 'children'),
     Output('peak-date', 'children'),
     Output('avg-correlation', 'children'),
     Output('top-events', 'children')],
    [Input('region-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('analysis-type', 'value')]
)
def update_dashboard(selected_regions, start_date, end_date, analysis_type):
    # 데이터 필터링
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    if selected_regions != 'all' and selected_regions:
        if isinstance(selected_regions, str):
            selected_regions = [selected_regions]
        filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]
    
    # 메인 차트 생성
    main_fig = create_main_chart(filtered_df, analysis_type)
    
    # 비교 차트 생성
    comparison_fig = create_comparison_chart(filtered_df)
    
    # 통계 계산
    total_searches = f"{filtered_df['ratio'].sum():.2f}"
    
    peak_idx = filtered_df['ratio'].idxmax()
    peak_region = filtered_df.loc[peak_idx, 'region'] if not pd.isna(peak_idx) else "N/A"
    peak_date = filtered_df.loc[peak_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(peak_idx) else "N/A"
    
    # 상관계수 계산
    pivot_df = filtered_df.pivot(index='date', columns='region', values='ratio')
    correlation_matrix = pivot_df.corr()
    avg_correlation = f"{correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean():.3f}"
    
    # 상위 이벤트 생성
    top_events = create_top_events(filtered_df)
    
    return main_fig, comparison_fig, total_searches, peak_region, peak_date, avg_correlation, top_events

def create_main_chart(df, analysis_type):
    if analysis_type == 'timeseries':
        fig = px.line(df, x='date', y='ratio', color='region',
                     title='시계열 검색 트렌드',
                     labels={'ratio': '검색 비율', 'date': '날짜'})
        
    elif analysis_type == 'monthly':
        monthly_data = df.groupby(['month', 'region'])['ratio'].mean().reset_index()
        fig = px.bar(monthly_data, x='month', y='ratio', color='region',
                    title='월별 평균 검색 비율',
                    labels={'ratio': '평균 검색 비율', 'month': '월'})
        
    elif analysis_type == 'weekday':
        df['day_of_week_num'] = df['date'].dt.dayofweek
        weekday_data = df.groupby(['day_of_week_num', 'region'])['ratio'].mean().reset_index()
        day_names = ['월', '화', '수', '목', '금', '토', '일']
        weekday_data['day_name'] = weekday_data['day_of_week_num'].map(lambda x: day_names[x])
        
        fig = px.bar(weekday_data, x='day_name', y='ratio', color='region',
                    title='요일별 평균 검색 비율',
                    labels={'ratio': '평균 검색 비율', 'day_name': '요일'})
        
    elif analysis_type == 'correlation':
        pivot_df = df.pivot(index='date', columns='region', values='ratio')
        correlation_matrix = pivot_df.corr()
        
        fig = px.imshow(correlation_matrix, 
                       title='지역간 검색 트렌드 상관관계',
                       color_continuous_scale='RdBu_r',
                       aspect='auto')
        fig.update_layout(
            xaxis_title='지역',
            yaxis_title='지역'
        )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        title_font_size=16
    )
    
    return fig

def create_comparison_chart(df):
    # 지역별 총 검색량 비교
    region_totals = df.groupby('region')['ratio'].agg(['sum', 'mean', 'max']).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='총합',
        x=region_totals['region'],
        y=region_totals['sum'],
        yaxis='y',
        offsetgroup=1
    ))
    
    fig.add_trace(go.Bar(
        name='평균',
        x=region_totals['region'],
        y=region_totals['mean'],
        yaxis='y2',
        offsetgroup=2
    ))
    
    fig.update_layout(
        title='지역별 검색량 비교',
        xaxis=dict(title='지역'),
        yaxis=dict(title='총 검색량', side='left'),
        yaxis2=dict(title='평균 검색량', side='right', overlaying='y'),
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_top_events(df):
    # 상위 10개 검색 이벤트
    top_events = df.nlargest(10, 'ratio')[['date', 'region', 'ratio']]
    
    events_list = []
    for _, row in top_events.iterrows():
        events_list.append(
            dbc.ListGroupItem([
                html.Div([
                    html.Strong(f"{row['region']}"),
                    html.Span(f" - {row['date'].strftime('%Y-%m-%d')}", className="text-muted"),
                    html.Br(),
                    html.Span(f"검색비율: {row['ratio']:.4f}", className="text-primary")
                ])
            ])
        )
    
    return dbc.ListGroup(events_list, flush=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
