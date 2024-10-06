import pandas as pd

# CSV 파일을 읽어옴
df = pd.read_csv('../data_HW_2/drinks.csv')

# 열 이름 변경
df.columns = ['국가', '맥주', '증류주', '와인', '알코올', '대륙']

# df 상위 5개 출력
print(df.head())

# 요약출력
print(df.info())

# 결측값 개수 확인 결측값 = null값
print(df.isnull().sum())

# '대륙' 열의 고유 값과 값의 개수 출력
print(df['대륙'].unique())
print(df['대륙'].value_counts())

# '대륙' 열이 결측값인 행 출력
print(df[df['대륙'].isnull()])

# '대륙' 열의 결측값을 'NA'로 대체
df.loc[df['대륙'].isnull(), '대륙'] = 'NA'
print(df['대륙'].unique())

# '맥주', '증류주', '와인', '알코올' 열을 선택해 새로운 df생성
df2 = df[['맥주', '증류주', '와인', '알코올']]

# 와인열 선택 맥주+증류주 선택 와인이 더 큰 값만 대륙별로 정렬
print(df[df['와인'] > (df['맥주'] + df['증류주'])].sort_values('대륙'))

# '맥주'와 '와인'이 둘 다(and) 230보다 큰 값 출력
print(df[(df['맥주'] > 230) & (df['와인'] > 230)])

# '대륙'이 'AS'인 값 출력
print(df[df['대륙'] == 'AS'])

# 주류 소비량1,2, 알코올비율 열 추가
df['주류소비량'] = df['맥주'] + df['증류주'] + df['와인']   #직관적 계산
df['주류소비량2'] = df[['맥주', '증류주', '와인']].sum(axis=1)  # 동적계산에 유리
df['알코올비율'] = df['알코올'] / df['주류소비량']
print(df.head())

# 알코올 비율이 가장 높은 상위 5개 국가 출력
print(df[['국가', '주류소비량', '알코올비율']].sort_values('알코올비율', ascending=False).head())

# 대륙별 맥주 소비량 평균 출력
print(df.groupby('대륙')[['맥주']].mean())

# 대륙별 맥주와 와인 소비량 평균 출력
print(df.groupby('대륙')[['맥주', '와인']].mean())
