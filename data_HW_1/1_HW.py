import pandas as pd
df = pd.read_csv('1번째_파이썬_교육.csv')
print(df)   #데이터 출력
print(df.head(5))   #상위 5개 출력
print(df.tail(5))   #하위 5개 출력
print(df.info())    #요약 정보 출력
print(df.shape)     #DF 모양 출력
print(df.dtypes)    #데이터 타입 출력
print(df['title'].dtypes)   #'title' 열의 데이터 타입 출력
df['subscriber'] = df['subscriber'].replace('만', '0000',  regex = True).astype('int64')
#만을 0000으로 치환 후 정수형(int64)로 변환
print(df['subscriber']) # subscriber 열을 모두 출력

# view열의 데이터를 억은 빈문자열로 바꾸고 만은 0000으로 치환
df['view'].head(10)
df['view'] = df['view'] = df['view'].replace(['억', '만'], ['', '0000'], regex=True).astype('int64')
# video열 데이터 상위 5개 출력
print(df['video'].head())

# video 열에서 ','와 '개' 문자를 제거한 후, int32 형식으로 변환
df['video'] = df['video'] = df['video'].replace([',', '개'], '', regex=True).astype('int32')

print(df.info())

# category 열에서 각 카테고리 값 개수를 출력
print(df['category'].value_counts())

# subscriber 열을 기준으로 내림차순 정렬하여 상위 5개 행 출력
print(df.sort_values('subscriber', ascending=False).head())

# 특정 열을 기준으로 내림차순(오름차순) 정렬하여 상위 5개 행 출력
print(df.sort_values('view', ascending=False).head())
print(df.sort_values('video', ascending=False).head())
print(df.sort_values('video', ascending=True).head())

# category 열을 기준으로 먼저 오름차순 정렬, 그 다음 subscriber 열을 기준으로 내림차순 정렬
print(df.sort_values(['category', 'subscriber'], ascending=[True, False]))

# category에서 행을 선택해 그 행에서 특정 값이 많은 순으로 상위 5개 출력
print(df[df['category'] == '[음악/댄스/가수]'].sort_values('subscriber', ascending=False).head())
print(df.loc[df['category'] == '[TV/방송]', :].sort_values('view', ascending=False).head())

# 특정 열 값을 데이터 범위를 정해서 출력
print(df['subscriber'].dtypes)
print(df[df['subscriber'] >= 30000000])
print(df[(df['video'] >= 30000) & (df['video'] <= 35000)])
print(df.loc[(df['subscriber'] >= 30000000) | (df['video'] >= 50000), :])


# 카테고리가 TV/방송이거나 게임인 행 선택
temp = df[(df['category'] == '[TV/방송]') | (df['category'] == '[게임]')]
print(len(temp))    #temp 길이 출력

# title 열에서 KBS가 포함된 행만 필터링 후 title 출력
print(df[df['title'].str.contains('KBS')]['title'])

# 위와 동일한 작업을 loc로 수행 (행과 열을 좀 더 명확히 선택 가능)
print(df.loc[df['title'].str.contains('KBS'), 'title'])

# title 열에 대문자를 소문자로 변환 후 위와 같은 작업 수행
print(df.loc[df['title'].str.lower().str.contains('kbs'), 'title'])