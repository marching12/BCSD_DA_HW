import pandas as pd

# 데이터 읽어오기
df = pd.read_csv('basic1.csv')

# 4-1. 'f5' 컬럼 처리하기
# 'f5' 컬럼에서 상위 10개의 값중 고유값을 구함
top_10_f5 = df['f5'].value_counts().nlargest(10).index  #상위 10개

# 상위 10개의 'f5' 값이 아닌 데이터는 상위 10개 중 최솟값으로 대체
min_f5 = df[df['f5'].isin(top_10_f5)]['f5'].min()
df['f5'] = df['f5'].apply(lambda x: x if x in top_10_f5 else min_f5)

# 'age' 컬럼이 80 이상인 데이터의 'f5' 컬럼 평균값 구하기
avg_f5_age_80_plus = df[df['age'] >= 80]['f5'].mean()
print(f"4-1. 'age' 80 이상인 데이터의 'f5' 평균값: {avg_f5_age_80_plus}")

# 4-2. 'f1' 컬럼 결측치 채우기 전후 표준편차 구하기
# 전체 데이터의 70%를 랜덤 샘플로 추출
df_sample = df.sample(frac=0.7, random_state=1)

# 결측치 채우기 전 표준편차 구하기
std_before = df_sample['f1'].std()

# 결측치를 중앙값으로 채움
median_f1 = df_sample['f1'].median()
df_sample['f1'] = df_sample['f1'].fillna(median_f1)

# 결측치 채운 후 표준편차 구하기
std_after = df_sample['f1'].std()

# 두 표준편차 차이 계산하기
std_diff = std_before - std_after
print(f"4-2. 표준편차 차이: {std_diff}")

# 4-3. 'age' 컬럼의 이상치 구하기
# 'age' 컬럼의 평균과 표준편차 구하기
mean_age = df['age'].mean()
std_age = df['age'].std()

# 이상치 기준 설정 표준편차를 활용해 설정
lower_bound = mean_age - 1.5 * std_age
upper_bound = mean_age + 1.5 * std_age

# 이상치 찾기
outliers = df[(df['age'] < lower_bound) | (df['age'] > upper_bound)]
print(f"4-3. 'age' 컬럼의 이상치:\n{outliers}")
