import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('e-commerce.csv')

# 2-1
# 피드백의 길이 계산
df['FeedbackLength'] = df['Feedback'].apply(len)
# 가장 긴 피드백을 작성한 UserID 찾기
longest_feedback_user = df.loc[df['FeedbackLength'].idxmax(), 'UserID']
# 해당 UserID의 주문 수 계산
user_order_count = df[df['UserID'] == longest_feedback_user].shape[0]
print(f"가장 긴 피드백을 작성한 UserID: {longest_feedback_user}")
print(f"해당 UserID가 주문한 수: {user_order_count}")


# 'OrderDate'와 'ArrivalDate'를 datetime으로 변환
df['OrderDate'] = pd.to_datetime(df['OrderDate'])
df['ArrivalDate'] = pd.to_datetime(df['ArrivalDate'])

# 대기 시간 계산
df['WaitTimeMinutes'] = (df['ArrivalDate'] - df['OrderDate']).dt.total_seconds() / 60

# 2-2
# '제품' 포함 여부 검사
df['ContainsProduct'] = df['Feedback'].str.contains('제품', case=False)

# '제품'이 가장 많이 포함된 카테고리 사이즈 저장
category_counts = df[df['ContainsProduct']].groupby('Category').size()
most_common_category = category_counts.idxmax()

# 해당 카테고리의 평균 배송 시간 계산
average_delivery_time = df[df['Category'] == most_common_category]['WaitTimeMinutes'].mean()

print(f"'제품'이라는 단어가 가장 많이 포함된 카테고리: {most_common_category}")
print(f"해당 카테고리의 평균 배송 시간(분): {average_delivery_time:.2f}")

