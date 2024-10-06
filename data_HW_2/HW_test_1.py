import pandas as pd
import numpy as np

# CSV 파일 읽기
df = pd.read_csv('website.csv')

# 'StartTime'과 'EndTime'을 datetime으로 변환
df['StartTime'] = pd.to_datetime(df['StartTime'])
df['EndTime'] = pd.to_datetime(df['EndTime'])

# 세션의 지속 시간을 분으로 계산하고 가장 긴 지속 시간을 출력
 #지속시간을 60으로 나누어 분으로 변환해 데이터프레임에 추가
df['Duration'] = (df['EndTime'] - df['StartTime']).dt.total_seconds() / 60

# 1-1. 세션의 지속 시간을 분으로 계산하고 가장 긴 지속시간을 출력하시오(반올림 후 총 분만 출력)
# round()로 반올림
longest_duration = round(df['Duration'].max())
print(f"가장 긴 지속 시간: {longest_duration}분")

# 1-2. 가장 많이 머무른 Page를 찾고 그 페이지의 평균 머문 시간을 구하기
 # 페이지당 평균 머문시간 찾기
page_durations = df.groupby('Page')['Duration'].mean()
 # 평균 방문 시간이 가장 긴 페이지의 이름 저장
most_visited_page = page_durations.idxmax()
 # 가장 긴 평균 방문시간을 반올림 해서 저장
average_duration = round(page_durations.max())
print(f"가장 많이 머무른 페이지: {most_visited_page}")
print(f"{most_visited_page} 페이지의 평균 머문 시간: {average_duration}분")

# 시간을 나누어 문자를 라벨링
df['Hour'] = df['StartTime'].dt.hour
bins = [0, 6, 12, 18, 24]
labels = ['새벽', '오전', '오후', '저녁']
# 'Hour' 열을 시간대 구간에 따라 분류
df['TimeOfDay'] = pd.cut(df['Hour'], bins=bins, labels=labels, right=False)

# 1-3. 사용자들이 가장 활발히 활동하는 시간대 분석
# 각 시간대별로 세션 수 계산
# observed를 활용해 모든 범주를 고려하여 각 시간대의 행 크기를 저장
sessions_per_time_of_day = df.groupby('TimeOfDay', observed=False).size()
# 가장 세견이 많은 시간대 이름 저장
most_active_time_of_day = sessions_per_time_of_day.idxmax()
# 가장 세션이 많은 시간대 세션 수 저장
most_active_sessions = sessions_per_time_of_day.max()
print(f"가장 많은 세션이 시작된 시간대: {most_active_time_of_day}")
print(f"가장 많은 세션 수: {most_active_sessions}")


# 3-2 시간대별 방문 패턴 분석
# 시간대별로 페이지 방문 수 계산
time_page_visits = df.groupby(['TimeOfDay', 'Page'],  observed=True).size().reset_index(name='Visits')
# 각 시간대별로 가장 많이 방문된 페이지 찾기
most_visited_pages = time_page_visits.loc[time_page_visits.groupby('TimeOfDay', observed=True)['Visits'].idxmax()]
# 결과 출력
print("시간대별 가장 많이 방문된 페이지와 방문 횟수:")
print(most_visited_pages)


# 1-4. user가 가장 많이 접속했던 날짜를 출력
df['Date'] = df['StartTime'].dt.date
most_active_date = df['Date'].value_counts().idxmax()
print(f"가장 많이 접속한 날짜: {most_active_date}")

# 3-1 사용자별 방문패턴 분석
# 사용자별 페이지별 평균 세션 시간 계산
user_page_avg_duration = df.groupby(['UserID', 'Page'])['Duration'].mean().reset_index()
# 각 페이지별로 가장 긴 평균 세션 시간을 가진 사용자 찾기
idx = user_page_avg_duration.groupby('Page')['Duration'].idxmax()
longest_avg_duration_users = user_page_avg_duration.loc[idx]
# 각 페이지별 가장 긴 평균 세션 시간을 모두 더하기
max_avg_duration_per_page = user_page_avg_duration.groupby('Page')['Duration'].max()
total_max_avg_duration = max_avg_duration_per_page.sum()
print(f"각 페이지별 가장 긴 평균 세션 시간을 모두 더한 값: {round(total_max_avg_duration)}분")

# 3-3 재방문이 많은 월 찾기
# 'Date' 열을 생성하여 날짜 정보만 추출
df['Date'] = df['StartTime'].dt.date
df['Month'] = df['StartTime'].dt.month

# 사용자와 날짜별로 방문한 페이지 수 계산
visits_per_day = df.groupby(['UserID', 'Date'])['Page'].nunique().reset_index(name='VisitCount')
# 여러 페이지를 방문한 날짜만 필터링 (재방문한 경우)
revisit_days = visits_per_day[visits_per_day['VisitCount'] > 1].copy()  # 복사본 생성(원본파일 유지)
# 재방문한 날짜의 월별 계산
revisit_days['Month'] = pd.to_datetime(revisit_days['Date']).dt.month
# 월별 재방문한 횟수를 계산
revisit_count_per_month = revisit_days['Month'].value_counts()
# 재방문이 가장 많은 월을 찾음
most_revisits_month = revisit_count_per_month.idxmax()
print(f"재방문이 가장 많은 월: {most_revisits_month}월")





