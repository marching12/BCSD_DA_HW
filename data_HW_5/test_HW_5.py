import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 한글 폰트 설정 (Windows: '맑은 고딕', Mac: 'AppleGothic')
plt.rc('font', family='Malgun Gothic')  # Windows
# plt.rc('font', family='AppleGothic')  # Mac

# 데이터 생성 (예제 데이터)
np.random.seed(0)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = (X[:, 0] > 0).astype(int)

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 로지스틱 회귀 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측 확률 계산
y_prob = model.predict_proba(X_test)[:, 1]

# 시각화
plt.scatter(X_test, y_test, color='blue', label='실제 값')
plt.plot(X_test, y_prob, 'r--', label='로지스틱 회귀 확률 예측')
plt.xlabel('특성 값')
plt.ylabel('클래스 확률')
plt.title('로지스틱 회귀 모델 예측')
plt.legend()
plt.show()
