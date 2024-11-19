import torch
import torch.nn as nn
import torch.optim as optim


class TitanicModel(nn.Module):
    """
    신경망 모델 정의 클래스.
    """
    def __init__(self, n_input, n_output, activation_fn):
        """
        TitanicModel 생성자.
        :param n_input: (int) 입력 특징 벡터의 크기.
        :param n_output: (int) 출력 노드 수 (분류 클래스의 수).
        :param activation_fn: (nn.Module) 사용할 활성화 함수.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 64),  # 첫 번째 은닉층
            activation_fn(),  # 활성화 함수
            nn.Linear(64, 32),  # 두 번째 은닉층
            activation_fn(),  # 활성화 함수
            nn.Linear(32, n_output),  # 출력층
        )

    def forward(self, x):
        """
        순전파 함수 (Forward Propagation).
        :param x: (torch.Tensor) 입력 데이터.
        :return: (torch.Tensor) 출력 데이터.
        """
        return self.model(x)


class TitanicTrainer:
    """
    TitanicModel 학습 및 검증을 관리하는 클래스.
    """
    def __init__(self, model, learning_rate=0.001, epochs=10):
        """
        TitanicTrainer 초기화.
        :param model: TitanicModel 인스턴스.
        :param learning_rate: (float) 학습률.
        :param epochs: (int) 에포크 수.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()  # 손실 함수
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # 옵티마이저

    def train(self, train_loader, val_loader, log_file="training_log.txt"):
        """
        모델 학습 및 검증 루프.
        """
        best_validation_accuracy = 0.0
        best_model_state = None
        best_epoch = 0

        # 학습 로그 파일 작성
        with open(log_file, "w") as log:
            for epoch in range(1, self.epochs + 1):
                # Training 단계
                self.model.train()
                train_loss = 0.0
                correct_train = 0
                total_train = 0

                for batch in train_loader:
                    inputs = batch['input']
                    targets = batch['target']

                    # 순전파 및 손실 계산
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    train_loss += loss.item()

                    # 역전파 및 가중치 갱신
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # 정확도 계산
                    predictions = torch.argmax(outputs, dim=1)
                    correct_train += (predictions == targets).sum().item()
                    total_train += targets.size(0)

                train_accuracy = correct_train / total_train

                # Validation 단계
                self.model.eval()
                val_loss = 0.0
                correct_val = 0
                total_val = 0

                with torch.no_grad():
                    for batch in val_loader:
                        inputs = batch['input']
                        targets = batch['target']

                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        val_loss += loss.item()

                        predictions = torch.argmax(outputs, dim=1)
                        correct_val += (predictions == targets).sum().item()
                        total_val += targets.size(0)

                validation_accuracy = correct_val / total_val

                # 최고 모델 상태 저장
                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_model_state = self.model.state_dict()
                    best_epoch = epoch

                # 로그 작성
                log_message = (f"Epoch {epoch}/{self.epochs}: "
                               f"Train Loss: {train_loss / total_train:.4f}, "
                               f"Train Accuracy: {train_accuracy:.4f}, "
                               f"Validation Loss: {val_loss / total_val:.4f}, "
                               f"Validation Accuracy: {validation_accuracy:.4f}\n")
                print(log_message.strip())
                log.write(log_message)

        print(f"Best model found at epoch {best_epoch} with validation accuracy {best_validation_accuracy:.4f}")
        return best_validation_accuracy, best_model_state, best_epoch

    def test(self, test_loader):
        """
        학습된 모델로 테스트 데이터 예측.
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['input']
                outputs = self.model(inputs)
                prediction = torch.argmax(outputs, dim=1)
                predictions.extend(prediction.tolist())

        return predictions
