import pandas as pd
from torch.utils.data import random_split, Dataset, DataLoader
import torch


class TitanicDataset(Dataset):
    """
    Titanic 데이터셋을 처리하는 클래스.
    """
    def __init__(self, X, y=None):
        """
        TitanicDataset의 생성자.

        :param X: (numpy 배열 또는 DataFrame) 특성 행렬 (입력 데이터).
        :param y: (numpy 배열 또는 DataFrame, optional) 레이블 (타겟 데이터). 학습 데이터에서만 필요.
        """
        self.X = torch.FloatTensor(X)  # torch.FloatTensor로 변환
        self.y = torch.LongTensor(y) if y is not None else None  # torch.LongTensor로 변환

    def __len__(self):
        """
        데이터셋의 총 샘플 수를 반환.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        주어진 인덱스에 해당하는 샘플을 반환.
        """
        feature = self.X[idx]
        if self.y is not None:
            target = self.y[idx]
            return {'input': feature, 'target': target}
        else:
            return {'input': feature}


class TitanicDatasetManager:
    """
    Titanic 데이터셋을 관리하고 전처리하며, DataLoader를 생성하는 클래스.
    """
    def __init__(self, train_path, test_path):
        """
        Titanic 데이터셋 관리 클래스 초기화.

        :param train_path: 학습 데이터 파일 경로.
        :param test_path: 테스트 데이터 파일 경로.
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    def load_and_preprocess(self):
        """
        데이터를 로드하고 전처리하여 학습, 검증, 테스트 데이터셋을 생성.
        """
        # 데이터 로드
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        # 데이터 크기 확인
        print(f"Original train data size: {train_df.shape[0]}")
        print(f"Original test data size: {test_df.shape[0]}")

        # 학습 및 테스트 데이터를 결합하여 전처리
        all_df = pd.concat([train_df, test_df], sort=False)
        all_df = self._preprocess_1(all_df)
        all_df = self._preprocess_2(all_df)

        # 학습/검증 데이터 분리
        train_X = all_df[~all_df["Survived"].isnull()].drop("Survived", axis=1).reset_index(drop=True)
        train_y = train_df["Survived"]
        test_X = all_df[all_df["Survived"].isnull()].drop("Survived", axis=1).reset_index(drop=True)

        # 데이터 유형 강제 변환
        train_X = train_X.apply(pd.to_numeric, errors="coerce")
        test_X = test_X.apply(pd.to_numeric, errors="coerce")

        # 테스트 데이터 크기 확인
        print(f"Test data size after preprocessing: {test_X.shape[0]}")
        if test_X.shape[0] != 418:
            raise ValueError("Test data size must be 418 rows after preprocessing.")

        # 학습/검증/테스트 데이터셋 생성
        dataset = TitanicDataset(train_X.values, train_y.values)
        self.train_dataset, self.validation_dataset = random_split(dataset, [0.8, 0.2])

        # 테스트 데이터는 타겟 없이 생성
        self.test_dataset = TitanicDataset(test_X.values)

    def _preprocess_1(self, all_df):
        """
        전처리 단계 1: 새로운 피처 생성 및 불필요한 컬럼 제거.
        """
        # 가족 수와 혼자 탑승 여부 생성
        all_df["family_num"] = all_df["Parch"] + all_df["SibSp"]
        all_df["alone"] = all_df["family_num"].apply(lambda x: 1 if x == 0 else 0)

        # 존재하는 열만 제거
        columns_to_drop = ["PassengerId", "Name", "family_name", "name", "Ticket", "Cabin"]
        existing_columns = [col for col in columns_to_drop if col in all_df.columns]
        all_df = all_df.drop(existing_columns, axis=1)
        return all_df

    def _preprocess_2(self, all_df):
        """
        전처리 단계 2: 범주형 변수를 LabelEncoder를 사용하여 수치값으로 변환.
        """
        category_features = all_df.columns[all_df.dtypes == "object"]
        from sklearn.preprocessing import LabelEncoder
        for category_feature in category_features:
            le = LabelEncoder()
            all_df.loc[:, category_feature] = all_df[category_feature].fillna("Missing")
            all_df.loc[:, category_feature] = le.fit_transform(all_df[category_feature])
        return all_df

    def get_data_loaders(self, batch_size=32, validation_batch_size=None):
        """
        학습 및 검증 데이터를 DataLoader로 반환.
        """
        if not self.train_dataset or not self.validation_dataset:
            raise ValueError("Datasets not initialized. Run load_and_preprocess() first.")

        validation_batch_size = validation_batch_size or len(self.validation_dataset)
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(self.validation_dataset, batch_size=validation_batch_size, shuffle=False)

        return train_loader, validation_loader

    def get_test_loader(self, batch_size=None):
        """
        테스트 데이터를 DataLoader로 반환.
        """
        if not self.test_dataset:
            raise ValueError("Datasets not initialized. Run load_and_preprocess() first.")

        batch_size = batch_size or len(self.test_dataset)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader
