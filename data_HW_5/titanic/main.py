from datetime import datetime
from titanic_dataset import TitanicDatasetManager
from titanic_model import TitanicModel, TitanicTrainer
import torch
import torch.nn as nn
import csv

def create_submission(predictions, output_file="best_submission.csv"):
    """
    제출용 CSV 파일 생성.
    """
    if len(predictions) != 418:
        raise ValueError(f"Submission must have 418 rows, but got {len(predictions)} rows.")

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["PassengerId", "Survived"])
        for i, prediction in enumerate(predictions):
            writer.writerow([892 + i, prediction])

    print(f"Submission file created: {output_file}")


def main():
    """
    다양한 활성화 함수로 모델 학습 및 검증 수행 후, 최적의 활성화 함수로 테스트 데이터 예측.
    """
    config = {
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 1e-3,
    }

    print("Loading and preprocessing data...")
    manager = TitanicDatasetManager("titanic_processed_train.csv", "titanic_processed_test.csv")
    manager.load_and_preprocess()
    train_loader, val_loader = manager.get_data_loaders(batch_size=config['batch_size'])
    test_loader = manager.get_test_loader()

    activation_functions = {
        "ReLU": nn.ReLU,
        "ELU": nn.ELU,
        "Leaky ReLU": nn.LeakyReLU,
        "PReLU": nn.PReLU,
        "Mish": nn.Mish
    }

    best_activation_function = None
    best_validation_accuracy = 0.0
    best_model_state = None
    best_epoch = 0

    for activation_name, activation_fn in activation_functions.items():
        print(f"Training with {activation_name} activation...")
        model = TitanicModel(
            n_input=train_loader.dataset[0]['input'].shape[0],
            n_output=2,
            activation_fn=activation_fn
        )
        trainer = TitanicTrainer(model, learning_rate=config['learning_rate'], epochs=config['epochs'])
        validation_accuracy, model_state, epoch = trainer.train(train_loader, val_loader)

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_activation_function = activation_name
            best_model_state = model_state
            best_epoch = epoch

    print(f"The best activation function is {best_activation_function} with validation accuracy {best_validation_accuracy:.4f}")

    print(f"Testing the best model with {best_activation_function} activation...")
    model.load_state_dict(best_model_state)
    trainer = TitanicTrainer(model)
    test_predictions = trainer.test(test_loader)
    create_submission(test_predictions)


if __name__ == "__main__":
    main()
