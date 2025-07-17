from training import run_training
from sklearn.model_selection import train_test_split
from data import DatasetUpsampler
import pandas as pd

if __name__ == "__main__":
    
    df = pd.read_csv("data.csv", sep=";")
    train, val = train_test_split(df, test_size=0.2, random_state=42)

    upsampler = DatasetUpsampler(
        df=train,
        class_columns=["crack", "inactive"],
        target_counts={
            "0_0": 600,
            "0_1": 200,
            "1_0": 400,
            "1_1": 300
        }
    )

    train_balanced = upsampler.upsample()

    for lr in [1e-4, 3e-4, 6e-4]:
        for bs in [64, 100]:
            for es in [25, -1]:
                run_training(
                    lr = lr,
                    bs = bs,
                    early_stopping_patience=es,
                    train = train_balanced,
                    val = val,
                    epochs = 200
                )