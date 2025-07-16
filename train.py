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
            "0_0": 400,
            "0_1": 400,
            "1_0": 400,
            "1_1": 400
        }
    )

    train_balanced = upsampler.upsample()

    run_training(
        train = train_balanced,
        val = val,
        epochs = 200
    )