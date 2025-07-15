from training import run_training
import pandas as pd
#from data import create_dataset

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    df = pd.read_csv("data.csv", sep=";")
    train, val = train_test_split(df, test_size=0.2, random_state=42)

    run_training(
        train = train,
        val = val,
        epochs = 200
    )