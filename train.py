from training import run_training
from data import create_dataset

if __name__ == "__main__":
    # run trainings here
    train, val = create_dataset(
        data_csv="data.csv",
        total_samples=500,
        output_dir="augmented",
        output_train="train.csv",
        output_val="val.csv"
    )


    run_training(
        val=val,
        epochs=1,
        train=train,
    )