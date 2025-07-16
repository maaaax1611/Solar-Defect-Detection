def run_training(
  train,
  val,
  epochs = 200,
  lr = 1e-4,
  bs = 64,
  optimizer = "SGD",
  weight_decay = 1e-4,
  momentum = 0.9,
  early_stopping_patience = 20,
  model_save_dir = "trainings"      
):

    import torch as t
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import datetime
    from pathlib import Path

    from data import ChallengeDataset
    from model import ResNet
    from trainer import Trainer
    from logger import Logger

    train_ds = ChallengeDataset(data=train, mode="train")
    val_ds = ChallengeDataset(data=val, mode="val")

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    model = ResNet()
    
    if optimizer == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum) # , weight_decay=weight_decay)
    else:
        raise ValueError("Choose a valid optimizer! Valid options are ADAM & SGD")
    
    criteria = nn.BCELoss()

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(model_save_dir) / f"run_{now}"
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model = model,
        crit = criteria,
        optim = optimizer,
        train_dl = train_loader,
        val_test_dl = val_loader,
        early_stopping_patience = early_stopping_patience,
        save_dir = run_dir,
        cuda=True
    )

    logger = Logger(save_dir=run_dir)
    logger._log("\nHyperparameters:")
    logger._log(f"Epoch: {epochs} | learning_rate: {lr} | batch_size: {bs} | weight_decay: {weight_decay} | early_stopping_patience: {early_stopping_patience} \n")
    logger._log("\n==============Logger metrics==============")

    res = trainer.fit(logger=logger, epochs=epochs)

    # ====================== plot results =======================
    # training and validation loss
    train_loss = res[0]
    val_loss = res[1]
    f1_crack = np.array(res[2])[:,0]
    f1_inactive = np.array(res[2])[:,1]

    epochs_trained = np.arange(1, len(train_loss) + 1, 1)

    plt.figure(figsize=(12, 6))
    plt.suptitle("Model Training", fontsize=14)
    
    # --- Subplot 1: F1 Scores ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_trained, f1_crack, label="F1 Crack", color="blue")
    plt.plot(epochs_trained, f1_inactive, label="F1 Inactive", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("F1 Scores")
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # --- Subplot 2: Loss ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs_trained, train_loss, label="Training Data", color="blue")
    plt.plot(epochs_trained, val_loss, label="Validation Data", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(run_dir / "f1_and_loss_plot.png")
    plt.close()

    # class distribution of training dataset after augmentation
    train["class_combo"] = train.apply(lambda row: f"{row['crack']}_{row['inactive']}", axis=1)
    sns.countplot(x="class_combo", data=train)
    plt.title("class distribution AFTER augmentation")
    plt.xlabel("crack | inactive")
    plt.ylabel("number of samples")
    plt.tight_layout()
    plt.savefig(run_dir / 'training_data_distribution_after_aug.png')
    plt.close()