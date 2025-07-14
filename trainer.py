import torch as t
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
# from tqdm.autonotebook import tqdm
import warnings
import os
import matplotlib.pyplot as plt
import shutil

warnings.simplefilter("ignore")


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=False,  # Whether to use the GPU
                 early_stopping_patience=-1,
                 save_dir='./checkpoints'):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience
        self.path = save_dir
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
            print("Existing Directory deleted")
        os.mkdir(self.path)
        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

        self.f1Score = 0
        self.all_loss = []
        self.no_progress_duration = 0

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, self.path + '/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load(self.path + '/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        self._optim.zero_grad()
        # -propagate through the network ==> forward pass
        pred = self._model(x)
        # -calculate the loss
        yy = t.squeeze(y).float()
        # RuntimeError: 1D target tensor expected, multi - target not supported
        loss = self._crit(pred, yy)
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss.item()

    def train_epoch(self):
        # set training mode
        # model.train() tells your model that you are training the model. So effectively layers like dropout, batchnorm etc.
        # which behave different on the train and test procedures know what is going on and hence can behave accordingly.
        self._model = self._model.train()
        # iterate through the training set
        loss = 0
        for img, label in self._train_dl:
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            # loss = 0
            if self._cuda:
                img = img.to('cuda')
                label = label.to('cuda')
            else:
                img = img.to('cpu')
                label = label.to('cpu')
            # perform a training step from line 52
            loss = loss + self.train_step(x=img, y=label)

        # calculate the average loss for the epoch and return it
        avg_loss = loss / len(self._train_dl)
        return avg_loss

    def val_test_step(self, x, y):
        # predict
        predict = self._model(x)
        # propagate through the network and calculate the loss and predictions
        yy = t.squeeze(y).float()
        loss = self._crit(predict, yy)
        # return the loss and the predictions
        return loss.item(), predict

    def val_test(self):
        # set eval mode
        self._model.eval()
        running_loss = 0
        n_batches = 0

        all_preds = []
        all_labels = []
        # disable gradient computation
        with t.no_grad(): 
            # iterate through the validation set
            for xb, yb in self._val_test_dl:
                # transfer the batch to the gpu if given
                if self._cuda:
                    xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
                loss, preds = self.val_test_step(xb, yb)
                running_loss += loss
                n_batches += 1

                # preds = t.sigmoid(probs)
                # use cpu
                preds = preds.cpu().numpy()
                labels = yb.cpu().numpy().squeeze()


                # binarize predictions
                preds_bin = (preds > 0.5).astype(int)

                # save the predictions and the labels for each batch
                all_preds.append(preds_bin)
                all_labels.append(labels)

        # calculate the average loss
        val_loss = running_loss / max(1, n_batches)

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_labels = all_labels.astype(int)

        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

        y_true_comb = [f"{c}_{i}" for c, i in all_labels]
        y_pred_comb = [f"{c}_{i}" for c, i in all_preds]
        cm = confusion_matrix(y_true_comb, y_pred_comb, labels=["0_0", "1_0", "0_1", "1_1"])


        # return the loss and print the calculated metrics
        print(f"Validation Loss: {val_loss} | F1 Score - Crack: {f1_per_class[0]} | F1 Score - Inactive: {f1_per_class[1]}")
        return val_loss, f1_per_class, cm

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        train_loss = []
        val_loss = []
        patience_count = 0
        self.cnt_epoch = 0
        f1_mean = []
        best_val_loss = float("inf")

        while True:
            # stop by epoch number
            if self.cnt_epoch == epochs:
                break

            self.cnt_epoch += 1
            # train for a epoch and then calculate the loss and metrics on the validation set
            avg_train_loss = self.train_epoch()

            # print(avg_train_loss)
            avg_val_loss, self.f1Score, confusion_matrix = self.val_test()
            # print(avg_train_loss, avg_val_loss)
            # print('***********', self.f1Score)

            f1_mean.append(self.f1Score)

            # mean_f1 = (self.f1Score[0]+self.f1Score[1])/2
            # print('mean f1 score :', mean_f1)
            print('Epoch: {} mean F1_Score: {} Validation Loss: {}'.format(self.cnt_epoch, np.mean(self.f1Score), avg_val_loss))

            # append the losses to the respective lists
            train_loss.append(avg_train_loss)
            val_loss.append(avg_val_loss)
            if avg_val_loss < best_val_loss:
                patience_count = 0
                best_val_loss = avg_val_loss                
            else:
                patience_count += 1

            if np.mean(self.f1Score) >= 0.70:
                self.save_checkpoint(self.cnt_epoch)

                self.save_best_model(self._save_dir)
                # logger._log(f"saved checkpoint model at epoch: {self.cnt_epoch}")

                # save confusion matrix for best model
                ConfusionMatrixDisplay(confusion_matrix, display_labels=["0_0", "1_0", "0_1", "1_1"]).plot(cmap="Blues", values_format="d")
                plt.title("Confusion Matrix")
                plt.savefig(self.path / 'confusion_matrix.png')
                plt.close()                
                # self.save_onnx(self.path+'/checkpoint_{:03d}.onnx'.format(self.cnt_epoch))
            # self.save_checkpoint(self.cnt_epoch)

            if patience_count >= self._early_stopping_patience or self.cnt_epoch >= epochs:
                print('Enough.. I have no Patience, I am STOPPING')
                break
        return train_loss, val_loss, f1_mean
