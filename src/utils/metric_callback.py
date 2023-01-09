import numpy as np
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import accuracy_score
from datetime import datetime


class MetricCallback(Callback):
    def __init__(self, n_classes, filename):
        self.n_classes = n_classes
        self.log_file_name = "logs/CORRECT_{}_{}".format(
            filename, datetime.now().strftime("%m_%d_%Y_%H_%M_%S.txt")
        )
        self.best = 0

    def on_validation_start(self, *args, **kwargs):
        self.correct_per_class = [0 for _ in range(self.n_classes + 1)]
        self.instances_per_class = [0 for _ in range(self.n_classes + 1)]
        self.preds = []
        self.true = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        # here is where you store everything that you need
        # for output_per_gpu in outputs:
        #     print(output_per_gpu)
        pred_label = outputs["outputs"].argmax(dim=1)
        for index in range(outputs["targets"].size(0)):
            label = outputs["targets"][index]
            self.index_ = pred_label[index]
            predicted_label = self.index_
            self.instances_per_class[label] += 1
            if label.item() == predicted_label.item():
                self.correct_per_class[label] += 1
        self.preds += pred_label.tolist()
        self.true += outputs["targets"].tolist()

    def on_validation_end(self, *args, **kwargs):
        accuracy_per_class = np.array(self.correct_per_class) / np.array(
            self.instances_per_class
        )
        closed_accuracy = accuracy_per_class[: self.n_classes].mean()
        open_accuracy = accuracy_per_class[-1]
        h_score = (
            2 * closed_accuracy * open_accuracy / (closed_accuracy + open_accuracy)
        )
        val_accuracy = accuracy_score(self.preds, self.true)

        print("H score (callback): {}".format(h_score))
        if h_score > self.best:
            self.best = h_score
            with open(self.log_file_name, "a") as logfile:
                logfile.write("    Validation accuracy: {}".format(val_accuracy))
                logfile.write("    Closed accuracy: {}\n".format(closed_accuracy))
                logfile.write("    Open accuracy: {}\n".format(open_accuracy))
                logfile.write("    H-score: {}\n".format(h_score))
                logfile.write("----------------------------------------\n")
