import numpy as np
from model import VisionTransformerModel
from dataset import Dataset
from softmax_cross_entropy import SoftmaxCrossEntropy
from sklearn.metrics import precision_score, recall_score, f1_score


def to_one_hot(index):
    one_hot = np.zeros((1, 10))
    one_hot[0][index] = 1.0

    return one_hot


def train(model: VisionTransformerModel, epochs: int, train_dataset: Dataset, valid_dataset: Dataset,
          learning_rate: float) -> None:
    '''
    Train the model.

    :param model: VisionTransformerModel
    :param epochs: number of epochs
    :param train_dataset: dataset used for training
    :param valid_dataset: dataset used for validation
    :param learning_rate: learning rate
    :return: None
    '''

    # loss function
    loss_function = SoftmaxCrossEntropy()

    # performance metrics
    global_train_loss = []
    global_train_acc = []

    global_loss = []
    global_acc = []
    global_precision = []
    global_recall = []
    global_f1 = []

    for epoch in range(epochs):
        train_loss = 0
        train_accuracy = 0

        for idx in range(len(train_dataset)):
            data, label = train_dataset[idx]

            # model output
            output = model.forward(data)

            # output_one_hot
            output_one_hot = to_one_hot(np.argmax(output, 1))

            # loss
            loss = loss_function.forward(output, label)

            # backpropagation
            delta_e_y = loss_function.backward(y_true=label)
            model.backward(delta_e_y=delta_e_y, learning_rate=learning_rate)

            acc = int(np.sum(np.abs(output_one_hot - label)))

            train_loss += loss / len(train_dataset)
            train_accuracy += acc / len(train_dataset)

        # evaluate the model

        # performance metrics
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        epoch_val_precision = 0
        epoch_val_recall = 0
        epoch_val_f1 = 0

        for idx in range(len(valid_dataset)):
            data, label = valid_dataset[idx]

            # model output
            val_output = model.forward(data)
            # output label
            val_output_label = to_one_hot(np.argmax(output, 1))
            # loss
            val_loss = loss_function.forward(val_output_label, label)

            acc = int(np.sum(np.abs(output_one_hot - label)))
            prec = precision_score(label, val_output_label, zero_division=0)
            rec = recall_score(label, val_output_label, zero_division=0)
            f1 = f1_score(label, val_output_label, zero_division=0)

            epoch_val_accuracy += acc / len(valid_dataset)
            epoch_val_loss += val_loss / len(valid_dataset)
            epoch_val_precision += prec / len(valid_dataset)
            epoch_val_recall += rec / len(valid_dataset)
            epoch_val_f1 += f1 / len(valid_dataset)

        # global performance metrics
        global_train_loss.append(train_loss)
        global_train_acc.append(train_accuracy)

        global_loss.append(epoch_val_loss)
        global_acc.append(epoch_val_accuracy)
        global_precision.append(epoch_val_precision)
        global_recall.append(epoch_val_recall)
        global_f1.append(epoch_val_f1)

        print(f'Epoch : {epoch + 1}')
        print()
        print(f'Train loss : {train_loss:.4f}')
        print(f'Train acc: {train_accuracy:.4f}')
        print()
        print(f'Val loss : {epoch_val_loss:.4f}')
        print(f'Val acc: {epoch_val_accuracy:.4f}')
        print(f'Val precision: {epoch_val_precision:.4f}')
        print(f'Val recall: {epoch_val_recall:.4f}')
        print(f'Val f1: {epoch_val_f1:.4f}')
        print('----------------------------------------------')
