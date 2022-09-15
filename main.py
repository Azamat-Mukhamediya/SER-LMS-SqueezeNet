
import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from squeezenet import squeezenet1_1
from sklearn.model_selection import train_test_split

from utils.training import get_mean_and_std, train, evaluate
from utils.preprocessing import get_audio_segmentation, SERDataset
from utils.load import get_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

seed = 42
np.random.seed(seed)
print('hello')

parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='emodb',
                    help='emodb | iemocap | savee')
parser.add_argument('--data_root', type=str,
                    default='dataset/EMODB/wav')

parser.add_argument('--k', type=int,
                    default=10, help='number of shuffles and splits')
parser.add_argument('--seg_len', type=int,
                    default=16000, help='segmentaion length in samples (16k = 1s)')
parser.add_argument('--win_len', type=int,
                    default=255, help='window length')
parser.add_argument('--hop_len', type=int,
                    default=32, help='hop length')
parser.add_argument('--n_mel', type=int,
                    default=128, help='number of mel bins')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch size')
parser.add_argument('--epochs', type=int,
                    default=100, help='number of epochs')


opt = parser.parse_args()


def main():

    data, num_classes = get_data(opt.data, opt.data_root)

    segmented_data, segmented_labels = get_audio_segmentation(data=data,
                                                              seg_len=opt.seg_len, win_len=opt.win_len, hop_len=opt.hop_len, n_mel=opt.n_mel)
    test_acc_list = []
    for i in range(0, opt.k):

        X_train, segmented_test, y_train, y_test = train_test_split(
            segmented_data, segmented_labels, test_size=0.2, shuffle=True, stratify=segmented_labels)
        segmented_train, segmented_val, y_training, y_valid = train_test_split(
            X_train, y_train, test_size=0.2, shuffle=True, stratify=y_train)

        print(len(segmented_test), 'testSet')
        print(len(segmented_train), 'trainingSet')
        print(len(segmented_val), 'validSet')

        mean1, std1 = get_mean_and_std(segmented_train, opt.batch_size)
        mean1 = mean1.numpy()
        std1 = std1.numpy()

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=mean1, std=std1)
                                        ])

        train_dataset = SERDataset(
            segmented_train, 'train', transform=transform)
        val_dataset = SERDataset(
            segmented_val, 'val', transform=transform)
        test_dataset = SERDataset(
            segmented_test, 'test', transform=transform)

        train_dataloader = DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(
            val_dataset, batch_size=1, shuffle=True, num_workers=0)
        test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=True, num_workers=0)

        model = squeezenet1_1(num_classes=num_classes)
        model = model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        eval_acc_max = 0

        for t in range(opt.epochs):

            print(f"Epoch {t+1}\n-------------------------------")
            train_loss, train_acc = train(
                train_dataloader, model, loss_fn, optimizer, device)
            eval_loss, eval_acc = evaluate(
                val_dataloader, model, loss_fn, device, 'Validation')

            if (eval_acc > eval_acc_max):
                eval_acc_max = eval_acc
                torch.save(model.state_dict(), 'model_{}.pth'.format(i))

        model.load_state_dict(torch.load('model_{}.pth'.format(i)))
        test_loss, test_acc = evaluate(
            test_dataloader, model, loss_fn, device, 'Test')

        test_acc_list.append(test_acc)
        print("Done!")

    test_acc_mean = np.mean(test_acc_list)
    test_acc_std = np.std(test_acc_list)

    result = {
        'n_mel': opt.n_mel,
        'seg_len': opt.seg_len,
        'win_len': opt.win_len,
        'hop_len': opt.hop_len,
        'acc': test_acc_mean,
        'std': test_acc_std,
    }
    print(result)


if __name__ == '__main__':
    main()
