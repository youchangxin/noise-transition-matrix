import argparse
import sys
import os
import torch
import pandas as pd
import torch.nn.functional as F
import random
from tqdm import tqdm


from utils.tools import transition_matrix_error
from pathlib import Path
from torch.utils.data import DataLoader
from dataloader import *
from models.lenet import LeNet
from models.resnet import resnet18, resnet34
from models.transitionMatrix import TransitionMatrix

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # arguments
    parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100', default='mnist')
    parser.add_argument('--flip_type', type=str, default='pair')
    parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.5)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', help='cuda device, i.e. 0 or 0,1,2,3 or cpu', default=0)
    parser.add_argument('--lam', type=float, default=0.0001)
    parser.add_argument('--without_vol', action='store_true', help='Not to use VolMinNet to train dataset')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def splite_train_valid(dataset, split_rate=0.1, seed=20):
    torch.manual_seed(seed=seed)
    valid_size = int(split_rate * len(dataset))
    train_size = len(dataset) - valid_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, valid_size])
    return train_set, val_set


def experiment_mnist(opt):
    # create Dataset
    dataset = Dataset_MNIST(noise_rate=opt.noise_rate, flip_type=opt.flip_type, random_seed=opt.seed)
    train_set, val_set = splite_train_valid(dataset, seed=opt.seed)
    test_set = Dataset_MNIST_TEST()

    model = LeNet()
    return train_set, val_set, test_set, model, dataset.transition_matrix


def experiment_cifar10(opt):
    # create Dataset
    dataset = Dataset_CIFAR(num_classes=10, noise_rate=opt.noise_rate, flip_type=opt.flip_type, random_seed=opt.seed)
    train_set, val_set = splite_train_valid(dataset, seed=opt.seed)
    test_set = Dataset_CIFAR_TEST(num_classes=10)

    model = resnet18(num_classes=10)
    return train_set, val_set, test_set, model, dataset.transition_matrix


def experiment_cifar100(opt):
    # create Dataset
    dataset = Dataset_CIFAR(num_classes=100, noise_rate=opt.noise_rate, flip_type=opt.flip_type, random_seed=opt.seed)
    train_set, val_set = splite_train_valid(dataset, seed=opt.seed)
    test_set = Dataset_CIFAR_TEST(num_classes=100)

    model = resnet34(num_classes=100)
    return train_set, val_set, test_set, model, dataset.transition_matrix


def train(opt, device):
    if opt.dataset == 'mnist':
        train_set, val_set, test_set, model, T = experiment_mnist(opt)
        EPOCH = 60
        num_classes = 10

    elif opt.dataset == 'cifar10':
        train_set, val_set, test_set, model, T = experiment_cifar10(opt)
        EPOCH = 150
        num_classes = 10
    else:
        train_set, val_set, test_set, model, T = experiment_cifar100(opt)
        EPOCH = 150
        num_classes = 100

    # create data_loader
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size,
                              shuffle=True, num_workers=4, drop_last=False)

    val_loader = DataLoader(dataset=val_set, batch_size=opt.batch_size,
                            shuffle=True, num_workers=4, drop_last=False)

    test_loader = DataLoader(dataset=test_set, batch_size=opt.batch_size,
                             shuffle=True, num_workers=4, drop_last=False)
    # loss
    loss_func_ce = F.nll_loss  # softmax is added in model

    # optimizer and StepLR
    optimizer_h = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_h, milestones=[30, 60], gamma=0.1)

    transition_matrix = TransitionMatrix(num_classes=num_classes, device=device)
    if opt.dataset == "cifar10":
        optimizer_trans = torch.optim.SGD(transition_matrix.parameters(), lr=10e-2, weight_decay=0, momentum=0.9)
    else:
        optimizer_trans = torch.optim.Adam(transition_matrix.parameters(), lr=10e-2)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_trans, milestones=[30, 60], gamma=0.1)

    model.to(device)
    transition_matrix.to(device)

    # Create DataFrame to log
    res = pd.DataFrame()

    for epoch in range(EPOCH):
        model.train()
        transition_matrix.train()

        train_loss = 0.
        train_vol_loss = 0.
        train_acc = 0.

        val_loss = 0.
        val_acc = 0.

        # ################ TRAIN ################
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for index, (batch_x, batch_y) in loop:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer_h.zero_grad()
            optimizer_trans.zero_grad()

            clean = model(batch_x)
            t_hat = transition_matrix()
            y_tilde = torch.mm(clean, t_hat)
            vol_loss = torch.abs(t_hat.slogdet().logabsdet)
            ce_loss = loss_func_ce(y_tilde.log(), batch_y.long())
            loss = ce_loss + opt.lam * vol_loss

            train_loss += loss.item()
            train_vol_loss += vol_loss.item()

            pred = torch.max(y_tilde, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()

            loss.backward()
            optimizer_h.step()
            optimizer_trans.step()

            # update tqdm
            loop.set_description(f'Epoch [{epoch}/{EPOCH}]')

        # train metric
        train_loss = train_loss / (len(train_set)) * opt.batch_size
        train_vol_loss = train_vol_loss / (len(train_set)) * opt.batch_size
        train_acc = train_acc / (len(train_set))

        print('Train Loss: {:.6f} Train vol loss: {:.6f}  Train Acc: {:.6f}'.format(train_loss,
                                                                                    train_vol_loss,
                                                                                    train_acc))

        scheduler1.step()
        scheduler2.step()

        # ################ VALIDATION ################
        with torch.no_grad():
            model.eval()
            transition_matrix.eval()
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                clean = model(batch_x)
                t_hat = transition_matrix()

                y_tilde = torch.mm(clean, t_hat)
                loss = loss_func_ce(y_tilde.log(), batch_y.long())
                val_loss += loss.item()
                pred = torch.max(y_tilde, 1)[1]
                val_correct = (pred == batch_y).sum()
                val_acc += val_correct.item()

        # validation metric
        val_loss = val_loss / len(val_set)
        val_acc = val_acc / len(val_set)

        print('Val Loss: {:.6f}  Val Acc: {:.6f}'.format(val_loss, val_acc))

        # record log
        res_row = pd.DataFrame({'train_loss': train_loss, 'train_vol_loss': train_vol_loss, 'train_acc': train_acc,
                                'val_loss': val_loss, 'val_acc': val_acc}, index=[epoch])
        res = res.append(res_row)

    # ################ TEST ################
    test_acc = 0.0
    with torch.no_grad():
        model.eval()
        transition_matrix.eval()

        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            clean = model(batch_x)
            pred = torch.max(clean, 1)[1]
            eval_correct = (pred == batch_y).sum()
            test_acc += eval_correct.item()
        test_acc /= len(test_set)
        print('Test Accuracy: {:.6f}'.format(test_acc))

    T_hat = t_hat.detach().cpu().numpy()
    matrix_error = transition_matrix_error(T, T_hat)
    print('Estimation Transition Matrix Error: {:.2f}'.format(matrix_error))

    # ##################### SAVE Result #####################
    folder = "output/" + str(opt.dataset)+"_"+str(opt.flip_type)+"_"+str(opt.noise_rate)+"/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    # train_val result
    res.to_csv(folder+"result.csv")
    # test result
    with open(folder+"test.txt", "w") as f:
        f.write(str(test_acc))
    #  Save transition matrix
    np.save(folder+'T', T)
    np.save(folder+'T_hat', T_hat)


def train_without_vol(opt, device):
    if opt.dataset == 'mnist':
        train_set, val_set, test_set, model, T = experiment_mnist(opt)
        EPOCH = 60
        num_classes = 10

    elif opt.dataset == 'cifar10':
        train_set, val_set, test_set, model, T = experiment_cifar10(opt)
        EPOCH = 150
        num_classes = 10
    else:
        train_set, val_set, test_set, model, T = experiment_cifar100(opt)
        EPOCH = 150
        num_classes = 100

    # create data_loader
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size,
                              shuffle=True, num_workers=4, drop_last=False)

    val_loader = DataLoader(dataset=val_set, batch_size=opt.batch_size,
                            shuffle=True, num_workers=4, drop_last=False)

    test_loader = DataLoader(dataset=test_set, batch_size=opt.batch_size,
                             shuffle=True, num_workers=4, drop_last=False)
    # loss
    loss_func_ce = F.nll_loss  # softmax is added in model

    # optimizer and StepLR
    optimizer_h = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_h, milestones=[30, 60], gamma=0.1)

    model.to(device)

    # Create DataFrame to log
    res = pd.DataFrame()

    for epoch in range(EPOCH):
        model.train()

        train_loss = 0.
        train_acc = 0.

        val_loss = 0.
        val_acc = 0.

        # ################ TRAIN ################
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for index, (batch_x, batch_y) in loop:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer_h.zero_grad()

            clean = model(batch_x)

            loss = loss_func_ce(clean.log(), batch_y.long())
            train_loss += loss.item()

            pred = torch.max(clean, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()

            loss.backward()
            optimizer_h.step()

            # update tqdm
            loop.set_description(f'Epoch [{epoch}/{EPOCH}]')

        # train metric
        train_loss = train_loss / (len(train_set)) * opt.batch_size
        train_acc = train_acc / (len(train_set))

        print('Train Loss: {:.6f} Train Acc: {:.6f}'.format(train_loss, train_acc))

        scheduler1.step()

        # ################ VALIDATION ################
        with torch.no_grad():
            model.eval()
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                clean = model(batch_x)

                loss = loss_func_ce(clean.log(), batch_y.long())
                val_loss += loss.item()
                pred = torch.max(clean, 1)[1]
                val_correct = (pred == batch_y).sum()
                val_acc += val_correct.item()

        # validation metric
        val_loss = val_loss / len(val_set)
        val_acc = val_acc / len(val_set)

        print('Val Loss: {:.6f}  Val Acc: {:.6f}'.format(val_loss, val_acc))

        # record log
        res_row = pd.DataFrame({'train_loss': train_loss, 'train_acc': train_acc,
                                'val_loss': val_loss, 'val_acc': val_acc}, index=[epoch])
        res = res.append(res_row)

    # ################ TEST ################
    test_acc = 0.0
    with torch.no_grad():
        model.eval()

        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            clean = model(batch_x)
            pred = torch.max(clean, 1)[1]
            eval_correct = (pred == batch_y).sum()
            test_acc += eval_correct.item()
        test_acc /= len(test_set)
        print('Test Accuracy: {:.6f}'.format(test_acc))


    # ##################### SAVE Result #####################
    folder = "output/" + str(opt.dataset)+"_"+str(opt.flip_type)+"_"+str(opt.noise_rate)+"_NoVol/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    # train_val result
    res.to_csv(folder+"result.csv")
    # test result
    with open(folder+"test.txt", "w") as f:
        f.write(str(test_acc))


def main(opt):
    setup_seed(opt.seed)
    device = torch.device("cuda:"+str(opt.device) if torch.cuda.is_available() else "cpu")
    if opt.without_vol:
        train_without_vol(opt, device)
    else:
        train(opt, device)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

