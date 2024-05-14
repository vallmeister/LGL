import configargparse
import torch
import torch.nn as nn
import torch.utils.data as Data
import tqdm

from datasets import continuum, graph_collate
from models import LGL, AFGN, PlainNet
from torch_util import EarlyStopScheduler, performance

## AFGN is LGL with attention; AttnPlainNet is the PlainNet with attention
nets = {'lgl': LGL, 'afgn': AFGN, 'plain': PlainNet}


def train(loader, net, criterion, optimizer, device):
    net.train()
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets, neighbor) in enumerate(tqdm.tqdm(loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        if not args.k:
            neighbor = [element.to(device) for element in neighbor]
        else:
            neighbor = [[element.to(device) for element in item] for item in neighbor]

        optimizer.zero_grad()
        outputs = net(inputs, neighbor)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

    return (train_loss / (batch_idx + 1), correct / total)


if __name__ == '__main__':
    # Arguements
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="learning rate")
    parser.add_argument("--model", type=str, default='LGL', help="LGL or SAGE")
    parser.add_argument("--load", type=str, default=None, help="load pretrained model file")
    parser.add_argument("--save", type=str, default=None, help="model file to save")
    parser.add_argument("--optm", type=str, default='SGD', help="SGD or Adam")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--factor", type=float, default=0.1, help="ReduceLROnPlateau factor")
    parser.add_argument("--min-lr", type=float, default=0.001, help="minimum lr for ReduceLROnPlateau")
    parser.add_argument("--patience", type=int, default=3, help="patience for Early Stop")
    parser.add_argument("--batch-size", type=int, default=10, help="number of minibatch size")
    parser.add_argument("--milestones", type=int, default=15, help="milestones for applying multiplier")
    parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("--early-stop", type=int, default=5, help="number of epochs for early stop training")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum of the optimizer")
    parser.add_argument("--gamma", type=float, default=0.1, help="learning rate multiplier")
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    parser.add_argument("--k", type=int, default=None, help='khop.')
    parser.add_argument("--hidden", type=int, nargs="+", default=[10, 10])
    parser.add_argument("--drop", type=float, nargs="+", default=[0, 0])
    args = parser.parse_args();
    print(args)
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    # Datasets
    train_data = continuum(root=args.data_root, name='ogbn-arxiv', data_type='train', download=True, k_hop=args.k)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=10, shuffle=True,
                                   collate_fn=graph_collate, drop_last=True)
    test_data = continuum(root=args.data_root, name='ogbn-arxiv', data_type='test', download=True, k_hop=args.k)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=10, shuffle=False,
                                  collate_fn=graph_collate, drop_last=True)
    valid_data = continuum(root=args.data_root, name='ogbn-arxiv', data_type='valid', download=True, k_hop=args.k)
    valid_loader = Data.DataLoader(dataset=valid_data, batch_size=10, shuffle=False,
                                   collate_fn=graph_collate, drop_last=True)

    Net = nets['plain']
    net = Net(feat_len=train_data.feat_len, num_class=train_data.num_class, hidden=args.hidden, dropout=args.drop).to(
        args.device)

    criterion = nn.CrossEntropyLoss()
    exec('optimizer = torch.optim.%s(net.parameters(), lr=%f)' % (args.optm, args.lr))
    scheduler = EarlyStopScheduler(optimizer, factor=args.factor, verbose=True, min_lr=args.min_lr,
                                   patience=args.patience)

    # Training
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train(train_loader, net, criterion, optimizer, args.device)
        test_acc = performance(test_loader, net, args.device, args.k)  # validate
        print("epoch: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f"
              % (epoch, train_loss, train_acc, test_acc))
        if scheduler.step(error=1 - test_acc):
            print('Early Stopping!')
            break
