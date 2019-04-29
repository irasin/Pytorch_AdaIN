import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import PreprocessDataset, denorm
from model import Model


def main():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer by Pytorch')
    parser.add_argument('--batch_size', '-b', type=int, default=8,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--learning_rate', '-lr', type=int, default=5e-5,
                        help='learning rate for Adam')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot to generate image')
    parser.add_argument('--train_content_dir', type=str, default='content',
                        help='content images directory for train')
    parser.add_argument('--train_style_dir', type=str, default='style',
                        help='style images directory for train')
    parser.add_argument('--test_content_dir', type=str, default='content',
                        help='content images directory for test')
    parser.add_argument('--test_style_dir', type=str, default='style',
                        help='style images directory for test')
    parser.add_argument('--save_dir', type=str, default='result',
                        help='save directory for result and loss')
    parser.add_argument('--reuse', default=None,
                        help='model state path to load for reuse')

    args = parser.parse_args()

    # create directory to save
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    loss_dir = f'{args.save_dir}/loss'
    model_state_dir = f'{args.save_dir}/model_state'
    image_dir = f'{args.save_dir}/image'

    if not os.path.exists(loss_dir):
        os.mkdir(loss_dir)
        os.mkdir(model_state_dir)
        os.mkdir(image_dir)

    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    print(f'# Minibatch-size: {args.batch_size}')
    print(f'# epoch: {args.epoch}')
    print('')

    # prepare dataset and dataLoader
    train_dataset = PreprocessDataset(args.train_content_dir, args.train_style_dir)
    test_dataset = PreprocessDataset(args.test_content_dir, args.test_style_dir)
    iters = len(train_dataset)
    print(f'Length of train image pairs: {iters}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_iter = iter(test_loader)

    # set model and optimizer
    model = Model().to(device)
    if args.reuse is not None:
        model.load_state_dict(torch.load(args.reuse))
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # start training
    loss_list = []
    for e in range(1, args.epoch + 1):
        print(f'Start {e} epoch')
        for i, (content, style) in tqdm(enumerate(train_loader, 1)):
            content = content.to(device)
            style = style.to(device)
            loss = model(content, style)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'[{e}/total {args.epoch} epoch],[{i} /'
                  f'total {round(iters/args.batch_size)} iteration]: {loss.item()}')

            if i % args.snapshot_interval == 0:
                content, style = next(test_iter)
                content = content.to(device)
                style = style.to(device)
                with torch.no_grad():
                    out = model.generate(content, style)
                content = denorm(content, device)
                style = denorm(style, device)
                out = denorm(out, device)
                res = torch.cat([content, style, out], dim=0)
                res = res.to('cpu')
                save_image(res, f'{image_dir}/{e}_epoch_{i}_iteration.png', nrow=args.batch_size)
        torch.save(model.state_dict(), f'{model_state_dir}/{e}_epoch.pth')
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.savefig(f'{loss_dir}/train_loss.png')
    with open(f'{loss_dir}/loss_log.txt', 'w') as f:
        for l in loss_list:
            f.write(f'{l}\n')
    print(f'Loss saved in {loss_dir}')


if __name__ == '__main__':
    main()
