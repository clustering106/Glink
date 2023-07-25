import sys

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.utils as vutils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2

from utils import cv_utils
from symbols import *
from utils import data_load
from utils import face_attr_loss

def train(config_file):
    print("into train func...")

    # read config    
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    print("get cfg...")
    print(cfg)

    epochs = cfg['total_epochs']

    # make save dir
    train_model = config_file.split('/')[-1].split('.')[0]
    save_root = '/home/zhangkai/Myproject/face_attr/checkpoints/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    tb_i = 0
    tb_root = save_root + "sample"
    while os.path.exists(tb_root + str(tb_i)): # 递增
        tb_i += 1

    save_root = tb_root + str(tb_i) + '/'
    print('all train info will save in:', save_root)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    tb_writer = SummaryWriter(save_root)  # Tensorboard

    last_model = save_root + 'last.pt'
    best_model = save_root + 'best.pt'
    print(last_model)
    print(best_model)

    # select device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set init seed
    cv_utils.init_seeds(seed=0)
    # get model
    # model = get_model.build_model(cfg).to(device)
    model = get_mbf().to(device)
    # model = get_mobilenetv3().to(device)

    writer = SummaryWriter(log_dir='/home/zhangkai/Myproject/log')  
    fake_input = torch.randn(1,3,112,112).to(device)
    writer.add_graph(model=model, input_to_model=fake_input)
    writer.close()

    # ema
    ModelEMA = cv_utils.ModelEMA(model)

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if 'adam' == cfg['optimizer']:
        optimizer = optim.Adam(parameters, lr=cfg['lr_base'], betas=(cfg['momentum'], 0.999))
        # optimizer = optim.Adam(pg0, lr=cfg['lr_base'], betas=(cfg['momentum'], 0.999))
    else:
        optimizer = optim.SGD(parameters, lr=cfg['lr_base'], momentum=cfg['momentum'], nesterov=True)

    if 'linear_lr' ==  cfg['scheduler']:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - cfg['lr_final']) + cfg['lr_final']  # linear
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    elif 'one_cycle' ==  cfg['scheduler']:
        lf = cv_utils.one_cycle(1, cfg['lr_final'], epochs)  # cosine 1->hyp['lr_final']
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    elif 'multi_step' ==  cfg['scheduler']:
        milestones = [int(epochs * 0.6), int(epochs * 0.9)]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones, 0.1)
    else:
        print('get', cfg['scheduler'], 'not support!!!')
    scheduler.last_epoch = - 1

    # data loader
    train_loader = data_load.create_dataloader(cfg=cfg, path=cfg['train'], imgsz=cfg['image_size'], batch_size=cfg['batch_size'], augment=True, workers=cfg['workers'])
    val_loader = data_load.create_dataloader(cfg=cfg, path=cfg['val'], imgsz=cfg['image_size'], batch_size=cfg['batch_size'], augment=False, workers=cfg['workers'])

    # loss func
    if 'face_attr' == train_model:
        compute_loss = face_attr_loss.ComputeLoss(model, cfg)

    # best acc
    best_acc = -1

    # start epoch
    for epoch in range(epochs):
        model.train()
        
        # Warmup
        nw = round(cfg['warmup_epochs'])
        if epoch < nw:
            for _, x in enumerate(optimizer.param_groups):
                x['lr'] = x['initial_lr'] * (epoch+1) / (nw+1)

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)

        mloss = torch.zeros(14, device=device)  # mean losses
        # ['pretty', 'blur', 'glass', 'makeup', 'gender', 'mouthopen', 'smile']
        print(('%4s' + '%10s' * 18) % ('', 'epoch', 'mem/G', 'l/pre', 'l/blu', 'l/gla', 'l/mak', 'l/gen', 
                                       'l/mou', 'l/smi','l/eye', 'l/isf', 'l/blo', 'l/yaw', 'l/pit', 'l/age', 'l/mask', 'batch', 'imgs'))
        
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=len(train_loader), desc='train')
        for i_batch, (imgs, targets) in pbar:

            imgs = imgs.to(device, non_blocking=True).float()
            targets = targets.to(device)
            if False:
                print(imgs.shape)
                # show target
                cls_names = cfg["names"]
                for i in range(imgs.shape[0]):
                    image = imgs[i]
                #     image = (image * 255)
                #     image = np.clip(image, 0, 255)
                    image = np.transpose(image.cpu().numpy(), (1, 2, 0))
                    
                    image = image.astype('uint8')
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    attrs = targets[i]
                    
                    image = cv2.resize(image, dsize=(256, 256))

                    skip = 10
                    for idx, attr_name in enumerate(cls_names):
                        cv2.putText(image, '%s: %.2f' % (attr_name, attrs[idx]), (5, 10 + idx * skip), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        if idx >=3 and idx <= 12 and idx % 2 == 0:
                            cv2.circle(image, (int(attrs[idx - 1] * image.shape[1]), int(attrs[idx] * image.shape[0])), 3, (0, 255, 0), -1)

                    b_show = True
                    if b_show:
                        cv2.imshow('line_split[0]', image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
            optimizer.zero_grad()

            # with torch.cuda.amp.autocast():
            preds = model(imgs)
            loss, loss_items = compute_loss(preds, targets)

            loss.backward()
            optimizer.step()
            ModelEMA.update(model)

            mloss = (mloss * i_batch + loss_items) / (i_batch + 1)  # update mean losses
            s = ('train: %3g/%3g' + '%10.3g' * 17) % (epoch + 1, epochs, mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)
            if 0 == i_batch and epoch < 3:
                img_grid = vutils.make_grid(imgs*0.5+0.5)
                tb_writer.add_image('imgs' + str(epoch), img_grid, epoch)

        # end epoch ----------------------------------------------------------------------------------------------------
        lr = [x["lr"] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # start eval ----------------------------------------------------------------------------------------------------
        model.eval()
        # model_eval = ModelEMA.ema
        # model_eval.eval()

        eloss = torch.zeros(14, device=device)  # mean losses
        eacc = torch.zeros(14, device=device)  # mean accs

        pbar = enumerate(val_loader)
        pbar = tqdm(pbar, total=len(val_loader), desc='val')
        for i_batch, (imgs, targets) in pbar:
            
            imgs = imgs.to(device, non_blocking=True).float()
            targets = targets.to(device)

            with torch.no_grad():
                preds = model(imgs)
                # preds = model_eval(imgs)

                loss, loss_items = compute_loss(preds, targets)
                eloss = (eloss * i_batch + loss_items) / (i_batch + 1)  # update mean losses

                acc_items = cv_utils.get_accuracy(preds, targets, eacc)
                eacc = (eacc * i_batch + acc_items) / (i_batch + 1)  # update mean acc

                s = ('  val: %3g/%3g' + '%10.3g' * 17) % (epoch + 1, epochs, mem, *eloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)
        
        # start log ----------------------------------------------------------------------------------------------------
        #names: ['pretty', 'blur', 'glass', 'makeup', 'gender', 'mouthopen', 'smile', 'eyeopen', 'isface', 'block', 'yaw', 'pitch']
        tags = ['train/pretty_loss', 'train/blur_loss', 'train/glass_loss', 'train/makeup_loss', 
        'train/gender_loss', 'train/mouthopen_loss', 'train/smile_loss', 'train/eyeopen_loss', 'train/isface_loss', 'train/block_loss',
        'train/yaw_loss', 'train/pitch_loss', 'train/age_loss','train/mask_loss',
        'val/pretty_loss', 'val/blur_loss', 'val/glass_loss', 'val/makeup_loss', 
        'val/gender_loss', 'val/mouthopen_loss', 'val/smile_loss', 'val/eyeopen_loss', 'val/isface_loss', 'val/block_loss',
        'val/yaw_loss', 'val/pitch_loss', 'val/age_loss','val/mask_loss',
        'acc/pretty', 'acc/blur', 'acc/glass', 'acc/makeup', 
        'acc/gender', 'acc/mouthopen', 'acc/smile', 'acc/eyeopen', 'acc/isface', 'acc/block', 'acc/yaw', 'acc/pitch','acc/age', 'acc/mask', 'lr/lr0']

        for x, tag in zip(list(mloss) + list(eloss) + list(eacc) + lr, tags):
            tb_writer.add_scalar(tag, x, epoch)

        print('%24s'%'' + '%10s' * 14 % ('a/pre', 'a/blu', 'a/gla', 'a/mak', 'a/gen', 'a/mou', 'a/smi','a/eye', 'a/isf', 'a/blo', 'a/yaw', 'a/pit', 'a/age', 'a/mask'))
        print('%24s'%'' + '%10.3g' * 14 % (eacc[0], eacc[1], eacc[2], eacc[3], eacc[4], eacc[5], eacc[6],eacc[7], eacc[8], eacc[9], eacc[10], eacc[11], eacc[12], eacc[13]))
        print('')

        # start save ----------------------------------------------------------------------------------------------------
        cv_utils.save_checkpoint(model, last_model)
        # cv_utils.save_checkpoint(model_eval, last_model)

        cur_acc = eacc[0] + eacc[1] + eacc[2] + eacc[3] + eacc[4] + eacc[5] + eacc[6] + eacc[7] + eacc[8] + eacc[9] + eacc[13] # age land 应该更合理的归一化
        if cur_acc > best_acc:
            best_acc = cur_acc
            cv_utils.save_checkpoint(model, best_model)
            # cv_utils.save_checkpoint(model_eval, best_model)

    # end epoch ----------------------------------------------------------------------------------------------------
    tb_writer.close()

    # end training
    print("end train func, all info saved in:", save_root)

if __name__ == "__main__":
    
    config_file = "/home/zhangkai/Myproject/face_attr/configs/face_attr.yaml"

    train(config_file=config_file)

    print("end all train !!!")