import os
import numpy as np
import argparse
from collections import OrderedDict

# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]

class Options:
    def __init__(self, isTrain):
        self.dataset = 'ganzhou_part1'
        # self.dataset = 'ganzhou_few'
        self.isTrain = isTrain

        # --- model hyper-parameters --- #
        self.model = dict()
        # self.model['name'] = 'resnet50'  # 可选: 'resnet50', 'resnet101', 'densenet121', 'densenet169', 'efficientnet_b0', 'efficientnet_b3', 'vit_base'
        # self.model['name'] = 'densenet121' 
        # self.model['name'] = 'vit_base' 
        # self.model['name'] = 'efficientnet_b3' 
        # self.model['name'] = 'convnext_tiny' 
        # self.model['name'] = 'resnext50'
        # self.model['name'] = 'swin_t' 
        self.model['name'] = 'regnet_y_3_2gf' 
        # self.model['name'] = 'mobilenet_v3_large' 
        self.model['num_classes'] = 2  # 二分类：良性/恶性
        self.model['pretrained'] = True  # 是否使用预训练权重

        # --- training params --- #
        self.train = dict()
        self.train['data_dir'] = './data/{:s}'.format(self.dataset)
        self.train['save_dir'] = './experiments/{:s}'.format(self.dataset)
        self.train['input_size'] = (256, 256)
        # self.train['input_size'] = (224, 224)
        # self.train['input_size'] = (300, 300)
        self.train['num_epochs'] = 100
        self.train['batch_size'] = 64
        self.train['lr'] = 0.0005 # initial learning rate
        self.train['weight_decay'] = 1e-4  # weight decay
        self.train['log_interval'] = 50    # iterations to print training results
        self.train['workers'] = 2        # number of workers to load images
        self.train['gpu'] = [0, ]
        # --- resume training --- #
        self.train['start_epoch'] = 0 
        self.train['checkpoint'] = ''
        # self.train['checkpoint'] = './experiments/{:s}/checkpoints/checkpoint.pth.tar'.format(self.dataset)

        # --- test parameters --- #
        self.test = dict()
        self.test['gpu'] = [0,]
        self.test['img_dir'] = './data/{:s}/img/test'.format(self.dataset)
        self.test['mask_color'] = './data/{:s}/mask_color/test'.format(self.dataset)
        self.test['save_dir'] = './experiments/{:s}/test'.format(self.dataset)   #用于多模型文件测试
        self.test['model_path'] = './experiments/{:s}/checkpoints'.format(self.dataset)    #用于多模型文件测试
        # self.test['save_dir'] = './experiments/{:s}/test/best'.format(self.dataset) 
        # self.test['model_path'] = './experiments/{:s}/checkpoints/checkpoint_best.pth.tar'.format(self.dataset)
      

    def parse(self):
        """ Parse the options, replace the default value if there is a new input """
        parser = argparse.ArgumentParser(description='')
        if self.isTrain:
            parser.add_argument('--batch-size', type=int, default=self.train['batch_size'], help='input batch size for training')
            parser.add_argument('--epochs', type=int, default=self.train['num_epochs'], help='number of epochs to train')
            parser.add_argument('--lr', type=float, default=self.train['lr'], help='learning rate')
            parser.add_argument('--log-interval', type=int, default=self.train['log_interval'], help='how many batches to wait before logging training status')
            parser.add_argument('--gpu', type=list, default=self.train['gpu'], help='GPUs for training')
            parser.add_argument('--data-dir', type=str, default=self.train['data_dir'], help='directory of training data')
            parser.add_argument('--save-dir', type=str, default=self.train['save_dir'], help='directory to save training results')
            parser.add_argument('--checkpoint-path', type=str, default=self.train['checkpoint'], help='directory to load a checkpoint')
            args = parser.parse_args()

            self.train['batch_size'] = args.batch_size
            self.train['num_epochs'] = args.epochs
            self.train['lr'] = args.lr
            self.train['log_interval'] = args.log_interval
            self.train['gpu'] = args.gpu
            self.train['checkpoint'] = args.checkpoint_path
            self.train['data_dir'] = args.data_dir
            self.train['img_dir'] = '{:s}/img'.format(self.train['data_dir'])
            self.train['label_dir'] = '{:s}/mask_color'.format(self.train['data_dir'])

            self.train['save_dir'] = args.save_dir
            if not os.path.exists(self.train['save_dir']):
                os.makedirs(self.train['save_dir'], exist_ok=True)
        else:
            parser.add_argument('--gpu', type=list, default=self.test['gpu'], help='GPUs for training')
            parser.add_argument('--img-dir', type=str, default=self.test['img_dir'], help='directory of test images')
            parser.add_argument('--save-dir', type=str, default=self.test['save_dir'], help='directory to save test results')
            parser.add_argument('--model-path', type=str, default=self.test['model_path'], help='train model to be evaluated')
            args = parser.parse_args()
            self.test['gpu'] = args.gpu
            self.test['img_dir'] = args.img_dir
            self.test['save_dir'] = args.save_dir
            self.test['model_path'] = args.model_path

            if not os.path.exists(self.test['save_dir']):
                os.makedirs(self.test['save_dir'], exist_ok=True)

    def print_options(self, logger=None):
        message = '\n'
        message += self._generate_message_from_options()
        if not logger:
            print(message)
        else:
            logger.info(message)

    def save_options(self):
        if self.isTrain:
            filename = '{:s}/train_options.txt'.format(self.train['save_dir'])
        else:
            filename = '{:s}/test_options.txt'.format(self.test['save_dir'])
        message = self._generate_message_from_options()
        file = open(filename, 'w')
        file.write(message)
        file.close()

    def _generate_message_from_options(self):
        message = ''
        message += '# {str:s} Options {str:s} #\n'.format(str='-'*25)
        train_groups = ['model', 'train', 'transform']
        test_groups = ['model', 'test', 'post', 'transform']
        cur_group = train_groups if self.isTrain else test_groups

        for group, options in self.__dict__.items():
            if group not in train_groups + test_groups:
                message += '{:>20}: {:<35}\n'.format(group, str(options))
            elif group in cur_group:
                message += '\n{:s} {:s} {:s}\n'.format('*' * 15, group, '*' * 15)
                if group == 'transform':
                    for name, val in options.items():
                        if (self.isTrain and name != 'test') or (not self.isTrain and name == 'test'):
                            message += '{:s}:\n'.format(name)
                            for t_name, t_val in val.items():
                                t_val = str(t_val).replace('\n', ',\n{:22}'.format(''))
                                message += '{:>20}: {:<35}\n'.format(t_name, str(t_val))
                else:
                    for name, val in options.items():
                        message += '{:>20}: {:<35}\n'.format(name, str(val))
        message += '# {str:s} End {str:s} #\n'.format(str='-'*26)
        return message
