import os
import numpy as np
import argparse
from collections import OrderedDict

# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]

class Options:
    def __init__(self, isTrain):
        self.dataset = 'CRAG'
        # self.dataset = 'GlaS'
        # self.dataset = 'ganzhou_part1'
        self.isTrain = isTrain

        # --- model hyper-parameters --- #
        self.model = dict()
        self.model['in_c'] = 3       # input channel
        self.model['out_c'] = 3      # output channel
        # self.model['os'] = 16
        # self.model['resnet_type'] = '50'  # 可选: '50' 或 '101'
        # self.model['备注'] = 'unet'


        # --- training params --- #
        self.train = dict()
        self.train['data_dir'] = './data/{:s}'.format(self.dataset)
        # self.train['save_dir'] = './experiments/{:s}/{}/'.format(self.dataset,self.model['os'])
        self.train['save_dir'] = './experiments/{:s}'.format(self.dataset)
        self.train['input_size'] = (512, 512)
        self.train['num_epochs'] = 250
        self.train['batch_size'] = 16
        self.train['val_overlap'] = 80   # overlap size of patches for validation
        self.train['lr'] = 0.0005 # initial learning rate
        self.train['weight_decay'] = 1e-4  # weight decay
        self.train['log_interval'] = 50    # iterations to print training results
        self.train['workers'] = 2        # number of workers to load images
        self.train['gpu'] = [0, ]
        self.train['alpha'] = 1.0        # weight for variance term
        self.train['checkpoint_freq'] = 100
        # --- resume training --- #
        self.train['start_epoch'] = 0 
        self.train['checkpoint'] = ''
        # self.train['checkpoint'] = './experiments/{:s}/checkpoints/checkpoint.pth.tar'.format(self.dataset)

        # --- data transform --- #
        self.transform = dict()
        # defined in parse function

        # --- test parameters --- #
        self.test = dict()
        self.test['epoch'] = ''
        self.test['hausdorff'] = True
        # self.test['hausdorff'] = False
        self.test['gpu'] = [0,]
        self.test['img_dir'] = './data/{:s}/images/test'.format(self.dataset)
        self.test['label_dir'] = './data/{:s}/labels_instance/test'.format(self.dataset)
        self.test['mask_color'] = './data/{:s}/labels_instance/val'.format(self.dataset)
        # self.test['img_dir'] = './data/{:s}/3zhangtu/img'.format(self.dataset)
        # self.test['label_dir'] = './data/{:s}/3zhangtu/mask'.format(self.dataset)
        # self.test['mask_color'] = './data/{:s}/3zhangtu/mask_color'.format(self.dataset)
        self.test['tta'] = True
        self.test['save_flag'] = True
        self.test['patch_size'] = 1024
        self.test['overlap'] = 512
        # self.test['save_dir'] = './experiments/{:s}/test'.format(self.dataset)   #用于多模型文件测试
        # self.test['model_path'] = './experiments/{:s}/checkpoints'.format(self.dataset)    #用于多模型文件测试
        self.test['save_dir'] = './experiments/{:s}_attunet/test/best_197'.format(self.dataset) 
        self.test['model_path'] = './experiments/{:s}_attunet/checkpoints/checkpoint_epoch_197_iou_0.8462.pth.tar'.format(self.dataset)
      
        # self.test['save_dir'] = './gland'.format(self.dataset) 

        # --- post processing --- #
        self.post = dict()
        self.post['min_area'] = 1290
        self.post['radius'] = 3


    def parse(self):
        """ Parse the options, replace the default value if there is a new input """
        parser = argparse.ArgumentParser(description='')
        if self.isTrain:
            parser.add_argument('--batch-size', type=int, default=self.train['batch_size'], help='input batch size for training')
            parser.add_argument('--alpha', type=float, default=self.train['alpha'], help='The weight for the variance term in loss')
            parser.add_argument('--epochs', type=int, default=self.train['num_epochs'], help='number of epochs to train')
            parser.add_argument('--lr', type=float, default=self.train['lr'], help='learning rate')
            parser.add_argument('--log-interval', type=int, default=self.train['log_interval'], help='how many batches to wait before logging training status')
            parser.add_argument('--gpu', type=list, default=self.train['gpu'], help='GPUs for training')
            parser.add_argument('--data-dir', type=str, default=self.train['data_dir'], help='directory of training data')
            parser.add_argument('--save-dir', type=str, default=self.train['save_dir'], help='directory to save training results')
            parser.add_argument('--checkpoint-path', type=str, default=self.train['checkpoint'], help='directory to load a checkpoint')
            args = parser.parse_args()

            self.train['batch_size'] = args.batch_size
            self.train['alpha'] = args.alpha
            self.train['num_epochs'] = args.epochs
            self.train['lr'] = args.lr
            self.train['log_interval'] = args.log_interval
            self.train['gpu'] = args.gpu
            self.train['checkpoint'] = args.checkpoint_path
            self.train['data_dir'] = args.data_dir
            self.train['img_dir'] = '{:s}/images'.format(self.train['data_dir'])
            self.train['label_dir'] = '{:s}/labels_instance'.format(self.train['data_dir'])
            # self.train['weight_map_dir'] = '{:s}/weight_maps'.format(self.train['data_dir'])

            self.train['save_dir'] = args.save_dir
            if not os.path.exists(self.train['save_dir']):
                os.makedirs(self.train['save_dir'], exist_ok=True)

            # define data transforms for training
            self.transform['train'] = OrderedDict()
            self.transform['val'] = OrderedDict()
            if self.dataset == 'GlaS':
                self.transform['train'] = {
                    # 'scale': 208+30,
                    'horizontal_flip': True,
                    'random_affine': 0.3,
                    'random_elastic': [6, 15],
                    'random_rotation': 90,
                    'random_crop': self.train['input_size'],
                    'label_encoding': 2,
                    'to_tensor': 1,
                    # 'normalize': np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
                }
                self.transform['train_sdm'] = {
                    # 'scale': 208+30,
                    # 'horizontal_flip': True,
                    # 'random_affine': 0.3,
                    # 'random_elastic': [6, 15],
                    # 'random_rotation': 90,
                    'random_crop': self.train['input_size'],
                    #                    'label_encoding': 2,
                    'to_tensor': 1,
                    # 'normalize': np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
                }


                self.transform['val'] = {
                    # 'scale': 208,
                    'label_encoding': 2,
                    'to_tensor': 1,
                    # 'normalize': np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
                }
            else:
                self.transform['train'] = {
                    'random_resize': [0.8, 1.25],
                    # 'scale': 512,
                    'horizontal_flip': True,
                    'random_affine': 0.3,
                    'random_elastic': [6, 15],
                    'random_rotation': 90,
                    'random_crop': self.train['input_size'],
                    'label_encoding': 1,
                    'to_tensor': 1,
                    # 'normalize': [IMAGENET_MEAN, IMAGENET_STD]
                    # 'normalize': np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
                }
                self.transform['val'] = {
                    'label_encoding': 1,
                    'to_tensor': 1,
                    # 'normalize': [IMAGENET_MEAN, IMAGENET_STD]
                    # 'normalize': np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
                }

        else:
            parser.add_argument('--epoch', type=str, default=self.test['epoch'], help='select the model used for testing')
            parser.add_argument('--save-flag', type=bool, default=self.test['save_flag'], help='flag to save the network outputs and predictions')
            parser.add_argument('--gpu', type=list, default=self.test['gpu'], help='GPUs for training')
            parser.add_argument('--img-dir', type=str, default=self.test['img_dir'], help='directory of test images')
            parser.add_argument('--label-dir', type=str, default=self.test['label_dir'], help='directory of labels')
            parser.add_argument('--save-dir', type=str, default=self.test['save_dir'], help='directory to save test results')
            parser.add_argument('--model-path', type=str, default=self.test['model_path'], help='train model to be evaluated')
            args = parser.parse_args()
            self.test['epoch'] = args.epoch
            self.test['gpu'] = args.gpu
            self.test['save_flag'] = args.save_flag
            self.test['img_dir'] = args.img_dir
            self.test['label_dir'] = args.label_dir
            self.test['save_dir'] = args.save_dir
            self.test['model_path'] = args.model_path

            if not os.path.exists(self.test['save_dir']):
                os.makedirs(self.test['save_dir'], exist_ok=True)

            self.transform['test'] = OrderedDict()
            if self.dataset == 'GlaS':
                self.transform['test'] = {
                    # 'scale': 208,
                    'to_tensor': 1,
                    # 'normalize': np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
                }
            else:
                self.transform['test'] = {
                    'to_tensor': 1,
                    # 'normalize': [IMAGENET_MEAN, IMAGENET_STD]
                    # 'normalize': np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
                }

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
