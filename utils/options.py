import argparse
import os 

parser = argparse.ArgumentParser(description='Adversarial Network Compression')

parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')

parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=('cifar10'),
    help='Dataset to train')

parser.add_argument(
    '--workers',
    type=int,
    default=2)

parser.add_argument(
    '--data_dir',
    type=str,
    default='./data/CIFAR10',
    help='The directory where the CIFAR-10 input data is stored.')

parser.add_argument(
    '--job_dir',
    type=str,
    default='result/tmp',
    help='The directory where the summaries will be stored.')

parser.add_argument(
    '--reset',
    action='store_true',
    help='Reset the directory')

parser.add_argument(
    '--resume', 
    type=str, 
    default=None,
    help='load the model from the specified checkpoint')


## Model

parser.add_argument(
    '--block_type',
    type=str,
    default='shift',
    choices=('Shift','DW','Group'),
    help='The block type')

parser.add_argument(
    '--group_num',
    type=int,
    default=1,
    help='The num of groups in each convolution')

parser.add_argument(
    '--expansion',
    type=int,
    default=1,
    help='The value of expansion')

parser.add_argument(
    '--num_stu',
    type=int,
    default=4,
    help='The number of student models')

parser.add_argument(
    '--epochs',
    type=int,
    default=300,
    help='The num of epochs to train.')

parser.add_argument(
    '--start_epoch',
    type=int,
    default=0,
    help='start epochs to train.')

parser.add_argument(
    '--scheduler', 
    type=str,
    default='multistep',
    help='training scheduler')

parser.add_argument(
    '--train_batch_size',
    type=int,
    default=128,
    help='Batch size for training.')

parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=100,
    help='Batch size for validation.')

parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='Momentum for MomentumOptimizer.')

parser.add_argument(
    '--lr',
    type=float,
    default=0.1
)

parser.add_argument(
    '--lr_decay_step',
    default='150,225'
)

parser.add_argument(
    '--lr_decay_factor',
    type=float,
    default=0.1
)

parser.add_argument(
    '--weight_decay', 
    type=float,
    default=2e-4,
    help='The weight decay of loss.'
)

parser.add_argument(
    '--t',
    type=float,
    default=4,
    help='temperature of soft targets')

## Result
parser.add_argument(
    '--print_freq', 
    type=int,
    default=50,
    help='The frequency to print loss.')

parser.add_argument(
    '--save_freq', 
    type=int,
    default=50,
    help='The frequancy to save model during training.')

parser.add_argument(
    '--eval_freq', 
    type=int,
    default=1,
    help='The frequancy to evaluate model during training.')

parser.add_argument(
    '--test_only', 
    action='store_true',
    help='test only')

parser.add_argument(
    '--adjust_ckpt',
    action='store_true',
    help='adjust ckpt')

args = parser.parse_args()
if args.resume is not None:
    if not os.path.isfile(args.resume):
        raise ValueError('No checkpoint found at {}'.format(args.resume))


