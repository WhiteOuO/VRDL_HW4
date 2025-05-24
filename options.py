import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--epochs', type=int, default=50, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int, default=1, help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')
parser.add_argument('--de_type', nargs='+', default=['derain', 'desnow'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--train_degraded_dir', type=str, default='data/train/degraded/',
                    help='where degraded training images (rain and snow) are saved.')
parser.add_argument('--train_clean_dir', type=str, default='data/train/clean/',
                    help='where clean training images (rain clean and snow clean) are saved.')
parser.add_argument('--test_dir', type=str, default='data/test/degraded/',
                    help='where test degraded images are saved.')
parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers.')
parser.add_argument('--data_file_dir', type=str, default='data/', help='where data files are saved.')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/", help='checkpoint save path')
parser.add_argument("--ckpt_dir", type=str, default="ckpt", help="Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for training")
# for test
parser.add_argument('--mode', type=int, default=1, help='1 for derain and desnow')
parser.add_argument('--ckpt_name', type=str, default=None, help='checkpoint file name (e.g., epoch=11-step=19200.ckpt)')

options = parser.parse_args()