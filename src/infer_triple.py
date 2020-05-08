

"""

CC BY-NC-ND 4.0 license
"""
import sys
from tools import *
from trainers import *
from datasets import *
import torchvision
import itertools
from common import *
import tensorboard
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
# from tensorboard import summary
from optparse import OptionParser

from loaders import dataset_loader
import constants
# for model parallel


parser = OptionParser()
parser.add_option('--gpu', type=int, help="gpu id", default=0)
parser.add_option('--resume', type=int, help="resume training?", default=0)
parser.add_option('--config', type=str, help="net configuration")
parser.add_option('--log', type=str, help="log path")

batch_size = 16
LOCATION = "D:/Users/delgallegon/Documents/GithubProjects/BridgeGAN/figures/"

def main(argv):
    (opts, args) = parser.parse_args(argv)

    # Load experiment setting
    assert isinstance(opts, object)
    print(opts.config)
    config = NetConfig(opts.config)

    max_iterations = config.hyperparameters['max_iterations']

    dataloader = dataset_loader.load_vemon_dataset(batch_size, -1)
    
    # Plot some training images
    name_batch, normal_batch = next(iter(dataloader))
    plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
    plt.axis("off")
    plt.title("VEMON Images")
    plt.imshow(np.transpose(vutils.make_grid(normal_batch.cuda()[:batch_size], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

    trainer = None
    #exec ("trainer=%s(config.hyperparameters)" % config.hyperparameters['trainer'])
    trainer = COCOGANTrainer_triple_res(config.hyperparameters)
    # Check if resume training
    iterations = 0
    if opts.resume == 1:
        iterations = trainer.resume(config.snapshot_prefix)
    trainer.cuda(opts.gpu)
    if config.hyperparameters['para'] == 1:
        trainer.parallel()
    # trainer = nn.DataParallel(trainer, device_ids=range(4), output_device=3)


    ######################################################################################################################
    # Setup logger and repare image outputs
    # train_writer = tensorboard.FileWriter("%s/%s" % (opts.log,os.path.splitext(os.path.basename(opts.config))[0]))
    image_directory, snapshot_directory = prepare_snapshot_and_image_folder(config.snapshot_prefix, iterations, config.image_save_iterations)
    file_number = 0
    for it, (name, normal_img) in enumerate(dataloader, 0):
      # if images_a.size(0) != batch_size or images_b.size(0) != batch_size or images_c.size(0) != batch_size:
      #   continue
        images_a = Variable(normal_img.cuda())
        
        file_number = file_number + 1
        image_output, shared = trainer.gen.forward_a2b(images_a)
        assembled_images = image_output.data
        fig, ax = plt.subplots(2, 1)
        fig.set_size_inches(15, 11)
        fig.tight_layout()
        
        ims = np.transpose(vutils.make_grid(images_a, nrow = 8, padding=2, normalize=True).cpu(),(1,2,0))
        ax[0].set_axis_off()
        ax[0].imshow(ims)
        
        ims = np.transpose(vutils.make_grid(assembled_images, nrow = 8, padding=2, normalize=True).cpu(),(1,2,0))
        ax[1].set_axis_off()
        ax[1].imshow(ims)
        
        plt.subplots_adjust(left = 0.06, wspace=0.0, hspace=0.03) 
        print("Save location: ", (LOCATION + "infer_" + str(file_number) + ".png"))
        plt.savefig(LOCATION + "result_" + str(file_number) + ".png")
        plt.show()
        

if __name__ == '__main__':
    main(sys.argv)

