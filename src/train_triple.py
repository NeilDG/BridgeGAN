

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

MAX_EPOCHS = 100000
PERFORM_INFER = True
batch_size = 32
LOCATION = "D:/Users/delgallegon/Documents/GithubProjects/BridgeGAN/figures/"

def main(argv):
    (opts, args) = parser.parse_args(argv)

    # Load experiment setting
    assert isinstance(opts, object)
    print(opts.config)
    config = NetConfig(opts.config)

    #batch_size = config.hyperparameters['batch_size']
    max_iterations = config.hyperparameters['max_iterations']

    # train_loader_a = get_data_loader(config.datasets['train_a'], batch_size) # for homo
    # train_loader_b = get_data_loader(config.datasets['train_b'], batch_size) # for gt
    # train_loader_c = get_data_loader(config.datasets['train_c'], batch_size) # for new
    dataloader = dataset_loader.load_dataset(batch_size, -1)
    # Plot some training images
    name_batch, normal_batch, homog_batch, topdown_batch = next(iter(dataloader))
    plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
    plt.axis("off")
    plt.title("Training - Normal Images")
    plt.imshow(np.transpose(vutils.make_grid(normal_batch.cuda()[:batch_size], nrow = 16, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

    plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
    plt.axis("off")
    plt.title("Training - Homog Images")
    plt.imshow(np.transpose(vutils.make_grid(homog_batch.cuda()[:batch_size], nrow = 16, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

    plt.figure(figsize=(constants.FIG_SIZE,constants.FIG_SIZE))
    plt.axis("off")
    plt.title("Training - Normal Images")
    plt.imshow(np.transpose(vutils.make_grid(topdown_batch.cuda()[:batch_size], nrow = 16, padding=2, normalize=True).cpu(),(1,2,0)))
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
    for ep in range(0, MAX_EPOCHS):
        for it, (name, normal_img, homog_img, topdown_img) in enumerate(dataloader, 0):
          # if images_a.size(0) != batch_size or images_b.size(0) != batch_size or images_c.size(0) != batch_size:
          #   continue
            images_a = Variable(homog_img.cuda())
            images_b = Variable(topdown_img.cuda())
            images_c = Variable(normal_img.cuda())

          # Main training code
            if(PERFORM_INFER is False):
                trainer.dis_update(images_a, images_b, images_c,  config.hyperparameters)
                image_outputs = trainer.gen_update(images_a, images_b, images_c, config.hyperparameters)
                assembled_images = trainer.assemble_outputs(images_a, images_b, image_outputs).data

                if (iterations+1) % config.display == 0:
                    write_loss(iterations, max_iterations, trainer)
    
                if (iterations+1) % config.image_save_iterations == 0:
                    img_filename = '%s/gen_%08d.jpg' % (image_directory, iterations + 1)
                    torchvision.utils.save_image(assembled_images.data / 2 + 0.5, img_filename, nrow=1)
                    write_html(snapshot_directory + "/index.html", iterations + 1, config.image_save_iterations, image_directory)
                elif (iterations + 1) % config.image_display_iterations == 0:
                    img_filename = '%s/gen.jpg' % (image_directory)
                    torchvision.utils.save_image(assembled_images.data / 2 + 0.5, img_filename, nrow=1)
                    
                # Save network weights
                if (iterations+1) % config.snapshot_save_iterations == 0:
                    trainer.save(config.snapshot_prefix, iterations)
                    
                iterations += 1
                if iterations >= max_iterations:
                    return
            else:
                file_number = file_number + 1
                image_output, shared = trainer.gen.forward_a2b(images_a)
                assembled_images = image_output.data
                fig, ax = plt.subplots(3, 1)
                fig.set_size_inches(15, 7)
                fig.tight_layout()
                
                ims = np.transpose(vutils.make_grid(images_c, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
                ax[0].set_axis_off()
                ax[0].imshow(ims)
                
                ims = np.transpose(vutils.make_grid(assembled_images, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
                ax[1].set_axis_off()
                ax[1].imshow(ims)
                
                ims = np.transpose(vutils.make_grid(images_b, nrow = 16, padding=2, normalize=True).cpu(),(1,2,0))
                ax[2].set_axis_off()
                ax[2].imshow(ims)
                
                plt.subplots_adjust(left = 0.06, wspace=0.0, hspace=0.03) 
                print("Save locatioN: ", (LOCATION + "result_" + str(file_number) + ".png"))
                plt.savefig(LOCATION + "result_" + str(file_number) + ".png")
                plt.show()
            
        if(PERFORM_INFER is True):
            break


if __name__ == '__main__':
    main(sys.argv)

