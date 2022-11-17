import torch
import argparse
from transformers import BertTokenizer
from PIL import Image

from models import caption
from datasets import coco, utils
from configuration import Config
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def parse_config():
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', type=str, help='path to the folder containing test images', default='captioning/images')
    parser.add_argument('--save_dir', type=str, help='path to the folder for visualization outputs', default='hw3/p2_output')
    parser.add_argument('--v', type=str, help='version', default='v3')
    parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)
    args = parser.parse_args()

    return args

def visualization(image_path, image_name, save_dir, sentence, att_mat, cap_mask):
    attn_shape_info = {'bike.jpg':19, 'girl.jpg':13, 'sheep.jpg':19, 'ski.jpg':14, 'umbrella.jpg':13}
    att_mat = torch.stack(att_mat).squeeze(1).cpu().detach()
    image1 = Image.open(image_path).convert('RGB')
    fig = plt.figure()
    axes = []
    
    if image_name == 'girl.jpg' or image_name == 'sheep.jpg':
        axes.append(fig.add_subplot(3,5,1))
    else:
        axes.append(fig.add_subplot(3,4,1))
    axes[-1].set_title('<start>')
    plt.imshow(image1)
    plt.axis('off')
    
    for i in range(len(sentence)+1):
        mask = att_mat[-1,i,:].reshape(attn_shape_info[image_name],19).detach().numpy()
        image1 = Image.open(image_path).convert('L')

        result = cv2.resize(mask / mask.max(), image1.size).astype("double")
        image1 = np.array(image1).astype("double")
        image1 = image1 / 255.0
        result = cv2.addWeighted(image1, 0.2, result, 0.8, 0)

        if image_name == 'girl.jpg' or image_name == 'sheep.jpg':
            axes.append(fig.add_subplot(3,5,i+2))
        else:
            axes.append(fig.add_subplot(3,4,i+2))
        # axes.append(fig.add_subplot(3,4,i+2))
        if i<len(sentence)-1:
            axes[-1].set_title(sentence[i])
        elif i == len(sentence)-1:
            axes[-1].set_title(sentence[i][:len(sentence[i])-1])
        else:
            axes[-1].set_title('<end>')
        plt.imshow(result, cmap=plt.get_cmap('jet'))
        plt.axis('off')
        
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, image_name.split('.')[0]+'.png'))

    # # from mpl_toolkits.axes_grid1 import make_axes_locatable
    # fig, ax= plt.subplots(figsize=(16, 16))
    # ax.set_title('Attention Map Last Layer')
    # im = ax.imshow(result, cmap=plt.get_cmap('jet'))
    # # divider = make_axes_locatable(ax)
    # # cax = divider.append_axes("right", size="5%", pad=0.05)
    # # plt.colorbar(im, cax=cax)
    # plt.show()

def main(args):
    # image_path = args.path
    version = args.v
    checkpoint_path = args.checkpoint

    config = Config()

    if version == 'v1':
        model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
    elif version == 'v2':
        model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
    elif version == 'v3':
        model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
    else:
        print("Checking for checkpoint.")
        if checkpoint_path is None:
            raise NotImplementedError('No model to chose from!')
        else:
            if not os.path.exists(checkpoint_path):
                raise NotImplementedError('Give valid checkpoint path')
        print("Found checkpoint! Loading!")
        model,_ = caption.build_model(config)
        print("Loading Checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for image_name in os.listdir(args.folder):
        start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
        end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

        image_path = os.path.join(args.folder, image_name)
        image = Image.open(image_path)
        image = coco.val_transform(image)
        image = image.unsqueeze(0)

        def create_caption_and_mask(start_token, max_length):
            caption_template = torch.zeros((1, max_length), dtype=torch.long)
            mask_template = torch.ones((1, max_length), dtype=torch.bool)

            caption_template[:, 0] = start_token
            mask_template[:, 0] = False

            return caption_template, mask_template


        caption, cap_mask = create_caption_and_mask(
            start_token, config.max_position_embeddings)


        @torch.no_grad()
        def evaluate():
            model.eval()
            for i in range(config.max_position_embeddings - 1):
                predictions, attens = model(image, caption, cap_mask)
                predictions = predictions[:, i, :]
                predicted_id = torch.argmax(predictions, axis=-1)

                if predicted_id[0] == 102:
                    visualization(image_path, image_name, args.save_dir, tokenizer.decode(caption[0].tolist(), skip_special_tokens=True).split(" "), attens, cap_mask)
                    return caption

                caption[:, i+1] = predicted_id[0]
                cap_mask[:, i+1] = False
                result = tokenizer.decode(caption[0].tolist(), skip_special_tokens=True)
                # print(i+1, result.capitalize())

            return caption

        output = evaluate()
        result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        #result = tokenizer.decode(output[0], skip_special_tokens=True)
        print(result.capitalize())

if __name__=='__main__':
    args = parse_config()
    main(args)