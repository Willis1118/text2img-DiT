from PIL import Image
import os

import torch
from torch import nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms

import nltk
import spacy
from pycocotools.coco import COCO
from transformers import DistilBertTokenizerFast, DistilBertModel
from tqdm import tqdm
from einops import rearrange


### difference between nltk / BERT

class CocoDataset(data.Dataset):
    '''
        Customized Coco Dataset compatible with DataLoader
    '''
    def __init__(self, root, json, vocab, transform=None):
        '''
            Set the path for images, captions and vocabulary wrapper

            input:
            root --> image dir
            json --> coco annotation file path
            vocab --> vocabulary wrapper
            transform --> image transformer
        '''

        self.root = root
        self.coco = COCO(json) # --> turn json file into COCO format
        self.ids = list(self.coco.anns.keys())
        self.vocab  = vocab
        self.transform = transform
    
    def __getitem__(self, index):
        '''
            returns {image, caption}
        '''
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name'] # --> the image with the corresponding index into coco datasets

        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        
        # Convert caption(string) into word ids
        # tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        tokens = vocab(caption) # --> will append start and end to string (101 & 102)
        target = torch.Tensor(tokens['input_ids'])

        return image, target # --> image, caption pair
    
    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    '''
        Create mini-batch with the (image, caption) pair
        Default collate_fn does not support merging caption(include padding), so we need to define a customized

        Inputs:
            data: list of tuple (image, caption)
                  - image: [3, image_size, image_size] (after transformation)
                  - caption: torch tensor of shape [?]; variable length depending on specific annotations
        
        Outputs:
            images: torch tensor of shape [batch_size, 3, 256, 256]
            targets: torch tensor of shape [batch_size, padded_length]
            lengths: list; valid length for each padding caption
    '''
    data.sort(key=lambda x: len(x[1]), reverse=True) # --> sort data list based on length of caption
    images, captions = zip(*data)

    # merge images to mini_batch
    images = torch.stack(images, dim=0)

    # merge captions with 0 padding (to be tested)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long() # --> create tensor of size [batch_size, max(lengths)]
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    return images, targets

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers=0):
    '''
        Returns a Dataloader object based on our customized dataset class and collate_fn
    '''
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    
    dataloader = DataLoader(dataset=coco,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
    return dataloader

class TextEmbedder(nn.Module):
    def __init__(self, hidden_size, text_channel_size, encoder_config):
        super().__init__()
        
        self.text_emb_size = text_channel_size * 768
        self.encoder = DistilBertModel.from_pretrained(encoder_config)

        if encoder_config == "distilbert-base-uncased":
            self.end_token = 102

        self.mlp = nn.Sequential(
            nn.Linear(self.text_emb_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )
    
    @staticmethod
    def encoding(caption, encoder, end_token):

        mask = torch.cumsum((cap == end_token), 1)
        mask[cap == end_token] = 0
        mask = (~mask.bool()).long()

        with torch.no_grad():
            emb = encoder(caption, attention_mask=mask)['last_hidden_state']
        
        emb = rearrange(emb, 'b c h -> b (c h)')
        return emb

    def forward(self, x):
        x_emb = self.encoding(x, self.encoder, self.end_token)
        x_mlp = self.mlp(x_emb)

        return x_mlp

if __name__ == '__main__':

    root = '/scratch/nm3607/datasets/coco/train2017'
    annFile = '/scratch/nm3607/datasets/coco/annotations/captions_train2017.json'
    transform=transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
        ])
    vocab = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    dataloader = get_loader(root=root, json=annFile, vocab=vocab, transform=transform, batch_size=64, shuffle=True)

    img, cap = next(iter(dataloader))
    print(img.shape, cap.shape)
    assert img.shape == (64, 3, 128, 128)

    ########## Add to Text embedder (maybe we can use Bert here) ##########
    print(cap.shape)
    textembedder = TextEmbedder(256, cap.shape[1], encoder_config="distilbert-base-uncased")
    print(textembedder(cap).shape)
    