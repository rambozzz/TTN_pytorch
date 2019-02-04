# import argparse
import torch
import torch.nn as nn
import numpy as np
import os
# import pickle
import io
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.utils.data as data
import simplejson as json
import shutil
import random
# import time
import torch.nn.functional as F
import gensim
import nltk
# from pycocotools.coco import COCO
from collections import defaultdict
# import shutil
import copy
import matplotlib.pyplot as plt
from collections import OrderedDict
import matplotlib.pyplot as plt

save_step = 1500
batch_size = 1
num_epoch = 60
log_step = 20
learning_rate = 1e-4
img_dir = '/home/jacopob/Desktop/TextTopicNet-master/data/ImageCLEF_Wikipedia/'
label_dir = '/home/jacopob/Desktop/TextTopicNet-master/LDA/training_labels40.json'
model_path = '/home/jacopob/Desktop/TextTopicNet-master/CNN/Pytorch/'
trained_model_filepath = model_path + 'alexnet-' + str(num_epoch) + '-' + str(save_step) + '.pkl'


class WikiDataset(data.Dataset):
    def __init__(self, root, json, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: wiki labels and file path.
            transform: image transformer.
        """
        self.root = root
        self.wiki = json
        self.path = []
        self.target = []
        for key, value in self.wiki.items():
            self.path.append(self.root + key)
            self.target.append(value)
        self.target = np.array(self.target)
        self.ids = range(len(self.wiki.keys()))
        self.transform = transform
        #self.mean = np.array([104.00698793, 122.67891434, 116.66876762])
        # self.count = 0

    def __getitem__(self, index):
        """Returns filename andone data pair (image and caption)."""
        filename = os.path.join(self.path[index])
        image = Image.open(filename).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

            image = np.array(image, dtype=np.float32)
            # image = image[:,:,::-1]# switch channels RGB -> BGR
            #image = image[:, :, ::-1] - np.zeros_like(image)  # --> modify "/home/jacopob/anaconda2/envs/ttn_jacopo/lib/python2.7/site-packages/torchvision/transforms.py"
            #image -= self.mean  # subtract mean
            #image = image.transpose((2,0,1)) # transpose to channel x height x width order
            image = transforms.ToTensor().__call__(image) #!!!This already transposes to channel x height x width!!!!
            image = image/255
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]).__call__(image)


            # print ("image: " + str(self.count))
            # self.count += 1
        target = torch.Tensor(self.target[index])
        return filename, image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    filenames, images, labels = zip(*data)

    # Merge image (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    labels = torch.stack(labels, 0)

    return filenames, images, labels


def get_loader(root, json, transform, batch_size, shuffle, num_workers):
    wiki = WikiDataset(root=root,
                       json=json,
                       transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=wiki,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


# The net architecture
'''
class AlexNet(nn.Module):
    def __init__(self, num_classes=40):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(5),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(5),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            #nn.Sigmoid()
        )

    def init_weights(self, m):
        # for m in self.modules():

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            
            
            import scipy.stats as stats
            stddev = m.stddev if hasattr(m, 'stddev') else 0.2
            X = stats.truncnorm(-2, 2, scale=stddev)
            values = torch.Tensor(X.rvs(m.weight.numel()))
            values = values.view(m.weight.size())
            m.weight.data.copy_(values)
            
            
            print ("Initializing weights for: " + str(m))
            nn.init.xavier_normal_(m.weight.data, 0.2)

        
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
'''

class AlexNet(nn.Module):
    def __init__(self, num_classes=40):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.model = models.alexnet(pretrained=False, num_classes=self.num_classes)

    def forward(self, images):
        features = self.model(images)
        return features

    def init_weights(self, m):
        # for m in self.modules():

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            '''
            import scipy.stats as stats
            stddev = m.stddev if hasattr(m, 'stddev') else 0.2
            X = stats.truncnorm(-2, 2, scale=stddev)
            values = torch.Tensor(X.rvs(m.weight.numel()))
            values = values.view(m.weight.size())
            m.weight.data.copy_(values)
            '''
            print("Initializing weights for: " + str(m))
            nn.init.xavier_normal_(m.weight.data, 0.2)

        '''
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        '''

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


# Create a subset of the "wiki" dictionary extracted from the LDA trained labels. This subset is based on the 10k file of Caffe's output. es: path = '/home/jacopob/Downloads/topic_probs.json'
def crate_wiki_subset(path, wiki):
    topic_probs = json.load(open(path))
    filenames = topic_probs['filenames']
    paths = wiki.keys()
    new_wikiList = []
    for filename in filenames:
        filename = '/' + filename
        for path in paths:
            if filename in path:
                new_wikiList.append((path, wiki[path]))

                '''
                images,dir,img = path.split('/')
                dest_path = '/home/jacopob/Desktop/TTN_Pytorch/data/10k_wikipedia_subset/' + images + '/' + dir
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                shutil.copy(img_dir+path, dest_path)
                '''

    new_wiki = OrderedDict(new_wikiList)
    print (len(new_wiki))
    '''
    with open('/home/jacopob/Desktop/TTN_Pytorch/10k_PYTORCH_samples/10k_subdataset.json', 'w') as fp:
        json.dump(new_wiki,fp)
    '''
    return new_wiki


# Function to use when we have to train the model
def train_model(wiki):
    transform = transforms.Compose([
        transforms.RandomCrop(227),
        transforms.RandomHorizontalFlip()
        # transforms.ToTensor()
    ])
    print('Getting Data')
    data_loader = get_loader(img_dir, wiki, transform, batch_size, shuffle=True, num_workers=4)
    print('Building Alexnet')
    model = AlexNet()
    model.model.features.apply(model.init_weights)
    model.model.classifier.apply(model.init_weights)
    #model.features.apply(model.init_weights)
    #model.classifier.apply(model.init_weights)
    if torch.cuda.is_available():
        model.cuda()
    print('Loss, optimizer, etc.')
    # criterion = nn.MultiLabelSoftMarginLoss()
    criterion = nn.BCEWithLogitsLoss()# also try with parameter: reduction='sum'
    #criterion = nn.BCELoss()
    params = list(model.parameters())
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
    total_step = len(data_loader)
    print('Starting training')

    losses = []
    LR_slow_decay = False
    count = 0
    loss_decay = 0.9
    model.train()

    for epoch in range(num_epoch): #USE xrange IF USING PYTHON 2.7
        for i, (filename, images, labels) in enumerate(data_loader):

            images = to_var(images)
            target = to_var(labels)

            output = model(images)
            #             output = F.sigmoid(output)
            #             loss = torch.sum(torch.mul( -F.logsigmoid(output), target))
            loss = criterion(output, target)
            model.zero_grad()
            loss.backward()
            optimizer.step()


            if i % log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, LR: %f'
                      % (epoch, num_epoch, i, total_step, loss.item(), learning_rate))

            losses.append(loss.item())

            if (i + 1) % save_step == 0:
                torch.save(model.state_dict(),
                           os.path.join(model_path, 'alexnet-%d-%d.pkl' % (epoch + 1, i + 1)))
            '''
            with open('/home/jacopob/Desktop/TTN_Pytorch/outputfile.txt', 'a') as file:
                output = torch.sigmoid(output).cpu().detach().numpy()
                np.savetxt(file, output, delimiter=';  ')
            with open('/home/jacopob/Desktop/TTN_Pytorch/targetfile.txt', 'a') as file:
                target = target.cpu().detach().numpy()
                np.savetxt(file, target, delimiter=';  ')
            '''

        #LR slow decay, (to be tested)
        '''
        if epoch+1 == 10:
            print ('Decreasing learning_rate of a factor 10')
            learning_rate = learning_rate*1e-1
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
            LR_slow_decay = True
        if LR_slow_decay == True:
            count += 1
            if (count%5) == 0:
                learning_rate = learning_rate*0.9
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate
        '''
    plt.plot(losses)
    plt.show()

    print('Training ended')


# Once the model is trained, extracts the output and save it to "path_to_vectors" file. es: path_to_vectors = '/home/jacopob/Downloads/tsne_python/outputVector.npy'
def extract_model_output(path_to_vectors, wiki):
    transform = transforms.Compose([
        transforms.RandomCrop(227),
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor()
    ])
    print('Getting Data')
    data_loader = get_loader(img_dir, wiki, transform, batch_size, shuffle=False, num_workers=4)
    print('Building Alexnet')
    model = AlexNet()
    model.load_state_dict(torch.load(trained_model_filepath, map_location="cuda:0"))
    if torch.cuda.is_available():
        model.cuda()
    params = list(model.parameters())
    total_step = len(data_loader)
    print('Feeding the net')
    if os.path.exists(path_to_vectors):
        os.remove(path_to_vectors)
    model.eval()
    with torch.no_grad():
        with open(path_to_vectors, 'a') as file:
            for i, (images, labels) in enumerate(data_loader):
                images = to_var(images)
                target = to_var(labels)
                output = torch.sigmoid(model(images))
                output = output.cpu().numpy()
                np.savetxt(file, output, delimiter=';  ')
    print('Output vector file created, terminated!')


##############################################################################
def main():
    train = False
    extract_output = False
    wiki_subset = True

    # Dictionary extracted from the LDA trained model, json format
    wiki = json.load(open(label_dir))

    if wiki_subset:
        # Filepath of 10k images together with their 40d vectorial representations, output of CAFFE MODEL (original TextTopicNet work)
        caffe_output = '/home/jacopob/Desktop/TTN_Pytorch/10k_CAFFE_samples/topic_probs.json'
        wiki = crate_wiki_subset(caffe_output, wiki)

    if train:
        train_model(wiki)

    if extract_output:
        # Destination path_to_file for the model's output, .npy format
        path_to_vectors = '/home/jacopob/Desktop/TTN_Pytorch/10k_PYTORCH_samples/outputVector.npy'
        extract_model_output(path_to_vectors, wiki)


##############################################################################


'''
main function run
'''

main()
