from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import argparse
import os
import ruamel.yaml as yaml
from pathlib import Path
from math import sqrt, pi
import matplotlib.pyplot as plt
import mmcv
from models.blips import blip_decoder, blip_feature_extractor, is_url

import cv2 as cv
import numpy as np
from matplotlib import gridspec

def load_demo_image(img_url, image_size, device):
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' if img_url is None else img_url
    if is_url(img_url):
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')  
    elif os.path.isfile(img_url):  
        raw_image = Image.open(img_url).convert('RGB') 
    else:
        raise RuntimeError('checkpoint url or path is invalid') 

    w,h = raw_image.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image

def obtain_features(args, config, img_url = '', device = 'cuda', image = None):
    
    # model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'      
    # model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    
    
    #### Model #### 
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                         vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                         prompt=config['prompt'], use_vit_layers = config['use_vit_layers'],
                         use_swin=config['use_swin'],
                         use_contrastive=config['use_contrastive'], )  
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
    text = caption[0]
    print('caption: '+ text)
            
    if args.blips == 'blip_feature_extractor':
        print("Creating model")
        model = blip_feature_extractor(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                             vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                             use_vit_layers = config['use_vit_layers'],
                             use_swin=config['use_swin'],
                             use_contrastive=config['use_contrastive'], )  
                             
        model.eval()
        model = model.to(device)
        mode = 'image'
        with torch.no_grad():
            image_embeds = model.forward(image, caption=text, mode=mode)
            print(image_embeds[0].shape)
            print(image_embeds[1].shape)
            
        mode = 'multimodal'
        with torch.no_grad():
            hidden_states = model.forward(image, caption=text, mode=mode)
            cls_token = hidden_states[-1][:,0,:]
            print(len(hidden_states))
            print('hidden_states {}'.format(hidden_states[-1].shape))
            
        return image_embeds[0], image_embeds[1], cls_token,
        
def plt_heatmap(image_embeds, cls):
    B, L, C = image_embeds.shape
    H = W = int(sqrt(L))
    
    # feature_map = image_embeds.view(B, 4, 4, C).permute(0, 3, 1, 2).contiguous()
    sim = image_embeds @ cls.t()
    print('sim shape {}'.format(sim.shape))
    # sim = sim / sim.mean()
    # sim = torch.nn.Softmax(dim=1)(sim)
    sim = sim.view(H, W).cpu().numpy()
    print('heatmap {}'.format(sim.shape))
    
    plt.imshow(sim, cmap='YlOrRd', aspect='auto')
    plt.colorbar()
    plt.title('Heatmap Example')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.savefig('tools/heatmap/vis_ours.png')
    
    return sim
    
def draw_feature_map(heatmap, img, save_dir = 'tools/heatmap/', name = 'heatmap_ours_', image_id = 0): 
    import cv2
    import numpy as np
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    print('img shape {}'.format(img.shape))
    heatmap = np.uint8(255 * heatmap) 
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
    superimposed_img = heatmap * 0.5 + img*0.8
    
    cv2.imwrite(os.path.join(save_dir,name +str(image_id)+'.png'), superimposed_img)
    
    
    
    
    
    
def fourier_transform(img, i, vis_type = 'abs', return_values = True):
    
    shape = (4,4) if i == 0 else (24,24)
    img = img.mean(2).view(shape).cpu().numpy() # cv.imread('messi5.jpg',0)
    
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    # freq = np.fft.fftfreq(shape[0]*shape[1],1/2*pi) # x freq (len(signal, 1/f))
    # freq_shift = np.fft.fftshift(freq)
    magnitude_spectrum = np.log(np.abs(fshift))
    print('magnitude_spectrum {} min {} max {}'.format(magnitude_spectrum.shape, np.min(magnitude_spectrum), np.max(magnitude_spectrum)))
    
    # https://stackoverflow.com/questions/45425355/two-dimensional-fft-using-python-results-in-slightly-shifted-frequency
    # x = np.linspace(0, 2.0 * pi, shape[0])
    # dx = x[1] - x[0] # spacing in x (and also y) direction (real space)
    # freq_x = np.fft.fftfreq(f.shape[0], d = dx) # return the DFT sample frequencies 
    # freq_y = np.fft.fftfreq(f.shape[1], d = dx)
    # freq_x = np.fft.fftshift(freq_x) # order sample frequencies, such that 0-th frequency is at center of spectrum 
    # freq_y = np.fft.fftshift(freq_y)
    
    # print('freq_x {}'.format(freq_x))
    # print('freq_y {}'.format(freq_y))
    
    # https://www.cnblogs.com/wojianxin/p/12531004.html
    
    (rows, cols) = shape
    magni_mean = []
    crow,ccol = int(rows/2), int(cols/2)
    
    w = 1 
    if i == 1 and vis_type == 'rel':
        w = 5 # change wideband bigger 
    for radius in range(0,crow+1+w,w):
        mask = np.ones((rows, cols, 2), np.uint8)
        for i in range(0, rows):
            for j in range(0, cols):
                d = sqrt(pow(i - crow, 2) + pow(j - ccol, 2))
                if radius - w / 2 < d < radius + w / 2:
                    mask[i, j, 0] = 0
                else:
                    mask[i, j, 0] = 1
        remain = 1 - mask[:,:,0]
        # print('remain {}'.format(remain))
        # print('remain.sum {}'.format(remain.sum()))
        # print('remain_val {}'.format(magnitude_spectrum * remain))
        magni_mean.append(np.sum(magnitude_spectrum * remain)/remain.sum())
    magni_mean = magni_mean - magni_mean[0] if vis_type == 'rel' else magni_mean
    print('magni_mean {}'.format(magni_mean))
    if return_values:
        return magni_mean, magnitude_spectrum
    # plt.subplot(121), plt.plot(np.linspace(0, pi, len(magni_mean)), magni_mean) # imshow(img.reshape(shape), cmap = 'gray')
    # plt.title('Log Amplitudes'), plt.xticks(np.linspace(0, pi, 5)), plt.yticks()
    # plt.subplot(222), plt.imshow(magnitude_spectrum, cmap = 'gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.savefig('tools/heatmap/magnitude_{}_{}_{}.png'.format(vis_type, shape[0],shape[1]))

def obtain_images(max_num = 100):
    import os
    path = "../dataset/coco/image/train2014/"
    files= os.listdir(path)
    s = []
    i = 0
    for file_ in files:
        if not os.path.isdir(file_):
            i += 1 
            if i % 88 == 0:
                s.append(os.path.join(path, file_)) 
            if len(s) == max_num:
                break
    return s
    
def obtain_features_images(args, config, model_blip_decoder, model_blip_feature_extractor, img_url = '', device = 'cuda', image = None,):
    #### Model #### 
    print("Creating model")
    model = model_blip_decoder
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
    text = caption[0]
    print('caption: '+ text)
            
    if args.blips == 'blip_feature_extractor':
        model = model_blip_feature_extractor
                             
        model.eval()
        model = model.to(device)
        mode = 'image'
        with torch.no_grad():
            image_embeds = model.forward(image, caption=text, mode=mode)
            print(image_embeds[0].shape)
            print(image_embeds[1].shape)
            
        mode = 'flops' #
        with torch.no_grad():
            hidden_states = model.forward(image, caption=text, mode=mode)
            cls_token = hidden_states[-1][:,0,:]
            
        return image_embeds[0], image_embeds[1], cls_token,
          
def main_images():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_coco_load_pretrain_scale.yaml')
    parser.add_argument('--output_dir', default='output/tools/flops/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--blips', default='blip_feature_extractor')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    model_blip_decoder = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                         vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                         prompt=config['prompt'], use_vit_layers = config['use_vit_layers'],
                         use_swin=config['use_swin'],
                         use_contrastive=config['use_contrastive'], )
    model_blip_feature_extractor = blip_feature_extractor(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                             vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                             use_vit_layers = config['use_vit_layers'],
                             use_swin=config['use_swin'],
                             use_contrastive=config['use_contrastive'], ) 
                         
    
    
    magni_means_0 = []
    magni_means_1 = []
    magnitude_spectrums_0 = []
    magnitude_spectrums_1 = []
    
    img_urls = obtain_images()  # image files
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for img_url in img_urls:
        image = load_demo_image(img_url, image_size=config['image_size'], device=device)
        image_embeds, image_embeds1, _ = obtain_features_images(args, config, model_blip_decoder, model_blip_feature_extractor, 
                                                                  img_url = img_url, device = device, image = image)
        vis_type = 'abs'
        magni_mean0, magnitude_spectrum0 = fourier_transform(image_embeds, i = 0, vis_type = vis_type)
        magni_mean1, magnitude_spectrum1 = fourier_transform(image_embeds1, i = 1, vis_type = vis_type)
        
        magni_means_0.append(np.array(magni_mean0))
        magni_means_1.append(np.array(magni_mean1))
        magnitude_spectrums_0.append(np.array(magnitude_spectrum0))
        magnitude_spectrums_1.append(np.array(magnitude_spectrum1))
        
    magni_mean0 = np.stack(magni_means_0, axis=0)
    magnitude_spectrum0 = np.stack(magnitude_spectrums_0, axis=0)
    magni_mean1 = np.stack(magni_means_1, axis=0)
    magnitude_spectrum1 = np.stack(magnitude_spectrums_1, axis=0)
    
    print(magni_mean0.shape, magnitude_spectrum0.shape, magni_mean1.shape, magnitude_spectrum1.shape)
    
    # mean for 100 images
    magni_mean0 = np.mean(magni_mean0,axis=0).tolist()
    magnitude_spectrum0 = np.mean(magnitude_spectrum0,axis=0)
    magni_mean1 = np.mean(magni_mean1,axis=0).tolist()
    magnitude_spectrum1 = np.mean(magnitude_spectrum1,axis=0)
    
    print(magni_mean0, magni_mean1,)
    
    plt.subplot(121),
    plt.plot(np.linspace(0, pi, len(magni_mean0)), magni_mean0, linestyle='-') 
    plt.plot(np.linspace(0, pi, len(magni_mean1)), magni_mean1, linestyle= '-.') 
    plt.legend(['Middle-scale','Local-scale'], fontsize='large')
    # plt.title('(a) Log amplitudes', y=-0.15, fontsize=12), 
    plt.xticks(np.linspace(0, pi, 5),  ['0.0$\pi$', '0.2$\pi$', '0.5$\pi$', '0.8$\pi$', '1.0$\pi$' ], size=12), plt.yticks(size=12), plt.xlabel('Frequency', fontsize=12), plt.ylabel('Log amplitudes', fontsize=12)
    
    plt.subplot(222), plt.imshow(magnitude_spectrum1, cmap = 'gray')
    plt.title('(b) Local-scale', y=-0.15, fontsize=10), plt.xticks([]), plt.yticks([]) # Magnitude spectrum of 
    plt.subplot(224), plt.imshow(magnitude_spectrum0, cmap = 'gray')
    plt.title('(c) Middle-scale', y=-0.15, fontsize=10), plt.xticks([]), plt.yticks([]) # Magnitude spectrum of
    
    plt.tight_layout()

    plt.savefig('tools/heatmap/magnitude_{}_{}_{}_allimages_withlegend.png'.format(vis_type, 24,24))
    
    
    
 
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_coco_load_pretrain_scale.yaml')
    parser.add_argument('--output_dir', default='output/tools/vis/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--blips', default='blip_feature_extractor')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' # 'http://images.cocodataset.org/train2017/000000491102.jpg'
    # text = 'a woman sitting on the beach with her dog'
    # image_size = 384
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = load_demo_image(img_url, image_size=config['image_size'], device=device)
    raw_img = mmcv.imread(img_url)
    
    image_embeds, image_embeds1, cls = obtain_features(args, config, img_url = img_url, device = device, image = image)
    
    # offical vis: https://colab.research.google.com/github/xxxnell/how-do-vits-work/blob/transformer/fourier_analysis.ipynb#scrollTo=a7350a20
    vis_type = 'rel'
    magni_mean0, magnitude_spectrum0 = fourier_transform(image_embeds, i = 0, vis_type = vis_type)
    magni_mean1, magnitude_spectrum1 = fourier_transform(image_embeds1, i = 1, vis_type = vis_type)
    
    fig = plt.figure()
    spec = gridspec.GridSpec(ncols=2, nrows=2,
                             width_ratios=[3, 1], wspace=0.1,
                             hspace=0.1, height_ratios=[1, 1])
    ax0 = fig.add_subplot(spec[:,0])
    ax0.plot(np.linspace(0, pi, len(magni_mean0)), magni_mean0) 
    ax0.plot(np.linspace(0, pi, len(magni_mean1)), magni_mean1) 
    ax0.set_title('Log Amplitudes'), ax0.set_xticks(np.linspace(0, pi, 5)), ax0.set_yticks() # 
    
    ax1 = fig.add_subplot(spec[0,1])
    ax1.imshow(magnitude_spectrum1, cmap = 'gray')
    ax1.set_xticks([]), ax1.set_yticks([])
    
    ax2 = fig.add_subplot(spec[1,1])
    ax2.imshow(magnitude_spectrum0, cmap = 'gray')
    ax2.set_xticks([]), ax2.set_yticks([])
                             
    # plt.subplot(121),
    # plt.plot(np.linspace(0, pi, len(magni_mean0)), magni_mean0) 
    # plt.plot(np.linspace(0, pi, len(magni_mean1)), magni_mean1) 
    # plt.title('Log Amplitudes'), plt.xticks(np.linspace(0, pi, 5)), plt.yticks()
    
    # plt.subplot(222), plt.imshow(magnitude_spectrum1, cmap = 'gray')
    # plt.title('Magnitude Spectrum of Local Scale'), plt.xticks([]), plt.yticks([])
    # plt.subplot(224), plt.imshow(magnitude_spectrum0, cmap = 'gray')
    # plt.title('Magnitude Spectrum of Middle Scale'), plt.xticks([]), plt.yticks([])
    
    # plt.tight_layout()

    plt.savefig('tools/heatmap/magnitude_{}_{}_{}.png'.format(vis_type, 24,24))
    
    # sim = plt_heatmap(image_embeds, cls)
    # sim1 =  plt_heatmap(image_embeds1, cls)
    # draw_feature_map(sim, img = raw_img, image_id=img_url.split('/')[-1].split('.')[0])      
    # draw_feature_map(sim1, img = raw_img, image_id=img_url.split('/')[-1].split('.')[0], name='heatmap_local_') 
    
if __name__ == '__main__':
    main_images()
    # main()
