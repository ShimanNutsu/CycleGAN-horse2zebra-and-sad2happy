import torch
from models import *

def init_models(lr, n_res):
    generator = {
        'A': Generator(n_res).cuda(),
        'B': Generator(n_res).cuda()
    }

    discriminator = {
        'A': PatchGAN().cuda(),
        'B': PatchGAN().cuda()
    }
    opt = {
        'discr': {
            'A': torch.optim.Adam(discriminator['A'].parameters(), lr=lr, betas=(0.5, 0.999)),
            'B': torch.optim.Adam(discriminator['B'].parameters(), lr=lr, betas=(0.5, 0.999))
        },
        'gen': {
            'A': torch.optim.Adam(generator['A'].parameters(), lr=lr, betas=(0.5, 0.999)),
            'B': torch.optim.Adam(generator['B'].parameters(), lr=lr, betas=(0.5, 0.999))
        }
    }
    return generator, discriminator, opt

def change_lr(opt, lr):
    for k in opt:
        for k2 in opt[k]:
            for g in opt[k][k2].param_groups:
                g['lr'] = lr

def save_models(generator, discriminator, opt, losses, file):
    torch.save({
        'generator': generator,
        'discriminator': discriminator,
        'optimizers': opt,
        'losses': losses
    }, file)

def load_models(file):
    all = torch.load(file)
    generator = all['generator']
    discriminator = all['discriminator']
    losses = all['losses']
    opt = all['optimizers']
    for k in generator:
        generator[k] = generator[k].cuda()
    for k in discriminator:
        discriminator[k] = discriminator[k].cuda()
    return generator, discriminator, opt, losses