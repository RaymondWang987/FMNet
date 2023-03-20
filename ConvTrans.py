import numpy as np
import torch
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt




class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual,self).__init__()
        self.fn = fn
    def forward(self, x):
        #residual = x.copy()
        
        return self.fn(x) + x #+ resudual.view(6,32,192,256)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm,self).__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)#,elementwise_affine = False)
        self.fn = fn 
    def forward(self, x):
        
        
        if self.dim[1] != x.shape[2]:
            newnorm = nn.LayerNorm([x.shape[1],x.shape[2],x.shape[3]],elementwise_affine = False)
            x = newnorm(x)
            return self.fn(x)

        x = self.norm(x)
        return self.fn(x)

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Conv2d(self.dim,2*self.dim,1,1,0,1,1),
            nn.LeakyReLU(),
            nn.Conv2d(2*self.dim,self.dim,1,1,0,1,1))
        
    def forward(self, x):
        
        return self.net(x)
class ConvAttention(nn.Module):
    def __init__(self,num_hidden):
        super(ConvAttention, self).__init__()
        self.num_hidden = num_hidden
        self.nheads = 1
        self.dim = self.num_hidden[-1] // self.nheads
        self.convQ = nn.Sequential(
            nn.Conv2d(in_channels=self.num_hidden[-1], out_channels=self.num_hidden[-1], kernel_size=1)
        )
        
        self.convK = nn.Sequential(
            nn.Conv2d(in_channels=self.num_hidden[-1], out_channels=self.num_hidden[-1], kernel_size=1)
        )
        self.convV = nn.Sequential(
            nn.Conv2d(in_channels=self.num_hidden[-1], out_channels=self.num_hidden[-1], kernel_size=1)
        )
        
        self.conv_atten = nn.Sequential(
            nn.Conv2d(in_channels=self.dim*2, out_channels=1, kernel_size=5,padding=2)  # add or concat
        )
    def forward(self, x):
        l,c,h,w = x.shape
        
        Q = self.convQ(x)#.view(l,self.nheads,self.dim,h,w)
        #Q,K,V= torch.split(QKV,1,dim = 1)#QKV[:,0,:,:]
        K = self.convK(x)
        V = self.convV(x)
        #print(QKV.shape,Q.shape,K.shape,V.shape)   
        #exit(1)
        '''
        for jj in range(16):
            print(jj)
            plt.imsave('/data1/wangyiran/mytrans/firstconv/atten/'+ str(jj)+'.png', Q[jj,0,:,:].squeeze().cpu().numpy(), cmap='inferno')
        exit(1)
        '''
        #print(Q.shape)
        #exit(1)
        V_all_out = []
        
        for i in range(l):
            
            Qi = Q[i,...]  
            for h in range(self.nheads):
                hi_head = []
                for j in range(l):
                    Kj = K[j,...]
                    #Vj = V[j,...]
                    Qi_head = Qi[h*self.dim:(h+1)*self.dim,:,:]
  
                    Kj_head = Kj[h*self.dim:(h+1)*self.dim,:,:]

                    #Vj_head = Vj[h*self.dim:(h+1)*self.dim,:,:]
                    #Hij_head = self.conv_atten((Qi_head + Kj_head).unsqueeze(0)).squeeze(0) 
                    Hij_head = self.conv_atten(torch.cat([Qi_head,Kj_head],dim = 0).unsqueeze(0)).squeeze(0)  
                    hi_head.append(Hij_head)
                hi_head = torch.stack(hi_head, dim = 0)
                hi_head = F.softmax(hi_head, dim=0)
                '''
                if i == 0:
                    for jj in range(16):
                        print(hi_head.shape)
                        plt.imsave('/data1/wangyiran/mytrans/firstconv/atten/'+str(i)+'_'+str(jj)+'.png', hi_head[jj,:,:,:].squeeze().cpu().numpy(), cmap='inferno')
                    exit(1)
                '''
                '''
                if i == 0 and h == 3:
                    for jj in range(16):
                        print('head:',h,i,'to',jj)
                        plt.imsave('/data1/wangyiran/mytrans/firstconv/atten/'+str(i)+'_'+str(jj)+'.png', hi_head[jj,:,:,:].squeeze().cpu().numpy(), cmap='inferno')
                    exit(1)
                '''
                
                
                vj_head = V[:,h*self.dim:(h+1)*self.dim,:,:]
                outi_head = torch.sum(torch.mul(hi_head,vj_head),dim = 0).unsqueeze(0)
                if h == 0:
                    outi = outi_head
                else:
                    outi = torch.cat([outi,outi_head],dim = 1)
            
            if i == 0:
                out_final =  outi
            else:
                out_final = torch.cat([out_final,outi],dim = 0)
        '''
        print(out_final.shape)
        for i in range(16):
            print(out_final[i,5,:,:])
        exit(1)
        '''
        
        return out_final                           #(b,n,h,w,l)

class PositionalEncoding(nn.Module):

    def __init__(self,num_hidden, batch_size ,img_width,img_height ,input_length,device):
        super(PositionalEncoding, self).__init__()
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.device = device
        self.input_length = input_length
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table())

    def _get_sinusoid_encoding_table(self):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):

            return_list = [torch.ones((self.batch_size,
                                       self.img_height,
                                       self.img_width)).to(self.device)*(position / np.power(10000, 2 * (hid_j // 2) / self.num_hidden[-1])) for hid_j in range(self.num_hidden[-1])]
                  
            return torch.stack(return_list, dim=1)
        sinusoid_table = [get_position_angle_vec(pos_i) for pos_i in range(self.input_length)]
        
        #sinusoid_table[:][0::2] = np.sin(sinusoid_table[:][0::2])  # dim 2i
        #sinusoid_table[:][1::2] = np.cos(sinusoid_table[:][1::2])  # dim 2i+1
        for pos_i in range(self.input_length):
            sinusoid_table[pos_i][:,0::2,:,:] = torch.sin(sinusoid_table[pos_i][:,0::2,:,:])  #2i
            sinusoid_table[pos_i][:,1::2,:,:] = torch.cos(sinusoid_table[pos_i][:,1::2,:,:])  #2i+1     
        #exit(1)
           

        return torch.stack(sinusoid_table, dim=-1)

    def forward(self, x):
        '''

        :param x: (b, channel, h, w, seqlen)
        :return:
        '''   
        if self.pos_table.clone().detach().shape != x.shape:
            pe = self.pos_table.clone().detach().squeeze()
            pe = pe.permute(3,0,1,2)
            pe = torch.nn.functional.interpolate(pe, size=(x.shape[2],x.shape[3]))
            pe = pe.permute(1,2,3,0)
            pe = pe.unsqueeze(0)
            
            return x + pe
            
        
        return x + self.pos_table.clone().detach()

class Encoder(nn.Module):
    def __init__(self, num_hidden, depth,img_height,img_width,input_length):
        super().__init__()
        self.input_length = input_length
        self.layers = nn.ModuleList([])
        self.num_hidden = num_hidden
        self.depth = depth
        self.img_height = img_height
        self.img_width = img_width
        for _ in range(self.depth):
            self.layers.append(nn.Sequential(
                Residual(PreNorm([self.num_hidden[-1],self.img_height,self.img_width],
                                 ConvAttention(self.num_hidden))),
                Residual(PreNorm([self.num_hidden[-1],self.img_height,self.img_width],
                                 FeedForward(self.num_hidden[-1])))
            ))
    def forward(self, x, mask = None):
        for attn in self.layers:
            
            x = attn(x)
            #x = ff(x)
        return x
'''
class Decoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.configs = configs
        self.num_hidden = [4,8,16,32]
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm([self.num_hidden[-1], self.configs.img_width, self.configs.img_width],
                                 ConvAttention(self.configs))),
                Residual(PreNorm([self.num_hidden[-1], self.configs.img_width, self.configs.img_width],
                                 FeedForward(self.configs)))
            ]))
    def forward(self, x, enc_out, mask=None):
        for attn, ff in (self.layers):
            x = attn(x,enc_out=enc_out,dec=True)
            x = ff(x)
        return x
'''

# def feature_embedding(img, configs):
#     generator = feature_generator(configs).to(configs.device)
#     gen_img = []
#     for i in range(img.shape[-1]):
#         gen_img.append(generator(img[:,:,:,:,i]))
#     gen_img = torch.stack(gen_img, dim=-1)
#     return gen_img

class Transformer(nn.Module):
    def __init__(self, num_hidden = [1],batch_size = 1 ,img_width = 256,img_height = 192 ,input_length = 16,encoder_depth = 6,device = torch.device("cuda")):
        super().__init__()
        #self.configs = configs
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.device = device
        self.input_length = input_length
        self.encoder_depth = encoder_depth
        self.pos_embedding = PositionalEncoding(self.num_hidden, self.batch_size ,self.img_width,self.img_height ,self.input_length ,self.device)
        self.Encoder = Encoder(self.num_hidden, self.encoder_depth,self.img_height,self.img_width,self.input_length)
        #self.Decoder = Decoder(dim,depth, heads, mlp_dim, dropout)
        self.device = device
        '''
        self.back_to_pixel = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1],1,kernel_size=1)
        )
        '''
    def forward(self, frames, mask = None):   
        #b,n,h,w,l = frames.shape
        #out_list=[]
        #feature_map = self.feature_embedding(img=frames)
        frames = frames.view(1,2048,6,8,16)
        #frames = frames.view(1,1,192,256,16)
        
        #frames = frames.view(1,256,96,128,8)  
        enc_in = self.pos_embedding(frames)   # 1 32 192 256 6
        
        #print(enc_in.shape)
        #exit(1)
        enc_in = enc_in.view(16,2048,6,8)
        
        #enc_in = enc_in.view(8,256,96,128)
        
        #enc_in = enc_in.view(16,1,192,256)
        enc_out = self.Encoder(enc_in)
        return enc_out

class fmnet_encoder(nn.Module):
    def __init__(self, num_hidden = [1],batch_size = 1 ,img_width = 256,img_height = 192 ,input_length = 16, small_length = 4,encoder_depth = 6,device = torch.device("cuda")):
        super().__init__()
        #self.configs = configs
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.device = device
        self.input_length = input_length
        self.small_length = small_length
        self.encoder_depth = encoder_depth
        self.pos_embedding = PositionalEncoding(self.num_hidden, self.batch_size ,self.img_width,self.img_height ,self.input_length ,self.device)
        self.Encoder = Encoder(self.num_hidden, self.encoder_depth,self.img_height,self.img_width,self.small_length)
        #self.Decoder = Decoder(dim,depth, heads, mlp_dim, dropout)
        self.device = device
        '''
        self.back_to_pixel = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1],1,kernel_size=1)
        )
        '''
    def forward(self, frames ):   
        #b,n,h,w,l = frames.shape
        #out_list=[]
        #feature_map = self.feature_embedding(img=frames) 
        frames = frames.permute(1,2,3,0)
        frames = frames.unsqueeze(0)
        enc_in = self.pos_embedding(frames)   # 1 32 192 256 6
        enc_in = enc_in.squeeze(0)
        enc_in = enc_in.permute(3,0,1,2) 
        choose_frames = np.array([4,8]) #random.sample(range(self.input_length), self.small_length))
        print(choose_frames)
        #random.sample(range(self.input_length), self.small_length))#[3,6,9])#random.sample(range(self.input_length), self.small_length)) #[0,1,2,3,4,5,6,7,8,9,10,11])#random.sample(range(self.input_length), self.small_length))  #[4,8]   
        #print(choose_frames)
        
        
        #enc_in = enc_in[choose_frames,...]  liuxia
        
        
        #enc_in = enc_in.view(8,256,96,128)
        #enc_in = enc_in.view(16,1,192,256)
        enc_out = self.Encoder(enc_in)
        return enc_out , choose_frames


class fmnet_decoder(nn.Module):
    def __init__(self, num_hidden = [1],batch_size = 1 ,img_width = 256,img_height = 192 ,input_length = 16, small_length = 4,encoder_depth = 2,device = torch.device("cuda")):
        super().__init__()
        #self.configs = configs
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.device = device
        self.input_length = input_length
        self.small_length = small_length
        self.encoder_depth = encoder_depth
        self.pos_embedding = PositionalEncoding(self.num_hidden, self.batch_size ,self.img_width,self.img_height ,self.input_length ,self.device)
        self.Encoder = Encoder(self.num_hidden, self.encoder_depth,self.img_height,self.img_width,self.input_length)
        
        self.device = device 
        self.mask_token = nn.Parameter(torch.zeros(1, self.num_hidden[0],self.img_height ,self.img_width)) #2048 
        
    def forward(self, frames, choose_frames):   

        #frames = frames.view(1,self.num_hidden[-1],self.img_height,self.img_width,self.input_length)
        if self.mask_token.shape[2] != frames.shape[2]:
            up_mask_token = torch.nn.functional.interpolate(self.mask_token, size=(frames.shape[2],frames.shape[3]))
            dec_in = up_mask_token.repeat(self.input_length,1,1,1)
        else:
            dec_in = self.mask_token.repeat(self.input_length,1,1,1)
        for i in range(self.small_length):
            dec_in[choose_frames[i]] = frames[i]

        dec_in = dec_in.permute(1,2,3,0)
        dec_in = dec_in.unsqueeze(0)

        dec_in = self.pos_embedding(dec_in)   # 1 32 192 256 6 
        dec_in = dec_in.squeeze(0)
        dec_in = dec_in.permute(3,0,1,2)

        dec_out = self.Encoder(dec_in)

        return dec_out

# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dropout = 0.):
#         super().__init__()
#         self.heads = heads
#         self.scale = dim ** -0.5
#
#         self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
#         self.to_out = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x,kv = None,first = True ,mask = None):  # x shape(1,65,1024)
#         b, n, _, h = *x.shape, self.heads
#         if first:
#             qkv = self.to_qkv(x)  #(1,65,1024*3)
#             q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) # (1,8,65,128)
#         else :
#             q = rearrange(x,'b n (h d) -> b h n d',h = h)
#             k = rearrange(kv,'b n (h d) -> b h n d',h = h)
#             v = rearrange(kv,'b n (h d) -> b h n d',h = h)
#         dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
#
#         if mask is not None:
#             mask = F.pad(mask.flatten(1), (1, 0), value = True)
#             assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
#             mask = mask[:, None, :] * mask[:, :, None]
#             dots.masked_fill_(~mask, float('-inf'))
#             del mask
#
#         attn = dots.softmax(dim=-1)
#
#         out = torch.einsum('bhij,bhjd->bhid', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out =  self.to_out(out)
#         return out


# infer
class mae_encoder_infer(nn.Module):
    def __init__(self, num_hidden = [1],batch_size = 1 ,img_width = 256,img_height = 192 ,input_length = 16, small_length = 4,encoder_depth = 6,device = torch.device("cuda")):
        super().__init__()
        #self.configs = configs
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.device = device
        self.input_length = input_length
        self.small_length = small_length
        self.encoder_depth = encoder_depth
        self.pos_embedding = PositionalEncoding(self.num_hidden, self.batch_size ,self.img_width,self.img_height ,self.input_length ,self.device)
        self.Encoder = Encoder(self.num_hidden, self.encoder_depth,self.img_height,self.img_width,self.small_length)
        #self.Decoder = Decoder(dim,depth, heads, mlp_dim, dropout)
        self.device = device
        '''
        self.back_to_pixel = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1],1,kernel_size=1)
        )
        '''
    def forward(self, frames, mask = None):   
        #b,n,h,w,l = frames.shape
        #out_list=[]
        #feature_map = self.feature_embedding(img=frames)
        frames = frames.view(1,self.num_hidden[-1],self.img_height,self.img_width,self.input_length)
        #frames = frames.view(1,1,192,256,16)
        #frames = frames.view(1,256,96,128,8)  
        enc_in = self.pos_embedding(frames)   # 1 32 192 256 6
        enc_in = enc_in.view(self.input_length,self.num_hidden[-1],self.img_height,self.img_width)

        choose_frames1 = np.array([0,2,4,6,8,10,12,14])#random.sample(range(self.input_length), self.small_length))
        #print(choose_frames)
        enc_in1 = enc_in[choose_frames1,...]

        #enc_in = enc_in.view(8,256,96,128)
        #enc_in = enc_in.view(16,1,192,256)
        enc_out1 = self.Encoder(enc_in1)

        choose_frames2 = np.array([1,3,5,7,9,11,13,15])#random.sample(range(self.input_length), self.small_length)) 
        #print(choose_frames)
        enc_in2 = enc_in[choose_frames2,...]

        #enc_in = enc_in.view(8,256,96,128)
        #enc_in = enc_in.view(16,1,192,256)
        enc_out2 = self.Encoder(enc_in2)

        return enc_out1 , choose_frames1,enc_out2 , choose_frames2


class mae_decoder_infer(nn.Module):
    def __init__(self, num_hidden = [1],batch_size = 1 ,img_width = 256,img_height = 192 ,input_length = 16, small_length = 4,encoder_depth = 2,device = torch.device("cuda")):
        super().__init__()
        #self.configs = configs
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.device = device
        self.input_length = input_length
        self.small_length = small_length
        self.encoder_depth = encoder_depth
        self.pos_embedding = PositionalEncoding(self.num_hidden, self.batch_size ,self.img_width,self.img_height ,self.input_length ,self.device)
        self.Encoder = Encoder(self.num_hidden, self.encoder_depth,self.img_height,self.img_width,self.input_length)
        
        self.device = device
        self.mask_token = nn.Parameter(torch.zeros(1, 2048, 6,8))
        
    def forward(self, frames, choose_frames,frames1, choose_frames1):   
    
        #frames = frames.view(1,self.num_hidden[-1],self.img_height,self.img_width,self.input_length)
        dec_in = self.mask_token.repeat(self.input_length,1,1,1)
        for i in range(self.small_length):
            dec_in[choose_frames[i]] = frames[i]
        dec_in = dec_in.view(1,self.num_hidden[-1],self.img_height,self.img_width,self.input_length)
        
        dec_in = self.pos_embedding(dec_in)   # 1 32 192 256 6

        dec_in = dec_in.view(self.input_length,self.num_hidden[-1],self.img_height,self.img_width)

        dec_out00 = self.Encoder(dec_in)

        dec_in = self.mask_token.repeat(self.input_length,1,1,1)
        for i in range(self.small_length):
            dec_in[choose_frames1[i]] = frames1[i]
        dec_in = dec_in.view(1,self.num_hidden[-1],self.img_height,self.img_width,self.input_length)
        
        dec_in = self.pos_embedding(dec_in)   # 1 32 192 256 6

        dec_in = dec_in.view(self.input_length,self.num_hidden[-1],self.img_height,self.img_width)

        dec_out01 = self.Encoder(dec_in)

        dec_out00[0::2] = dec_out01[0::2]
        

        return dec_out00
if __name__=="__main__":
    Mae_Encoder = mae_encoder(num_hidden = [2048],batch_size = 1 ,img_width = 8,img_height = 6 ,input_length = 8,small_length = 2,encoder_depth = 6).to(torch.device("cuda:0"))
    Mae_Decoder = mae_decoder(num_hidden = [2048],batch_size = 1 ,img_width = 8,img_height = 6 ,input_length = 8,small_length = 2,encoder_depth = 1).to(torch.device("cuda:0"))
    
    #checkpoint = torch.load('/data1/wangyiran/mytrans/firstconv/allNYU/mae_16_2/epoch_100.pth',map_location = 'cpu')  
    #Mae_Encoder.load_state_dict(checkpoint['Mae_Encoder'])
    #Mae_Decoder.load_state_dict(checkpoint['Mae_Decoder'])
    

    img_seq = torch.ones(8,2048,15,20).to(torch.device("cuda:0"))
    #print(img_seq.shape)
    enc_out,choose_frames = Mae_Encoder(img_seq)
    dec_out = Mae_Decoder(enc_out,choose_frames)
    print(dec_out.shape,'hhhhh')

