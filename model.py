import torch
import torch.nn as nn
import torch.nn.functional as F
 
class CRCNN(nn.Module):
    def __init__(self, args):
        super(CRCNN, self).__init__()
        self.args = args
        
        vocab = args.vocab
        Vocab = len(vocab)
        posemb = args.posemb
        Dim = args.embed_dim
        Pos_dim = args.pos_dim
        Cla = args.class_num
        Ci = 1
        Knum = args.kernel_num
        Ks = args.kernel_sizes
        
        #self.sent_len = args.sent_len
        self.embed = nn.Embedding(len(vocab),Dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.pos_embed = nn.Embedding(len(posemb),Pos_dim)
        
        self.convs = nn.ModuleList([nn.Conv2d(Ci,Knum,(K,Dim+2*Pos_dim),padding=((K-1)//2,0)) for K in Ks])
        #self.dropout1 = nn.Dropout(args.dropout)
        #self.dropout2 = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(Ks)*Knum,Cla)
        
    def forward(self,cx):
        x = cx[0]
        #print(x.shape)
        pos = cx[1]
        #print(pos.shape)
        x = self.embed(x) #(N,W,D)
        #print(x.shape)
        #x = self.dropout1(x)
        #print(x.shape)
        
    
        p = self.pos_embed(pos) #(N,2*W,pD)
        #print(p.shape)
        p = p.view(p.shape[0],p.shape[1]//2,-1)
        #print(p.shape)
        #print("111111")
        #print(x.shape)
        #print(p.shape)
        x = torch.cat((x,p),2) #(N,W,D+2*pD)
        
        x = x.unsqueeze(1) #(N,Ci,W,D+2*pD)
        #print(x.shape)
        #xx = [F.relu(conv(x)) for conv in self.convs]
        #print(xx[0].shape)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # (len(Ks),N,Knum,W)
        #print(x[0].shape)
        x = [F.max_pool1d(line,line.size(2)).squeeze(2) for line in x]  # (len(Ks),N,Knum)
        
        x = torch.cat(x,1) #(N, len(Ks)*Knum)
        #x = self.dropout2(x)
        logit = self.fc(x) # (N,Cla)

        return logit
