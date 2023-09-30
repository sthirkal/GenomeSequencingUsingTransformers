
import os
import copy
import torch
import torch.nn as nn 
from torch.nn import functional as F
import numpy as np
device = 'cuda' 

block_size = 256
batch_size = 64
learning_rate = 1e-4
max_iters = 5000
eval_interval = 500
eval_iters = 200
n_embd = 512
n_head = 8
n_layer = 6
dropout = 0.2

path = "/home/harisshragav/Projects/GPT/dataCOV/"
dir_list = os.listdir(path)
text = ""
finaldata =[]

#print(dir_list)

for dname in dir_list:
    buffer = []
    with open("/home/harisshragav/Projects/GPT/dataCOV/"+dname,'r') as txt:
        text = txt.read().replace("\n","").replace('\ufeff', '')
        
    for i in range(0,len(text)-3,3):
        test =text[i:i+3]
        if(test == 'T'):
            print(dname)
        buffer.append(test)
        
    finaldata.append(buffer)

chars = sorted(list(set(finaldata[0])))
print(chars)
vocab_size = len(chars)

stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}
#print(stoi)


def encode(s):
    geneencode = []
    for i in s:
        geneencode.append(stoi[i])
    return geneencode

def decode(l):
    genedecode = []
    for i in l:
        genedecode.append(itos[i])
    return "".join(genedecode)

    
print(finaldata)
    
data = copy.deepcopy(finaldata)

for y in range(len(finaldata)):
    data[y] = encode(finaldata[y])

n = int( 0.9* len(data))
train_data = data[:n]
val_data  = data[n:]
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X,Y)
            losses[k] = loss.item()
        
        out[split] = losses.mean()

    model.train()
    return out


def get_batch(split):
    data = train_data if split == 'train' else val_data
    x = torch.Tensor()
    y = torch.Tensor()
    for _data in data:
        ix = torch.randint(len(_data)- block_size, (batch_size,))
        x = torch.stack([torch.tensor(_data[i: i+ block_size]) for i in ix])
        y = torch.stack([torch.tensor(_data[i+1 : i + block_size + 1]) for  i in ix])
    x,y = x.to(device), y.to(device)
    print(x,y)
    return x,y

class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads,n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out


class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__() 
        self.key = nn.Linear(n_embd,head_size, bias=False)
        self.query = nn.Linear(n_embd,head_size, bias=False)
        self.value = nn.Linear(n_embd,head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2,-1) * C**-0.5 
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        return out

class FeedForward(nn.Module):
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,n_embd * 4),
            nn.ReLU(),
            nn.Linear(4 * n_embd,n_embd ),
            nn.Dropout(dropout),
        )
    
    def forward(self, x): 
        return self.net(x)

class Block(nn.Module):
    
    def __init__(self,n_embd, n_head):
        super().__init__()
        head_size = n_embd //n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        

    def forward(self, x):
        
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.pos_embedding_table =  nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.lmhead = nn.Linear(n_embd,vocab_size)
    
    def forward(self, idx, targets=None):
        
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx) 
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device)) ## T, C
        ##(B,T,C) B = Batch_size, Time = No.chars or block_zie , C = The embedding value list returned by the embedding table.
        x = token_emb + pos_emb 
        x = self.blocks(x)
        logits = self.lmhead(x)

        if targets == None:
            loss = None
        else:

            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets) ## cross_entropy or negative log likelihood equals loss ## logits are output of neural net
        return logits, loss

    
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
    
            logits = logits[:,-1,:]

            probs = F.softmax(logits, dim =-1)

            idx_next = torch.multinomial(probs, num_samples = 1)

            idx = torch.cat((idx, idx_next), dim = 1)
        return idx


model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate, betas=(0.9,0.98), eps=1e-9)

for iter in range(max_iters):
    if (iter != 0):
        learning_rate = (n_embd)**(-0.5)*min(iter **(-0.5), iter * (500)**(-1.5))
    if iter % eval_interval == 0:
        losses = estimate_loss()           
        print(learning_rate)
        print(f"step:{iter} : train loss: {losses['train']:.4f} : val loss: {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

idx = torch.zeros((1,1), dtype = torch.long, device = device)

print(decode(m.generate(idx, max_new_tokens=10110)[0].tolist()))

          
        