import torch
import torch.nn as nn
from torch.nn import functional as F



torch.manual_seed(1337)

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
n_embd = 384
n_heads = 6
n_layer = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
print(f'device :{device}')
with open("input.txt" , 'r', encoding='utf-8') as f :
  text = f.read()


chars =  sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i , ch in enumerate(chars)}
itos = { i : ch for i, ch in enumerate(chars)}

encode = lambda  s: [stoi[c] for c in s]
decode = lambda  l: ''.join([itos[c] for c in l])


data = torch.tensor(encode(text) , dtype= torch.long)
n = int(0.9* len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split) :
  data = train_data if split=='train' else val_data
  ix = torch.randint(len(data) - block_size , (batch_size,))
  x  = torch.stack([data[i: i+block_size] for i in ix])
  y  = torch.stack([data[i+1: i+block_size+1]  for i in ix])
  x , y = x.to(device) , y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss() :
  out = {}
  model.eval()
  for split in ['train' , 'val'] :
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(split)
        logits , loss = model(x,y)
        losses[k] = loss.item()
    out[split] = losses.mean() 
  model.train()
  return out



class Head(nn.Module) :
  def __init__(self, head_size , use_cache= False) :
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias= False)
    self.query = nn.Linear(n_embd , head_size, bias= False)
    self.value = nn.Linear(n_embd, head_size , bias= False)
    self.dropout = nn.Dropout(dropout)
    self.k_cache = None
    self.v_cache = None
    self.use_cache = use_cache
    self.register_buffer('tril' , torch.tril(torch.ones(block_size, block_size)))


  def reset_cache(self):
    self.k_cache = None
    self.v_cache = None

  def forward(self , x) :
    B, T, C = x.shape
    k = self.key(x)  # B T C entire sequence , we can  cache this.
    q = self.query(x) # B T C
    v = self.value(x)
    if self.use_cache :
    #  prefill  stage  :
      if self.k_cache is None :
        self.k_cache = k.detach()
        self.v_cache = v.detach() 
      else :    
        # next token stage generation 
        self.k_cache = torch.cat([self.k_cache , k.detach()] , dim=1)
        self.v_cache = torch.cat([self.v_cache , v.detach()] , dim=1)
      k = self.k_cache
      v = self.v_cache
    wei = q @ k.transpose(-2,-1)*k.shape[-1]**-0.5     # B T C @ B C T  -> B T T 
    if self.use_cache :
      past_len = k.shape[1]-T
      q_pos = torch.arange(past_len , past_len + T , device = x.device)
      k_pos = torch.arange(k.shape[1], device= x.device)
      mask = k_pos[None,:] <= q_pos[:,None]
      wei = wei.masked_fill(~mask, float('-inf'))
    else :
      wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    out = wei @ v
    return out

class MultiHeadAttention(nn.Module) :
  def __init__(self, num_heads, head_size , use_cache) :
    super().__init__()
    self.head = nn.ModuleList([Head(head_size, use_cache) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x) :
    out = torch.cat([h(x) for h in self.head] , dim =-1)
    out = self.proj(out)
    out = self.dropout(out)
    return out


class FeedForward(nn.Module) :
  def __init__(self , n_embd) :
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4*n_embd),
      nn.ReLU(),
      nn.Linear(4*n_embd, n_embd),
      nn.Dropout(dropout)
    )
  def forward(self, x) :
    return self.net(x)



class Block(nn.Module):
  def __init__(self, num_heads, n_embd , use_cache) :
    super().__init__()
    head_size = n_embd//num_heads
    self.sa = MultiHeadAttention(num_heads , head_size, use_cache)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x) :
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class BigramLanguageModel(nn.Module) :
  def __init__(self , use_cache=False):
    super().__init__()
    self.use_cache = use_cache
    self.toeken_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size,  n_embd)
    self.blocks = nn.Sequential(*[Block(n_heads, n_embd , use_cache) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) 
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None) :
    B, T = idx.shape
    token_emb = self.toeken_embedding_table(idx)   # b - batch , t - time  , c - channel
    past_len = 0
    first_head = self.blocks[0].sa.head[0]
    if first_head.use_cache  and first_head.k_cache is not None :
      past_len = first_head.k_cache.shape[1]

    if past_len + T > block_size :
      raise ValueError("KV cache exceeds block_size")
    position = torch.arange(past_len , past_len+T, device= idx.device)
    pos_emb = self.position_embedding_table(position) # (T,C)
    x = token_emb + pos_emb
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)

    if targets is None :
      loss = None
    else :
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits , targets)
    return logits , loss
  
  def set_cache(self, use_cache):
    self.use_cache = use_cache

    for block in self.blocks:
        for head in block.sa.head:
            head.reset_cache()
            head.use_cache = use_cache
  
  def generate(self, idx , max_new_tokens) :
    for _ in range(max_new_tokens) :
      idx_cond = idx[:, -block_size:]
      # idx = (B, T)
      logits , loss = self(idx_cond)
      logits = logits[: , -1 , :]
      probs = F.softmax(logits , dim= 1)  # (B, C)
      idx_next = torch.multinomial(probs , num_samples=1) # (B, 1)
      idx = torch.concat((idx, idx_next) , dim=1)
    return idx 



model = BigramLanguageModel(use_cache=False)
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters() , lr= 1e-3)


def train() :
  for iter in range(max_iters) :
    if iter % eval_interval ==0 :
      losses = estimate_loss()
      print(f"step {iter} - train loss :{losses['train']:.4f} , val loss {losses['val']:.4f}")
    xb , yb = get_batch('train')
    logits , loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1,1) , dtype=torch.long , device=device)
print(decode(m.generate(context ,max_new_tokens = 500)[0].tolist()))