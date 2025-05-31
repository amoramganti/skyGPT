import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 32
block_size = 16  # context window
max_iters = 5000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-3
n_embd = 32
n_head = 4
n_layer = 4
dropout = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# -------------------------

torch.manual_seed(1234)

with open('dataset/tinyShakespeare.txt','r',encoding='utf-8') as f:
    text=f.read()

##==== tokenization
# create a simple character level tokenizer, using global lookup tables
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
tokenize_encode = lambda s: [stoi[c] for c in s]
tokenize_decode = lambda l: ''.join([itos[i] for i in l])

# tokenize dataset
data = torch.tensor(tokenize_encode(text), dtype=torch.long)
##==== 


##==== Split and prep dataset
# train test val split
split = 0.9
split_index = int(split*len(data))
train_data = data[:split_index]
val_data = data[split_index:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    inputs = torch.stack([data[i:i+block_size] for i in ix])

    # at every index has the next token for the input token in 'inputs' at that index 
    targets = torch.stack([data[i+1:i+block_size+1] for i in ix])
    inputs, targets = inputs.to(device), targets.to(device)

    return inputs, targets
##==== 


##==== Self Attention Head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)     # B,T,C (16)
        q = self.query(x)   # B,T,C

        # compute attention scores 
        # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # -1 is attn dim, -2 is T/Context size
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # (B, T, T)
        wei = F.softmax(wei, dim=-1)    # (B, T, T)

        wei = self.dropout(wei)

        # perform weighted aggregation of values based on qk dot products
        v = self.value(x)  # B, T, C
        out = wei @ v   # (B, T, T) @ (B, T, C) ---> (B, T, C)
        return out
##==== 


##==== Multi Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))    # projection back into the residual pathway
        return out
##====


##==== Feed Forward NN
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # projection layer
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
##==== 


##==== Attention-FF Block
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dim
        # n_head: number of attention heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # residual connections and layer norms included 
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
##==== 


##==== Model definition
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # Vocab Size x Embedding Dimension
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)    # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, inputs, targets=None):
        # inputs and targets are both initially (B,T)
        B, T = inputs.shape

        # Format: (B) Batch Size, (T) Context\Block Len, (C) embedding size
        tok_emb = self.token_embedding_table(inputs) # Predictions
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            # reshape logits to work with cross_entropy loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)  # now contains list of words, ideally most likely has the largest score
            targets = targets.view(B*T) # ground truth next word

            # calculate loss
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, inputs, max_new_tokens):
        # idx it (B, T) array of indices in the current context 

        for _ in range(max_new_tokens):
            input_cond = inputs[:, -block_size:]

            logits, loss = self(input_cond) # get predictions (calls forward function)

            # softmax over the last token in all of the batches
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1)

            next_input = torch.multinomial(probs, num_samples=1)  # (B, 1)
            inputs = torch.cat((inputs, next_input), dim=1)
        return inputs

model = BigramLanguageModel()
m = model.to(device)
##====


##==== Model evaluation through epochs/steps
@torch.no_grad()    # do not backprop over this
def estimate_loss():
    out = {}
    model.eval()    # good practice to set model to eval or train
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
##==== 


##==== Optimization and backprop
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):
  
  if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  # sample a batch of data
  inputs, targets = get_batch('train')

  # evaluate the loss
  logits, loss = model(inputs, targets)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
##====


##==== Generate using model
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # 1 batch, basic starter input
output = tokenize_decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(output)
##====