import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm

class Dictionary(object):
    def __init__(self):
        # Constructor for the Dictionary class
        self.word2idx = {}  # A dictionary to map words to indices
        self.idx2word = {}  # A dictionary to map indices to words
        self.idx = 0  # Initialize index counter

    def add_word(self, word):
        # Method to add a word to the dictionary
        if word not in self.word2idx:
            self.word2idx[word] = self.idx  # Map the word to its index
            self.idx2word[self.idx] = word  # Map the index to the word
            self.idx += 1  # Increment the index counter

    def __len__(self):
        # Method to return the length of the dictionary
        return len(self.word2idx)
    
class TextProcess(object):
    def __init__(self):
        # Constructor for the TextProcess class
        self.dictionary = Dictionary()  # Initialize a dictionary object

    def get_data(self, path, batch_size = 20):
        # Method to preprocess text data and convert it into a tensor
        with open(path, 'r') as f:
            tokens = 0  # Initialize token counter
            for line in f:
                words = line.split() + ['<eos>']  # Split the line into words and add end-of-sentence token
                tokens += len(words)  # Increment the token counter
                for word in words:
                    self.dictionary.add_word(word)  # Add each word to the dictionary
        # Create a 1-D tensor that contains the index of all the words in the file
        rep_tensor = torch.LongTensor(tokens)
        index = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    rep_tensor[index] = self.dictionary.word2idx[word]  # Convert words to indices
                    index += 1
        # Find out how many batches we need
        num_batches = rep_tensor.shape[0]
        # Remove the remainder (filter out the ones that don't fit)
        rep_tensor = rep_tensor[:num_batches*batch_size]
        rep_tensor = rep_tensor.view(batch_size, -1)  # Reshape the tensor into batches
        return rep_tensor
    
# Hyperparameters
embed_size = 128  # Input features to the LSTM
hidden_size = 1024  # Number of LSTM units
num_layers = 1
num_epochs = 20
batch_size = 20
timesteps = 30
learning_rate = 0.002

# Create an instance of the TextProcess class
corpus = TextProcess()

# Preprocess the text data and convert it into a tensor
rep_tensor = corpus.get_data('alice.txt', batch_size)

print(rep_tensor.shape)

# Get the size of the vocabulary
vocab_size = len(corpus.dictionary)
print(vocab_size)

# Calculate the number of batches
num_batches = rep_tensor.shape[1] // timesteps
print(num_batches)

# Define the TextGenerator model
class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(TextGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        out = self.linear(out)
        return out, (h, c)
    
# Create an instance of the TextGenerator model
model = TextGenerator(vocab_size, embed_size, hidden_size, num_layers)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def detach(states):
    return [state.detach() for state in states]

# Training loop
for epoch in range(num_epochs):
    states = (torch.zeros(num_layers, batch_size, hidden_size),
              torch.zeros(num_layers, batch_size, hidden_size))
    
    for i in range(0, rep_tensor.size(1)-timesteps, timesteps):
        inputs = rep_tensor[:, i:i+timesteps]
        targets = rep_tensor[:, (i+1):(i+1)+timesteps]
        outputs,_ = model(inputs, states)
        loss = loss_fn(outputs, targets.reshape(-1))

        model.zero_grad()
        loss.backward()

        clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()

        step = (i+1) // timesteps
        if step % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Generating text
with torch.no_grad():
    with open ('results.txt', 'w') as f:
        state = (torch.zeros(num_layers, 1, hidden_size),
                 (torch.zeros(num_layers, 1, hidden_size)))
        input = torch.randint(0, vocab_size, (1,)).long().unsqueeze(1)

        for i in range(500):
            output,_ = model(input, state)
            print(output.shape)

            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()
            print(word_id)

            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i+1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i+1, 500, 'results.txt'))
