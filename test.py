import hw3.charnn as charnn
import torch
corpus_path = r'C:\Users\Assaf\.pytorch-datasets\shakespeare.txt'
device = "cuda" if torch.cuda.is_available() else "cpu"

with open(corpus_path, 'r', encoding='utf-8') as f:
    corpus = f.read()

char_to_idx, idx_to_char = charnn.char_maps(corpus)
x, y = charnn.chars_to_labelled_samples(corpus[:129], char_to_idx, 64, device)

num_layers = 1
gru = charnn.MultilayerGRU(len(char_to_idx), 256, len(char_to_idx), num_layers, device)
gru.cuda()


for _ in range(3):
    text = charnn.generate_from_model(gru, "foobar", 50, (char_to_idx, idx_to_char), T=0.5)
    print(text)
