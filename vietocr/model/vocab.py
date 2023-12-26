class Vocab():
    def __init__(self, chars):
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.mask_token = 3

        self.chars = chars

        self.c2i = {c:i+4 for i, c in enumerate(chars)}

        self.i2c = {i+4:c for i, c in enumerate(chars)}
        
        self.i2c[0] = '<pad>'
        self.i2c[1] = '<sos>'
        self.i2c[2] = '<eos>'
        self.i2c[3] = '*'

    def encode(self, chars):
        return [self.go] + [self.c2i[c] for c in chars] + [self.eos]
    
    def decode(self, ids):
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        sent = ''.join([self.i2c[i] for i in ids[first:last]])
        return sent
    
    def __len__(self):
        return len(self.c2i) + 4
    
    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars
    

if __name__ == '__main__':
    def read_txt_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read().replace('\n', '')  # Remove leading/trailing whitespace

        return file_content

    # Example usage:
    txt_file_path = 'vocab.txt'
    file_string = read_txt_file(txt_file_path)
    # file_string = pd.read_csv(txt_file_path,header=None,on_bad_lines='skip')[0]
    # file_string = file_string[0].tolist()

    # print(file_string)
    chars = '?'
    a = Vocab(file_string)
    # print(a.c2i)
    # for c in a.c2i:
        # print(c)
        # chars = c
    b = a.encode(chars)
    print(b)
