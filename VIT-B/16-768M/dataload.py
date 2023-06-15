import torch
import torch.nn as nn
from PIL import Image
import jieba
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from torch.utils.data import DataLoader, Dataset
import os
import pickle
device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载模型
model, preprocess = load_from_name("ViT-B-16",device=device, download_root='https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-b-16.pt')
model.eval()

#文本解码器
class TextDecoder(nn.Module):
    def __init__(self, input_size,  output_size):
        super(TextDecoder, self).__init__()
        self.out = nn.Linear(input_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq):
        output = self.softmax(self.out(input_seq))
        return output

#clip模型文本输入解码器测试

def generate_text(decoder, input_text):
    with torch.no_grad():
        # 使用 CLIP 模型的文本编码器编码输入文本
        text_input = clip.tokenize(input_text).to(device)
        encoded_text1 = model.encode_text(text_input)
        
        # 解码文本特征生成文本
        decoder_input = encoded_text1.float() 
        output_seq = []
        max_length = 1
        for _ in range(max_length):  # 设置最大生成长度
            decoder_output = decoder(decoder_input)  # 使用 CLIP 的文本特征作为解码器的输入
            topv, topi = decoder_output.topk(1)
            topi = topi.squeeze().item()
            word = target_idx2token[topi]
            output_seq.append(word)
        return ' '.join(output_seq)



# #clip模型图像输入解码器测试

def generate_image(decoder, encoder_image):
    with torch.no_grad():
        encoder_image = encoder_image.to(device)
        
        output_seq = []
        max_length = 1
        decoder_input = encoder_image.float()
        for _ in range(max_length):  # 设置最大生成长度
            decoder_output = decoder(decoder_input)
            topv, topi = decoder_output.topk(1)
            topi = topi.squeeze().item()
            word = target_idx2token[topi]
            output_seq.append(word)

            # if topi == target_token2idx['<EOS>']:
            #     break
            # else:
            #     word = target_idx2token[topi]
            #     output_seq.append(word)
            #     decoder_input = text_tokenizer(word, return_tensors='pt', padding=True)['input_ids'].permute(1, 0)

        return ' '.join(output_seq)
    

# 打开文件并读取内容
with open('/home/zhangkai/Myproject/文本数据集/train1.txt', 'r', encoding='utf-8') as f:
    content = f.read()
# 使用jieba库进行分词
words = jieba.cut(content)
# 删除分词结果里的“/”，“。”,以及空格字符
# del_list = ['、','/', '。','（','）','？', ' ','o','','！','，','“','”','；','’','：','—','一一','《','》','…','１','２','３','４','５','６','７','８','９','０','ns','nt','nr']
# words = [word.strip() for word in words if word.strip() not in del_list]

# 只保留中文文字
new_words = []
for word in words:
    if '\u4e00' <= word <= '\u9fff':
        new_words.append(word)

# 使用set()函数去掉重复字符，并转换回列表
vocab = list(set(new_words))
target_token2idx = {token: i for i, token in enumerate(vocab) }
target_idx2token = {i: token for i, token in enumerate(vocab) }
vocab_size = len(vocab)



# 定义训练超参数
learning_rate = 1
num_epochs = 150
batch_size = 2048

# 初始化解码器网络和损失函数
decoder = TextDecoder(input_size=512, output_size=vocab_size).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)

# 定义自己的数据集类
class MyDataset(Dataset):
    def __init__(self, input_text, target_text):
        self.input_text = input_text
        self.target_text = target_text

    def __getitem__(self, index):
        input_text = self.input_text[index]
        target_text = self.target_text[index]
        return input_text, target_text

    def __len__(self):
        return len(self.input_text)

# 输入文本和目标文本
input_text = vocab
target_text = vocab

# 创建数据集对象
train_dataset = MyDataset(input_text, target_text)


# 创建dataloader对象
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

if os.path.isfile('/home/zhangkai/Myproject/图片分类/VIT-B/16-768M/checkpoint/checkpoint.pt'):
    checkpoint = torch.load('/home/zhangkai/Myproject/图片分类/VIT-B/16-768M/checkpoint/checkpoint.pt',map_location=device)
    start_epoch = checkpoint['epoch'] + 1
    decoder.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    decoder.train()
    
    print('decoder模型加载完成')
else:
    start_epoch = 0
    decoder.train()
# 训练循环
for epoch in range(start_epoch,num_epochs):
    for batch_data in train_loader:
        optimizer.zero_grad()
        # 获取 CLIP 模型的文本特征
        with torch.no_grad():
            text_input = clip.tokenize(batch_data[0]).to(device)
            encoded_text1 = model.encode_text(text_input)
            decoder_input = encoded_text1.float()

        # 解码文本特征并计算损失
        decoder_output = decoder(decoder_input)
        target_tensor = torch.tensor([target_token2idx[token] for token in batch_data[1]]).to(device)
        loss = criterion(decoder_output, target_tensor)

        # 反向传播和参数更新
        loss.backward()
        optimizer.step()

    # 打印训练信息
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    # 保存模型的参数和优化器状态
    torch.save({
        'epoch': epoch,
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 其他需要保存的信息
    }, '/home/zhangkai/Myproject/图片分类/VIT-B/16-768M/checkpoint/checkpoint.pt')

torch.save(decoder.state_dict(), '/home/zhangkai/Myproject/图片分类/VIT-B/16-768M/models/VIT-B16-768M.pt')


# 使用已训练的解码器从文本中生成文本
input_text = "书本"
generated_text = generate_text(decoder, input_text)
print('Generated Text:', generated_text)

# 使用已训练的解码器从图像中生成文本
# 设置图片路径和类别文本路径
image_dir = "/home/zhangkai/Myproject/图片分类/photoes/image30.jpg"
image = preprocess(Image.open(image_dir)).unsqueeze(0).to(device)
image_features = model.encode_image(image)
encoder_image = image_features.to(device)
generated_text1 = generate_image(decoder, encoder_image)
print('Generated Text1:', generated_text1)


# 文本和图像相似度测试
image_embedding = image_features / torch.norm(image_features)
input = ["自然风光"]
text1_input = clip.tokenize(input).to(device)
text_features = model.encode_text(text1_input)
text_embedding = text_features / torch.norm(text_features)


# # # #测试所保存的模型
# model_test = torch.load('/home/zhangkai/Myproject/图片分类/VIT-B/16-768M/models/VIT-B16-768M.pt')
# new_m = TextDecoder(input_size=512, output_size=vocab_size)
# new_m.load_state_dict(model_test)
# new_m.to(device)
# image_dir = "/home/zhangkai/Myproject/图片分类/photoes/image31.jpg"
# image = preprocess(Image.open(image_dir)).unsqueeze(0).to(device)
# image_features = model.encode_image(image)
# text = generate_image(new_m, image_features)
# print("图像-图像输出为：",text)

# input_text = "圆珠笔"
# text1 = generate_text(new_m, input_text)
# print("文本-文本输出为：",text1)
