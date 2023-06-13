import torch
import torch.nn as nn
from PIL import Image
import re
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
print("Available models:", available_models()) 
device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载模型
model, preprocess = load_from_name("RN50",device=device, download_root='https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_rn50.pt')
model.eval()

#文本解码器
class TextDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TextDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq, hidden):
        output, hidden = self.gru(input_seq, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

#clip模型文本输入解码器测试

def generate_text(decoder, input_text):
    with torch.no_grad():
        # 使用 CLIP 模型的文本编码器编码输入文本
        text_input = clip.tokenize(input_text).to(device)
        encoded_text1 = model.encode_text(text_input)
        
        # 解码文本特征生成文本
        decoder_input = encoded_text1.unsqueeze(0).float() 
        hidden = torch.zeros(1, 1, decoder.hidden_size).to(device)
        output_seq = []
        max_length = 1
        for _ in range(max_length):  # 设置最大生成长度
            decoder_output, hidden = decoder(decoder_input, hidden)  # 使用 CLIP 的文本特征作为解码器的输入
            topv, topi = decoder_output.topk(1)
            topi = topi.squeeze().item()
            word = target_idx2token[topi]
            output_seq.append(word)
        return ' '.join(output_seq)



# #clip模型图像输入解码器测试

def generate_image(decoder, encoder_image):
    with torch.no_grad():
        encoder_image = encoder_image.unsqueeze(0).to(device)
        hidden = torch.zeros(1, 1, decoder.hidden_size).to(device)
        output_seq = []
        max_length = 1
        decoder_input = encoder_image.float()
        for _ in range(max_length):  # 设置最大生成长度
            decoder_output, hidden = decoder(decoder_input, hidden)
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
    
import re
# 打开文件
file = open("/home/zhangkai/Myproject/图片分类/text.txt", "r", encoding="utf-8") 
# 读取文件内容
content = file.read()
text = re.findall(r'\w+', content)
vocab = text
# # 关闭文件
# file.close()
# vocab = list(word)
target_token2idx = {token: i for i, token in enumerate(vocab) }
target_idx2token = {i: token for i, token in enumerate(vocab) }
vocab_size = len(vocab)


# # 定义训练超参数
# learning_rate = 0.1
# num_epochs = 300

# # 初始化解码器网络和损失函数
# decoder = TextDecoder(input_size=1024, hidden_size=vocab_size, output_size=vocab_size).to(device)
# criterion = nn.CrossEntropyLoss().to(device)
# optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)


# # 输入文本和目标文本
# input_text = vocab
# target_text = vocab

# # 获取 CLIP 模型的文本特征
# with torch.no_grad():
#     text_input = clip.tokenize(input_text).to(device)
#     encoded_text1 = model.encode_text(text_input)
#     decoder_input = encoded_text1.unsqueeze(0).float()



# # 训练循环
# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#     # 解码文本特征并计算损失
#     hidden = torch.zeros(1, vocab_size, decoder.hidden_size).to(device)
#     decoder_output, _ = decoder(decoder_input, hidden)
#     target_tensor = torch.tensor([target_token2idx[token] for token in target_text]).to(device)
#     loss = criterion(decoder_output, target_tensor)
#     # 反向传播和参数更新
#     loss.backward()
#     optimizer.step()

#     # 打印训练信息
#     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
# torch.save(decoder.state_dict(), '/home/zhangkai/Myproject/图片分类/Resnet50-368M/model/Resnet50-368M.pt')



# # 使用已训练的解码器从文本中生成文本
# input_text = "书本"
# generated_text = generate_text(decoder, input_text)
# print('Generated Text:', generated_text)

# # 使用已训练的解码器从图像中生成文本
# # 设置图片路径和类别文本路径
# image_dir = "/home/zhangkai/Myproject/图片分类/photoes/11.jpg"
# image = preprocess(Image.open(image_dir)).unsqueeze(0).to(device)
# image_features = model.encode_image(image)
# encoder_image = image_features.to(device)
# generated_text1 = generate_image(decoder, encoder_image)
# print('Generated Text1:', generated_text1)


# # 文本和图像相似度测试
# image_embedding = image_features / torch.norm(image_features)
# input = ["自然风光"]
# text1_input = clip.tokenize(input).to(device)
# text_features = model.encode_text(text1_input)
# text_embedding = text_features / torch.norm(text_features)

# # 计算余弦相似度
# similarity = nn.functional.cosine_similarity(image_embedding, text_embedding,dim=1)
# print("相似度:", similarity)





#测试所保存的模型
model_test = torch.load('/home/zhangkai/Myproject/图片分类/Resnet50-368M/model/Resnet50-368M.pt')
new_m = TextDecoder(input_size=1024, hidden_size=vocab_size, output_size=vocab_size)
new_m.load_state_dict(model_test)
new_m.to(device)
image_dir = "/home/zhangkai/Myproject/图片分类/photoes/2.jpg"
image = preprocess(Image.open(image_dir)).unsqueeze(0).to(device)
image_features = model.encode_image(image)
text = generate_image(new_m, image_features)
print("图像-图像输出为：",text)

input_text = "豪华"
text1 = generate_text(new_m, input_text)
print("文本-文本输出为：",text1)