import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import re
from PIL import Image
# 加载图像编码器
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 打开文件
file = open("/home/zhangkai/Myproject/图片分类/text.txt", "r", encoding="utf-8") 
content = file.read()
vocab = re.findall(r'\w+', content)

# 关闭文件
file.close()
target_idx2token = {i: token for i, token in enumerate(vocab) }
vocab_size = len(vocab)
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
    
# #clip模型图像输入解码器测试

def generate_image(decoder, encoder_image):
    with torch.no_grad():
        encoder_image = encoder_image.unsqueeze(0)
        hidden = torch.zeros(1, 1, decoder.hidden_size)
        output_seq = []
        max_length = 1
        decoder_input = encoder_image
        for _ in range(max_length):  # 设置最大生成长度
            decoder_output, hidden = decoder(decoder_input, hidden)
            topv, topi = decoder_output.topk(1)
            topi = topi.squeeze().item()
            word = target_idx2token[topi]
            output_seq.append(word)

        return ' '.join(output_seq)
    
#测试所保存的模型
model_test = torch.load('/home/zhangkai/Myproject/图片分类/VIT-L/14-1.4G/model/model.pth')
new_m = TextDecoder(input_size=768, hidden_size=vocab_size, output_size=vocab_size)
new_m.load_state_dict(model_test)
image_dir = "/home/zhangkai/Myproject/图片分类/photoes/3.jpg"
image = processor(images=Image.open(image_dir), return_tensors="pt")
image_features = clip_model.get_image_features(**image)

text = generate_image(new_m, image_features)
print("图像-图像输出为：",text)


# input_text = "自然风光"
# text1 = generate_text(new_m, input_text)
# print("文本-文本输出为：",text1)