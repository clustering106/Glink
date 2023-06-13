import torch 
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import re
# 打开文件
file = open("/home/zhangkai/Myproject/图片分类/text.txt", "r", encoding="utf-8") 
# 读取文件内容
content = file.read()
text = re.findall(r'\w+', content)
vocab = text
target_token2idx = {token: i for i, token in enumerate(vocab) }
target_idx2token = {i: token for i, token in enumerate(vocab) }

print("Available models:", available_models())  
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("RN50", device=device, download_root='https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_rn50.pt')
model.eval()
image = preprocess(Image.open("/home/zhangkai/Myproject/图片分类/photoes/3.jpg")).unsqueeze(0).to(device)
text =  clip.tokenize(vocab).to(device)
output_seq = []
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
    image_features /= image_features.norm(dim=-1, keepdim=True) 
    text_features /= text_features.norm(dim=-1, keepdim=True)    

    logits_per_image, logits_per_text = model.get_similarity(image, text)
    probs = logits_per_image.softmax(dim=-1)
    topv, topi = probs.topk(1)
    topi = topi.squeeze().item()
    word = target_idx2token[topi]
    output_seq.append(word)

print(' '.join(output_seq))  