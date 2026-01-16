# -*- encoding: utf-8 -*-
# @File         : main.py
# @Date         : 2025/02/28 08:43:00
# @Author       : Eliwii_Keeya
# @Modified from: yuunnn-w, et al., 2024 -- RWKV_Pytorch

import torch
from model import RWKV_RNN

if __name__ == "__main__":
    args = {
        "MODEL_NAME": "./rwkv7-g1a-0.1b-20250728-ctx4096",  # 模型文件的名字，pth结尾的权重文件。
        "vocab_size": 65536,  # 词表大小
        "batch_size": 1,
        "device": "cpu",
    }
    device = args["device"]

    # 加载模型和分词器
    print("Loading model...")
    model = RWKV_RNN(args)
    print("Done.")

    # 编码初始字符串
    token = torch.ones([1], dtype=torch.int64, device=device)
    with torch.no_grad():
        out = model.forward(token)
        out = out.tolist()[0]
        print("Result: ", out[0:50])
