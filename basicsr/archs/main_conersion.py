import torch
from rdg_arch import RDG

if __name__ == '__main__':
    # 加载预训练模型参数
    crt_net = torch.load('/data0/zhengmingjun/RDG/result/GIF_BASE.pth')['params']

    # 初始化目标模型
    model = RDG()
    ori_net = model.state_dict()

    # 定义保存路径
    save_path = '/data0/zhengmingjun/RDG/experiments/pretrained_models/RDG_Base.pth'

    # 创建一个字典来保存更新后的参数
    updated_params = {}

    # 遍历预训练模型的参数
    for crt_k, crt_v in crt_net.items():
        # 根据键名进行映射
        if 'gmlp' in crt_k:
            ori_k = crt_k.replace('gmlp', 'dfm')
            print('replace gmlp -> dfm:', ori_k, crt_k)
        elif 'cg' in crt_k:
            ori_k = crt_k.replace('cg', 'hfb')
            print('replace cg -> hfb:', ori_k, crt_k)
        elif 'atten' in crt_k:
            ori_k = crt_k.replace('atten', 'ctr')
            print('replace atten -> ctr:', ori_k, crt_k)
        else:
            ori_k = crt_k
            print(ori_k, crt_k)

        # 检查目标模型中是否存在对应的键
        if ori_k in ori_net:
            updated_params[ori_k] = crt_v
        else:
            print(f"Warning: Key {ori_k} not found in the target model's state_dict.")

    # 更新目标模型的参数
    ori_net.update(updated_params)

    # 保存更新后的参数
    save_dict = {'params': ori_net}
    torch.save(save_dict, save_path)
    print('Saved at:', save_path)