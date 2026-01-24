# 测试代码
import torch
from ip_src.models import PointNetPlusPlusEncoder

encoder = PointNetPlusPlusEncoder(freeze=True)
encoder.load_pretrained_weights("./model.pt", verbose=True)