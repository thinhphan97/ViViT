import torch

from vivit import *

print(ATTENTION_REGISTRY.get_list())


def test_VanillaSelfAttention():

    test_tensor1 = torch.randn(2, 65, 1024)
    test_tensor2 = torch.randn(2, 257, 1024)

    attention = ATTENTION_REGISTRY.get("VanillaSelfAttention")(dim=1024)
    out = attention(test_tensor1)
    assert out.shape == (2, 65, 1024)
    del attention

    attention = ATTENTION_REGISTRY.get("VanillaSelfAttention")(dim=1024, num_heads=16)
    out = attention(test_tensor2)
    assert out.shape == (2, 257, 1024)
    del attention

def test_CrossStageSelfAttention():
    
    test_tensor1 = torch.randn(3, 65, 1024)
    test_tensor2 = torch.randn(3, 257, 1024)
    test_A1 = torch.randn(3,8,65, 65)
    test_A2 = torch.randn(3,16,257, 257)

    attention = ATTENTION_REGISTRY.get("CrossStageSelfAttention")(dim=1024)
    out, dot = attention(test_tensor1, test_A1)
    assert out.shape == (3, 65, 1024) and dot.shape == test_A1.shape
    del attention
    
    attention = ATTENTION_REGISTRY.get("CrossStageSelfAttention")(dim=1024, num_heads=16)
    out, dot = attention(test_tensor2, test_A2)
    assert out.shape == (3, 257, 1024)and dot.shape == test_A2.shape
    del attention
    
if __name__ == "__main__":
    test_VanillaSelfAttention()
    print("pass !")
    test_CrossStageSelfAttention()
    print("pass !")