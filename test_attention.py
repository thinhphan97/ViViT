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

    
if __name__ == "__main__":
    test_VanillaSelfAttention()
    print("pass !")
   