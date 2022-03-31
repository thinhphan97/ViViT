import torch
import torch.nn as nn

from vivit import *
print(ENCODER_REGISTRY.get_list())
def test_VanillaEncoder():

    test_tensor = torch.randn(2, 65, 1024)
    encoder = ENCODER_REGISTRY.get("VanillaEncoder")(
        embedding_dim=1024, depth=6, num_heads=16, head_dim=64, mlp_dim=2048
    )
    out = encoder(test_tensor)
    assert out.shape == test_tensor.shape  # shape remains same
    del encoder, test_tensor

def test_TubeletEmbedding():
    test_tensor = torch.randn(
        7, 20, 3, 224, 224
    )  # batch_size,time,in_channels,height,width
    embedding = TubeletEmbedding(
        embedding_dim=192, tubelet_w=16, tubelet_t=5, tubelet_h=16, in_channels=3
    )
    out = embedding(test_tensor)
    assert out.shape == (
        7,
        4,
        196,
        192,
    )  # batch,time/tubelet_t,height*width/(tubelet_h,tubelet_w),embeeding_dim
    del embedding

    test_tensor = torch.randn(11, 15, 1, 28, 28)
    embedding = TubeletEmbedding(96, 5, 7, 7, 1)
    out = embedding(test_tensor)
    assert out.shape == (11, 3, 16, 96)
    del embedding


def test_ViViTEncoder():

    encoder = ENCODER_REGISTRY.get("ViViTEncoderModel3")(
        dim=192, num_heads=3, head_dim=64, p_dropout=0.0, depth=3
    )

    test_tensor = torch.randn(7, 20, 196, 192)
    logits = encoder(test_tensor)
    assert logits.shape == (7, 3920, 192)

def test_ViViTEncoderBlockModel4():
    
    encoder = ENCODER_REGISTRY.get("ViViTEncoderBlockModel4")(
        dim=192, num_heads=3, head_dim=64, p_dropout=0.0
    )

    test_tensor = torch.randn(7, 20, 196, 192)
    logits = encoder(test_tensor)
    assert logits.shape == (7, 20, 196, 192)

def test_ViViTEncoderModel4():
    
    encoder = ENCODER_REGISTRY.get("ViViTEncoderModel4")(
        dim=192, num_heads=3, head_dim=64, p_dropout=0.0, depth=3
    )
    test_tensor = torch.randn(7, 20, 196, 192)
    logits = encoder(test_tensor)
    assert logits.shape == (7, 3920, 192)
    
if __name__ == '__main__':
    
    test_VanillaEncoder()
    print("pass !")
    test_TubeletEmbedding()
    print("pass !")
    test_ViViTEncoder()
    print("pass !")
    test_ViViTEncoderBlockModel4()
    print("pass !")
    test_ViViTEncoderModel4()
    print("pass !")
    
    