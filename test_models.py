import torch
import torch.nn as nn

from vivit import MODEL_REGISTRY

print(MODEL_REGISTRY.get_list())

def test_ViViT():
    test_tensor1 = torch.randn([1, 16, 3, 224, 224])
    test_tensor2 = torch.randn([3, 16, 3, 224, 224])

    model = MODEL_REGISTRY.get("FactorisedEncoder")(
        img_size=224,
        in_channels=3,
        patch_t=2,
        patch_h=16,
        patch_w=16,
        tubelet=True, 
        embedding_dim=192,
        depth=4,
        num_heads=3,
        head_dim=64,
        num_frames=16,
        n_classes=10,
        mlp_dim=256,
    )
    out = model(test_tensor1)
    assert out.shape == (1, 10)

    out = model(test_tensor2)
    assert out.shape == (3, 10)
    del model
    
    print("pass 1")
    model = MODEL_REGISTRY.get("FactorisedEncoder")(
        img_size=224,
        in_channels=3,
        patch_t=2,
        patch_h=16,
        patch_w=16,
        tubelet=False, 
        embedding_dim=192,
        depth=4,
        num_heads=3,
        head_dim=64,
        num_frames=16,
        n_classes=10,
        mlp_dim=256,
    )

    out = model(test_tensor1)
    assert out.shape == (1, 10)

    out = model(test_tensor2)
    assert out.shape == (3, 10)
    del model
    
    print("pass 2")
    
    model = MODEL_REGISTRY.get("FactorisedSelfAttention")(
        num_frames=32,
        img_size=(64, 64),
        patch_t=8,
        patch_h=4,
        patch_w=4,
        tubelet = True,
        n_classes=10,
        embedding_dim=124,
        depth=3,
        num_heads=4,
        head_dim=32,
        p_dropout=0.0,
        in_channels=3,
        mlp_dim=256,
    )
    test_tensor3 = torch.randn(32, 32, 3, 64, 64)
    logits = model(test_tensor3)
    assert logits.shape == (32, 10)
    del model
    
    print("pass 3")

    model = MODEL_REGISTRY.get("FactorisedSelfAttention")(
        num_frames=16,
        img_size=(64, 64),
        patch_t=8,
        patch_h=4,
        patch_w=4,
        tubelet = False,
        n_classes=10,
        embedding_dim=124,
        depth=3,
        num_heads=4,
        head_dim=32,
        p_dropout=0.0,
        in_channels=1,
        mlp_dim=256,
    )

    test_tensor4 = torch.randn(7, 16, 1, 64, 64)
    logits = model(test_tensor4)
    assert logits.shape == (7, 10)
    
    del model
    
    print("pass 4")
    
    model = MODEL_REGISTRY.get("FactorisedDotProduct")(
        num_frames=16,
        img_size=(64, 64),
        patch_t=8,
        patch_h=4,
        patch_w=4,
        tubelet = False,
        n_classes=10,
        embedding_dim=124,
        depth=3,
        num_heads=4,
        head_dim=32,
        p_dropout=0.0,
        in_channels=1,
        mlp_dim=256,
    )

    test_tensor4 = torch.randn(7, 16, 1, 64, 64)
    logits = model(test_tensor4)
    assert logits.shape == (7, 10)
    
    del model
    
    print("pass 5")

    model = MODEL_REGISTRY.get("FactorisedDotProduct")(
        num_frames=16,
        img_size=(64, 64),
        patch_t=8,
        patch_h=4,
        patch_w=4,
        tubelet = True,
        n_classes=10,
        embedding_dim=124,
        depth=3,
        num_heads=4,
        head_dim=32,
        p_dropout=0.0,
        in_channels=1,
        mlp_dim=256,
    )

    test_tensor4 = torch.randn(7, 16, 1, 64, 64)
    logits = model(test_tensor4)
    assert logits.shape == (7, 10)
    
    del model
    
    print("pass 6")
    
if __name__ == "__main__":
    test_ViViT()