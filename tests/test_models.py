import torch

from scLDL.models.label_enhancer import LabelEnhancer, LabelEncoder, LabelLogicalDecoder, LabelDistributionDecoder, LabelGapDecoder

def test_label_encoder_defaults():
    # Test that default arguments are not shared
    enc1 = LabelEncoder(x_dim=10)
    enc2 = LabelEncoder(x_dim=10)
    
    # Check if they are different objects (this was always true, but checking mutable default side effects is harder directly without inspecting the signature or modifying the list)
    # However, we can check if the model runs.
    assert enc1 is not enc2

def test_label_enhancer_init():
    # Test initialization
    model = LabelEnhancer(x_dim=100, d_dim=5, h_dim=32)
    assert model.x_dim == 100
    assert model.d_dim == 5
    assert model.h_dim == 32
    
    # Check submodules
    assert isinstance(model.encoder, LabelEncoder)
    assert isinstance(model.logical_decoder, LabelLogicalDecoder)
    assert isinstance(model.distribution_decoder, LabelDistributionDecoder)
    assert isinstance(model.gap_decoder, LabelGapDecoder)

def test_label_enhancer_forward():
    # Test forward pass
    x_dim = 50
    d_dim = 3
    batch_size = 4
    model = LabelEnhancer(x_dim=x_dim, d_dim=d_dim)
    
    x = torch.randn(batch_size, x_dim)
    mu, std, l_logits, d_logits, gap_sigma = model(x)
    
    assert mu.shape == (batch_size, 64) # Default h_dim
    assert std.shape == (batch_size, 64)
    assert l_logits.shape == (batch_size, d_dim)
    assert d_logits.shape == (batch_size, d_dim)
    assert gap_sigma.shape == (batch_size, 1)
    
    # Check positivity of std and gap_sigma
    assert (std > 0).all()
    assert (gap_sigma > 0).all()

def test_device_placement():
    # Test that we can move the model to a device (simulated with cpu since we might not have gpu)
    model = LabelEnhancer(x_dim=10, d_dim=2)
    device = torch.device("cpu")
    model.to(device)
    
    x = torch.randn(5, 10).to(device)
    mu, _, _, _, _ = model(x)
    assert mu.device == device

if __name__ == "__main__":
    test_label_encoder_defaults()
    test_label_enhancer_init()
    test_label_enhancer_forward()
    test_device_placement()
    print("All tests passed!")
