import pytest
import torch
import os
import sys

# Add the project root to the Python path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_import_mamba_and_samplefactory():
    """Test that we can import mamba_ssm and sample_factory together."""
    try:
        import mamba_ssm
        import sample_factory
        # If we get here, both imports succeeded
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")

def test_mamba_basic_usage():
    """Test basic usage of mamba_ssm."""
    from mamba_ssm import Mamba
    
    # Create a simple Mamba model
    model = Mamba(
        d_model=16,  # Model dimension
        d_state=8,   # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Expansion factor
    )
    
    # Check that the model is an instance of Mamba
    assert isinstance(model, Mamba)

def test_samplefactory_basic_usage():
    """Test basic usage of sample_factory."""
    import sample_factory
    from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
    from sample_factory.envs.env_utils import register_env
    from sample_factory.model.encoder import Encoder
    from sample_factory.model.decoder import Decoder
    from sample_factory.model.actor_critic import ActorCritic
    
    # Check that we can parse arguments (this is a light-weight test)
    # We don't actually run training, just check that the imports work
    # and we can create a basic config.
    try:
        # This is just to see if the module loads correctly
        assert hasattr(sample_factory, '__file__')
    except Exception as e:
        pytest.fail(f"Samplefactory basic usage failed: {e}")