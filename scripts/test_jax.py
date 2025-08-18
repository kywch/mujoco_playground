import os
import jax
import jax.numpy as jnp

try:
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    gpu_devices = [d for d in jax.devices() if d.platform == 'gpu']
    if not gpu_devices:
        print("No GPU devices found by JAX!")
    else:
        print(f"JAX GPU devices: {gpu_devices}")

    try:
        from jax.extend.backend import get_backend
        print(f"XLA backend: {get_backend().platform}")
        print(f"XLA backend version: {get_backend().platform_version}")
    except Exception as e:
        print(f"XLA backend error: {e}")

    # Simple GPU computation
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1000, 1000))
    y = jnp.dot(x, x)
    y.block_until_ready() # Ensure computation completes
    print("JAX GPU test successful: Able to perform a dot product on GPU.")

except Exception as e:
    print(f"JAX GPU test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2
# https://github.com/jax-ml/jax/issues/8916#issuecomment-2703894042

key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (3, 3))

# Perform a Cholesky decomposition (uses cuSOLVER)
L = jnp.linalg.cholesky(A @ A.T) # Ensure A is positive definite
print("Cholesky decomposition:\n", L)