import jax
import jax.numpy as jnp
print("jax", jax.__version__)
devs = jax.devices()
print("devices:", [d.device_kind for d in devs])
gpus = [d for d in devs if "AMD Instinct" in d.device_kind]
assert gpus, f"JAX has no ROCm GPU! devices={devs}"
a = jax.random.normal(jax.random.PRNGKey(0), (2048, 2048))
c = float((a @ a).sum())
print("matmul 2048^3 ok, checksum:", round(c, 1))
print("JAX: PASS")
