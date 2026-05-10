from mujoco_playground._src import registry
import jax
env = registry.load('ParaNontendonFR3Grasp')
state = env.reset(jax.random.PRNGKey(0))
state = env.step(state, jax.numpy.zeros(env.action_size))
print('OK', state.obs.shape if hasattr(state.obs,'shape') else {k:v.shape for k,v in state.obs.items()})