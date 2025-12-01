from flax import linen as nn
import jax.numpy as jnp

class TaskEmbedding(nn.Module):
    num_tasks: int
    embedding_size: int

    def setup(self):
        self.embeddings = nn.Embed(self.num_tasks, self.embedding_size)

    def __call__(self, x: jnp.ndarray):
        emb = self.embeddings(x)
        norm = jnp.linalg.norm(emb, axis=-1, keepdims=True)
        emb = emb / norm
        return emb