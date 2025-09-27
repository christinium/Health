import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

# Load and preprocess the Iris dataset
data = load_iris()
X = data['data']
y = data['target'].reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a 2-layer feedforward neural network using Flax
class FeedForwardNN(nn.Module):
    hidden_dim: int = 32
    output_dim: int = 3

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

# Create a training state
def create_train_state(rng, learning_rate, model):
    params = model.init(rng, jnp.ones([1, 4]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Define the loss function
def cross_entropy_loss(logits, labels):
    return optax.softmax_cross_entropy(logits, labels).mean()

def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    return {'loss': loss, 'accuracy': accuracy}

# Training step
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['X'])
        loss = cross_entropy_loss(logits, batch['y'])
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch['y'])
    return state, metrics

# Evaluation step
def eval_step(state, batch):
    logits = state.apply_fn({'params': state.params}, batch['X'])
    return compute_metrics(logits, batch['y'])

# Training loop
def train_model():
    rng = jax.random.PRNGKey(0)
    model = FeedForwardNN()
    state = create_train_state(rng, 0.001, model)
    num_epochs = 100
    batch_size = 32
    num_train = X_train.shape[0]
    for epoch in range(num_epochs):
        perm = np.random.permutation(num_train)
        for i in range(0, num_train, batch_size):
            idx = perm[i:i+batch_size]
            batch = {'X': jnp.array(X_train[idx]), 'y': jnp.array(y_train[idx])}
            state, metrics = train_step(state, batch)
        if (epoch + 1) % 10 == 0:
            test_batch = {'X': jnp.array(X_test), 'y': jnp.array(y_test)}
            test_metrics = eval_step(state, test_batch)
            print(f"Epoch {epoch+1}, Test Loss: {test_metrics['loss']:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}")
    return state

if __name__ == "__main__":
    train_model()
