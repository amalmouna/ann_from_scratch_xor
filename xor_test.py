import numpy as np
from network import Network
from layers import FullyConnected
from activations import tanh, d_tanh
from loss import mse, d_mse

class TanhActivation:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = tanh(x)
        return self.output

    def backward(self, grad_output, lr):
        return grad_output * d_tanh(self.output)

# Données XOR (-1 pour 0, 1 pour 1)
X = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]], dtype=np.float32)
Y = np.array([[[-1]], [[1]], [[1]], [[-1]]], dtype=np.float32)

# Création du réseau
net = Network()
net.add(FullyConnected(2, 4))
net.add(TanhActivation())
net.add(FullyConnected(4, 1))
net.add(TanhActivation())

# Configuration de la perte
net.use(mse, d_mse)

# Entraînement
print("Début de l'entraînement...")
net.train(X, Y, epochs=20000, learning_rate=0.1)

# Test
print("\nRésultats après entraînement:")
for i in range(len(X)):
    pred = net.predict(X[i])
    print(f"Input: {X[i][0]} -> Prédiction: {pred[0][0]:.4f} (Attendu: {Y[i][0][0]})")
import matplotlib.pyplot as plt
plt.plot(net.loss_history)
plt.title("Courbe d'apprentissage")
plt.show()