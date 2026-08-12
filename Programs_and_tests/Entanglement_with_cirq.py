import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torchvision import datasets, transforms
import itertools


# Cirq 
import cirq
import sympy #serve per creare simboli che rappresentano parametri variabili nei circuiti quantistici


# Variabili globali
n_qubits = 3
n_shots = 1000
shift = np.pi / 2
learning_rate = 0.01

class QuantumCircuit:
    def __init__(self, n_qubits, shots):
        self.n_qubits= n_qubits
        self.shots = shots

        #Definisco i qubit e il circuito
        self.qubits= cirq.LineQubit.range(n_qubits)
        self.circuit = cirq.Circuit()

        #definisco i parametri variazionali usando i simboli di sympy
        self.params = [sympy.Symbol(f'p{i}') for i in range(n_qubits)]

         # Applico Hadamard su q0 e CNOT a cascata
        self.circuit.append(cirq.H(self.qubits[0]))
        for i in range(n_qubits-1):
            self.circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i+1]))
            print(i)

         # Applico Rz su ogni qubit usando i parametri SymPy
         # in cirq uso cirq.rz(angolo di rotazione)(qubit che voglio ruotare)
        for k in range(self.n_qubits):
            self.circuit.append(cirq.rz(self.params[k])(self.qubits[k]))

        # Uncomputation dell'entanglement
        for i in range(n_qubits-1):
                    self.circuit.append(cirq.CNOT(self.qubits[n_qubits-1-i], self.qubits[n_qubits-i-2]))
        self.circuit.append(cirq.H(self.qubits[0]))
        
        # Misura sul primo qubit (qubit 0) con chiave 'm'
        self.circuit.append(cirq.measure(self.qubits, key='m'))

        

    def print_circuit(self):
        print("Circuit:")
        print(self.circuit)

C=QuantumCircuit(n_qubits, n_shots)
C.print_circuit()
