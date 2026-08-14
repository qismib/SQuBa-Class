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
                    self.circuit.append(cirq.CNOT(self.qubits[n_qubits-2-i], self.qubits[n_qubits-i-1]))
        self.circuit.append(cirq.H(self.qubits[0]))
        
        # Misura sul primo qubit (qubit 0) con chiave 'm'
        self.circuit.append(cirq.measure(self.qubits[0], key='m'))


    def expectation_Z(self, counts, shots, n_qubits):
         expects = np.zeros(1)
         for key in counts.keys():
              percentage = counts[key] / shots
              check = np.array([(float(key))*percentage ])
              expects += check
         return expects

    def run(self, thetas):
         thetas = thetas.squeeze()

         #creo un vocabolario con cui associo dei parametri PyTorch ai simboli di SyimPy 
         param_resolver = {self.params[k]: float(thetas[k].item()) for k in range(self.n_qubits)}

         #simulazione del circuito per un numero di volte pari a self.shots, usando i parametri risolti
         #ovvero i parametri ai quali viene associato un valore numerico
         result = cirq.Simulator().run(self.circuit, param_resolver=param_resolver, repetitions=self.shots)

         #istogramma dei conteggi
         counts = result.histogram(key='m') #conta quante volte il risultato è stato 0 e quante 1
         print(counts)
         #calcolo del valore di aspettazione
         expectation = self.expectation_Z(counts, self.shots, self.n_qubits)
         return expectation


class HybridFunction(Function):
     @staticmethod #Nota: trovi spiegazioni su questo comando nelle note salvate su Notebook llm
     def forward(ctx, input, quantum_circuit, shift):
         # ctx è un oggetto che viene creato ed inserito automaticamente da PyTorch 
         # quando eseguiamo il forward pass. Serve come canale di comunicazione per conservare
         # le informazioni del forward che saranno indispensabili per calcolare il gradiente nel backward.
         # Salvo nel contesto 'ctx' i parametri necessari per il calcolo dei gradienti (backward)
         ctx.shift = shift
         ctx.quantum_circuit = quantum_circuit

         #salvo l'input come tensore PyTorch per poterlo usare nel backward pass
         ctx.save_for_backward(input) #è un metodo integrato dell'oggetto ctx, fornito nativamente da Autograd

         #eseguo il circuito quantistico di Cirq passando gli angoli e calcolo il valore di aspettazione
         expectation_z = ctx.quantum_circuit.run(input)

         #converto il risultato ottenuto in un tensore
         result = torch.tensor([expectation_z], dtype=torch.float32)

         return result

     @staticmethod
     def backward(ctx, grad_output): # grad_output è il gradiente proveniente dai layer classici successivi
         # calcolo i gradienti del circuito quantitico tramitre la Parameter-Shift Rule
         input, = ctx.saved_tensors #recupero l'input salvato nel forward pass
         input_list = input.squeeze().tolist()  #converto il tensore in una lista di float

         gradients = []

         #per ciascun qubit applico lo shift a destra (+shift) e sinistra (-shift)
         for k in range(len(input_list)):
              shift_right = list(input_list)
              shift_right[k] += ctx.shift
              expectation_right = ctx.quantum_circuit.run(shift_right)

              shift_left = list(input_list)
              shift_left[k] -= ctx.shift
              expectation_left = ctx.quantum_circuit.run(shift_left)

              gradients.append((expectation_right - expectation_left)/2.0)

         #converto i gradienti in un tensore pytorch
         gradients_tensor = torch.tensor(gradients, dtype=torch.float32)

         #Per la Chain Rule moltiplico per il gradiente in uscita dai layer successivi 
         # per i gradienti del layer quantistico
         grad_input = gradients_tensor * grad_output

         # Ritorno il gradiente con lo stesso shape dell'input iniziale,
         # e 'None' per gli altri argomenti non addestrabili di forward() (quantum_circuit e shift)
         return grad_input.view_as(input), None, None



C=QuantumCircuit(n_qubits, n_shots)
print(C.circuit)
#print(C.expectation_Z(counts,n_shots, n_qubits))
rotations = torch.tensor([np.pi/4 for i in range(n_qubits)])
#rot = torch.tensor([np.pi / 4] * 3, dtype=torch.float32)
exp = C.run(rotations)

print(f"Valore atteso (probabilità P(1)) per rotazione pi/4: {exp}")
