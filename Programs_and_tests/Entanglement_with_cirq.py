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

"""import warnings
# Forza Python a lanciare un'eccezione (crash) appena incontra un UserWarning
warnings.filterwarnings('error', category=UserWarning)"""

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

     # gaia lo chiama "expectation_Z" ma non calcola davvero il valore di aspettazione, 
     # calcola la probabilità che lo stato finale sia |1>
    def expectation_Z(self, counts, shots, n_qubits):
         expects = np.zeros(1)
         for key in counts.keys():
              percentage = counts[key] / shots
              check = np.array([(float(key))*percentage ])
              expects += check
         return expects

    def run(self, thetas):
          # trasformo thetas in una lista di float, per farlo uso:
          # .as_tensor() che trasforma qualunque cosa riceva come input in un tensore PyThorch
          # .detach() scollega il tensore dal grafo (niente warning!)
          # .squeeze() rimuove le dimensioni di batch inutili (es. da [[0.5]] a [0.5])
          # .tolist() lo trasforma in una lista classica di float di Python

          thetas_list= torch.as_tensor(thetas, dtype=torch.float32).detach().squeeze().tolist()

          #creo un vocabolario con cui associo dei parametri PyTorch ai simboli di SyimPy 
          param_resolver = {self.params[k]: thetas_list[k] for k in range(self.n_qubits)}

          #simulazione del circuito per un numero di volte pari a self.shots, usando i parametri risolti
          #ovvero i parametri ai quali viene associato un valore numerico
          result = cirq.Simulator().run(self.circuit, param_resolver=param_resolver, repetitions=self.shots)

          #istogramma dei conteggi
          counts = result.histogram(key='m') #conta quante volte il risultato è stato 0 e quante 1
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
         result = torch.tensor([[expectation_z.item()]], dtype=torch.float32)
         # l'uso di .item() ha lo scopo di alleggerire il codice, un po' come è satato fatto
         # per gradient_tensor usando np.array(). Il fatto che usiamo le doppie parentesi quadre
         # è perchè la crossentropyloss richiede una fromattazione specifica (vedi note Notebook llm)
         return result

     @staticmethod
     def backward(ctx, grad_outputs): # grad_outputs è il gradiente proveniente dai layer classici successivi
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

         # La funzione expectation_Z restituisce degli array NumPy quindi gradients
         # sarà un array di NumPy array ognguno in un punto diverso della memoria. 
         # Questo rallenta il processo, per ottimizzarlo scriviamo np.array(gradients) 
         # così da avere un unico array NumPy. Così facendo, trasformarlo in tensore con 
         # torch.tensor() sarà molto più efficiente.
         gradients_tensor = torch.tensor(np.array(gradients), dtype=torch.float32)

         #Per la Chain Rule moltiplico per il gradiente in uscita dai layer successivi 
         # per i gradienti del layer quantistico
         grad_input = gradients_tensor * grad_outputs

         # Ritorno il gradiente con lo stesso shape dell'input iniziale,
         # e 'None' per gli altri argomenti non addestrabili di forward() (quantum_circuit e shift)
         return grad_input.view_as(input), None, None

class Hybrid(nn.Module):
     def __init__(self, n_qubits, shots, shift):
          super().__init__()
          # creo un istanza della classe QuantumCircuit per poter usare il circuito quantistico
          self.quantum_circuit = QuantumCircuit(n_qubits, shots) 
          self.shift = shift

     def forward(self, input):
          return HybridFunction.apply(input, self.quantum_circuit, self.shift)
          # .apply() crea l'oggetto ctx, nel quale salviamo i parametri 
          # e le vaire istanze (.shift, .quantum_circuit) che servono 
          # per il calcolo dei gradienti, ed esegue il forward

def show_img(X):
    image, label = X
    print(f"Image shape: {image.shape}")
    plt.imshow(image.squeeze(), cmap="gray") # image shape is [1, 28, 28] (colour channels, height, width)  
    plt.title(f"Label: {label} ")
    plt.show()
    print(image.squeeze().shape, image.shape) # the squeeze() function removes the colour channel dimension

# definisco il numero di campioni per classe per l'addestramento
n_samples = 100

# carico il dataset MNIST 
x_train = datasets.MNIST(root='./data', train=True, download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))

# filtro solo le cifre 0 e 1 con un numero di elementi pari a n_samples per ogni classe
# il comando np.where ci da liste di indici  
idx = np.append(
    np.where(x_train.targets == 0)[0][:n_samples],
    np.where(x_train.targets == 1)[0][:n_samples] )
# i dataset supportano la funzione di indexing avanzato, passando un numpy array o un tensore
# verrà interpretato come lista di indici. Così facendo i training data saranno solo quelli
# selezionati
x_train.data= x_train.data[idx]
x_train.targets = x_train.targets[idx]
#show_img(x_train[107])
#print(x_train.targets[101])

#creo il DataLoader per pytorch
# Nota: trovi spiegazioni su questo comando nelle note salvate su Notebook llm
train_loader = torch.utils.data.DataLoader(x_train, batch_size=1, shuffle= True)

#costruisco la rete neurale
class Net(nn.Module):
     def __init__(self, n_qubits, shots, shift):
          super().__init__()
          self.layer_1 = nn.Linear(784,128) #passa 784 pixel a 128 neuroni
          self.layer_2 = nn.Linear(128,64) #riduco ulteriormente le dimensioni a 64 neuroni
          self.layer_3 = nn.Linear(64, n_qubits) #l'ultima riduzione fa si che i parametri
          #di output del secondo layer diventino dello stesso numero dei qubit del circuito quantistico
          self.hybrid = Hybrid(n_qubits, shots, shift)
     def forward(self, x):
          # Appiattisco l'immagine da (1, 28, 28) a (1, 784)
          # Il 728 dice a PyTorch in quante colonne ridistribuire i dati,
          # il -1 gli dice di calcolare da solo quante righe servono per farlo. 
          x = x.view(-1, 784)
          x = F.relu(self.layer_1(x))
          x = F.relu(self.layer_2(x))
          # Non applichiamo la ReLU qui perché gli angoli di rotazione quantistici 
          # devono poter assumere anche valori negativi (da -pi a pi).
          x = self.layer_3(x) 
          # Passaggio nel simulatore quantistico Cirq per estrarre la probabilità P(1)
          x = self.hybrid(x)
          # restituisco il vettore di probabilità bidimensionale (P(1), P(0)), .cat() concatena
          # la probabilità del qubit 0 di collassare in |1> con la sua complementare 1 - P(1)
          return torch.cat((x, 1.0 - x), -1) #il -1 fa si che si ottenga un vettore (p1 p0)
          # orizzontale, se avessi scritto 0 al posto di -1, .cat avrebbe dato un vettore verticale

def training_loop(n_epochs, optim, model, loss_fn, train_loader):
     loss_values = [] #memorizzza la loss media per ogni epoca
     for epoch in range(n_epochs):
          total_loss = []  #memorizzza la loss media per ogni batch
          for batch, (data,target) in enumerate(train_loader):
               # zero grad
               optim.zero_grad()
               #Forward Pass
               output = model(data)
               #calculate the loss
               loss = loss_fn(output, target)
               #Backward Pass
               loss.backward()
               #minimize the loss
               optim.step()

               total_loss.append(loss.item())

          # calcolo la loss media dell'epoca
          avg_loss = sum(total_loss)/len(total_loss)
          loss_values.append(avg_loss)

          #stampo l'avanzamento
          percent_done = 100 * (epoch + 1) / n_epochs
          print(f"Epoch {epoch+1:2d}/{n_epochs} [{percent_done:3.0f}%] ---- Loss Media: {avg_loss:.4f}")

     return loss_values


model = Net(n_qubits, n_shots, shift)
params = list(model.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

model.train()
epochs = 15
loss_list = training_loop(epochs, optimizer, model, loss_func, train_loader)

plt.figure(figsize=(8,5))
plt.plot(loss_list )
plt.xlabel('Epoche')
plt.ylabel('Cross Entropy Loss')
plt.show()

"""C=QuantumCircuit(n_qubits, n_shots)
print(C.circuit)
#print(C.expectation_Z(counts,n_shots, n_qubits))
rotations = torch.tensor([np.pi/4 for i in range(n_qubits)])
#rot = torch.tensor([np.pi / 4] * 3, dtype=torch.float32)
exp = C.run(rotations)

print(f"Valore atteso (probabilità P(1)) per rotazione pi/4: {exp}")"""
