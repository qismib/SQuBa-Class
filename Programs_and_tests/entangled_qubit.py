import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torchvision import datasets, transforms
import itertools
import time

# Cirq 
import cirq
import sympy #serve per creare simboli che rappresentano parametri variabili nei circuiti quantistici


# Variabili globali
n_qubits = 3
n_shots = 1000
shift = np.pi / 2
learning_rate = 0.0005

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
        for i in range(self.n_qubits-1):
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
        print(self.circuit)
 
    def expectation_Z(self, counts, shots, n_qubits):
         expects = np.zeros(1)
         for key in counts.keys():
              percentage = counts[key] / shots
              check = np.array([(float(key))*percentage ])
              expects += check
         return expects

    def run(self, thetas):

          thetas_list= torch.as_tensor(thetas, dtype=torch.float32).detach().squeeze().tolist()

          #creo un vocabolario con cui associo dei parametri PyTorch ai simboli di SyimPy 
          param_resolver = {self.params[k]: thetas_list[k] for k in range(self.n_qubits)}

          #simulazione del circuito per un numero di volte pari a self.shots
          result = cirq.Simulator().run(self.circuit, param_resolver=param_resolver, repetitions=self.shots)

          #istogramma dei conteggi
          counts = result.histogram(key='m') #conta quante volte il risultato è stato 0 e quante 1
          #calcolo del valore di aspettazione
          expectation = self.expectation_Z(counts, self.shots, self.n_qubits)
          return expectation


class HybridFunction(Function):
     @staticmethod
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

         return result

     @staticmethod
     def backward(ctx, grad_outputs):
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
    plt.imshow(image.squeeze(), cmap="gray") # la dimensione dell'immagine è [1, 28, 28] (colour channels, height, width)  
    plt.title(f"Label: {label} ")
    plt.show()
    print(image.squeeze().shape, image.shape) # la funzione squeeze() rimuove la dimensione del colour channels

# definisco il numero di campioni per classe per l'addestramento
n_samples = 100

# carico il dataset MNIST 
x_train = datasets.MNIST(root='./data', train=True, download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))

# filtro solo le cifre 0 e 1 con un numero di elementi pari a n_samples per ogni classe 
idx = np.append(
    np.where(x_train.targets == 0)[0][:n_samples],
    np.where(x_train.targets == 1)[0][:n_samples] )

x_train.data= x_train.data[idx]
x_train.targets = x_train.targets[idx]


#creo il DataLoader per pytorch

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
          return torch.cat(( 1.0 - x, x), -1) #il -1 fa si che si ottenga un vettore (p1 p0)
          # orizzontale, se avessi scritto 0 al posto di -1, .cat avrebbe dato un vettore verticale

def training_loop(n_epochs, optim, model, loss_fn, train_loader):
     loss_values = [] #memorizzza la loss media per ogni epoca
     for epoch in range(n_epochs):

          start_time = time.perf_counter()  # 1. Registra il tempo di inizio epoca

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

          end_time = time.perf_counter()    # Registra il tempo di fine epoca
          elapsed_time = end_time - start_time # Tempo totale dell'epoca in secondi
          epoch_times.append(elapsed_time)

          # calcolo la loss media dell'epoca
          avg_loss = sum(total_loss)/len(total_loss)
          loss_values.append(avg_loss)

          #stampo l'avanzamento
          percent_done = 100 * (epoch + 1) / n_epochs
          print(f"Epoch {epoch+1:2d}/{n_epochs} [{percent_done:3.0f}%] ---- Loss Media: {avg_loss:.4f} ---- Time:{elapsed_time:.2f}")

     return loss_values


#Fisso un seed manualmente per abilitare la riproducibilità 
torch.manual_seed(42) 

avg_time_perf = []
tot_time_perf = []
acc_perf = []

number_of_test=10

for i in range(number_of_test):

     model = Net(n_qubits, n_shots, shift)
     params = list(model.parameters())
     optimizer = torch.optim.Adam(params, lr=learning_rate)
     loss_func = nn.CrossEntropyLoss()

     model.train()
     epochs = 15
     epoch_times = []
     loss_list = training_loop(epochs, optimizer, model, loss_func, train_loader)
     average_epoch_times = sum(epoch_times)/epochs
     print(f"\n=================== TIME PERFORMANCE {i} ===================")
     print(f"Total training time:{sum(epoch_times):.2f} --- Average time per epoch: {average_epoch_times:.2f}")
     print("==========================================================\n")
     avg_time_perf.append(average_epoch_times)
     tot_time_perf.append(sum(epoch_times))

     # VALIDAZIONE DEL MODELLO

     #raccolgo 1000 immagini per ogni classe (0, 1)
     n_val_samples = 1000

     x_test = datasets.MNIST(root='./data', train=False, download=True,
                         transform=transforms.Compose([transforms.ToTensor()]))

     idx_test = np.append(np.where(x_test.targets == 0)[0][:n_val_samples],
                         np.where(x_test.targets == 1)[0][:n_val_samples] 
                         )
     x_test.data = x_test.data[idx_test]
     x_test.targets = x_test.targets[idx_test]

     #creo il dataloader con shuffle=False dato che non abbiamo bisogno di mescolare i dati di test
     test_loader = torch.utils.data.DataLoader(x_test, batch_size=1, shuffle=False)

     def validate(model, test_loader, loss_func):
          model.eval()# Disattiva Dropout e imposta la rete in modalità test
          test_loss = 0
          correct = 0
          with torch.no_grad(): #disattivo il calcolo dei gradienti
               for data, target in test_loader:
                    output=model(data)
                    loss = loss_func(output, target)
                    test_loss += loss.item()

                    #calcolo la predizione, la probabilità più alta che il qubit sia 0 o 1
                    pred = output.argmax(dim=1, keepdim=True)
                    #confrontiamola con la label reale
                    correct += pred.eq(target.view_as(pred)).sum().item()
          #calcoliamo la loss media e l'accuracy 
          test_loss /= len(test_loader)
          accuracy = 100. * correct / len(test_loader.dataset)
          print("\n=================== VALIDATION RESULTS ===================")
          print(f"Loss Media sul Validation Set: {test_loss:.4f}")
          print(f"Accuratezza Finale: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
          print("==========================================================\n")
          acc_perf.append(accuracy)
          return test_loss, accuracy

     validate(model, test_loader,loss_func)

for i in range(number_of_test):
    print(f"performance of test number {i}: total train time={tot_time_perf[i]:.2f} s --- average time per epoch={avg_time_perf[i]:.2f} s --- {acc_perf[i]:.2f}%")


#testo formattato per essere inserito in un file scritto in latex

print('tot,time_vec-------------------------------------------------------------')
print(" & ".join(f"{x:.0f}" for x in tot_time_perf))
print('avg_per epoch time_vec---------------------------------------------------'  )
print(" & ".join(f"{x:.2f}" for x in avg_time_perf))
print('acc_vec------------------------------------------------------------------'  )
print(" & ".join(f"{x:.2f}" for x in acc_perf))


