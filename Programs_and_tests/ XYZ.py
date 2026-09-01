import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torchvision import datasets, transforms
import itertools
from collections import Counter
import time

# Cirq 
import cirq
from cirq import Gate
import sympy


from typing import Union


#importo i gates per i qudit di dimensione d
#(la dimensione d è impostata di default uguale a 3)
from my_gates import H_d, X_P_ij, Y_P_ij, Z_P_ij

# Variabili globali
global_dimension = 3
global_shots = 1000
shift = np.pi / 2
learning_rate = 0.01
epochs = 15

class QuantumCircuit:
    def __init__(self, dimension =3, shots = 500):
        self.dimension= dimension
        self.shots = shots

        # Definisco il quidit e il circuito
        self.qudit= cirq.LineQid(0,dimension =self.dimension)
        self.circuit = cirq.Circuit()

        # Definisco i parametri variazionali usando i simboli di sympy.
        self.params = [sympy.Symbol(f'p{i}') for i in range(self.dimension**2)]

        # Definisco il curcuito
      
        self.circuit.append(X_P_ij(0,1,self.params[0], 3).on(self.qudit))
        self.circuit.append(Y_P_ij(0,1,self.params[1], 3).on(self.qudit))
        self.circuit.append(Z_P_ij(0,1,self.params[2], 3).on(self.qudit))
        self.circuit.append(X_P_ij(1,2,self.params[3], 3).on(self.qudit))
        self.circuit.append(Y_P_ij(1,2,self.params[4], 3).on(self.qudit))
        self.circuit.append(Z_P_ij(1,2,self.params[5], 3).on(self.qudit))
        self.circuit.append(X_P_ij(1,0,self.params[6], 3).on(self.qudit))
        self.circuit.append(Y_P_ij(1,0,self.params[7], 3).on(self.qudit))
        self.circuit.append(Z_P_ij(1,0,self.params[8], 3).on(self.qudit))
        

        # Misuro il qudit 0 con chiave 'm'
        self.circuit.append(cirq.measure(self.qudit, key='m'))
        print(self.circuit)
        
        # Calcolo le probabilità che qudit=|0>,|1>,|2> ...
    def calc_prob(self, counts, shots, dimension):
        self.vocabulary_of_values_and_probabilities = {}
        vector_probabilities = [0.0]*dimension # fondamentale, se nel circuito non venisse misurato un parametro non verrebbe passato, la dimensione del vocabolario sarebbe inferiore e darebbe errore
        for key in counts.keys():
            if key < dimension:
                vector_probabilities[key] = counts[key] / shots
            self.vocabulary_of_values_and_probabilities[key] =  counts[key] / shots
        return vector_probabilities

    def run(self, thetas):

            thetas_list= torch.as_tensor(thetas, dtype=torch.float32).detach().squeeze().tolist()

            # creo un vocabolario con cui associo dei parametri PyTorch ai simboli di SyimPy
            param_resolver = {self.params[k]: thetas_list[k] for k in range(self.dimension**2) }

            # simulazione del circuito per un numero di volte pari a self.shots, usando i parametri risolti
            result = cirq.Simulator().run(self.circuit, param_resolver=param_resolver, repetitions=self.shots)

            #conto dei risultati
            counts = Counter(result.measurements['m'][:,0]) #conta quante volte il risultato è stato 0, 1 e 2
            counts_ordinati = {int(k): v for k, v in sorted(counts.items())} #ordino e pulisco l'output
            #calcolo della probabilità
            probability = self.calc_prob(counts_ordinati, self.shots, self.dimension)
            return probability
    def view_val_prob(self):
        return self.vocabulary_of_values_and_probabilities

class HybridFunction(Function):
     @staticmethod 
     def forward(ctx, input, quantum_circuit, shift, dimension):
        # ctx è un oggetto che viene creato ed inserito automaticamente da PyTorch 
        # quando eseguiamo il forward pass. 
        ctx.quantum_circuit = quantum_circuit
        ctx.dimension = dimension   
         
        # salvo l'input come tensore PyTorch per poterlo usare nel backward pass
        ctx.save_for_backward(input) 
         
        # eseguo il circuito quantistico di Cirq passando gli angoli e calcolo la probabilità
        vector_prob = ctx.quantum_circuit.run(input)
             
        # converto il risultato ottenuto in un tensore
        result = torch.tensor([vector_prob], dtype=torch.float32)
        # Il fatto che usiamo una parentesi quadra oltre quella di vector_prob
        # è perchè la loss function richiede una formattazione spefica
        return result
         
     @staticmethod
     def backward(ctx, grad_outputs): # grad_outputs è il gradiente proveniente dai layer classici successivi
        # calcolo i gradienti del circuito quantitico tramitre la Parameter-Shift Rule
        input, = ctx.saved_tensors #recupero l'input salvato nel forward pass
        input_list = input.squeeze().tolist()  #converto il tensore in una lista di float
        dimension = ctx.dimension

        gradients = []

        # Per ciascun parametro applico lo shift a destra (+shift) e sinistra (-shift).
        for k in range(len(input_list)):
            shift_right = list(input_list)
            shift_right[k] += ctx.shift
            expectation_right = ctx.quantum_circuit.run(shift_right)

            shift_left = list(input_list)
            shift_left[k] -= ctx.shift
            expectation_left = ctx.quantum_circuit.run(shift_left)
            
            for i in range(dimension): 
                gradients.append((expectation_right[i] - expectation_left[i])/2.0)

        gradients_tensor = torch.tensor(np.array(gradients), dtype=torch.float32)

        #Per la Chain Rule moltiplico il gradiente in uscita dai layer successivi 
        # per la matrice jacobiana del layer quantistico
        Jacobian_matrix =  gradients_tensor.reshape(len(input_list), dimension)
        grad_input_matrix = Jacobian_matrix* grad_outputs.squeeze()
        grad_input = torch.sum(grad_input_matrix, -1) # Il -1 fa si che la somma sia solo per gli elementi delle righe.
        
        # Ritorno il gradiente con lo stesso shape dell'input iniziale,
        # e 'None' per gli altri argomenti non addestrabili di forward() (quantum_circuit, shift e dimension)
        return grad_input.view_as(input), None, None, None

class Hybrid(nn.Module):
     def __init__(self, dimension, shots, shift):
          super().__init__()
          # creo un istanza della classe QuantumCircuit per poter usare il circuito quantistico
          self.dimension = dimension
          self.quantum_circuit = QuantumCircuit(self.dimension, shots) 
          self.shift = shift
          

     def forward(self, input):
          return HybridFunction.apply(input, self.quantum_circuit, self.shift, self.dimension)
          # .apply() crea l'oggetto ctx
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

# filtro solo le cifre 0, 1 e 2 con un numero di elementi pari a n_samples per ogni classe
# il comando np.where ci da liste di indici  
idx = np.concatenate([
    np.where(x_train.targets == 0)[0][:n_samples],
    np.where(x_train.targets == 1)[0][:n_samples],
    np.where(x_train.targets == 2)[0][:n_samples]
    ])

x_train.data= x_train.data[idx]
x_train.targets = x_train.targets[idx]


#creo il DataLoader per pytorch
train_loader = torch.utils.data.DataLoader(x_train, batch_size=1, shuffle= True)

#costruisco la rete neurale
class Net(nn.Module):
     def __init__(self, dimension, shots, shift):
          super().__init__()
          self.layer_1 = nn.Linear(784,128) #passa 784 pixel a 128 neuroni
          self.layer_2 = nn.Linear(128,64) #riduco ulteriormente le dimensioni a 64 neuroni
          self.layer_3 = nn.Linear(64, 9) #l'ultima riduzione fa si che i parametri di output del secondo layer diventino dello stesso numero dei parametri del circuito
          self.hybrid = Hybrid(global_dimension,shots, shift)
     def forward(self, x):
          # Appiattisco l'immagine da (1, 28, 28) a (1, 784)
          x = x.view(-1, 784)
          x = F.relu(self.layer_1(x))
          x = F.relu(self.layer_2(x))
          # Non applico la ReLU qui perché gli angoli di rotazione quantistici devono poter assumere anche valori negativi (da -pi a pi).
          x = self.layer_3(x) 
          # Passaggio nel simulatore quantistico Cirq 
          x = self.hybrid(x)
          # restituisco il vettore di probabilità (P(0), P(1) ...)
          return x 

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


# Fisso il seed di pytorch per introdurre riproducibilità
torch.manual_seed(42) 

avg_time_perf = []
tot_time_perf = []
acc_perf = []

number_of_test=10

# costruisco un ciclo per testare più volte l'addestramento della rete neurale
for i in range(number_of_test):
    model = Net(global_dimension, global_shots, shift)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()


    model.train()
    epoch_times = []
    loss_list = training_loop(epochs, optimizer, model, loss_func, train_loader)
    average_epoch_times = sum(epoch_times)/epochs
    print(f"\n=================== TIME PERFORMANCE {i} ===================")
    print(f"Total training time:{sum(epoch_times):.2f} --- Average time per epoch: {average_epoch_times:.2f}")
    print("==========================================================\n")
    avg_time_perf.append(average_epoch_times)
    tot_time_perf.append(sum(epoch_times))

    #plt.figure(figsize=(8,5))
    #plt.plot(loss_list )
    #plt.xlabel('Epoche')
    #plt.ylabel('Cross Entropy Loss')
    #plt.show()


    # VALIDAZIONE DEL MODELLO

    #raccolgo 1000 immagini per ogni classe (0, 1)
    n_val_samples = 1000

    x_test = datasets.MNIST(root='./data', train=False, download=True,
                            transform=transforms.Compose([transforms.ToTensor()]))

    idx_test = np.concatenate([
                            np.where(x_test.targets == 0)[0][:n_val_samples],
                            np.where(x_test.targets == 1)[0][:n_val_samples],
                            np.where(x_test.targets == 2)[0][:n_val_samples]
                            ])
    x_test.data = x_test.data[idx_test]
    x_test.targets = x_test.targets[idx_test]

    #creo il dataloader con shuffle=False dato che non abbiamo bisogno di mescolare i dati di test
    test_loader = torch.utils.data.DataLoader(x_test, batch_size=1, shuffle=False)

    def validate(model, test_loader, loss_func):
        model.eval()# 1. Disattiva Dropout e imposta la rete in modalità test
        test_loss = 0
        correct = 0
        with torch.no_grad(): #disattivo il calcolo dei gradienti
            for data, target in test_loader:
                output=model(data)
                loss = loss_func(output, target)
                test_loss += loss.item()

                #calcolo la predizione, la probabilità più alta
                pred = output.argmax(dim=1, keepdim=True)
                #confrontiamola con la label reale
                correct += pred.eq(target.view_as(pred)).sum().item()
        #calcolo la loss media e l'accuracy 
        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f"\n=================== VALIDATION RESULTS {i} ===================")
        print(f"Loss Media sul Validation Set: {test_loss:.4f}")
        print(f"Accuratezza Finale: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
        print("==========================================================\n")
        acc_perf.append(accuracy)
        return test_loss, accuracy

    validate(model, test_loader,loss_func)

for i in range(number_of_test):
    print(f"performance of test number {i}: total train time={tot_time_perf[i]:.2f} s --- average time per epoch={avg_time_perf[i]:.2f} s --- {acc_perf[i]:.2f}%")


print('tot,time_vec', tot_time_perf )
print('avg_time_vec', avg_time_perf )
print('acc_vec', acc_perf )




