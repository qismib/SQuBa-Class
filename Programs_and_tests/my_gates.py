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


# Cirq 
import cirq
from cirq import Gate
import sympy #serve per creare simboli che rappresentano parametri variabili nei circuiti quantistici


from typing import Union, Tuple


# rotation around Z for the ij couple
class Z_P_ij(Gate):
    def __init__(self, i, j ,phase : Union[float, sympy.Symbol] ,dimension=3,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.i = i
        self.j = j
        self.d= dimension
        self.phase = phase

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        matrix = np.eye(self.d, dtype=complex) 
        matrix[self.i][self.i]= np.exp(-1j*self.phase/2)
        matrix[self.j][self.j]= np.exp(1j*self.phase/2)
        return matrix

    def _circuit_diagram_info_(self, args):
        return f"Z({self.phase})_{self.i}{self.j}"
        #return f"Z({'Theta'})_{self.i}{self.j}"
    # ─── AGGIUNGO QUESTI METODI PER INSEGNARE A CIRQ COME GESTIRE I SIMBOLI ───
    def _is_parameterized_(self) -> bool:
        """Dice a Cirq se il gate contiene un parametro simbolico."""
        return isinstance(self.phase, sympy.Basic)

    def _parameter_names_(self):
        """Fornisce i nomi dei parametri simbolici nel gate."""
        if isinstance(self.phase, sympy.Symbol):
            return (self.phase.name,)
        elif isinstance(self.phase, sympy.Basic):
            return tuple(str(param) for param in self.phase.free_symbols)
        return ()

    def _resolve_parameters_(self, resolver, recursive):
        """Restituisce un nuovo gate Z_P_ij sostituendo il simbolo con il float risolto."""
        resolved_phase = resolver.value_of(self.phase, recursive)
        return Z_P_ij(self.i, self.j, dimension=self.d, phase=resolved_phase)
# rotation around Y for the ij couple
class Y_P_ij(Gate):

    def __init__(self, i, j, phase : Union[float, sympy.Symbol] , dimension=3,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.i = i
        self.j = j
        self.d= dimension
        self.phase = phase

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        matrix = np.eye(self.d, dtype=complex)
        matrix[self.i][self.i] = matrix[self.j][self.j] = np.cos(self.phase / 2)
        matrix[self.j][self.i] = np.sin(self.phase / 2)
        matrix[self.i][self.j] = -np.sin(self.phase / 2)
        return matrix

    def _circuit_diagram_info_(self, args):
        return f"Y({self.phase})_{self.i}{self.j}"
# ─── AGGIUNGO QUESTI METODI PER INSEGNARE A CIRQ COME GESTIRE I SIMBOLI ───
    def _is_parameterized_(self) -> bool:
        """Dice a Cirq se il gate contiene un parametro simbolico."""
        return isinstance(self.phase, sympy.Basic)

    def _parameter_names_(self):
        """Fornisce i nomi dei parametri simbolici nel gate."""
        if isinstance(self.phase, sympy.Symbol):
            return (self.phase.name,)
        elif isinstance(self.phase, sympy.Basic):
            return tuple(str(param) for param in self.phase.free_symbols)
        return ()

    def _resolve_parameters_(self, resolver, recursive):
        """Restituisce un nuovo gate Y_P_ij sostituendo il simbolo con il float risolto."""
        resolved_phase = resolver.value_of(self.phase, recursive)
        return Y_P_ij(self.i, self.j, dimension=self.d, phase=resolved_phase)


# rotation around X for the ij couple
class X_P_ij(Gate):

    def __init__(self, i, j, phase : Union[float, sympy.Symbol] , dimension=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.i = i
        self.j = j
        self.d = dimension
        self.phase = phase

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        matrix = np.eye(self.d, dtype=complex)
        matrix[self.i][self.i] = matrix[self.j][self.j] = np.cos(self.phase / 2)
        matrix[self.j][self.i] = matrix[self.i][self.j] = -1j * np.sin(self.phase / 2)
        return matrix

    def _circuit_diagram_info_(self, args):
        return f"X({self.phase})_{self.i}{self.j}"
# ─── AGGIUNGO QUESTI METODI PER INSEGNARE A CIRQ COME GESTIRE I SIMBOLI ───
    def _is_parameterized_(self) -> bool:
        """Dice a Cirq se il gate contiene un parametro simbolico."""
        return isinstance(self.phase, sympy.Basic)

    def _parameter_names_(self):
        """Fornisce i nomi dei parametri simbolici nel gate."""
        if isinstance(self.phase, sympy.Symbol):
            return (self.phase.name,)
        elif isinstance(self.phase, sympy.Basic):
            return tuple(str(param) for param in self.phase.free_symbols)
        return ()

    def _resolve_parameters_(self, resolver, recursive):
        """Restituisce un nuovo gate X_P_ij sostituendo il simbolo con il float risolto."""
        resolved_phase = resolver.value_of(self.phase, recursive)
        return X_P_ij(self.i, self.j, dimension=self.d, phase=resolved_phase)

    
# Hadamard with dimension d
class H_d(Gate):
    def __init__(self,  dimension=3, *args, **kwargs):
        """Init and save attributes."""
        super().__init__(*args, **kwargs)
        self.d = dimension
    
    def _qid_shape_(self):
        return (self.d,)
    
    def _unitary_(self):
        w = np.exp(1j*2*np.pi/self.d)
        matrix = np.zeros((self.d,self.d), dtype=complex)
        for i in range(self.d):
            for j in range(self.d):
                if i == 0:
                    matrix[i][j]=1
                else:
                    matrix[i][j] = w**(j*(self.d-i))
        return matrix*(1/np.sqrt(self.d))
    
    def _circuit_diagram_info_(self, args):
        return f"H_d"


# rotation for the ij couple
class R_ij(Gate):

    def __init__(self, i, j, theta : Union[float, sympy.Symbol], phi : Union[float, sympy.Symbol], dimension=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.i = i
        self.j = j
        self.d = dimension
        self.theta = theta
        self.phi = phi

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        matrix = np.eye(self.d, dtype=complex)
        matrix[self.i][self.i] = matrix[self.j][self.j] = np.cos(self.theta/2 )
        matrix[self.i][self.j] = -1j * np.sin(self.theta / 2)*np.exp(1j*self.phi)
        matrix[self.j][self.i] = -1j * np.sin(self.theta / 2)*np.exp(-1j*self.phi)
        return matrix

    def _circuit_diagram_info_(self, args):
        return f"R({self.theta, self.phi})_{self.i}{self.j}"
    # ─── METODI DI RISOLUZIONE DEI PARAMETRI AGGIORNATI PER 2 PARAMETRI ───

    def _is_parameterized_(self) -> bool:
        """Dice a Cirq se il gate contiene almeno un parametro simbolico (in theta o phi)."""
        return isinstance(self.theta, sympy.Basic) or isinstance(self.phi, sympy.Basic)

    def _parameter_names_(self) -> Tuple[str, ...]:
        """Raccoglie i nomi di tutti i parametri simbolici liberi presenti in theta e phi."""
        names = set()
        for param in (self.theta, self.phi):
            if isinstance(param, sympy.Symbol):
                names.add(param.name)
            elif isinstance(param, sympy.Basic):
                # Se il parametro è una formula simbolica più complessa, estrae i simboli liberi
                names.update(str(symbol) for symbol in param.free_symbols)
        return tuple(names)

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> 'R_ij':
        """Risolve sia theta che phi usando il resolver e restituisce un nuovo gate numerico."""
        resolved_theta = resolver.value_of(self.theta, recursive)
        resolved_phi = resolver.value_of(self.phi, recursive)
        return R_ij(
            i=self.i, 
            j=self.j, 
            theta=resolved_theta, 
            phi=resolved_phi, 
            dimension=self.d
        )

