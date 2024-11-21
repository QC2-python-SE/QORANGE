import numpy as np

class GateClass:
    def __init__(self, matrix, gate_type):
        self._matrix = matrix
        self._gate_type = gate_type # 1 for single, 2 for double - for later identification in Circuits

class singleQubitGate(GateClass):
    def __init__(self, matrix):
        if np.shape(matrix) != (2, 2):
            raise Exception("Matrix not suitable for single qubit operation!")

        GateClass.__init__(self, matrix, 1)


class twoQubitGate(GateClass):
    def __init__(self, matrix):
        if np.shape(matrix) != (4, 4):
            raise Exception("Matrix not suitable for two qubit operation!")
        
        GateClass.__init__(self, matrix, 2)

def PauliX(singleQubitGate):

    def __init__(self):
        singleQubitGate.__init__(self, np.array([[0, 1], [1, 0]]))

def PauliY(singleQubitGate):

    def __init__(self):
        singleQubitGate.__init__(self, np.array([[0, -1j], [1j, 0]]))

def PauliZ(singleQubitGate):

    def __init__(self):
        singleQubitGate.__init__(self, np.array([[1, 0], [0, 1]]))

def Hadamard(singleQubitGate):

    def __init__(self):
        singleQubitGate.__init__(self, np.array([[1, 1], [1, -1]])/np.sqrt(2))

def S(singleQubitGate):

    def __init__(self):
        singleQubitGate.__init__(self, np.array([[1, 0], [0, 1j]]))

def T(singleQubitGate):

    def __init__(self):
        singleQubitGate.__init__(self, np.array([[1, 0], [0, 1/np.sqrt(2)+1j/np.sqrt(2)]]))

def CNOT(twoQubitGate):

    def __init__(self):
        twoQubitGate.__init__(self, np.array([
            [1, 0, 0, 0],  # |00⟩ -> |00⟩
            [0, 1, 0, 0],  # |01⟩ -> |01⟩
            [0, 0, 0, 1],  # |10⟩ -> |11⟩
            [0, 0, 1, 0]   # |11⟩ -> |10⟩
        ]))

def CZ(twoQubitGate):

    def __init__(self):
        twoQubitGate.__init__(self, np.array([
            [1, 0, 0, 0],  
            [0, 1, 0, 0],  
            [0, 0, 1, 0],  
            [0, 0, 0, -1]   
        ]))
    
def SWAP(twoQubitGate):

    def __init__(self):
        twoQubitGate.__init__(self, np.array([
            [1, 0, 0, 0],  # |00⟩ -> |00⟩
            [0, 0, 1, 0],  # |01⟩ -> |01⟩
            [0, 1, 0, 0],  # |10⟩ -> |11⟩
            [0, 0, 0, 1]   # |11⟩ -> |10⟩
        ]))



if __name__=="__main__":
    a = np.ones((2,3)) # this should throw error
    b = singleQubitGate(a) # qubit id not implemented so None type is ok for now