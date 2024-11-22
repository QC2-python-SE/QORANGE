import numpy as np

class GateClass:
    def __init__(self, matrix, gate_type):
        self._matrix = matrix
        self._gate_type = gate_type # 1 for single, 2 for double - for later identification in Circuits

    def get_gate_type(self):
        return self._gate_type

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

class Identity(singleQubitGate):

    def __init__(self):
        singleQubitGate.__init__(self, np.array([[1, 0], [0, 1]]))

class PauliX(singleQubitGate):

    def __init__(self):
        singleQubitGate.__init__(self, np.array([[0, 1], [1, 0]]))



class PauliY(singleQubitGate):

    def __init__(self):
        singleQubitGate.__init__(self, np.array([[0, -1j], [1j, 0]]))

class PauliZ(singleQubitGate):

    def __init__(self):
        singleQubitGate.__init__(self, np.array([[1, 0], [0, 1]]))

class Hadamard(singleQubitGate):

    def __init__(self):
        singleQubitGate.__init__(self, np.array([[1, 1], [1, -1]])/np.sqrt(2))

class S(singleQubitGate):

    def __init__(self):
        singleQubitGate.__init__(self, np.array([[1, 0], [0, 1j]]))

class T(singleQubitGate):

    def __init__(self):
        singleQubitGate.__init__(self, np.array([[1, 0], [0, 1/np.sqrt(2)+1j/np.sqrt(2)]]))

class CNOT(twoQubitGate):

    def __init__(self):
        twoQubitGate.__init__(self, np.array([
            [1, 0, 0, 0],  # |00⟩ -> |00⟩
            [0, 1, 0, 0],  # |01⟩ -> |01⟩
            [0, 0, 0, 1],  # |10⟩ -> |11⟩
            [0, 0, 1, 0]   # |11⟩ -> |10⟩
        ]))

class CZ(twoQubitGate):

    def __init__(self):
        twoQubitGate.__init__(self, np.array([
            [1, 0, 0, 0],  
            [0, 1, 0, 0],  
            [0, 0, 1, 0],  
            [0, 0, 0, -1]   
        ]))
    
class SWAP(twoQubitGate):

    def __init__(self):
        twoQubitGate.__init__(self, np.array([
            [1, 0, 0, 0],  
            [0, 0, 1, 0],  
            [0, 1, 0, 0],  
            [0, 0, 0, 1]   
        ]))



if __name__=="__main__":
    a = np.ones((2,3)) # this should throw error
    b = singleQubitGate(a) # qubit id not implemented so None type is ok for now