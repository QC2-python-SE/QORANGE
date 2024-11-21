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
    



if __name__=="__main__":
    a = np.ones((2,3)) # this should throw error
    b = singleQubitGate(a) # qubit id not implemented so None type is ok for now