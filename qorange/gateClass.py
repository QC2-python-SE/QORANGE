import numpy as np

class GateClass:
    def __init__(self, qubit_id_list, matrix, gate_type):
        self._matrix = matrix
        self._act_on = qubit_id_list
        self._gate_type = gate_type # 1 for single, 2 for double - for later identification in Circuits

class singleQubitGate(GateClass):
    def __init__(self, qubit_id, matrix):
        if np.shape(matrix) != (2, 2):
            raise Exception("Matrix not suitable for single qubit operation!")

        GateClass.__init__(self, qubit_id, matrix, 1)


class twoQubitGate(GateClass):
    def __init__(self, qubit_id, matrix):
        if np.shape(matrix) != (4, 4):
            raise Exception("Matrix not suitable for two qubit operation!")
        
        GateClass.__init__(self, qubit_id, matrix, 2)


if __name__=="__main__":
    a = np.ones((2,3)) # this should throw error
    b = singleQubitGate(None, a) # qubit id not implemented so None type is ok for now