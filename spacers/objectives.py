from abc import ABC, abstractmethod


class VaccineObjective(ABC):
    ''' base class for the milp objective
    '''

    @abstractmethod
    def insert_objective(self, model):
        ''' insert the objective in the model
        '''

