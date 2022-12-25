class Operations:
    '''
    Operations base class
    
    '''
    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args):
        raise NotImplementedError()

    
class Add(Operations):
    '''
    Element wise addition between two Tensors
    '''
    def __call__(self, other):
        pass