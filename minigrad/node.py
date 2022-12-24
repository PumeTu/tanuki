class Node:
    '''
    
    '''
    def __init__(self, tensor):
        self.tensor = tensor
        self.childrens = []
        self.parents = []
        self.backward_fn = lambda: None
        self.visited = False
