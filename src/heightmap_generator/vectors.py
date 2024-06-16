class vec:
    def __init__(self, *args) -> None:
        self.scalars : list[float] = args

    def __len__(self) -> int:
        return len(self.scalars)
    
    def __getitem__(self, key) -> int:
        return self.scalars[key]
    
    def __str__(self) -> str:
        return f"({', '.join([str(i) for i in self.scalars])})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __add__(self, b):
        if len(self) != len(b):
            raise ValueError("Vectors must be of same length")
        
        return vec(*[self[i] + b[i] for i in range(len(self))])
    
    def __sub__(self, b):
        if len(self) != len(b):
            raise ValueError("Vectors must be of same length")
        
        return vec(*[self[i] - b[i] for i in range(len(self))])
    
    def __mul__(self, b : float | int):
        if type(b) != float and type(b) != int:
            raise ValueError("Vector must be multiplied by a scalar type")
        
        return vec(*[self[i] * b for i in range(len(self))])
    
    def __truediv__(self, b : float | int):
        if type(b) != float and type(b) != int:
            raise ValueError("Vector must be multiplied by a scalar type")
        
        return vec(*[self[i] / b for i in range(len(self))])

def dot(a : vec, b : vec) -> float:
    return sum([a[i] * b[i] for i in range(len(a))])
