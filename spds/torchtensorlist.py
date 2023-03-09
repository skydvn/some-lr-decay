import torch

class TorchTensorList:
    def __init__(self, 
                 device:torch.device = None) -> None:

        if device:
            self.device = device
        else:
            self.device = "cpu"

        self.count = 0
        self.express_line = []
        self.arr = None
    
    def __len__(self):
        return self.count
    
    def size(self):
        if self.arr == None:
            return None
        return self.arr.shape
    
    def get_epr_line(self):
        return self.express_line
        
    @staticmethod
    def _check_index(self, index):
        if index is None:
            raise Exception("index cannot be None")
        elif not isinstance(index, int):
            raise Exception(f"index must be an integer but found {type(index)} instead")
        elif not 0 <= index < self.count:
            return IndexError('index is out of bounds !')
    
    @staticmethod
    def _check_input(input):
        if input is None:
            raise Exception("node cannot be None")
        elif not isinstance(input, torch.Tensor):
            raise Exception(f"node must be an integer but found {type(input)} instead")
        # elif not input.shape == self.ele_shape:
        #     raise Exception(f"Torch Tensor List cannot take another shape vector")
    
    def __setitem__(self, index: int , x: torch.Tensor):
        raise NotImplementedError    

        self._check_index(index=index)
        self._check_input(input=x)
    
    def __getitem__(self, index: int):
        self._check_index(self, index=index)

        if index == 0:
            return self.arr[: self.express_line[index]]
        else:
            return self.arr[sum(self.express_line[:index]) : sum(self.express_line[:index+1])]
    
    def append(self, x:torch.Tensor):
        self._check_input(x)

        if self.count == 0:
            self.arr = x.to(device=self.device)
            self.count += 1
            self.express_line.append(x.shape[0])
        else:
            self.count += 1
            self.express_line.append(x.shape[0])
            self.arr = torch.cat((self.arr, x.to(device=self.device)))
    
    def pop(self):
        if self.count == 0:
            raise Exception("There is no data in Torch Tensor List")
        else:
            self.count -= 1
            self.express_line.pop()
            self.arr = self.arr[self.express_line[0]:]
        
    def insert(self):
        raise NotImplementedError

if __name__ == "__main__":

    test_list = TorchTensorList(device="cuda")

    print(test_list.size())

    for idx in range(9):
        test_list.append(torch.randn((idx + 1, 4 , 32, 32)))

    print(test_list.size())

    print(len(test_list))

    print(test_list.get_epr_line())

    print(test_list[6].shape)