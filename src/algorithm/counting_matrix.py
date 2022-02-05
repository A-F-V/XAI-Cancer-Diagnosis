
class CountingMatrix:
    def __init__(self, width):
        self.matrix = [0]*width
        self.cum_matrix = [0]*width
        self.size = 0
        self.width = width

    def add(self, index):
        self.matrx[index] += 1

    def add_many(self, index, amt):
        self.matrix[index] += amt

    def cumulate(self):
        for i in range(1, self.width):
            self.cum_matrix[i] = self.cum_matrix[i-1] + self.matrix[i-1]

    def __getitem__(self, index):
        # Binary Search Alogrithm

    def __len__(self):
        return self.size
