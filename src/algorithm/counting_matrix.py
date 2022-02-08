
class CountingMatrix:
    def __init__(self, width):
        self.matrix = [0]*width
        self.cum_matrix = [0]*width
        self.size = 0
        self.width = width
        self._dirty = False

    def add(self, index):
        self.matrix[index] += 1
        self._dirty = True
        self.size += 1

    def add_many(self, index, amt):
        self.matrix[index] += amt
        self._dirty = True
        self.size += amt

    def cumulate(self):
        for i in range(1, self.width):
            self.cum_matrix[i] = self.cum_matrix[i-1] + self.matrix[i-1]
        self._dirty = False

    def __getitem__(self, index):
        # Binary Search Alogrithm
        if self._dirty:
            self.cumulate()
        if index < 0 or index >= self.cum_matrix[-1]+self.matrix[-1]:
            raise IndexError
        bot, top = 0, self.width
        while bot + 1 < top:
            mid = (bot + top) // 2
            if self.cum_matrix[mid] <= index:
                bot = mid
            else:
                top = mid
        return (bot, index-self.cum_matrix[bot])

    def __len__(self):
        return self.size
