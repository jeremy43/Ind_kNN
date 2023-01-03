import numpy as np

class HashTable:
    def __init__(self, hash_size, inp_dimensions):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions  # input data dimension
        self.hash_table = dict()
        self.projections = np.random.randn(self.hash_size, inp_dimensions)

    def generate_hash(self, inp_vector):
        bools = (np.dot(inp_vector, self.projections.T)>0).astype('int')
        return ''.join(bools.astype('str'))
        
    def __setitem__(self, inp_vec, label):
        hash_value = self.generate_hash(inp_vec)
        self.hash_table[hash_value] = self.hash_table.get(hash_value, list()) + [label]

    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        # return all items that have the same hash code
        return self.hash_table.get(hash_value, [])

class LSH:
    def __init__(self, num_tables, hash_size, inp_dimensions):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_tables = list()
        for i in range(self.num_tables):
            self.hash_tables.append(HashTable(self.hash_size, self.inp_dimensions))
    
    def __setitem__(self, inp_vec, label):
        for table in self.hash_tables:
            table[inp_vec] = label
    
    def __getitem__(self, inp_vec):
        results = list()
        for table in self.hash_tables:
            results.extend(table[inp_vec])
        return list(set(results))

    def query_neighbor(self, inp_vec, dist=1):
        # return all data such that the hash distance is within dish.
        # the length of hash_code is hash_size * num_tables
        same_hash_neighbors = self.__getitem__(inp_vec)
        #print(f'hash_code is {same_hash_neighbors}')
        return same_hash_neighbors
        result = []
        select_neighbor = {}
        for table in self.hash_tables:
            same_value_neighbor = [i for i,j in table.hash_table.items() if j == table[inp_vec]]
            for i in same_value_neighbor:
                if i not in select_neighbor:
                    select_neighbor[i]+=1
                else:
                    select_neighbor[i]=0
        #threshold = self.hash_size -1 # only select neighbors such that the hamming distance is less than 1.
        top_neighbor = [i for i, j in select_neighbor.items()]
        return top_neighbor
        #for i in range(hash_size):
            # generate a neighbor hash code

