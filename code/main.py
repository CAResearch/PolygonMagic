import numpy as np

class Solution:

    """ Constructor definition """
    def __init__(self):
        self.a, self.b = 2, 300

    """ Function definition """
    @staticmethod
    def compute_matrix_a(vertices: int) -> list:

        matrixA = []
        for v in range(vertices - 1):

            leftZeroArray = [0] * v
            rightZeroArray = [0] * (vertices - v - 2)
            matrixRow = leftZeroArray + [1, 1] + rightZeroArray
            matrixA.append(matrixRow)

        matrixA.append([1] + [0] * (vertices - 2) + [1])

        return matrixA

    @staticmethod
    def compute_matrix_b(weights: list, vertices: int) -> list:

        matrixB = []
        for i in range(vertices - 2):

            weightSum = weights[i] + weights[i + 2]
            matrixB.append(weightSum)

        matrixB.append(weights[vertices - 2] + weights[0])
        matrixB.append(weights[vertices - 1] + weights[1])

        return matrixB

    def compute_maps(self) -> tuple:

        limitVector = []
        verticeLimitMap = []
        for v in range(self.a, self.b):

            # If vertices = 2 * v, then the
            # solution matrix can't be computed,
            # because det(matrixA) = 0.
            vertices = 2 * v + 1

            matrixA = self.compute_matrix_a(vertices)
            inverseA = np.linalg.inv(matrixA)

            weights = [(i + 1) for i in range(vertices)]
            weightSum = vertices * (vertices + 1) / 2
            currentWeightSum = weightSum

            limit = 0
            while currentWeightSum == weightSum:

                matrixB = self.compute_matrix_b(weights, vertices)
                matrixX = np.dot(inverseA, matrixB)
                matrixX = [int(weight) for weight in matrixX]

                currentWeightSum = sum(matrixX)
                if currentWeightSum != weightSum: break

                limit += 1
                weights = matrixX

            verticeLimitMap.append([vertices, limit])
            limitVector.append(limit)

        return verticeLimitMap, limitVector

    def get_limit_frequency(self, verticeLimitMap: list, limitVector: list) -> list:

        keys = self.make_unique(limitVector)
        keys.reverse()

        limitFrequency = []
        for key in keys:

            timesKeyWasFound = 0
            for region in verticeLimitMap:

                keyFound = (region[1] == key)
                if not keyFound: continue

                timesKeyWasFound += 1

            limitFrequency.append([key, timesKeyWasFound])

        return limitFrequency

    @staticmethod
    def make_unique(vector: list) -> list:

        keys = list()
        vector.sort()
        [keys.append(key) for key in vector if key not in keys]

        return keys

    def solve(self):

        verticeLimitMap, limitVector = self.compute_maps()

        print()
        print(verticeLimitMap)

        """ Observations """
        ######################################################
        # For a given number of vertices v, of the form
        # 2 * k + 1, we encounter at each new set of
        # vertices - only once - the number N, which is
        # given by the formula: (v + 1) / 2. Moreover,
        # that number, is then lost from the set, if the
        # sum of the weights of the vertices != v * (v + 1) / 2.
        #######################################################

        #######################################################
        # It seems that the number N, at a given number
        # of vertices of the form 2 * v + 1, jumps inside
        # the set, at each iteration exactly t steps, which
        # is given by the formula: (v - 1).
        #######################################################

        #######################################################
        # Now the frequency of occurrence, of each of the
        # steps, that the sum broke, given a number of vertices
        # v, doesn't seem to have any closed form that we
        # can use in order to compute it.
        #######################################################

        limitFrequency = self.get_limit_frequency(verticeLimitMap, limitVector)

        print()
        print(limitFrequency)

if __name__ == "__main__":

    solution = Solution()
    solution.solve()
