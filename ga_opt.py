'''
Python implementation of a genetic algorithm for optimisation of a provided n-dimensional cost function
'''

import struct
import numpy as np
from random import seed
from random import randint
from copy import deepcopy
from costfunctions import Costfuncs
from bucketsort import bsort

class Genetic(object):

    def __init__(self, domain, N, nruns, nelite, costf, ndim=2, crossfrac=0.8, beta=0.05, globalopt=None, \
            thresh = 1.0E-03, randseed=20):
        seed(randseed)
        np.random.seed(randseed)
        self.domain = domain # 2D tuple of min/max bounds (applies to all variables) for initialising positions randomly
        self.N = N # no. of individuals in population
        self.nruns = nruns# max. no. of iterations
        self.ndim = ndim # dimensionality of cost function (no. of 'chromosomes' of each individual)
        self.nelite = nelite # no. of elite individuals selected in each generation (elite individuals are guaranteed to be passed to the next population)
        self.costf = costf # choice of cost function
        self.crossfrac = crossfrac # crossover fraction - proportion of individuals in the next generation that are created by crossover
        self.beta = beta # exponent coeff in calculating expectation values
        self.globalopt = globalopt # n-dimensional tuple of known optimal coords. Allows error to be calculated
        self.thresh = thresh # error threshold
        return

    # function to return float as bit string
    def floattobits(self, f):
        s = struct.pack('>f', f)
        return str(struct.unpack('>l', s)[0])

    # function to return bit string as float
    def bitstofloat(self, b):
        s = struct.pack('>l', int(b))
        return struct.unpack('>f', s)[0]

    def initialise_posns(self):
        r = np.random.uniform(self.domain[0], self.domain[1], size = self.N*self.ndim)
        return r

    def calc_err(self, r):
        avgerr = 0.0
        besterr = float("inf")
        try:
            for i in range(self.N):
                err = np.sum(abs(self.globalopt - r[i*self.ndim:i*self.ndim+self.ndim]))
                avgerr += err
                if err < besterr:
                    besterr = err
        except TypeError:
            pass
        avgerr = avgerr / self.N
        return avgerr, besterr

    # function to divide expectation values of each individual into a 'roulette wheel'
    def make_rwheel(self, expecvalues):
        for i in range(1, self.N):
            expecvalues[i] += expecvalues[i-1]
        return expecvalues

    # function to calculate expectation values based on raw cost values, and divide into 'roulette wheel'
    def calc_expecvalues(self, costs, avgcost):
        expecvalues = np.exp(-self.beta*(costs-avgcost))
        sum_expecvalues = np.sum(expecvalues)
        expecvalues = expecvalues / sum_expecvalues
        return expecvalues

    # function to determine the population's average fitness (cost)
    def grade(self, costs):
        avgcost = np.sum(costs)/np.shape(costs)
        return avgcost

    # Function to choose partners to share genetic information, based on a probability distribution defined by the expectation values
    def choose_partners(self, expecvalues):
        n1 = np.random.uniform(0,1)
        n2 = np.random.uniform(0,1)
        i1 = self.choose_individual(expecvalues, n1)
        i2 = self.choose_individual(expecvalues, n2)
        return i1, i2

    # Function to choose (index of) individual, based on a probability distribution defined by the expectation values
    def choose_individual(self, expecvalues, n):
        i = 0
        while expecvalues[i] <= n:
            i += 1
            continue
        return i

    # Function to perform crossover of genetic info between two chromosomes of different individuals
    def crossover(self, ri, rj):
        j = 0
        for fi, fj in zip(ri, rj):
            bi = list(self.floattobits(fi))
            bj = list(self.floattobits(fj))
            bi_neg = False
            bj_neg = False
            if bi[0] == "-":
                bi_neg = True
                bi.remove("-")
            if bj[0] == "-":
                bj_neg = True
                bj.remove("-")
            bk = deepcopy(bi)
            m = randint(2,len(bk)-2) # (internal) position at which crossover takes place
            bi = bi[0:m+1] + bj[m+1:len(bk)+1]
            bj = bj[0:m+1] + bk[m+1:len(bk)+1]
            if bi_neg: bi.insert(0,"-")
            if bj_neg: bj.insert(0,"-")
            fi = self.bitstofloat("".join(str(i) for i in bi))
            fj = self.bitstofloat("".join(str(i) for i in bj))
            ri[j] = fi
            rj[j] = fj
            j += 1
        return ri, rj

    # Function to perform random permutation of bit strings and return new float
    def mutation(self, ri):
        j = 0
        for f in ri:
            b = self.floattobits(f)
            b = list(b)
            m = randint(2,len(b)-2) # position in bit to be permuted, chosen from range such that permutation is of reasonable size
            n = b[m]
            while int(n) == int(b[m]):
                n = randint(0,9) # new integer in permutation
            b[m] = n
            f = self.bitstofloat("".join(str(i) for i in b))
            ri[j] = f
            j += 1
        return ri

    # Function to bucket sort individuals according to their cost and return the indicies of the nelite fittest candidates
    def find_elite(self, expecvalues):
        elites = bsort(expecvalues)
        elites = elites[-self.nelite:]
        return elites

    # Function to drive genetic algorithm optimisation
    def genetic_optimise(self):
        r = self.initialise_posns()
        rnew = deepcopy(r)
        costs = np.zeros(self.N)
        j = 0
        besterr = float("inf")
        while j < self.nruns and besterr >= self.thresh:
#            print "r...\n", r
            for i in range(self.N):
                costs[i] = self.costf(r[i*self.ndim:i*self.ndim+self.ndim])
#            print "costs...\n", costs
            avgcost = self.grade(costs)
#            print "avgcost...\n", avgcost
            expecvalues = self.calc_expecvalues(costs, avgcost)
#            print "expecvalues...\n", expecvalues
            if self.nelite > 0: # find elite individuals and allow them place in the next generation
                elites = self.find_elite(expecvalues)
#                print "elites", elites
                for i, elite in enumerate(elites):
                    rnew[i*self.ndim:i*self.ndim+self.ndim] = r[elite*self.ndim:elite*self.ndim+self.ndim]
            expecvalues = self.make_rwheel(expecvalues)
            for k in range(self.nelite, int(self.N*self.crossfrac), 2): # perform crossovers
                p1, p2 = self.choose_partners(expecvalues)
                rnew[k*self.ndim:k*self.ndim+self.ndim], rnew[(k+1)*self.ndim:(k+1)*self.ndim+self.ndim] = \
                    self.crossover(r[p1*self.ndim:p1*self.ndim+self.ndim], r[p2*self.ndim:p2*self.ndim+self.ndim])
            for p in range(int(self.N*self.crossfrac)+self.nelite, self.N): # rest of next generation population is formed via mutations
                n1 = np.random.uniform(0,1)
                m1 = self.choose_individual(expecvalues, n1)
                rnew[p*self.ndim:p*self.ndim+self.ndim] = self.mutation(r[m1*self.ndim:m1*self.ndim+self.ndim])
            if self.globalopt is not None:
                avgerr, besterr = self.calc_err(rnew)
                print j, avgerr, besterr
            r = deepcopy(rnew)
#            print "new r...\n", r
            j += 1
        return


### DRIVER CODE
if __name__ == "__main__":

    # set params
    domain = (-3.0,3.0)
    N = 20
    nruns = 5000
    nelite = 2

    # EXAMPLE 1
    costfunc1 = Costfuncs()
    globaloptcoords = costfunc1.coscos(x=None) # n-dimensional vector giving coords at global optimimum.
    costf = costfunc1.coscos


    genetic1 = Genetic(domain, N, nruns, nelite, costf, globalopt=globaloptcoords)
    genetic1.genetic_optimise()
