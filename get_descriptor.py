#!/usr/bin/env python2

# MIT License
# 
# Copyright (c) 2016 Anders Steen Christensen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import numpy as np
from fml import Molecule
from wellfareSTO import SlaterOverlapCartesian
EV_TO_HARTREE = 0.0367493
# EV_TO_HARTREE = 1.0 #0.0367493
ANGS_TO_BOHR = 1.889725989
E2H = EV_TO_HARTREE


# CNDO/S parameters from 
# http://dx.doi.org/10.1063/1.1679498


NUCLEAR_CHARGE = dict()
NUCLEAR_CHARGE["H"] = 1.0
NUCLEAR_CHARGE["C"] = 6.0
NUCLEAR_CHARGE["N"] = 7.0
NUCLEAR_CHARGE["O"] = 8.0
NUCLEAR_CHARGE["F"] = 9.0
NUCLEAR_CHARGE["Si"] = 14.0
NUCLEAR_CHARGE["P"] = 15.0
NUCLEAR_CHARGE["S"] = 16.0
NUCLEAR_CHARGE["Cl"] = 17.0
NUCLEAR_CHARGE["Ge"] = 32.0

ZNUC = dict()
ZNUC["H"]  = 1
ZNUC["C"]  = 4
ZNUC["N"]  = 5
ZNUC["O"]  = 6
ZNUC["F"]  = 7
ZNUC["Cl"] = 7


# Number of shells per alom
MAXSHELL = dict()
MAXSHELL["H"]  = 2
MAXSHELL["C"]  = 2
MAXSHELL["N"]  = 2
MAXSHELL["O"]  = 2
MAXSHELL["F"]  = 2
MAXSHELL["Cl"] = 2

# UU = IP - EA = IP - AP
UU = dict()
UU["H"]  = [ 14.35 * E2H,  0.00 * E2H]
UU["C"]  = [ 29.92 * E2H, 11.61 * E2H] 
UU["N"]  = [ 40.97 * E2H, 16.96 * E2H] 
UU["O"]  = [ 54.51 * E2H, 21.93 * E2H]
UU["F"]  = [ 53.96 * E2H, 24.36 * E2H]
UU["Cl"] = [ 35.00 * E2H, 18.76 * E2H]

# BETA = -\beta_{A}
BETA = dict()
BETA["H"]  = 12.0 * E2H 
BETA["C"]  = 17.5 * E2H 
BETA["N"]  = 26.0 * E2H 
BETA["O"]  = 45.0 * E2H
BETA["F"]  = 50.0 * E2H
BETA["Cl"] = 15.0 * E2H

# GAA = \gamma_{AA}
GAA = dict()
GAA["H"]  = 12.85 * E2H 
GAA["C"]  = 10.93 * E2H 
GAA["N"]  = 13.10 * E2H 
GAA["O"]  = 15.27 * E2H
GAA["F"]  = 17.36 * E2H
GAA["Cl"] = 11.30 * E2H

# Orbital exponents in AU [ZS, ZP]
# From AM1
ZETA = dict()
ZETA["H"]  = [ 1.18807800, 0.00000000]
ZETA["C"]  = [ 1.80866500, 1.68511600] 
ZETA["N"]  = [ 2.31541000, 2.15794000] 
ZETA["O"]  = [ 3.10803200, 2.52403900]
ZETA["F"]  = [ 3.77008200, 2.49467000]
ZETA["Cl"] = [ 3.63137600, 2.07679900]

def triangular_to_vector(M, uplo="U"):

    if not (M.shape[0] == M.shape[1]):
        print "ERROR: Not a square matrix."
        exit(1)

    n = M.shape[0]
    l = (n + 1) * n // 2 # Explicit integer division
    v = np.empty((l))

    index = 0
    for i in range(n):
        for j in range(n):
            
            if j > i:
                continue

            v[index] = M[i, j]

            index += 1

    return v

def twoe(rab, gaa, gbb):
    # Mataga-Nishimoto approximation
    small_g = 0.0001
    if (gaa < small_g) or (gbb < small_g):
        return 0.0

    gab = (gaa + gbb) / (2.0 + rab * (gaa + gbb))

    return gab

def distance(a, b):

    rab = np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)
    return rab * ANGS_TO_BOHR


def overlap(rij, zetai, shelli, zetaj, shellj):

    small_zeta = 0.0001
    if (zetai < small_zeta) or (zetaj < small_zeta):
        return 0.0

    return SlaterOverlapCartesian(shelli + 1, 0, 0, zetai, 0.00000, 0.00000, 0.00000, \
        shellj + 1, 0, 0, zetaj, 0.00000, 0.000000, rij)


# Assemble one-electron and two-electron matrices
def get_matrix(atomtypes, coordinates, size=23):
 
    nshell = [MAXSHELL[x] for x in atomtypes]
    totshell = sum(nshell)

    # One-electron terms
    # G = np.zeros((totshell,totshell))
    G = np.zeros((size*2,size*2))

    # Two-electron terms
    # H = np.zeros((totshell,totshell))
    H = np.zeros((size*2,size*2))

    for i, iatom in enumerate(atomtypes):
        for j, jatom in enumerate(atomtypes):

            for ishell in range(nshell[i]):
                iorb = sum(nshell[:i]) + ishell

                for jshell in range(nshell[j]):
                    jorb = sum(nshell[:j]) + jshell

                    rij = distance(coordinates[i], coordinates[j])

                    # Two-electron terms
                    G[iorb,jorb] -= 0.5 * twoe(rij, UU[jatom][jshell], UU[iatom][ishell])
                    if not (iorb == jorb):
                        G[iorb,iorb] += twoe(rij, UU[jatom][jshell], UU[iatom][ishell])


                    # One-electron terms
                    if (iorb == jorb):
                        H[iorb,jorb] = GAA[iatom]
                        for k, katom in enumerate(atomtypes):

                            if (k == i):
                                continue

                            rik = distance(coordinates[i], coordinates[k])
                            H[iorb,jorb] += ZNUC[iatom] * twoe(rik, UU[iatom][ishell], UU[katom][0])
                            
                    elif (i == j):
                        pass # No on-site terms in CNDO currently
                    else:
                        Sab = overlap(rij, ZETA[iatom][ishell], ishell, ZETA[jatom][jshell], jshell)
                        # print Sab, iatom, jatom
                        H[iorb,jorb] = (BETA[iatom] + BETA[jatom]) / 2.0 * Sab

    
    return triangular_to_vector(G), triangular_to_vector(H)

def sort_atoms(atomtypes, coordinates, idx=0):

    atomtype_i = atomtypes[idx]

    distances = []

    for j, atomtype_j in enumerate(atomtypes):

        distance = np.linalg.norm(coordinates[idx] - coordinates[j])

        distances.append((distance, j))

    distances.sort()

    atomtypes_sorted = []
    coordinates_sorted = []

    for (dist_j, j) in distances:

        atomtypes_sorted.append(atomtypes[j])
        coordinates_sorted.append(coordinates[j])

    return atomtypes_sorted, coordinates_sorted



def get_coulomb_matrix(atomtypes, coordinates, size=23):

    Mij = np.zeros((size, size))

    for i, atomtype_i in enumerate(atomtypes):
        for j, atomtype_j in enumerate(atomtypes):
            if i == j:
                Mij[i, j] = 0.5 * NUCLEAR_CHARGE[atomtype_i] ** 2.4

            elif j > i:
                continue

            else:
                Mij[i, j] = NUCLEAR_CHARGE[atomtype_i] * NUCLEAR_CHARGE[atomtype_j] \
                            / np.linalg.norm(coordinates[i] - coordinates[j])

    return triangular_to_vector(Mij)


def get_ndodesc(mol, size=23):

    atomtypes, coordinates = sort_atoms(mol.atomtypes, mol.coordinates)
    M = get_coulomb_matrix(atomtypes, coordinates, size=size)
    (G, H) = get_matrix(atomtypes, coordinates, size=size)

    X = np.concatenate([M, G, H])

    return X

if __name__ == "__main__":


    np.set_printoptions(precision=10,linewidth=500)
    filename = sys.argv[1]
    mol = Molecule()
    mol.read_xyz(filename)

    X = get_ndodesc(mol, size=4)

    print X
    print X.shape

