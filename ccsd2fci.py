from functools import lru_cache, partial
from itertools import chain, combinations, product
from multiprocessing import Pool

import numpy as np
from pyscf import cc, gto, scf
from pyscf.fci.addons import cistring
from scipy.special import comb


class IndicesX:
    def __init__(self, spin, str_orb, str_gs):
        self.spin = spin
        str_x = bin(str_orb ^ str_gs)
        indx_x = np.array(list(str_x[:1:-1]), dtype=int).nonzero()
        self.occ, self.virt = np.split(indx_x[0], 2)

def t1_ixer(indx, t1, n_occ):
    offset = int(indx.spin == 'b')
    return [(amp, (1<<(2 * i + offset)) | (1<<(2 * a + offset)))
            for i in indx.occ for a in indx.virt if (amp := t1[i, a - n_occ]) != 0]


def amp_assembler(indx_a, indx_b, t2, n_occ):
    indxs = [indx_a.occ, indx_b.occ, indx_a.virt, indx_b.virt]
    amps = amp_gather(indxs, [0, 1]*2, t2[0], n_occ)
    amps += amp_gather(indxs[:2] + indxs[:1:-1], [0, 1, 1, 0], t2[1], n_occ)
    amps += amp_gather(indxs[1::-1] + indxs[-2:], [1, 0, 0, 1], t2[1], n_occ)
    amps += amp_gather(indxs[1::-1] + indxs[:1:-1], [1, 0]*2, t2[0], n_occ)
    return amps


def amp_gather(indx, offset, t2, n_occ):
    return [(amp, (1<<ix) | (1<<jx) | (1<<ax) | (1<<bx))
            for i in indx[0] for j in indx[1]
            if (ix := 2 * i + offset[0]) < (jx := 2 * j + offset[1])
            for a in indx[2] for b in indx[3]
            if ((amp := t2[i, j, a - n_occ, b - n_occ]) !=0
                and ((ax := 2 * a + offset[2]) < (bx := 2 * b + offset[3])))]


def fci_coefs(data, parallel=False):
    t1 = relevant_amps(np.array(data["CC T1 amplitudes"]))
    t2 = relevant_amps(np.array(data["CC T2 amplitudes"]))
    t2t = -t2.transpose(1, 0, 2, 3)
    t2 = (t2, t2t, t2 + t2t)
    nocc, nvirt = t1.shape
    spat_orbs = cistring.gen_strings4orblist(range(nocc + nvirt), nocc)
    part_fn = partial(amp2coef, str_gs=spat_orbs[0], t1=t1, t2=t2, nocc=nocc)
    if parallel:
        with Pool(processes=8) as pool:
            phi_map = pool.starmap(part_fn, product(spat_orbs, repeat=2))
    else:
        phi_map = (part_fn(*pair) for pair in product(spat_orbs, repeat=2))
    phi_ccsd = np.fromiter(phi_map, count=len(spat_orbs)**2, dtype=float)
    return phi_ccsd / np.linalg.norm(phi_ccsd)


def amp2coef(str_orba, str_orbb, str_gs, t1, t2, nocc):
    indx_a, indx_b = mapper(IndicesX, zip(['a', 'b'], [str_orba, str_orbb]),
                            str_gs=str_gs)
    num_x = len(indx_a.occ) + len(indx_b.occ)
    if num_x == 0:
        return 1.0
    
    t1a, t1b = mapper(t1_ixer, zip([indx_a, indx_b]), t1=t1, n_occ=nocc)
    t1_amps = t1a + t1b
    if len(t1_amps) > 0:
        t1_coeffs, t1_orbs = zip(*t1_amps)
        t1_amps = (np.array(t1_coeffs), np.array(t1_orbs))
    else:
        t1_amps = (np.empty(0), np.empty(0))
    
    t2_idxz = zip([[indx_a.occ]*2 + [indx_a.virt]*2,
                   [indx_b.occ]*2 + [indx_b.virt]*2],
                  [[0]*4, [1]*4])
    t2aa, t2bb = mapper(amp_gather, t2_idxz, t2=t2[2], n_occ=nocc)
    t2ab = amp_assembler(indx_a, indx_b, t2, nocc)
    t2_amps = t2aa + t2bb + t2ab
    if len(t2_amps) > 0:
        t2_coeffs, t2_orbs = zip(*t2_amps)
        t2_amps = (np.array(t2_coeffs), np.array(t2_orbs))
    else:
        t2_amps = (np.empty(0), np.empty(0))

    coeffs = (np.sum([combine_amps(t1_amps, t2_amps, num_x, k)
                      for k in range(num_x // 2 + 1)])
                      if t1_amps[0].any() or t2_amps[0].any() else 0)

    return coeffs


def combine_amps(t1_amps, t2_amps, num_x, n2):
    n1 = num_x - 2 * n2

    if n1 > len(t1_amps[0]) or n2 >len(t2_amps[0]):
        return 0

    t1_prods, t1_orbs = (prodofcombs(t1_amps, n1) if n1 != 0
                         else (np.array(1), np.array([0])))
    if not t1_prods.any():
        return 0
    
    t2_prods, t2_orbs = (prodofcombs(t2_amps, n2) if n2 != 0
                         else (np.array(1), np.array([0])))
    if not t2_prods.any():
        return 0

    mask = ~np.bitwise_and.outer(t1_orbs, t2_orbs).astype(bool)

    total_prod = np.outer(t1_prods, t2_prods)[mask]
    return np.sum(total_prod)


@lru_cache(maxsize=256)
def comb_index(n: int, k: int):
    count = comb(n, k, exact=True)
    index = np.fromiter(chain.from_iterable(combinations(range(n), k)),
                        np.uint16, count=count*k)
    return index.reshape(-1, k)


def prodofcombs(amps, n):
    coeffs, indxs = amps
    if np.prod(np.sort(np.abs(coeffs))[-n:]) < 1e-8:
        return np.empty(0), np.empty(0, dtype=int)

    combix = comb_index(len(coeffs), n)
    orbs = indxs[combix]
    xorbs = np.bitwise_xor.reduce(orbs, axis=1)
    mask = np.bitwise_or.reduce(orbs, axis=1) == xorbs

    vecofprods = np.prod(coeffs[combix][mask], axis=1)[:, None]
    vecoforbs = xorbs[mask]

    return vecofprods, vecoforbs


def prodofcombs_0(amps, n, m):
    return (tuple([np.prod([entry[0] for entry in comb])]) + orbs
            for comb in combinations(amps, n)
            if len(set(orbs := sum([entry[1:] for entry in comb], ()))) == m * n)


def filter_orbs(ind_1,ind_2):
    return not any(elem in ind_2 for elem in ind_1)


def relevant_amps(input_array):
    sorted_values = np.sort(np.abs(input_array.flatten()))[::-1]
    norm_old = norm = 0
    for value in sorted_values:
        norm += value**2
        if norm == norm_old:
            break
        norm_old = norm
    return input_array * (np.abs(input_array) > value)

def mapper(func, *args, **kwargs):
    return tuple(map(lambda var: func(*var, **kwargs), *args))

def main():
    xyz = '''N 0 0 0; N 0 1 0'''
    basis = 'sto-6g'
    mol = gto.M(atom=xyz, basis=basis, spin=0, symmetry=True)
    hf = scf.RHF(mol).run()
    myccsd = cc.CCSD(hf).run()
    df = {"CC T1 amplitudes": myccsd.t1, "CC T2 amplitudes": myccsd.t2}
    return fci_coefs(df)


if __name__ == '__main__':
    main()
