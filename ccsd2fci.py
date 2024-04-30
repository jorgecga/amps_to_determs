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


class Determinant:
    def __init__(self, nocc, indx_a, indx_b) -> None:
        self.nocc= nocc
        occs = np.hstack((2 * indx_a.occ, 2 * indx_b.occ + 1))
        occs.sort()
        self.occ = {val: idx for idx, val in enumerate(occs)}
        virts = np.hstack((2 * indx_a.virt, 2 * indx_b.virt + 1))
        virts.sort()
        self.virt = {val: idx for idx, val in enumerate(virts)}


def t1_ixer(indx, t1, det):
    offset = int(indx.spin == 'b')
    return [t1_gen(amp, i, a, offset, det)
            for i in indx.occ for a in indx.virt if (amp := t1[i, a - det.nocc]) != 0]


def t1_gen(amp, i, a, offset, det):
    ix = 2 * i + offset
    ax = 2 * a + offset
    return (amp, (1<<ix) | (1<<ax), (det.occ[ix], det.virt[ax]))


def t2_gen(amp, ix, jx, ax, bx, det):
    return (amp, (1<<ix) | (1<<jx) | (1<<ax) | (1<<bx),
            ((det.occ[ix], det.virt[ax]), (det.occ[jx], det.virt[bx])))


def amp_assembler(indx_a, indx_b, t2, det):
    indxs = [indx_a.occ, indx_b.occ, indx_a.virt, indx_b.virt]
    amps = amp_gather(indxs, [0, 1]*2, t2[0], det)
    amps += amp_gather(indxs[:2] + indxs[:1:-1], [0, 1, 1, 0], t2[1], det)
    amps += amp_gather(indxs[1::-1] + indxs[-2:], [1, 0, 0, 1], t2[1], det)
    amps += amp_gather(indxs[1::-1] + indxs[:1:-1], [1, 0]*2, t2[0], det)
    return amps


def amp_gather(indx, offset, t2, det):
    return [t2_gen(amp, ix, jx, ax, bx, det)
            for i in indx[0] for j in indx[1]
            if (ix := 2 * i + offset[0]) < (jx := 2 * j + offset[1])
            for a in indx[2] for b in indx[3]
            if ((amp := t2[i, j, a - det.nocc, b - det.nocc]) !=0
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
        
    det = Determinant(nocc, indx_a, indx_b)
    
    t1a, t1b = mapper(t1_ixer, zip([indx_a, indx_b]), t1=t1, det=det)
    t1_amps = t1a + t1b
    if len(t1_amps) > 0:
        t1_amps = tuple([np.array(col) for col in zip(*t1_amps)])
    else:
        t1_amps = (np.empty(0), np.empty(0), ())
    
    t2_idxz = zip([[indx_a.occ]*2 + [indx_a.virt]*2,
                   [indx_b.occ]*2 + [indx_b.virt]*2],
                  [[0]*4, [1]*4])
    t2aa, t2bb = mapper(amp_gather, t2_idxz, t2=t2[2], det=det)
    t2ab = amp_assembler(indx_a, indx_b, t2, det)
    t2_amps = t2aa + t2bb + t2ab
    t2_amps = (tuple([np.array(col) for col in zip(*t2_amps)]) if len(t2_amps) > 0
               else (np.empty(0), np.empty(0), ()))

    coeffs = (np.sum([combine_amps(t1_amps, t2_amps, num_x, k)
                      for k in range(num_x // 2 + 1)])
                      if t1_amps[0].any() or t2_amps[0].any() else 0)

    return coeffs



def combine_amps(t1_amps, t2_amps, num_x, n2):
    n1 = num_x - 2 * n2

    if n1 > len(t1_amps[0]) or n2 >len(t2_amps[0]):
        return 0

    t1_out = (prodofcombs(t1_amps, n1, 2) if n1 != 0
              else (np.array(1), np.array([0]), np.empty((1, 0, 2), dtype=int)))
    if not t1_out[0].any():
        return 0
    
    t2_out = (prodofcombs(t2_amps, n2, 4) if n2 != 0
              else (np.array(1), np.array([0]), np.empty((1, 0, 2, 2), dtype=int)))
    if not t2_out[0].any():
        return 0

    mask = ~np.bitwise_and.outer(t1_out[1], t2_out[1]).astype(bool)

    total_prod = np.outer(t1_out[0], t2_out[0])[mask]

    if total_prod.size == 0:
        return 0
    
    sign_array = np.array([signed(t1_out[2][i], t2_out[2][j])
                           for i, j in np.argwhere(mask)])
    
    return np.sum(total_prod * sign_array)


def signed(t1_perms, t2_perms):
    sources = np.concatenate((t1_perms[:, 0], t2_perms[:, :, 0].ravel()))
    targets = np.concatenate((t1_perms[:, 1], t2_perms[:, :, 1].ravel()))

    max_node = max(np.max(sources), np.max(targets)) + 1

    adjacency_list = [[] for _ in range(max_node)]
    for source, target in zip(sources, targets):
        adjacency_list[source].append(target)

    visited = set()
    chains = []

    def dfs(start):
        visited.add(start)
        chain = [start]
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in adjacency_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    chain.append(neighbor)
                    stack.append(neighbor)
        return chain

    for node in range(max_node):
        if node not in visited and adjacency_list[node]:
            chain = dfs(node)
            chains.append(chain)

    return (-1) ** np.array([len(chain) - 1 for chain in chains]).sum()



@lru_cache(maxsize=256)
def comb_index(n: int, k: int):
    count = comb(n, k, exact=True)
    index = np.fromiter(chain.from_iterable(combinations(range(n), k)),
                        np.uint16, count=count*k)
    return index.reshape(-1, k)


def prodofcombs(amps, n, m):
    coeffs, indxs, permix = amps
    if np.prod(np.sort(np.abs(coeffs))[-n:]) < 1e-8:
        return np.empty(0), np.empty(0, dtype=int)

    combix = comb_index(len(coeffs), n)
    orbs = indxs[combix]
    rorbs = np.bitwise_or.reduce(orbs, axis=1)
    porbs = np.unpackbits(rorbs.view(np.uint8)).reshape(rorbs.shape + (-1,))
    mask = porbs.sum(axis=1) == n * m

    vecofprods = np.prod(coeffs[combix][mask], axis=1)[:, None]
    vecoforbs = rorbs[mask]
    vecofperms = permix[combix][mask]

    return vecofprods, vecoforbs, vecofperms


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
