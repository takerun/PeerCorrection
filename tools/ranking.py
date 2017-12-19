# coding: utf-8

import numpy as np
import collections
import scipy.stats

class RankingMeasures:
    def __init__(self, hypos, refs):
        assert len(hypos) == len(refs)
        self.hypos = np.array(hypos)
        self.refs = np.array(refs)
        self.hypo_ranking = self.get_ranking(hypos)
        self.ref_ranking = self.get_ranking(refs)

    def get_ranking(self, values):
        # ranking = (rank of i_th item)
        idxs = np.argsort(values)[::-1]
        ranking = np.zeros(len(idxs))
        for i, idx in enumerate(idxs): ranking[idx] = i
        ties = collections.defaultdict(list)
        for idx in idxs: ties[values[idx]].append(idx)
        for tie in ties.values():
            s = np.mean([ranking[idx] for idx in tie])
            for idx in tie: ranking[idx] = s
        return ranking

    def mrr(self):
        ref_best = np.argsort(self.refs)[::-1][0]
        hypo_guesses = np.argsort(self.hypos)[::-1]
        return 1. / (list(hypo_guesses).index(ref_best) + 1.)

    def precision_to_find_best_with_k_guess(self, k):
        assert k > 0 and k <= len(self.refs)
        ref_best = np.argsort(self.refs)[::-1][0]
        hypo_guesses = np.argsort(self.hypos)[::-1][:k]
        if ref_best in hypo_guesses: return 1.
        else: return 0.

    def precision_at_one(self):
        hypo_best = np.argsort(self.hypos)[::-1][0]
        ref_best = np.argsort(self.refs)[::-1][0]
        if self.refs[hypo_best] == self.refs[ref_best]: return 1.
        else: return 0.

    def nDCG(self, k):
        idxs = np.argsort(self.hypos)[::-1][:k]
        refs_k = self.refs[idxs]
        return self.DCG(refs_k) / self.DCG(sorted(self.refs, reverse=True)[:k])

    def DCG(self, values):
        n = len(values)
        s = values[0]
        for i in range(1, n): s += values[i] / np.log2(i+1)
        return s

    def spearman_corr(self):
        return scipy.stats.spearmanr(self.hypo_ranking, self.ref_ranking)[0]

    def corr(self):
        return np.corrcoef(self.hypo_ranking, self.ref_ranking)[1, 0]

    def tie_groups(self, values):
        ties = collections.defaultdict(list)
        for i in range(len(values)): ties[values[i]].append(i)
        group = []
        for tie in ties.values():
            if len(tie) > 1: group.append(len(tie))
        return np.array(group)

    def kendall(self):
        x = self.hypo_ranking
        y = self.ref_ranking
        n = len(x)
        n0 = n * (n-1) * 0.5
        ti = self.tie_groups(x)
        uj = self.tie_groups(y)
        n1 = 0.5 * sum([t * (t-1) * 0.5 for t in ti])
        n2 = 0.5 * sum([u * (u-1) * 0.5 for u in uj])

        nc = 0
        nd = 0

        for i in range(n):
            for j in range(n):
                if i >= j: continue
                if x[i] > x[j] and y[i] > y[j]: nc += 1
                elif x[i] < x[j] and y[i] < y[j]: nc += 1
                elif x[i] > x[j] and y[i] < y[j]: nd += 1
                elif x[i] < x[j] and y[i] > y[j]: nd += 1

        return (nc - nd) / np.sqrt((n0-n1) * (n0-n2))

if __name__ == '__main__':
    hypos = [6, 5, 4, 3, 2, 1]
    refs = [3, 2, 3, 0, 1, 2]
    rm = RankingMeasures(hypos, refs)
    print rm.nDCG(1)
