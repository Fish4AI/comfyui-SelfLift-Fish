"""Regression for saturated score bins. Does not require a GPU."""
import unittest

def bin_of(relative):
    return int(relative / .25) if relative < 24 else min(127,96+int((relative-24)/2))

def cutoff(histogram,budget):
    count=0
    for b in range(127,-1,-1):
        count+=histogram[b]
        if count>budget:return b+1
    return 0

class TokenBudgetTests(unittest.TestCase):
    def check(self,scores,budget=256):
        bins=[bin_of(s+8) if s+8>=0 else -1 for s in scores]
        histogram=[bins.count(b) for b in range(128)]
        edge=cutoff(histogram,budget)
        selected=[i for i,b in enumerate(bins) if b>=edge]
        self.assertLessEqual(len(selected),budget)
        return edge,selected

    def test_saturated_top_bin_cannot_be_partially_admitted(self):
        # Old code turned cutoff bin 128 into score 80. Scores above it were
        # still admitted, even though the ENTIRE saturated bin overflowed.
        edge,selected=self.check([100.0]*18711)
        self.assertEqual(edge,128);self.assertEqual(selected,[])
        self.assertEqual(sum(s>=80 for s in [100.0]*18711),18711)

    def test_exact_budget_remains_eligible(self):
        _,selected=self.check([100.0]*256+[-9.0]*100)
        self.assertEqual(selected,list(range(256)))

    def test_whole_bins_and_original_score_window(self):
        _,selected=self.check([20.0]*200+[10.0]*200+[-9.0]*300)
        self.assertEqual(selected,list(range(200)))

if __name__=='__main__':unittest.main()
