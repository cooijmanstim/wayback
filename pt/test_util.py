import unittest
import numpy as np, scipy.stats
import pt

class SampleTest(unittest.TestCase):
  def test_choice(self):
    return
    # chi-squared test using numpy's multinomial sampler for comparison
    p = np.exp(np.random.randn(10))
    p = p / p.sum()
    n = 10000
    q = 0
    for _ in range(n):
      q = q + np.random.multinomial(1, p)
    print("test_choice p", p)
    print("test_choice q", q / q.sum())
    chisq, pval = scipy.stats.chisquare(q, p * n)
    np.testing.assert_array_less(0.05, pval)

  def test_sample1(self):
    p = np.exp(np.random.randn(10))
    n = 10000
    q = 0
    for _ in range(n):
      q = q + pt.to_numpy(pt.sample(pt.from_numpy(p.astype(np.float32)), onehotted=True))
    p = p / p.sum()
    print("test_sample1 p", p)
    print("test_sample1 q", q / q.sum())
    chisq, pval = scipy.stats.chisquare(q, p * n)
    np.testing.assert_array_less(0.05, pval)

  def test_sample2(self):
    return
    axis = 1
    p = np.exp(np.random.randn(2, 3))
    n = 10000
    q = 0
    for _ in range(n):
      q = q + pt.to_numpy(pt.sample(pt.from_numpy(p.astype(np.float32)), dim=axis, onehotted=True))
    p = p / p.sum(axis=axis, keepdims=True)
    print("test_sample2 p", p)
    print("test_sample2 q", q / q.sum(axis=axis, keepdims=True))
    chisq, pval = scipy.stats.chisquare(q, p * n, axis=axis)
    np.testing.assert_array_less(0.05, pval)

  def test_sample_onehot(self):
    d = 10
    for p in np.eye(d):
      x = pt.to_numpy(pt.sample(pt.from_numpy(p.astype(np.float32)), onehotted=True))
      np.testing.assert_allclose(p, x)

if __name__ == "__main__":
  unittest.main()
