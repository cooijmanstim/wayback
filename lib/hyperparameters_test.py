import unittest, functools as ft
import lib.hyperparameters as hyperparameters
from lib.namespace import Namespace as NS

class Test(unittest.TestCase):
  def setUp(self):
    pass

  def test_parse(self):
    self.assertEqual(
      hyperparameters.parse_value("""{a: 1, b: [2e0, 3., four], c: {d: "five", "e": False}}"""),
      NS(a=1, b=[2., 3., "four"], c=NS(d="five", e=False)))

    self.assertRaises(hyperparameters.ParseError, ft.partial(hyperparameters.parse_value, """{a:1, b: [fn()]}"""))
    self.assertRaises(hyperparameters.ParseError, ft.partial(hyperparameters.parse_value, """{a:1, b: dict(c=2)}"""))

if __name__ == "__main__":
  unittest.main()
