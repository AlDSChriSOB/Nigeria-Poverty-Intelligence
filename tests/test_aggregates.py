import csv, unittest
from pathlib import Path
class TestAggregates(unittest.TestCase):
 def test_baseline(self):
  with (Path(__file__).resolve().parents[1]/'nigeria_poverty_preliminary.csv').open() as f:
   rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),9)
  n=next(r for r in rows if r['dimension']=='National')
  self.assertEqual(int(n['households']),4715)
  self.assertAlmostEqual(float(n['poverty_pct']),48.28825241967344)
  for dim in ('Zone','Sector'):
   self.assertEqual(sum(int(r['households']) for r in rows if r['dimension']==dim),4715)
  for r in rows:
   self.assertTrue(0<=float(r['poverty_pct'])<=100)
