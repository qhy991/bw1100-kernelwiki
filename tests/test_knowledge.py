import copy,importlib.util,json,re,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import validate,get_page
spec=importlib.util.spec_from_file_location('evidence_checks',ROOT/'scripts/validate-evidence.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
class KnowledgeContract(unittest.TestCase):
 def test_strict_corpus_validation_and_references(self):
  result=subprocess.run([str(ROOT/'bwiki'),'validate'],capture_output=True,text=True)
  self.assertEqual(result.returncode,0,result.stdout+result.stderr)
 def test_retrieve_actual_mechanism_by_id(self):
  result=subprocess.run([str(ROOT/'bwiki'),'get','technique-execution-groups'],capture_output=True,text=True)
  self.assertEqual(result.returncode,0);self.assertIn('物理线程分摊',result.stdout);self.assertIn('exp-width-qualification',result.stdout)
 def test_filtered_query_reaches_moe_page(self):
  result=subprocess.run([str(ROOT/'bwiki'),'query','--architecture','gfx938','--kernel-type','moe','--type','kernel','--limit','20'],capture_output=True,text=True)
  self.assertEqual(result.returncode,0);self.assertIn('kernel-bw-moe-fp32',result.stdout)
 def test_example_spilling_query_has_actual_matches(self):
  result=subprocess.run([str(ROOT/'bwiki'),'query','--tag','register-spilling','--type','pattern'],capture_output=True,text=True)
  self.assertEqual(result.returncode,0);self.assertIn('pattern-fp32-staging',result.stdout)
 def test_keyword_and_type_filter_are_both_applied(self):
  result=subprocess.run([str(ROOT/'bwiki'),'query','RMSNorm','--type','kernel','--limit','20'],capture_output=True,text=True)
  self.assertEqual(result.returncode,0);self.assertIn('kernel-bw-rmsnorm',result.stdout);self.assertNotIn('kernel-bw-moe-fp32',result.stdout)
 def test_generated_manifest_matches_page_owner(self):
  pages=get_page.build_index();manifest=json.loads((ROOT/'queries/pages.json').read_text())
  self.assertEqual({p['id'] for p in manifest},set(pages));self.assertEqual(len(manifest),len(pages))
  for p in manifest:self.assertTrue((ROOT/p['path']).is_file())
 def test_compile_only_cannot_carry_a_speed_claim(self):
  fm,_=validate.extract_frontmatter(ROOT/'sources/experiments/exp-aiter-audit.md');bad=copy.deepcopy(fm);bad['performance_claims']=[dict(value=2.0)]
  self.assertIn('non-performance scope carries a performance claim',e.check_page(bad))
 def test_local_verified_does_not_bypass_scope(self):
  fm,_=validate.extract_frontmatter(ROOT/'sources/experiments/exp-width-qualification.md');bad=copy.deepcopy(fm);bad['confidence']='verified'
  self.assertIn('local source cannot acquire broad official/upstream verified confidence',e.check_page(bad))
 def test_paired_record_needs_its_denominator(self):
  fm,_=validate.extract_frontmatter(ROOT/'sources/experiments/exp-width-qualification.md');bad=copy.deepcopy(fm);del bad['baseline']
  self.assertIn('paired scope missing baseline',e.check_page(bad))
 def test_navigation_links_resolve(self):
  for p in [ROOT/'README.md',*list((ROOT/'queries').glob('*.md'))]:
   for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
    if link.startswith(('http','#')):continue
    self.assertTrue((p.parent/link.split('#')[0]).exists(),str(p)+': '+link)
if __name__=='__main__':unittest.main()
