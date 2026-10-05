#!/usr/bin/env python3
"""BW1100 evidence-scope checks layered on the retained upstream validator."""
import math,re,sys
from pathlib import Path
from validate import collect_all_ids,extract_frontmatter
ROOT=Path(__file__).resolve().parents[1]
SCOPES={'environment-snapshot','compile-only','original-device-correctness','paired-component','paired-original-task','component-only','partial-runtime-triage','invalid','protocol-only'}
def check_page(fm):
 errors=[]
 if fm.get('type')!='source-experiment':return errors
 scope=fm.get('evidence_scope')
 if scope not in SCOPES:errors.append('unknown evidence_scope')
 if not isinstance(fm.get('evidence_root'),str) or not fm['evidence_root'].strip():errors.append('missing evidence root')
 if not isinstance(fm.get('artifacts'),list) or not fm['artifacts'] or any(not isinstance(p,str) or not p for p in fm['artifacts']):errors.append('missing concrete artifact paths')
 if fm.get('confidence')=='verified':errors.append('local source cannot acquire broad official/upstream verified confidence')
 if scope in {'compile-only','protocol-only','invalid'} and fm.get('performance_claims'):errors.append('non-performance scope carries a performance claim')
 if scope in {'paired-component','paired-original-task'}:
  for key in ['compiler','dtype','shape','baseline','measurement']:
   if not isinstance(fm.get(key),str) or not fm[key].strip():errors.append('paired scope missing '+key)
 for p in fm.get('performance_claims',[]):
  if not isinstance(p,dict):errors.append('malformed performance claim');continue
  for key in ['gpu','dtype','shape','baseline','metric','value','source_id']:
   if key not in p:errors.append('performance missing '+key)
  if isinstance(p.get('value'),(int,float)) and not math.isfinite(p['value']):errors.append('nonfinite metric')
 return errors

def main():
 errors=[]
 for directory in ['sources','wiki']:
  for p in (ROOT/directory).rglob('*.md'):
   fm,body=extract_frontmatter(p)
   if fm:
    errors += [str(p.relative_to(ROOT))+': '+e for e in check_page(fm)]
    if fm.get('type','').startswith('wiki-') and not fm.get('sources'):errors.append(str(p)+': synthesis lacks sources')
   if re.search(r'(?i)(ANTHROPIC_AUTH_TOKEN\s*=|sk-[a-z0-9]{12,}|password\s*[:=]\s*\S+)',body):errors.append(str(p)+': possible credential material')
 print('Evidence checks:', 'passed' if not errors else 'failed')
 for e in errors:print(e)
 return bool(errors)
if __name__=='__main__':sys.exit(main())
