"""Check the task metadata and the scoped, recorded submission audit."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
spec=json.loads((root/'task.json').read_text())
required=('title','category','instruction','setup','scoring','run_limit','fidelity','scores','media')
issues=[f'Missing required field: {key}' for key in required if not spec.get(key)]
if len(spec.get('fidelity',''))>500:
    issues.append('Fidelity note exceeds the task board limit of500characters.')
if any(word in spec.get('scores','').lower() for word in ('not run','pending')):
    issues.append('Required model evaluation is pending.')
for name in ('yam/task.py','yam/evaluate.sh','yam/dream_continuity.py','yam/analyze_evaluation.py'):
    if not (root/name).is_file():issues.append(f'Missing task-local reproduction file: {name}')
audit_file=root/'yam/submission-audit.json'
if not audit_file.is_file():
    issues.append('Submission audit has not been recorded.')
else:
    audit=json.loads(audit_file.read_text())
    required_checks=('task_board_review','scored_media_verified','isolated_task_reproduction',
                     'press_only_interlock_probe','table_support_regrade','live_support_measurement')
    for check in required_checks:
        if audit.get('checks',{}).get(check) is not True:issues.append(f'Unfinished audit: {check}')
    for name,expected in audit.get('current_source_sha256',{}).items():
        path=root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            issues.append(f'Source changed since audit: {name}')
    if not audit.get('current_source_sha256'):issues.append('Audit source hashes are missing.')
if issues:
    print('Submission is not ready:')
    for issue in issues:print(f'- {issue}')
    raise SystemExit(1)
print('Submission metadata, task-local files and scoped audit pass. Reviewer acceptance is not guaranteed.')
