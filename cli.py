import json, sys
from icd_suggester import IcdSuggester

note = " ".join(sys.argv[1:]) or sys.stdin.read()
print(json.dumps(IcdSuggester().fit().suggest(note), indent=2))
