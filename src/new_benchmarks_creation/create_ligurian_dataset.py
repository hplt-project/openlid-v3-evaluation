import os
import json
from iso639 import Lang
from iso639.exceptions import DeprecatedLanguageValue
out = '../../data/ITDI_2022'
os.makedirs(out, exist_ok=True)

with open(os.path.join(out, 'test.jsonl'), 'w') as out:
    with open(os.path.expanduser("ITDI_2022/task/test_gold_standard.txt"), "r") as f:
        for line in f:
            script = "Latn"
            lang = line[:3].lower()
            text = line[4:].strip()
            try:
                code = Lang(lang)
            except DeprecatedLanguageValue as e:
                lang = "eml"
                print(e)
            out.write(
                json.dumps(
                    {
                        "id": lang + "_" + script,
                        "sentence": text,
                        "language": code.name,
                        "iso639-3": lang,
                        "iso15924": script,
                    }, ensure_ascii=False,
                ) + '\n',
            )
