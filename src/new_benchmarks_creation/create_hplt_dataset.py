import pandas as pd
import os
import json
import math
from collections import defaultdict

languages = {
    "bos_Latn": [0, 1], "srp_Cyrl": [0], "hrv_Latn": [0, 1], "spa_Latn": [0, 1], "cmn_Hans": [0], "cat_Latn": [0],
    "ast_Latn": [0], "jpn_Jpan": [0],
    "hin_Deva": [0], "fin_Latn": [2, 3], "ita_Latn": [0], "glg_Latn": [0], "slk_Latn": [1], "yor_Latn": [0],
    "fra_Latn": [0], "por_Latn": [0],
    "nob_Latn": [1, 2], "ces_Latn": [0],
    "deu_Latn": [1],
    "eng_Latn": [0], "pes_Arab": [3, 4], "rus_Cyrl": [2, 4], "ell_Grek": [0, 1, 2, 3, 4],
}
out = '../evaluation/data/hplt'
os.makedirs(out, exist_ok=True)
counter = defaultdict(int)
with open(os.path.join(out, 'test.jsonl'), 'w') as test:
    for lang, batches in languages.items():
        # with open(os.path.join(out, f'{lang}-was-wrong.jsonl'), 'w') as wrong_out:
        for batch in batches:
            dirpath = f"release3_inspection/annot_round1/{lang}/batch{batch}.tsv"
            data = pd.read_csv(dirpath, sep='\t')
            for line in data.iterrows():
                line = line[1]
                porn = line[0]
                try:
                    if not math.isnan(porn):
                        continue
                except TypeError:
                    continue
                id_ = ""
                bad = False
                if (line[3] == 1) or (isinstance(line[3], str) and line[3].strip() == "1"):  # "lang correct?\n0/1"
                    id_ = lang
                unnatural = line[2]
                try:
                    if not math.isnan(unnatural):
                        id_ = ""
                        bad = True
                except TypeError:
                    id_ = ""
                    bad = True
                artifacts = line[1]
                try:
                    if not math.isnan(artifacts):
                        id_ = ""
                        bad = True
                except TypeError:
                    id_ = ""
                    bad = True

                if id_:
                    test.write(
                        json.dumps(
                            {
                                "id": id_,
                                "sentence": line["text_show"].replace('\n', ' ').strip(),
                                "language": lang,
                                "iso639-3": lang,
                                "iso15924": lang.split('_')[-1],
                            }, ensure_ascii=False,
                        ) + '\n',
                    )
                    counter[id_] += 1
                # else:
                #     if not bad:
                #         wrong_out.write(
                #             json.dumps(
                #                 {
                #                     "sentence": line["text_show"].replace('\n', ' ').strip(),
                #                 }, ensure_ascii=False,
                #             ) + '\n',
                #         )
print(counter)
