import json
from glob import glob

with open("../results/maria/ensemble-hplt-hbs.jsonl","w") as out:
    for file in glob("../results/maria/bos_annotated/*was-wrong.jsonl"):
        print(file)
        v3 = file.replace("bos_annotated", "hbs_not_annotated_ensemble").replace(
            "glotlid", "v3"
        ).replace(".jsonl", "-ensemble-at1.jsonl")
        with open(file, 'r') as f:
            with open(v3, "r") as v3_f:
                for line, line_v3 in zip(f, v3_f):
                    line = json.loads(line)
                    text = line["sentence"]

                    line_v3 = json.loads(line_v3)
                    text_v3 = line_v3["sentence"]

                    assert text == text_v3
                    line_v3["id"] = line["id"]
                    out.write(json.dumps(line_v3) + "\n")
