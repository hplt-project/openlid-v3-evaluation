import json
import os
import re

from alphabet_detector import AlphabetDetector
from iso639 import Lang
import pandas as pd
from glob import glob
from datasets import load_dataset

ad = AlphabetDetector()


def make_twi():
    out = '../../data/twi_hbs'
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(out, 'test.jsonl'), 'w') as twi_hbs:
        with open(os.path.expanduser("~/Downloads/Twitter-HBS/Twitter-HBS.json"), 'r') as twi_f:
            twi = json.load(twi_f)
            for line in twi:
                if line["language"] != "me":
                    code = Lang(line["language"])
                    iso_639 = code.pt3

                    for tweet in line["tweets"]:
                        script = "Latn"
                        if line["language"] == "sr":
                            scripts = ad.detect_alphabet(tweet)
                            if 'CYRILLIC' in scripts:
                                script = "Cyrl"
                        twi_hbs.write(
                            json.dumps(
                                {
                                    "id": iso_639 + "_" + script,
                                    "sentence": tweet,
                                    "language": code.name,
                                    "iso639-3": iso_639,
                                    "iso15924": script,
                                }, ensure_ascii=False,
                            ) + '\n',
                        )


def make_setimes():
    out = '../../data/setimes'
    os.makedirs(out, exist_ok=True)
    srp = {text.strip().lower() for text in pd.read_parquet(os.path.expanduser(f"/data/srp_Latn.parquet"))['text'].unique()}
    hrv = {text.strip().lower() for text in pd.read_parquet(os.path.expanduser(f"/data/hrv_Latn.parquet"))['text'].unique()}
    bos = {text.strip().lower() for text in pd.read_parquet(os.path.expanduser(f"/data/bos_Latn.parquet"))['text'].unique()}
    srp_cyrl = {text.strip().lower().rstrip('.') for text in pd.read_parquet(os.path.expanduser(
        f"/data/srp_Cyrl.parquet"))['text'].unique()}
    all = srp | hrv | bos | srp_cyrl

    glotlid = set()

    for file in glob("../../../../glotlid-corpus/v3.1/hbs_Latn/*.txt"):
        with open (file, 'r') as f:
            for line in f:
                glotlid.add(line.strip().lower().rstrip('.'))
    for file in glob("../../../../glotlid-corpus/v3.1/srp_Cyrl/*.txt"):
        with open(file, 'r') as f:
            for line in f:
                glotlid.add(line.strip().lower().rstrip('.'))

    total = 0
    dirty = 0
    glotlid_dirty = 0
    with open(os.path.join(out, 'test.jsonl'), 'w') as hbs_out:
        with open(os.path.expanduser("~/Downloads/SETimes.HBS/SETimes.HBS.json"), 'r') as f:
            hbs = json.load(f)
            for line in hbs:
                result = ''
                sentences = line['text'].split('.')
                n_sent = len(sentences)
                total += n_sent
                cyrl = 0
                for sent in sentences:
                    dirty_flag = False
                    proc = sent.strip().lower()
                    if proc in all:
                        dirty += 1
                        dirty_flag = True
                    if proc in glotlid:
                        glotlid_dirty += 1
                        dirty_flag = True
                    if not dirty_flag:
                        result += sent.strip() + '. '
                    if line["language"] == "sr":
                        scripts = ad.detect_alphabet(sent)
                        if scripts == {'CYRILLIC'}:
                            cyrl += 1
                if result:
                    code = Lang(line["language"])
                    iso_639 = code.pt3
                    script = "Latn"
                    if cyrl/n_sent > 0.5:
                        script = "Cyrl"
                    hbs_out.write(
                        json.dumps(
                            {
                                "id": iso_639 + "_" + script,
                                "sentence": result.strip(),
                                "language": code.name,
                                "iso639-3": iso_639,
                                "iso15924": script,
                            }, ensure_ascii=False,
                        ) + '\n',
                    )

    print(total)
    print(dirty/total)
    print(glotlid_dirty/total)


def make_parlasent():
    data = load_dataset("classla/ParlaSent", "BCS", split='train')
    out = '../../data/parlasent'
    os.makedirs(out, exist_ok=True)
    mapping = {"HR": "hrv", "SRB": "srp", "BiH": "bos"}
    with open(os.path.join(out, 'test.jsonl'), 'w') as hbs_out:
        for sample in data:
            script = "Latn"
            iso_639 = mapping[sample["country"]]
            hbs_out.write(
                json.dumps(
                    {
                        "id": iso_639 + "_" + script,
                        "sentence": sample["sentence"],
                        "language": sample["country"],
                        "iso639-3": iso_639,
                        "iso15924": script,
                    }, ensure_ascii=False,
                ) + '\n',
            )


def make_copa():
    out = '../../data/hs_copa'
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'test.jsonl'), 'w') as hbs_out:
        for file in glob(os.path.expanduser("~/Downloads/Choice of plausible alternatives dataset in */*.jsonl")):

                with open(file, 'r') as f:
                    for line in f:
                        line = json.loads(line)
                        sentence = ' '.join((line["premise"], line["choice1"], line["choice2"]))
                        script = "Latn"
                        iso_639 = "srp"
                        if "Croatian" in file:
                            iso_639 = "hrv"
                        else:
                            scripts = ad.detect_alphabet(sentence)
                            if scripts == {'CYRILLIC'}:
                                script = "Cyrl"
                        hbs_out.write(
                            json.dumps(
                                {
                                    "id": iso_639 + "_" + script,
                                    "sentence": sentence,
                                    "language": iso_639,
                                    "iso639-3": iso_639,
                                    "iso15924": script,
                                }, ensure_ascii=False,
                            ) + '\n',
                        )


# with open("../../results/maria/v3-setimes.jsonl", 'r') as f:
#     with open("../../results/maria/v3-setimes-correct.jsonl", 'w') as out:
#         for line in f:
#             line = json.loads(line)
#             if line["id"] == line["predictions"]["retrained"]:
#                 out.write(json.dumps(line, ensure_ascii=False)+'\n')
def heritage():
    pat = re.compile(r"\d+ \w: ")
    pat2 = re.compile(r"\d+")
    out = '../../data/he_bcs_ge_full'
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'test.jsonl'), 'w') as hbs_out:
        for file in glob(os.path.expanduser("~/Downloads/He-BCS-Ge/T*.txt")):
            current_paragraph = ""
            with open(file, 'r') as f:
                started = False

                for line in f:
                    if "Staatsangehörigkeit" in line:
                        if "Kroatin" in line:
                            iso_639 = "hrv"
                        elif "kroatisch" in line.lower():
                            iso_639 = "hrv"
                        elif "serbisch" in line.lower():
                            iso_639 = "srp"
                        elif "bosnisch" in line.lower():
                            iso_639 = "bos"

                    match = re.match(pat, line)
                    if (match is not None) and not (started) and ("B: " in line):
                        started = True
                        script = "Latn"

                        current_paragraph += line[match.span()[-1]:].rstrip('\n')
                    elif started and (match is None):
                        current_paragraph += re.sub(pat2, "", line).rstrip()
                    elif started and (match is not None) and ("B: " not in line):
                        started = False


            hbs_out.write(
                json.dumps(
                    {
                        "id": iso_639 + "_" + script,
                        "sentence": current_paragraph.rstrip(),
                        "language": iso_639,
                        "iso639-3": iso_639,
                        "iso15924": script,
                    }, ensure_ascii=False,
                ) + '\n',
            )

make_setimes()