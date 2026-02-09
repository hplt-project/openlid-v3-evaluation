from huggingface_hub import hf_hub_download
import fasttext
from iso639 import Lang

#model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
model_path = "/home/m/PycharmProjects/OpenLID-v2/new_data/oci_no_pilar_frp/train/openlid-v3.bin"
model = fasttext.load_model(model_path)
labels = model.get_labels()
labels.sort()
for label in labels:
    label = label.removeprefix("__label__")
    langcode, script = label.split("_")
    lang = Lang(langcode)
    name = lang.name
    if lang.other_names():
        try:
            name += f" ({', '.join(lang.other_names())})"
        except TypeError:
            pass
    print("\langlabel{"+label.replace('_', "\_") + "} & " + name + " \\\\")
print(len(labels))
