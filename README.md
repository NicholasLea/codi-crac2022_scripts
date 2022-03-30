# codi2022_scripts
Scripts to convert the [Universal Anaphora](https://github.com/UniversalAnaphora/UniversalAnaphora/blob/main/documents/UA_CONLL_U_Plus_proposal_v1.0.md) format to jsonlines

1. `helper.py` contains scripts to convert UA to jsonlines format

2. `preprocess.py` contains scripts to parse annotation structure from UA documents

## Commands for conversion

```import helper```

#### Identity Anaphora

1. UA to jsonlines 
```helper.convert_coref_ua_to_json(UA_PATH, JSON_PATH, MODEL="coref-hoi", SEGMENT_SIZE=512, TOKENIZER_NAME="bert-base-cased")```

2. jsonlines to UA 
```helper.convert_coref_json_to_ua(JSON_PATH, UA_PATH, MODEL="coref-hoi")```

> **NOTE:** Currently, these scripts only support conversion to and from the format used by models that use bert/spanbert embeddings. E.g. [coref-hoi](https://github.com/lxucs/coref-hoi/).


#### Bridging

1. UA to jsonlines 
```helper.convert_bridg_ua_to_json(UA_PATH, JSON_PATH, MODEL="dali_bridging")```

2. jsonlines to UA 
```helper.convert_bridg_json_to_ua(JSON_PATH, UA_PATH, MODEL="dali-bridging")```

> **NOTE:** Currently, these scripts only support conversion to and from the format used by [dali-bridging](https://github.com/juntaoy/dali-bridging).


#### Discourse Deixis

1. Previous Utterance Baseline (for "this", "that")
```helper.discourse_deixis_baseline(IN_UA_PATH, PRED_UA_PATH, MODEL="previous-utterance")```

## Baseline Performance

|                                   | Model | AMI   | LIGHT | Persuasion | Swbd  | ARRAU (Trains91) |
| --------------------------------- | ----- | ----- | ----- | ---------- | ----- | ---------------- |
| Identity Anaphora (CoNLL Avg. F1) | [coref-hoi](https://github.com/lxucs/coref-hoi/) | TODO | TODO | TODO       | TODO | TODO            |
| Bridging (Entity F1)              | [dali-bridging](https://github.com/juntaoy/dali-bridging) | TODO  | TODO  | TODO       | TODO | TODO            |
| Discourse Deixis (CoNLL Avg. F1)  | [prev-utterance](https://github.com/sopankhosla/codi2021_scripts/blob/3509e2c588cd5097b4778b7754b0b1a89b06b478/helper.py#L377) | TODO | TODO | TODO       | TODO | TODO            |
