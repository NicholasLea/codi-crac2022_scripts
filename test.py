import helper

# 对于Bridging 和 Discourse Deixis，后面再说。我本次可能只是对Identity Anaphora进行微调
# 2023-03-15 目前看起来，数据处理就没什么问题哎。而且其余暂时不用到的部分就先不用看了，否则看起来也是看不完看不动。

###
segment_size = 384
model = "coref-hoi"
tokenize_name = "bert-base-cased"

#light
helper.convert_coref_ua_to_json('/Users/nicholas/Documents/Dataset/CODI-CRAC22-Corpus-main/light_train.2022.CONLLUA', \
                                '/Users/nicholas/Documents/Dataset/CODI-CRAC22-Corpus-main/light.train.2022.384.bert-base-cased.jsonlines', \
                                MODEL=model, SEGMENT_SIZE=segment_size, TOKENIZER_NAME=tokenize_name)
helper.convert_coref_ua_to_json('/Users/nicholas/Documents/Dataset/CODI-CRAC22-Corpus-main/light_dev.2022.CONLLUA', \
                                '/Users/nicholas/Documents/Dataset/CODI-CRAC22-Corpus-main/light.dev.2022.384.bert-base-cased.jsonlines', \
                                MODEL=model, SEGMENT_SIZE=segment_size, TOKENIZER_NAME=tokenize_name)
helper.convert_coref_ua_to_json('/Users/nicholas/Documents/Dataset/CODI-CRAC22-Corpus-main/light_test.2022.CONLLUA', \
                                '/Users/nicholas/Documents/Dataset/CODI-CRAC22-Corpus-main/light.test.2022.384.bert-base-cased.jsonlines', \
                                MODEL=model, SEGMENT_SIZE=segment_size, TOKENIZER_NAME=tokenize_name)

#AMI
helper.convert_coref_ua_to_json('/Users/nicholas/Documents/Dataset/CODI-CRAC22-Corpus-main/AMI_train_v2.2022.CONLLUA', \
                                '/Users/nicholas/Documents/Dataset/CODI-CRAC22-Corpus-main/AMI.train.2022.384.bert-base-cased.jsonlines', \
                                MODEL=model, SEGMENT_SIZE=segment_size, TOKENIZER_NAME=tokenize_name)
helper.convert_coref_ua_to_json('/Users/nicholas/Documents/Dataset/CODI-CRAC22-Corpus-main/AMI_dev_v2.2022.CONLLUA', \
                                '/Users/nicholas/Documents/Dataset/CODI-CRAC22-Corpus-main/AMI.dev.2022.384.bert-base-cased.jsonlines', \
                                MODEL=model, SEGMENT_SIZE=segment_size, TOKENIZER_NAME=tokenize_name)
helper.convert_coref_ua_to_json('/Users/nicholas/Documents/Dataset/CODI-CRAC22-Corpus-main/AMI_test.2022.CONLLUA', \
                                '/Users/nicholas/Documents/Dataset/CODI-CRAC22-Corpus-main/AMI.test.2022.384.bert-base-cased.jsonlines', \
                                MODEL=model, SEGMENT_SIZE=segment_size, TOKENIZER_NAME=tokenize_name)

#Persuasion
helper.convert_coref_ua_to_json('/Users/nicholas/Documents/Dataset/CODI-CRAC22-Corpus-main/Persuasion_train.2022.CONLLUA', \
                                '/Users/nicholas/Documents/Dataset/CODI-CRAC22-Corpus-main/Persuasion.train.2022.384.bert-base-cased.jsonlines', \
                                MODEL=model, SEGMENT_SIZE=segment_size, TOKENIZER_NAME=tokenize_name)
helper.convert_coref_ua_to_json('/Users/nicholas/Documents/Dataset/CODI-CRAC22-Corpus-main/Persuasion_dev.2022.CONLLUA', \
                                '/Users/nicholas/Documents/Dataset/CODI-CRAC22-Corpus-main/Persuasion.dev.2022.384.bert-base-cased.jsonlines', \
                                MODEL=model, SEGMENT_SIZE=segment_size, TOKENIZER_NAME=tokenize_name)
helper.convert_coref_ua_to_json('/Users/nicholas/Documents/Dataset/CODI-CRAC22-Corpus-main/Persuasion_test.2022.CONLLUA', \
                                '/Users/nicholas/Documents/Dataset/CODI-CRAC22-Corpus-main/Persuasion.test.2022.384.bert-base-cased.jsonlines', \
                                MODEL=model, SEGMENT_SIZE=segment_size, TOKENIZER_NAME=tokenize_name)
###