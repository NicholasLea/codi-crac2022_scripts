import helper

# helper.convert_coref_ua_to_json('D:\Dataset\AMI\AMI_train_v2.2022.CONLLUA', 'D:\Dataset\AMI\AMI_train_v2.2022.jsonlines', \
#                                 MODEL="coref-hoi", SEGMENT_SIZE=512, TOKENIZER_NAME="spanbert-base-cased")

# helper.convert_coref_ua_to_json('D:\Dataset\AMI\AMI_dev_v2.2022.CONLLUA', 'D:\Dataset\AMI\AMI_dev_v2.2022.jsonlines', \
#                                 MODEL="coref-hoi", SEGMENT_SIZE=512, TOKENIZER_NAME="spanbert-base-cased")


# 对于Bridging 和 Discourse Deixis，后面再说。我本次可能只是对Identity Anaphora进行微调

# helper.convert_coref_ua_to_json('D:\Dataset\LIGHT\light_train.2022.CONLLUA', \
#                                 'D:\Dataset\LIGHT\light_train.2022.jsonlines', \
#                                 MODEL="coref-hoi", SEGMENT_SIZE=512, TOKENIZER_NAME="spanbert-base-cased")


# helper.convert_coref_ua_to_json('D:\Dataset\LIGHT\light_dev.2022.CONLLUA', \
#                                 'D:\Dataset\LIGHT\light_dev.2022.jsonlines', \
#                                 MODEL="coref-hoi", SEGMENT_SIZE=512, TOKENIZER_NAME="spanbert-base-cased")

# mac代码
helper.convert_coref_ua_to_json('/Users/nicholas/Documents/Dataset/LIGHT/light_train.2022.CONLLUA', \
                                '/Users/nicholas/Documents/Dataset/LIGHT/light_train.2022.bert-base-cased.jsonlines', \
                                MODEL="coref-hoi", SEGMENT_SIZE=512, TOKENIZER_NAME="bert-base-cased")

helper.convert_coref_ua_to_json('/Users/nicholas/Documents/Dataset/LIGHT/light_dev.2022.CONLLUA', \
                                '/Users/nicholas/Documents/Dataset/LIGHT/light_dev.2022.bert-base-cased.jsonlines', \
                                MODEL="coref-hoi", SEGMENT_SIZE=512, TOKENIZER_NAME="bert-base-cased")

# 2023-03-15 目前看起来，数据处理就没什么问题哎。而且其余暂时不用到的部分就先不用看了，否则看起来也是看不完看不动。