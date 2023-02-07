import helper

# helper.convert_coref_ua_to_json('D:\Dataset\AMI\AMI_train_v2.2022.CONLLUA', 'D:\Dataset\AMI\AMI_train_v2.2022.jsonlines', \
#                                 MODEL="coref-hoi", SEGMENT_SIZE=512, TOKENIZER_NAME="spanbert-base-cased")

# helper.convert_coref_ua_to_json('D:\Dataset\AMI\AMI_dev_v2.2022.CONLLUA', 'D:\Dataset\AMI\AMI_dev_v2.2022.jsonlines', \
#                                 MODEL="coref-hoi", SEGMENT_SIZE=512, TOKENIZER_NAME="spanbert-base-cased")


# 对于Bridging 和 Discourse Deixis，后面再说。我本次可能只是对Identity Anaphora进行微调

helper.convert_coref_ua_to_json('D:\Dataset\LIGHT\light_train.2022.CONLLUA', \
                                'D:\Dataset\LIGHT\light_train.2022.jsonlines', \
                                MODEL="coref-hoi", SEGMENT_SIZE=512, TOKENIZER_NAME="spanbert-base-cased")


# helper.convert_coref_ua_to_json('D:\Dataset\LIGHT\light_dev.2022.CONLLUA', \
#                                 'D:\Dataset\LIGHT\light_dev.2022.jsonlines', \
#                                 MODEL="coref-hoi", SEGMENT_SIZE=512, TOKENIZER_NAME="spanbert-base-cased")
