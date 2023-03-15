# 测试 get_document
from preprocess import get_document, get_tokenizer, get_all_docs, UADocumentState, normalize_word, split_into_segments

UA_PATH, JSON_PATH, MODEL, SEGMENT_SIZE, TOKENIZER_NAME = \
'/Users/nicholas/Documents/Dataset/LIGHT/light_dev.2022.CONLLUA', \
'/Users/nicholas/Documents/Dataset/LIGHT/light_train.2022.bert-base-cased.jsonlines', \
"coref-hoi", 512,  "bert-base-cased"

tokenizer = get_tokenizer(TOKENIZER_NAME)
key_docs, key_doc_sents = get_all_docs(UA_PATH)

# TOKENIZER_NAME = "bert-base-uncased"

tokenizer = get_tokenizer(TOKENIZER_NAME)
doc = 'light_dev/episode_7296'

# 设置某个指定的词。用Nikkei这个词，uncased和cased就会得出不一样的结果
# word_test = 'Nikkei'
# subtokens_test = tokenizer.tokenize(word_test)
# print('word_test', word_test)
# print('subtokens_test', subtokens_test)

# def get_document1(doc_key, doc_lines, language, seg_len, tokenizer):
#     """ Process raw input to finalized documents """
#     document_state = UADocumentState(doc_key) #这个对象刚初始化完的时候，只有self.doc_key是有值的，其他的都是空的
#     word_idx = -1
#
#     # Build up documents
#     for line in doc_lines:
#         # print('line', line)
#         row = line.split()  # Columns for each token
#         # print('row', row)
#         if len(row) == 0:
#             document_state.sentence_end[-1] = True
#         else:
#             word_idx += 1
#             word = normalize_word(row[1], language) # 从1开始是因为0是行index
#             subtokens = tokenizer.tokenize(word)
#
#             # if word != subtokens[0]:
#             #     print('word', word)
#             #     print('subtokens', subtokens)
#
#             document_state.tokens.append(word)  # 注意这里的tokens是会有重复的！
#             document_state.token_end += [False] * (len(subtokens) - 1) + [True]
#
#             for idx, subtoken in enumerate(subtokens):
#                 document_state.subtokens.append(subtoken)
#                 info = None if idx != 0 else (row + [len(subtokens)])
#                 document_state.info.append(info)
#                 document_state.sentence_end.append(False)
#                 document_state.subtoken_map.append(word_idx)
#
#     # Split documents
#     constraits1 = document_state.sentence_end if language != 'arabic' else document_state.token_end
#     split_into_segments(document_state, seg_len, constraits1, document_state.token_end, tokenizer)
#     document = document_state.finalize()
#     return document

print("key_docs[doc]", key_docs[doc])
document = get_document(doc, key_docs[doc], 'english', SEGMENT_SIZE, tokenizer)
print('document:', document) # OJBK，到这里就正确跑出来了。下面拆开来看看

a = '))'
print(a)
print(a.split('('))

a = 'abc'
print(a)
print(a.split('d'))






