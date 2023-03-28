import json
import os
from os import walk, makedirs
from os.path import isfile, join
import collections
from collections import deque
import sys
import logging
import re
import numpy as np
import random
from transformers import BertTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

def flatten(l):
    return [item for sublist in l for item in sublist]


def get_tokenizer(bert_tokenizer_name):
    return BertTokenizer.from_pretrained(bert_tokenizer_name)


def skip_doc(doc_key):
    return False

def normalize_word(word, language):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?": # 这里应该是没有转义的意思，就是单纯的如果是/.，那么就只取.
        # print('normalize_word 0', word)
        # print('normalize_word 1', word)
        return word[1:]
    else:
        return word


def get_sentence_map(segments, sentence_end):
    assert len(sentence_end) == sum([len(seg) - 2 for seg in segments])  # of subtokens in all segments
    sent_map = []
    sent_idx, subtok_idx = 0, 0
    for segment in segments:
        sent_map.append(sent_idx)  # [CLS]
        for i in range(len(segment) - 2):
            sent_map.append(sent_idx)
            sent_idx += int(sentence_end[subtok_idx])
            subtok_idx += 1
        sent_map.append(sent_idx)  # [SEP]
    return sent_map


class Markable:
    def __init__(self, doc_name, start, end, MIN, is_referring, words,is_split_antecedent=False,split_antecedent_members=set()):
        self.doc_name = doc_name
        self.start = start
        self.end = end
        self.MIN = MIN
        self.is_referring = is_referring
        self.words = words
        self.is_split_antecedent = is_split_antecedent
        self.split_antecedent_members = split_antecedent_members

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            # for split-antecedent we check all the members are the same
            if self.is_split_antecedent or other.is_split_antecedent:
                return self.split_antecedent_members == other.split_antecedent_members
            # MIN is only set for the key markables
            elif self.MIN:
                return (self.doc_name == other.doc_name
                        and other.start >= self.start
                        and other.start <= self.MIN[0]
                        and other.end <= self.end
                        and other.end >= self.MIN[1])
            elif other.MIN:
                return (self.doc_name == other.doc_name
                        and self.start >= other.start
                        and self.start <= other.MIN[0]
                        and self.end <= other.end
                        and self.end >= other.MIN[1])
            else:
                return (self.doc_name == other.doc_name
                        and self.start == other.start
                        and self.end == other.end)
        return NotImplemented

    def __neq__(self, other):
        if isinstance(other, self.__class__):
            return self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        if self.is_split_antecedent:
            return hash(frozenset(self.split_antecedent_members))
        return hash(frozenset((self.start, self.end)))

    def __short_str__(self):
        return ('({},{})'.format(self.start,self.end))

    def __str__(self):
        if self.is_split_antecedent:
            return str([cl[0].__short_str__() for cl in self.split_antecedent_members])
        return self.__short_str__()
            # ('DOC: %s SPAN: (%d, %d) String: %r MIN: %s Referring tag: %s'
            #	 % (
            #		 self.doc_name, self.start, self.end, ' '.join(self.words),
            #		 '(%d, %d)' % self.MIN if self.MIN else '',
            #		 self.is_referring))

class DocumentState(object):
    def __init__(self, key):
        self.doc_key = key
        self.tokens = []

        # Linear list mapped to subtokens without CLS, SEP
        self.subtokens = []
        self.subtoken_map = []
        self.token_end = []
        self.sentence_end = []
        self.info = []  # Only non-none for the first subtoken of each word

        # Linear list mapped to subtokens with CLS, SEP
        self.sentence_map = []

        # Segments (mapped to subtokens with CLS, SEP)
        self.segments = []
        self.segment_subtoken_map = []
        self.segment_info = []  # Only non-none for the first subtoken of each word
        self.speakers = []

        # Doc-level attributes
        self.pronouns = []
        self.clusters = collections.defaultdict(list)  # {cluster_id: [(first_subtok_idx, last_subtok_idx) for each mention]}
        self.coref_stacks = collections.defaultdict(list)

    def finalize(self):
        print('调用父类的finalize')
        """ Extract clusters; fill other info e.g. speakers, pronouns """
        # Populate speakers from info
        subtoken_idx = 0
        for seg_info in self.segment_info:
            speakers = []
            for i, subtoken_info in enumerate(seg_info):
                if i == 0 or i == len(seg_info) - 1:
                    speakers.append('[SPL]')
                elif subtoken_info is not None:  # First subtoken of each word
                    speakers.append(subtoken_info[9])
                    # if subtoken_info[4] == 'PRP':  # Uncomment if needed
                    #     self.pronouns.append(subtoken_idx)
                else:
                    speakers.append(speakers[-1])
                subtoken_idx += 1
            self.speakers += [speakers]

        # Populate cluster
        first_subtoken_idx = 0  # Subtoken idx across segments
        subtokens_info = flatten(self.segment_info)
        while first_subtoken_idx < len(subtokens_info):
            subtoken_info = subtokens_info[first_subtoken_idx]
            coref = subtoken_info[-2] if subtoken_info is not None else '-'
            if coref != '-':
                last_subtoken_idx = first_subtoken_idx + subtoken_info[-1] - 1
                for part in coref.split('|'):
                    if part[0] == '(':
                        if part[-1] == ')':
                            cluster_id = int(part[1:-1])
                            self.clusters[cluster_id].append((first_subtoken_idx, last_subtoken_idx))
                        else:
                            cluster_id = int(part[1:])
                            self.coref_stacks[cluster_id].append(first_subtoken_idx)
                    else:
                        cluster_id = int(part[:-1])
                        start = self.coref_stacks[cluster_id].pop()
                        self.clusters[cluster_id].append((start, last_subtoken_idx))
            first_subtoken_idx += 1

        # Merge clusters if any clusters have common mentions
        merged_clusters = []
        for cluster in self.clusters.values():
            existing = None
            for mention in cluster:
                for merged_cluster in merged_clusters:
                    if mention in merged_cluster:
                        existing = merged_cluster
                        break
                if existing is not None:
                    break
            if existing is not None:
                print("Merging clusters (shouldn't happen very often)")
                existing.update(cluster)
            else:
                merged_clusters.append(set(cluster))

        merged_clusters = [list(cluster) for cluster in merged_clusters]
        all_mentions = flatten(merged_clusters)
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = flatten(self.segment_subtoken_map)

        # Sanity check
        assert len(all_mentions) == len(set(all_mentions))  # Each mention unique
        # Below should have length: # all subtokens with CLS, SEP in all segments
        num_all_seg_tokens = len(flatten(self.segments))
        assert num_all_seg_tokens == len(flatten(self.speakers))
        assert num_all_seg_tokens == len(subtoken_map)
        assert num_all_seg_tokens == len(sentence_map)

        return {
            "doc_key": self.doc_key,
            "tokens": self.tokens,
            "sentences": self.segments,
            "speakers": self.speakers,
            "constituents": [],
            "ner": [],
            "clusters": merged_clusters,
            'sentence_map': sentence_map,
            "subtoken_map": subtoken_map,
            'pronouns': self.pronouns
        }

class UADocumentState(DocumentState):
    def __init__(self, key):
        self.doc_key = key
        self.tokens = []

        # Linear list mapped to subtokens without CLS, SEP
        self.subtokens = []
        self.subtoken_map = []
        self.token_end = []
        self.sentence_end = []
        self.info = []  # Only non-none for the first subtoken of each word

        # Linear list mapped to subtokens with CLS, SEP
        self.sentence_map = []

        # Segments (mapped to subtokens with CLS, SEP)
        self.segments = []
        self.segment_subtoken_map = []
        self.segment_info = []  # Only non-none for the first subtoken of each word
        self.speakers = []

        # Doc-level attributes
        self.pronouns = []
        self.clusters = collections.defaultdict(list)  # {cluster_id: [(first_subtok_idx, last_subtok_idx) for each mention]}
        self.coref_stacks = []#collections.defaultdict(list)

    # 这个类里只有这一个方法
    def finalize(self):
        print('finalize 开始')
        # print('self.coref_stacks 开始', self.coref_stacks) # 此时为空
        """ Extract clusters; fill other info e.g. speakers, pronouns """
        # Populate speakers from info
        subtoken_idx = 0

        # 处理speakers
        for seg_info in self.segment_info:
            # print('seg_info', seg_info)
            speakers = []
            # 回忆一下seg_info里面放的是是row一整行（list）以及subtoken的长度
            for i, subtoken_info in enumerate(seg_info):
                # print('subtoken_info', subtoken_info)
                if i == 0 or i == len(seg_info) - 1: # 如果是第一个或者最后一个，添加分隔符
                    speakers.append('[SPL]')
                elif subtoken_info is not None:  # First subtoken of each word（回忆一下对于subtoken的后面几个，它是空的）
                    # print('subtoken_info', subtoken_info)
                    speakers.append(subtoken_info[9]) # subtoken_info[9] 是对话者的名字，比如fisherman
                    # print('subtoken_info[9]', subtoken_info[9]) # subtoken_info[9] 是对话者的名字，比如fisherman
                    # if subtoken_info[4] == 'PRP':  # Uncomment if needed
                    #     self.pronouns.append(subtoken_idx)

                    # print('subtoken_info[-1]', subtoken_info[-1])
                else:
                    speakers.append(speakers[-1]) # speakers[-1]就是上一个speaker的名字
                subtoken_idx += 1

            # print('speakers', speakers)
            self.speakers += [speakers] #顾名思义，speakers就是讲话者
            # print('self.speakers', self.speakers)

        # Populate cluster
        first_subtoken_idx = 0  # Subtoken idx across segments（这个备注这里的across就是跨越不同的segment的意思）
        # print('len(self.segment_info)', len(self.segment_info))
        subtokens_info = flatten(self.segment_info)
        # print('len(subtokens_info)', len(subtokens_info))
        # print('subtokens_info', subtokens_info)

        # print('self.coref_stacks 中间', self.coref_stacks)
        # 看到while，还是2个点，一个while后面的部分，这里是停止条件，一个是看变量first_subtoken_idx的更改条件
        # print('subtokens_info', subtokens_info)
        while first_subtoken_idx < len(subtokens_info):
            # print('------开始while----------')
            subtoken_info = subtokens_info[first_subtoken_idx]
            # print('first_subtoken_idx', first_subtoken_idx)
            # 这里的第10个位置正好对应的就是coref那一列。另外subtoken_info里的第一个一定是None，这是在split_into_segments那里手工写的每个seg的开头是None。所以第一个coref是-
            coref = subtoken_info[10] if subtoken_info is not None else '-'
            # print('coref', coref) # 如 coref (EntityID=72|MarkableID=markable_175|Min=548|SemType=do)

            # print('self.coref_stacks 中间2', self.coref_stacks)
            if coref != '-' and coref != '' and coref != '_':
                # 最后一个子token的id = 第一个子token的id + subtoken_info[-1]（即subtoken的数量）-1
                last_subtoken_idx = first_subtoken_idx + subtoken_info[-1] - 1

                # 因为有一些coref是 ')(EntityID=72|MarkableID=markable_159|Min=524|SemType=do)'，多出来一个')'，所以要做这一步处理
                parts = coref.split('(') # coref.split('(')配合len(parts[0])，用来判断这个entity是不是起始的一个，比如下述第7行的a
                # 左括号(代表着开始，右括号)代表着关闭，如下面，a polished gold是一个
                # 7     a          _  _  _  _  _  _  _  _  (EntityID=4|MarkableID=markable_4|Min=9|SemType=dn    _  _  _  (MarkableID=markable_4|Entity_Type=substance|Genericity=undersp-substance
                # 8     polished   _  _  _  _  _  _  _  _  _                                                     _  _  _
                # 9     gold       _  _  _  _  _  _  _  _  )

                # print('---')
                # print('parts', parts)
                # print('len(parts[0])', len(parts[0]))

                # if len(parts[0])是为了处理这种不是以"("开头的，如果以"("开头，那么就parts[0]就是空，而不以（开头的，就不是空，包括以
                # 其他开头但是包含(的，如)(EntityID=1，也包括其中不包含(的，如)))，注意对于str.split('(')，如果str中不包含(，那么会返回str本身（这个我之前居然忘记了）
                # 不管以上是哪两种情况，都是以)开头，有一种情况是上述说到的)))，这种是嵌套的entity，而另外一种情况是下述这种
                # 如果是这个，那么就代表是类似于)(EntityID=2|MarkableID=markable_160|Min=4|SemType=dn)这种，这种代表着这个本身是一个entity,但是它又是一个更大的entity的结尾，如
                # 1     a           _  _  _  _  _  _  _  _  (EntityID=1|MarkableID=markable_1|Min=4|SemType=dn
                # 2     suit        _  _  _  _  _  _  _  _  _
                # 3     of          _  _  _  _  _  _  _  _  _
                # 4     armor       _  _  _  _  _  _  _  _  )(EntityID=2|MarkableID=markable_160|Min=4|SemType=dn)

                # if len(parts[0])>1:
                #     print('aaaaa')
                # if len(parts[0])<1:
                #     print('bbbbb')
                if len(parts[0]): # the close bracket
                    for _ in range(len(parts[0])):
                        # print('-')
                        # print('self.coref_stacks', self.coref_stacks)
                        # 当我刚看到这里这个 self.coref_stacks的时候，我不知道是从哪里来的。后来测试后才发现，它一开始是空的，然后通过下面的句子去获得。
                        # 这是一个栈，后进先出，用来处理这种嵌套的
                        cluster_id, start = self.coref_stacks.pop()
                        self.clusters[cluster_id].append((start, last_subtoken_idx))

                for part in parts[1:]:
                    # print(coref)
                    entity = part.split("|")[0].split("=")[1].split("-")[0] #这里entity写出entityid更好
                    # print('entity', entity)
                    cluster_id = int(entity)

                    # 如果是以')'结尾的，就代表这个entity是由一个token组成的。那么就把subtoken是信息存储到clusters，否则就存储到栈！
                    # print('part[-1]', part[-1])
                    if part[-1] == ')':
                        # print('first_subtoken_idx', first_subtoken_idx)
                        # print('last_subtoken_idx', last_subtoken_idx)
                        # print('self.subtokens[first_subtoken_idx]', self.subtokens[first_subtoken_idx])
                        # print('self.subtokens[last_subtoken_idx]', self.subtokens[last_subtoken_idx])
                        # print('self.subtokens[first_subtoken_idx:last_subtoken_idx]', self.subtokens[first_subtoken_idx-1:last_subtoken_idx-1])
                        self.clusters[cluster_id].append((first_subtoken_idx, last_subtoken_idx))
                        #TODO：这里是有问题的，打印self.subtokens[first_subtoken_idx]和self.subtokens[last_subtoken_idx]，出现的不对。
                        # 我感觉这里是有问题的！！！具体要看看后面是怎么用的。难不成问题就出来这里？
                        # 后续打印出来观察后发现，存储的first_subtoken_idx, last_subtoken_idx都是从1开始计数的，所以可能在使用的时候考虑过这种情况
                    else:
                        # print('(cluster_id, first_subtoken_idx)', (cluster_id, first_subtoken_idx))
                        self.coref_stacks.append((cluster_id, first_subtoken_idx))

                    # print(self.coref_stacks)

#                 else:
#                     parts = coref.split(")")
#                     for part in parts[1:]:
#                         cluster_id, start = self.coref_stacks.pop()
#                         print("Popped")
#                         print(self.coref_stacks)
#                         self.clusters[cluster_id].append((start, last_subtoken_idx))

            first_subtoken_idx += 1

        # Merge clusters if any clusters have common mentions
        merged_clusters = []
        # print('self.clusters', self.clusters)
        # print('self.subtokens', self.subtokens)
        # print('self.clusters.values()', self.clusters.values())
        # 算了，先不看了，等到遇到了再看吧。暂时不太用得到！
        for cluster in self.clusters.values():
            existing = None
            for mention in cluster:
                for merged_cluster in merged_clusters:
                    if mention in merged_cluster:
                        existing = merged_cluster
                        break
                if existing is not None:
                    break
            if existing is not None:
                print("Merging clusters (shouldn't happen very often)")
                existing.update(cluster)
            else:
                merged_clusters.append(set(cluster))

        merged_clusters = [list(cluster) for cluster in merged_clusters]
        all_mentions = flatten(merged_clusters)
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = flatten(self.segment_subtoken_map)

        # Sanity check
        assert len(all_mentions) == len(set(all_mentions))  # Each mention unique
        # Below should have length: # all subtokens with CLS, SEP in all segments
        num_all_seg_tokens = len(flatten(self.segments))
        assert num_all_seg_tokens == len(flatten(self.speakers))
        assert num_all_seg_tokens == len(subtoken_map)
        assert num_all_seg_tokens == len(sentence_map)

        return {
            "doc_key": self.doc_key,
            "tokens": self.tokens,
            "sentences": self.segments,
            "speakers": self.speakers,
            "constituents": [],
            "ner": [],
            "clusters": merged_clusters,
            'sentence_map': sentence_map,
            "subtoken_map": subtoken_map,
            'pronouns': self.pronouns
        }

def split_into_segments(document_state: DocumentState, max_seg_len, constraints1, constraints2, tokenizer):
    """ Split into segments.
        Add subtokens, subtoken_map, info for each segment; add CLS, SEP in the segment subtokens
        Input document_state: tokens, subtokens, token_end, sentence_end, utterance_end, subtoken_map, info
    """
    curr_idx = 0  # Index for subtokens
    prev_token_idx = 0

    # ！！！这种while的核心就看两个地方，一个是停止条件里的变量（curr_idx)，一个是它是怎么变动的（curr_idx = end_idx + 1）
    # 此处通过这两个可以看到，它是curr_idx在循环，循环的重点是document_state.subtokens，步长是每一个seg
    # print('split_into_segments document_state.subtokens', document_state.subtokens)

    print('-----')
    print('document_state.doc_key', document_state.doc_key)
    print('len(document_state.subtokens)', len(document_state.subtokens))

    while curr_idx < len(document_state.subtokens):
        # Try to split at a sentence end point

        # 这里的max_seg_len的意思是单个seg的subtoken最大的设置长度，比如128。下面这个min的意思是：
        # 这个的意思是如果subtokens（比如是100）比max_seg_len（比如128）小，那么就把99当作是end_idx（即最后一个subtokens)
        # 如果是subtokens还剩下很多（比如1000），那么就截取max_seg_len长度的
        # 这里-1是因为下面segment那里写的是end_idx + 1，如果下面那里直接写end_idx，那么这里也不再需要-1。这里-2是因为添加了tokenizer.cls_token和tokenizer.sep_token
        end_idx = min(curr_idx + max_seg_len - 1 - 2, len(document_state.subtokens) - 1)  # Inclusive
        # print('curr_idx', curr_idx)
        # print('end_idx 1', end_idx)

        # print('constraints1', constraints1)
        # not constraints1[end_idx]的意思是如果end_idx不是句子的结尾（英语）；整个句子的意思是这个end_idx不是句子的结尾，且
        # 目前end_idx仍然大于等于curr_idx，那么回退end_idx，直到到达句子的末尾，或者是end_idx已经小于end_idx了，就停止
        # ！注意while这里是执行条件，里面就是什么情况下执行（反过来就是什么情况下停止）。这里是把"意外终止条件"和"逻辑条件"写在一起了，所以有点难懂
        while end_idx >= curr_idx and not constraints1[end_idx]:
            end_idx -= 1
            # print('end_idx go back')
        # print('end_idx 2', end_idx)

        # 如果上面那个while的end_idx一直执行直到end_idx < curr_idx（因为数据里没有句子末尾），那么就重新进行seg拆分，新方式采用
        if end_idx < curr_idx:
            logger.info(f'{document_state.doc_key}: no sentence end found; split at token end')
            # If no sentence end point, try to split at token end point
            end_idx = min(curr_idx + max_seg_len - 1 - 2, len(document_state.subtokens) - 1)
            # 这里的constraints2根据调用它的部分可以知道就是document_state.token_end
            while end_idx >= curr_idx and not constraints2[end_idx]:
                end_idx -= 1
            if end_idx < curr_idx:
                logger.error('Cannot split valid segment: no sentence end or token end')

        # 生成一个segment
        # print('curr_idx 3', curr_idx)
        # print('end_idx 3', end_idx)
        # 由于subtoken的限制，segment的长度不一定是512，可能略少（但是不会更多，因为end_idx是往回走）
        segment = [tokenizer.cls_token] + document_state.subtokens[curr_idx: end_idx + 1] + [tokenizer.sep_token]
        # print('len(segment)', len(segment))
        document_state.segments.append(segment)

        # print('curr_idx', curr_idx)
        # print('end_idx', end_idx)
        # print('subtoken_map', document_state.subtoken_map)
        subtoken_map = document_state.subtoken_map[curr_idx: end_idx + 1]
        # print('subtoken_map', subtoken_map)
        # print('len(ubtoken_map)', len(subtoken_map))
        # 这里添加prev_token_idx和subtoken_map[-1]是因为上面加了[tokenizer.cls_token]和[tokenizer.sep_token]
        document_state.segment_subtoken_map.append([prev_token_idx] + subtoken_map + [subtoken_map[-1]])
        # print('new subtoken_map', [prev_token_idx] + subtoken_map + [subtoken_map[-1]])
        # print('document_state.segment_subtoken_map', document_state.segment_subtoken_map)

        document_state.segment_info.append([None] + document_state.info[curr_idx: end_idx + 1] + [None])

        curr_idx = end_idx + 1
        prev_token_idx = subtoken_map[-1]


def get_doc_markables(doc_name, doc_lines, extract_MIN, keep_bridging, word_column=1, markable_column=10, bridging_column=11, print_debug=False):
    markables_cluster = {}
    markables_start = {}
    markables_end = {}
    markables_MIN = {}
    markables_coref_tag = {}
    markables_split = {} # set_id: [markable_id_1, markable_id_2 ...]
    bridging_antecedents = {}
    all_words = []
    stack = []
    for word_index, line in enumerate(doc_lines):
        columns = line.split()
        all_words.append(columns[word_column])

        if columns[markable_column] != '_':
            markable_annotations = columns[markable_column].split("(")
            if markable_annotations[0]:
                #the close bracket
                for _ in range(len(markable_annotations[0])):
                    markable_id = stack.pop()
                    markables_end[markable_id] = word_index

            for markable_annotation in markable_annotations[1:]:
                if markable_annotation.endswith(')'):
                    single_word = True
                    markable_annotation = markable_annotation[:-1]
                else:
                    single_word = False
                markable_info = {p[:p.find('=')]:p[p.find('=')+1:] for p in markable_annotation.split('|')}
                markable_id = markable_info['MarkableID']
                cluster_id = markable_info['EntityID']
                markables_cluster[markable_id] = cluster_id
                markables_start[markable_id] = word_index
                if single_word:
                    markables_end[markable_id] = word_index
                else:
                    stack.append(markable_id)

                markables_MIN[markable_id] = None
                if extract_MIN and 'Min' in markable_info:
                    MIN_Span = markable_info['Min'].split(',')
                    if len(MIN_Span) == 2:
                        MIN_start = int(MIN_Span[0]) - 1
                        MIN_end = int(MIN_Span[1]) - 1
                    else:
                        MIN_start = int(MIN_Span[0]) - 1
                        MIN_end = MIN_start
                    markables_MIN[markable_id] = (MIN_start,MIN_end)

                markables_coref_tag[markable_id] = 'referring'
                if cluster_id.endswith('-Pseudo'):
                    markables_coref_tag[markable_id] = 'non_referring'

                if 'ElementOf' in markable_info:
                    element_of = markable_info['ElementOf'].split(',') # for markable participate in multiple plural using , split the element_of, e.g. ElementOf=1,2
                    for ele_of in element_of:
                        if ele_of not in markables_split:
                            markables_split[ele_of] = []
                        markables_split[ele_of].append(markable_id)
        if keep_bridging and columns[bridging_column] != '_':
            bridging_annotations = columns[bridging_column].split("(")
            for bridging_annotation in bridging_annotations[1:]:
                if bridging_annotation.endswith(')'):
                    bridging_annotation = bridging_annotation[:-1]
                bridging_info = {p[:p.find('=')]:p[p.find('=')+1:] for p in bridging_annotation.split('|')}
                bridging_antecedents[bridging_info['MarkableID']] = bridging_info['MentionAnchor']




    clusters = {}
    id2markable = {}
    for markable_id in markables_cluster:
        m = Markable(
                doc_name, markables_start[markable_id],
                markables_end[markable_id], markables_MIN[markable_id],
                markables_coref_tag[markable_id],
                all_words[markables_start[markable_id]:
                        markables_end[markable_id] + 1])
        id2markable[markable_id] = m
        if markables_cluster[markable_id] not in clusters:
            clusters[markables_cluster[markable_id]] = (
                    [], markables_coref_tag[markable_id],doc_name,[markables_cluster[mid] for mid in markables_split.get(markables_cluster[markable_id],[])])
        clusters[markables_cluster[markable_id]][0].append(m)

    bridging_pairs = {}
    for anaphora, antecedent in bridging_antecedents.items():
        if not anaphora in id2markable or not antecedent in id2markable:
            print('Skip bridging pair ({}, {}) as markable_id does not exist in identity column!'.format(antecedent,anaphora))
            continue
        bridging_pairs[id2markable[anaphora]] = id2markable[antecedent]

    #print([(str(ana),str(ant)) for ana,ant in bridging_pairs.items()])
    # for cid in clusters:
    #	 cl = clusters[cid]
    #	 print(cid,[str(m) for m in cl[0]],cl[1],cl[2],cl[3] )
    return clusters, bridging_pairs

def process_clusters(clusters, keep_singletons, keep_non_referring, keep_split_antecedent):
    removed_non_referring = 0
    removed_singletons = 0
    processed_clusters = []
    processed_non_referrings = []

    for cluster_id, (cluster, ref_tag, doc_name, split_cid_list) in clusters.items():
        #recusively find the split singular cluster
        if split_cid_list and keep_split_antecedent:
            # if using split-antecedent, we shouldn't remove singletons as they might be used by split-antecedents
            assert keep_singletons
            split_clusters = set()
            queue = deque()
            queue.append(cluster_id)
            while queue:
                curr = queue.popleft()
                curr_cl, curr_ref_tag, doc_name, curr_cid_list = clusters[curr]
                #non_referring shouldn't be used as split-antecedents
                # if curr_ref_tag != 'referring':
                #	 print(curr_ref_tag, doc_name, curr_cid_list)
                if curr_cid_list:
                    for c in curr_cid_list:
                        queue.append(c)
                else:
                    split_clusters.add(tuple(curr_cl))
            split_m = Markable(
                doc_name, -1,
                -1, None,
                'referring',
                '',
                is_split_antecedent=True,
                split_antecedent_members=split_clusters)

            cluster.append(split_m) #add the split_antecedents

        if ref_tag == 'non_referring':
            if keep_non_referring:
                processed_non_referrings.append(cluster[0])
            else:
                removed_non_referring += 1
            continue

        if not keep_singletons and len(cluster) == 1:
            removed_singletons += 1
            continue

        processed_clusters.append(cluster)

    if keep_split_antecedent:
        #step 2 merge equivalent split-antecedents clusters
        merged_clusters = []
        for cl in processed_clusters:
            existing = None
            for m in cl:
                if m.is_split_antecedent:
                #only do this for split-antecedents
                    for c2 in merged_clusters:
                        if m in c2:
                            existing = c2
                            break
            if existing:
                # print('merge cluster ', [str(m) for m in cl], ' and ', [str(m) for m in existing])
                existing.update(cl)
            else:
                merged_clusters.append(set(cl))
        merged_clusters = [list(cl) for cl in merged_clusters]
    else:
        merged_clusters = processed_clusters

    return (merged_clusters, processed_non_referrings,
            removed_non_referring, removed_singletons)

def get_all_docs(path):
    all_doc_sents = {}
    all_docs = {}
    doc_lines = []
    sentences = []
    sentence = []
    doc_name = None
    print('path:', path)
    # https://stackoverflow.com/questions/40997603
    for line in open(path, encoding='utf-8'): # LK add encoding='utf-8'
        # print('get_all_docs line 0:', line) #就是一行一行的
        # print('len(get_all_docs line 0):', len(line))  # 就是一行一行的
        line = line.strip() #去除两边的空格
        # print('get_all_docs line 1:', line) #就是一行一行的
        # print('len(get_all_docs line 1):', len(line))  # 就是一行一行的
        # 这里就涉及到这里的一个数据结构，以# newdoc id = 开头的小段落，它都是紧跟一个setting，这个setting里包含一句话，这句话
        # 的sent_id是newdoc id-1，即第一句话
        if line.startswith('# newdoc'):
            # print('line with # newdoc:', line)
            # print('doc_name:', doc_name)
            # print('doc_lines:', doc_lines)

            # 这里的意思是，如果之前已经有doc_name和doc_lines了，则对之前的进行下存档，并且初始化doc_lines和sentences
            # 是一种前置判断！
            if doc_name and doc_lines:
                # print('aaaaa')
                all_docs[doc_name] = doc_lines
                all_doc_sents[doc_name] = sentences
                doc_lines = []
                sentences = []
            # 初始化cur_spk和doc_name
            cur_spk = "_"
            doc_name = line[len('# newdoc id = '):] #这里写的非常好，用len('# newdoc id = ')而不是14，可读性非常好

        # 每一个新的turn_id都会有speaker，代表着新一轮对话。而如果下一句话还是这个人说的，那么就只会有sent_id和text，不会有turn_id
        # 和speaker。第一个文档light_dev/episode_7296就是个很好的体会这个数据结构的例子。
        elif "# speaker = " in line:
            cur_spk = line[len('# speaker = '):]

        # 如果一个行以#开头，那么就跳过，这些就是不需要处理的普通行，这些行包括有# global.columns/# setting/# sent_id/# text/# turn_id等等（懒得列举了，总之就是剩余所有行）
        elif line.startswith('#'):
            continue

        # 如果是空行，那么就采取结束操作：把这个sentence的list添加到sentences里，然后把sentence给清空
        elif len(line) == 0:
            sentences.append(sentence)
            sentence = []
            continue

        # 以上的空行或者以#开头的，都是一些特殊的行。而else这里都是一些内容行，就是那些以数字开头的每一个token的行
        else:
            # print('line 0', line)
            splt_line = line.split()
            # print('splt_line', splt_line)
            # 第9个是MISC这一列，从文档https://github.com/UniversalAnaphora/UniversalAnaphora/blob/main/documents/UA_CONLL_U_Plus_proposal_v1.0.md
            # 里可以看出来，MISC是用来"using the Misc column to encode all information ("CONLL-U-Compact") proposed by Anna Nedoluzhko and Amir Zeldes."
            # 包含所有信息的。之类是把current_speaker给加进去了。可能是因为这个字段是把所有东西都往里塞的吧。
            splt_line[9] = "_".join(cur_spk.split()) # MISC
            # print('"_".join(cur_spk.split())', "_".join(cur_spk.split()))
            # print('splt_line[9]', splt_line[9])

            # 这里又把splt_line给重新合起来变成line了，这个和else那一行下面那个原始的line还是稍微有点不一样。之前那个各个行之间是有多个空格的。而这个新弄的各个字段之间只有一个空格
            line = " ".join(splt_line)
            # print('line 1', line)
            doc_lines.append(line)
            sentence.append(splt_line[1]) #在sentence这个列表里加入splt_line的第一个，即FORM（即token)
                
    sentences.append(sentence)
    if doc_name and doc_lines:
        all_docs[doc_name] = doc_lines
        all_doc_sents[doc_name] = sentences
    return all_docs, all_doc_sents

def get_markable_assignments(clusters):
    markable_cluster_ids = {}
    for cluster_id, cluster in enumerate(clusters):
        for m in cluster:
            markable_cluster_ids[m] = cluster_id
    return markable_cluster_ids

def get_document(doc_key, doc_lines, language, seg_len, tokenizer):
    """ Process raw input to finalized documents """
    document_state = UADocumentState(doc_key)
    word_idx = -1

    # Build up documents
    for line in doc_lines:
        # print('line', line)
        row = line.split()  # Columns for each token
        # print('row', row)
        # print('len(row)', len(row))
        # 只有这一行是一个完全的空行，它才会等于0，然后sentence_end才会是True。所以这个sentence_end的意思是"这个句子的末尾一行"。像是用来处理那种留空的行。
        # 像这种行其实不会很多，因此大部分的sentence_end都是False
        # Todo  # 在light_dev.2022.CONLLUA这个文件里是没有sentence_end的，全部为False.light_train也没有。
        if len(row) == 0:
            document_state.sentence_end[-1] = True # 这个只有空行的时候才会是True
            # print('document_state.sentence_end', document_state.sentence_end)
        else:
            word_idx += 1
            word = normalize_word(row[1], language)
            subtokens = tokenizer.tokenize(word) # !!! subtokens就是tokenizer后的TOKEN。tokens就是原始的word
            document_state.tokens.append(word) # 注意这里的tokens是会有重复的！

            # if subtokens[0] != word:
            #     print('word', word)
            #     print('subtokens', subtokens)

            # 感觉这里的名字起的不好，应该叫subtoken_end。不过暂时先这样吧
            document_state.token_end += [False] * (len(subtokens) - 1) + [True] # 这里我没有细致测试，就认为它是对的好了，看意思不难明白它的作用
            # print('document_state.tokens', document_state.tokens)
            # print('document_state.token_end', document_state.token_end)

            # 这里是处理subtokens
            for idx, subtoken in enumerate(subtokens):
                document_state.subtokens.append(subtoken)
                # print('row', row)
                # print('subtokens', subtokens)

                # info就是把整个行row的末尾加上subtokens的长度，比如1、2。只有idx=0的时候才会放进去，比如word='strive', subtokens= ['s', '##tri', '##ve']
                # 那么就只有's'这一个有info信息，其他的都是None
                info = None if idx != 0 else (row + [len(subtokens)])
                # if len(subtokens)>1 : print('info', info)
                # print('info', info)
                document_state.info.append(info)
                document_state.sentence_end.append(False) #从这里可以看出来，sentence_end是针对subtoken的
                # word_idx是会有重复的，当subtokens有大于等于两个的时候，word_idx就产生了重复。word_idx就是这个subtokens对应的word的idx！
                document_state.subtoken_map.append(word_idx) #subtoken_map放着这个subtoken的index

        # 上述逻辑完成后的检验：
        # if int(row[0]) <= 20:
        #     print('------')
        #     print('line', line)
        #     print('document_state.tokens', document_state.tokens)
        #     print('document_state.subtokens', document_state.subtokens)
        #     print('document_state.token_end', document_state.token_end)
        #     print('document_state.info', document_state.info)
        #     print('document_state.sentence_end', document_state.sentence_end)
        #     print('document_state.subtoken_map', document_state.subtoken_map)


    # Split documents
    constraits1 = document_state.sentence_end if language != 'arabic' else document_state.token_end
    # print('constraits1', constraits1)
    # 看名字就知道是把文档拆分为很多个segment。这里是针对每个doc(newdoc id)来拆分的，因为现在所在的这个函数get_document是限定了doc_key的
    split_into_segments(document_state, seg_len, constraits1, document_state.token_end, tokenizer)
    document = document_state.finalize()
    return document
