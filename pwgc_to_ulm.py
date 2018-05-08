import argparse
import sys
import os
import pickle
from lxml import etree
from nltk.corpus import wordnet as wn
from collections import defaultdict
from copy import deepcopy
import string

from my_data_classes import Ctoken, Cinstance, Clexelt
from my_classes import Sentence, Token
from sensekey_utils import add_sense_info_to_clexelt, load_mapping_sensekey2offset

__here__ = os.path.realpath(os.path.dirname(__file__))


def process_wf(child_node):
    #returns text, lemma, pos, list_skeys
    text = lemma = pos = skeys = None
    if child_node.get('tag') == 'un': #unnanotated
        text = child_node.text
        long_lemma = child_node.get('lemma')
        p = long_lemma.find('|')
        if p != -1:
            long_lemma = long_lemma[p+1:]
        p = long_lemma.find('%')
        lemma = long_lemma[:p]
        pos = child_node.get('pos')
        ##SAVE DATA
        #print(text,lemma,pos)
    elif child_node.get('tag') == 'ignore': #ignore
        text = child_node.text
        lemma = child_node.get('lemma')
        pos = child_node.get('pos')
        ##SAVE DATA
        #print(text,lemma,pos)
    elif child_node.get('tag') == 'man' or child_node.get('tag') == 'auto': #manual or auto annotation
        #Look for the <id> taht contains a valid lemma (non empty)
        pos = child_node.get('pos')
        skeys = []
        for id_node in child_node:
            skeys.append(id_node.get('sk'))
        
        #The text and lemma are taken from the last 'id'
        text = id_node.tail
        lemma = id_node.get('lemma')
        if pos is None:
            pos = get_pos_from_skey(skeys[0])
        ##SAVE DATA
        #print(text, lemma, pos, skeys)
    return text, lemma, pos, skeys
       
    
def get_pos_from_skey(this_skey):
    m = {'1':'NN', '2': 'VB', '3':'JJ', '4':'R','5':'J'}
    this_pos = None
    if this_skey is not None:
        p = this_skey.find('%')
        if p!= -1:  
            this_pos = m.get(this_skey[p+1], None)
    return this_pos
            
def process_node(this_root_node, this_id):
    is_first_cf = True
    info_for_cf = None
    list_of_tokens = []
    num_token = 0
    type_tag_for_token_id = {}
    sense_keys_for_token_id = {}
    
    children = []
    for child_node in this_root_node:
        if child_node.tag == 'qf':
            children.extend(child_node.getchildren())
        else:
            children.append(child_node)
        
    
    #for child_node in this_root_node:
    for child_node in children:
        if child_node.tag == 'wf':    
            if info_for_cf is not None:  #there is info from a previous cf
                text, lemma, pos, type_tag, sensekeys = info_for_cf
                is_first_cf = True
                info_for_cf = None

                token_id = '%s_%d' % (this_id,num_token)
                num_token += 1
                new_token = Ctoken(token_id)
                new_token.set_pos(pos)
                new_token.set_text(text)
                if lemma is None: lemma = text.lower()
                new_token.set_lemma(lemma)
                list_of_tokens.append(new_token)
                type_tag_for_token_id[token_id] = type_tag
                sense_keys_for_token_id[token_id] = sensekeys
                
            text, lemma, pos, sensekeys = process_wf(child_node)
            token_id = '%s_%d' % (this_id,num_token)
            num_token += 1
            new_token = Ctoken(token_id)
            new_token.set_pos(pos)
            new_token.set_text(text)
            if lemma is None: lemma = text.lower()
            new_token.set_lemma(lemma)
            list_of_tokens.append(new_token)
            type_tag_for_token_id[token_id] = child_node.get('tag')
            sense_keys_for_token_id[token_id] = sensekeys
            
        elif child_node.tag == 'cf':
            if is_first_cf:
                glob_node = child_node.find('glob')
                if glob_node is None:
                    #case of putting ...... together
                    continue
                type_tag = glob_node.get('tag')
                pos = child_node.get('pos')
                sensekeys = []
                if type_tag == 'un':
                    lemma = glob_node.get('lemma')
                    p = lemma.find('|')
                    if p!=-1:
                        lemma = lemma[p+1:]
                    p = lemma.rfind('%')
                    lemma = lemma[:p]
                else:  
                    id_node = glob_node.find('id')
                    lemma = id_node.get('lemma').replace(' ','_')
                    sensekeys.append(id_node.get('sk'))
                    fixed_pos = get_pos_from_skey(id_node.get('sk'))
                    if fixed_pos is not None:
                        pos = fixed_pos
                text = glob_node.tail
                info_for_cf = [text, lemma, pos, type_tag, sensekeys]
                is_first_cf = False
            else:
                second_text = child_node.text
                text, lemma, pos, type_tag, sensekeys = info_for_cf
                text = text+'_'+second_text
                info_for_cf = [text, lemma, pos, type_tag, sensekeys]
        elif child_node.tag == 'mwf':
            #multiword
            for this_wf_node in child_node:
                text, lemma, pos, sensekeys = process_wf(this_wf_node)
                token_id = '%s_%d' % (this_id,num_token)
                num_token += 1
                new_token = Ctoken(token_id)
                new_token.set_pos(pos)
                new_token.set_text(text)
                if lemma is None: lemma = text.lower()
                new_token.set_lemma(lemma)
                list_of_tokens.append(new_token)
                type_tag_for_token_id[token_id] = this_wf_node.get('tag')
                sense_keys_for_token_id[token_id] = sensekeys
        elif child_node.tag == 'aux':
            #Auxiliar information, do nothing
            pass

    if list_of_tokens:
        if list_of_tokens[-1].text == ';':
            list_of_tokens = list_of_tokens[:-1]

    return list_of_tokens, sense_keys_for_token_id, type_tag_for_token_id
        
def generate_instances(this_id, list_tokens, sense_keys_for_token_id, type_tag_for_token_id, data_lexelt, my_wn_reader):
    
    for index_token, token in enumerate(list_tokens):
        token_id = token.get_id()
        gold_lexkeys = sense_keys_for_token_id.get(token_id)
        if gold_lexkeys is not None and len(gold_lexkeys) != 0:
            lemma = token.get_lemma() 
            # The pos is fixed according to the lexkeys
            pos = get_pos_from_skey(gold_lexkeys[0])
            if pos is None:
                #<id id="n00037200_id.5" lemma="purposefully ignored" sk="purposefully_ignored%0:00:00::"/>credit<
                continue
            
            #pos  = token.get_pos()
            
            this_lemma_key = '%s.%s' % (lemma,pos[0].lower())
            if this_lemma_key not in data_lexelt:
                data_lexelt[this_lemma_key] = Clexelt(lemma,pos)
                data_lexelt[this_lemma_key].set_wn_possible_skeys(my_wn_reader)
            
            #For this line we need to have called first to set_wn_possible_skeys
            if True or data_lexelt[this_lemma_key].contains_valid_lexkey(gold_lexkeys):
                new_instance = Cinstance()
                new_instance.set_lemma(lemma)
                new_instance.set_pos(pos)
                new_instance.set_id(token_id)
                new_instance.set_docsrc(this_id)
                new_instance.set_lexkeys(gold_lexkeys)
                new_instance.set_confidence_for_senses({skey: 1.0 for skey in gold_lexkeys})
                new_instance.set_annotation_type(type_tag_for_token_id[token_id])
                new_instance.set_tokens(list_tokens)
                new_instance.set_index_head_list([index_token])
                data_lexelt[this_lemma_key].add_instance(new_instance) 
            else:
                print('Token %s in file %s not valid with lexkeys %s' % (token_id, file_id, str(gold_lexkeys)), file=sys.stderr)     
            
            
            
def process_file(this_file, my_wn_reader, data_lexelt):
    my_tree = etree.parse(this_file)
    n=0
    for synset_node in my_tree.findall('synset'):
        synset_id = synset_node.get('id')
        # Find the gloss node
        gloss_node = None
        for gloss_node in synset_node.findall('gloss'):
            if gloss_node.get('desc') == 'wsd':
                break
            
        ####################################################
        # Process the definition and generate an example   #
        ####################################################
        def_node = gloss_node.find('def')
        
        list_of_tokens, sense_keys_for_token_id, type_tag_for_token_id = process_node(def_node, def_node.get('id'))
        generate_instances(def_node.get('id'), list_of_tokens, sense_keys_for_token_id, type_tag_for_token_id, data_lexelt, my_wn_reader)
        
        
        #The examples
        for example_node in gloss_node.findall('ex'):
            list_of_tokens, sense_keys_for_token_id, type_tag_for_token_id = process_node(example_node, example_node.get('id'))
            generate_instances(example_node.get('id'), list_of_tokens, sense_keys_for_token_id, type_tag_for_token_id, data_lexelt, my_wn_reader)
  


def instances_with_all_annotations(the_data_lexelt, sensekey2offset):
    """
    create dictionary of instances in which all annotations for all tokens
    of an instances are stored

    :param dict the_data_lexelt: mapping lemma -> my_data_classes.Clexelt

    :rtype: tuple
    :return: total_lemmas, total_instances, instance_id2instance
    """
    for item_key, lexelt in the_data_lexelt.items():

        # First we call to add_sense_info_to_clexelt to create all the sensekey related info

        if len(lexelt) > 0:
            # First we call to add_sense_info_to_clexelt to create all the senskey related info
            # We do not need to assign the return value to a variable, as the object is passed by reference
            # and it's modified already

            # print('\tLexical item: %s' % item_key, file=sys.stderr)

            add_sense_info_to_clexelt(lexelt, my_wn_reader, debug=False)

            for instance in lexelt.instances:

                # clean tokens, remove underscore before or after token

                for index, token in enumerate(instance.tokens):
                    token.text = token.text.replace('\n     ', '')
                    token.text = token.text.strip('_')

                    assert not token.text.startswith('_')
                    assert not token.text.endswith('_')
                    assert '\n' not in token.text

                    if token.text == '':
                        if token.lemma:
                            token.text = token.lemma
                        else:
                            token.text = '-'


                assert all([token.text != '' for token in instance.tokens]), 'space in %s' % the_instance_id

                annotation_id = instance.id
                the_instance_id, token_id = annotation_id.rsplit('_', 1)

                if the_instance_id in instance_id2instance:
                    # retrieve instance that will contain annotations for all annotations in sentence
                    the_instance = instance_id2instance[the_instance_id]
                else:

                    the_tokens = []

                    for token in instance.tokens:

                        new_token_obj = Token(token_id=token.token_id,
                                              text=token.text,
                                              lemma=token.lemma,
                                              lexkeys=set(),
                                              synsets=set(),
                                              treebank_pos=token.pos)
                        the_tokens.append(new_token_obj)

                    the_instance = Sentence(instance.id, the_tokens)
                    instance_id2instance[the_instance_id] = the_instance

                # add annotations to token
                tokens = the_instance.tokens
                index = instance.index_head[0]

                tokens[index].annotation_type = instance.annotation_type

                for lexkey in instance.lexkeys:

                    found = False
                    mapped_3_to_5 = lexkey.replace('%3', '%5')

                    for key in [lexkey, mapped_3_to_5]:

                        if key in sensekey2offset:
                            synset_id = sensekey2offset[key]

                            tokens[index].lexkeys.add(key)
                            tokens[index].synsets.add(synset_id)

                            found = True
                            break

                    if not found:
                        print(lexkey, 'could not be mapped to synset id')
                
    return instance_id2instance


def split_instances_on_semicolon(the_instances):
    """
    split instances up on semicolon
    but split up if there are semicolons in the instance
    e.g.
    able to swim ; able to run
    will become two instances

    :param dict the_instances: mapping identifier -> my_data_classes.Cinstance

    :rtype: dict
    :return: identifier -> my_data_classes.Cinstance
    """
    splitted_instances = dict()

    for instance_id, the_instance in the_instances.items():

        sent = [token.text for token in the_instance.tokens]
        if ';' not in sent:
            splitted_instances[instance_id] = the_instance
        else:
            token_ids = {token.token_id for token in the_instance.tokens
                         if token.text != ';'}
            token_ids_added = set()

            anchors = [-1]
            semi_colons = [index
                           for index, token in enumerate(the_instance.tokens)
                           if token.text == ";"]
            anchors.extend(semi_colons)
            anchors.append(10000)

            for suffix, start, end in zip(string.ascii_lowercase,
                                          anchors,
                                          anchors[1:]):
                part = the_instance.tokens[start + 1: end]

                new_instance = deepcopy(the_instance)
                new_instance.id = '%s_%s' % (new_instance.id, suffix)
                new_instance.tokens = part

                added = {token.token_id for token in part}
                token_ids_added.update(added)

                part_sent = [token.text for token in part]
                assert ';' not in part_sent, part_sent

                if len(part) >= 2:
                    splitted_instances[new_instance.id] = new_instance


            assert token_ids_added == token_ids, token_ids - token_ids_added

    return splitted_instances


def index_meanings2sentences(instances,
                             debug=0):
    """
    load index of meaning (sensekey | synset) -> my_data_classes.Sentence objects

    :param dict instances: mapping id_ -> my_data_classes.Sentence instances

    :rtype: tuple
    :return: (sensekey -> instance ids, synset -> instance ids)
    """
    sensekey2instance_ids = defaultdict(set)
    synset2instance_ids = defaultdict(set)

    failed_mappings = set()
    for id_, instance_obj in instances.items():

        for token in instance_obj.tokens:
            assert len(token.synsets) <= len(token.lexkeys)

            for lexkey in token.lexkeys:
                sensekey2instance_ids[lexkey].add(id_)

            for synset in token.synsets:
                synset2instance_ids[synset].add(id_)

    stats = dict()
    for label, d in [('sensekey', sensekey2instance_ids),
                     ('synset', synset2instance_ids)]:
        counts = [len(value)
                  for value in d.values()]

        avg = round(sum(counts) / len(counts), 2)
        stats[label] = avg

    if debug >= 1:
        print('stats for sensekeys: #%s avg of %s' % (len(sensekey2instance_ids),
                                                      stats['sensekey']))
        print('stats for synsets: #%s avg of %s' % (len(synset2instance_ids),
                                                    stats['synset']))

    return sensekey2instance_ids, synset2instance_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts WordNet gloss corpus to our intermediate ULM format')
    parser.add_argument('-v',  action='version', version='1.0')
    parser.add_argument('-i',  dest='input_folder', required=True, help='Path to original wordnet gloss corpus main folder')
    parser.add_argument('-o',  dest='output_folder', required=True, help='Output folder')
    args = parser.parse_args()

    data_lexelt = {}
    my_wn_reader = wn


    files = [('adj.xml','a'),
             ('adv.xml','r'),
             ('noun.xml','n'),
             ('verb.xml','v')
             ]

    for this_file, short_pos in files:
        path_to_file = os.path.join(args.input_folder,'merged', this_file)
        print('Processing %s' % path_to_file)
        process_file(path_to_file, my_wn_reader, data_lexelt)

    print('Creating training data...', file=sys.stderr)
    total_instances = 0
    total_lemmas = 0
    instance_id2instance = dict()

    path_to_wn_dict_folder = str(wn._get_root())  # change this for other wn versions
    path_to_wn_index_sense = os.path.join(path_to_wn_dict_folder, 'index.sense')  # change this for other wn versions
    sensekey2offset = load_mapping_sensekey2offset(path_to_wn_index_sense,
                                                   '30')

    instance_id2instance = instances_with_all_annotations(data_lexelt,
                                                          sensekey2offset)

    splitted_instances = split_instances_on_semicolon(instance_id2instance)

    sensekey2instance_ids, synset2instance_ids = index_meanings2sentences(splitted_instances,
                                                                          debug=1)


    for basename, info in [('instances.bin', splitted_instances),
                           ('sensekey_index.bin', sensekey2instance_ids),
                           ('synset_index.bin', synset2instance_ids)]:

        # Save the instance object
        output_path = os.path.join(args.output_folder, basename)
        with open(output_path, 'wb') as outfile:
            pickle.dump(info, outfile, protocol=3)

    print('Total number of instances: %d' % len(instance_id2instance))
    print('Total number of splitted instances: %s' % len(splitted_instances))
