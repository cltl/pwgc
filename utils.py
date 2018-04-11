


def generate_training_instances_v2(sentence_tokens,
                                   sentence_lemmas,
                                   sentence_pos,
                                   annotations):
    """
    given the lemmas in a sentence with its annotations (can be more than one)
    generate all training instances for that sentence

    e.g.
    sentence_tokens = ['the', 'man',            'meets',   'women']
    sentence_lemmas = ['the', 'man',            'meet',    'woman']
    sentence_pos    = ['',    'n',              'v',       'n']
    annotations =     [[],    ['1', '2' ],      ['4'],     ['5', '6']]

    would result in
    ('man', 'n', '1', ['the', 'man', 'meets', 'women'], 'the man---1 meets women', 1)
    ('man', 'n', '2', ['the', 'man', 'meets', 'women'], 'the man---2 meets women', 1)
    ('meet', 'v', '4', ['the', 'man', 'meets', 'women'], 'the man meets---4 women', 2)
    ('woman', 'n', '5', ['the', 'man', 'meets', 'women'], 'the man meets women---5', 3)
    ('woman', 'n', '6', ['the', 'man', 'meets', 'women'], 'the man meets women---6', 3)

    :param list sentence_tokens: see above
    :param list sentence_lemmas: see above
    :param list sentence_pos: see above
    :param list annotations: see above

    :rtype: generator
    :return: generator of (target_lemma,
                           target_pos,
                           token_annotation,
                           sentence_tokens,
                           training_example,
                           target_index)
    """
    for target_index, token_annotations in enumerate(annotations):

        target_lemma = sentence_lemmas[target_index]
        target_pos = sentence_pos[target_index]

        for token_annotation in token_annotations:

            if token_annotation is None:
                continue

            a_sentence = []
            for index, token in enumerate(sentence_tokens):

                if index == target_index:
                    a_sentence.append(token + '---' + token_annotation)
                else:
                    a_sentence.append(token)

            training_example = ' '.join(a_sentence)

            yield (target_lemma,
                   target_pos,
                   token_annotation,
                   sentence_tokens,
                   training_example,
                   target_index)