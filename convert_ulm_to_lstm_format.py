import pickle
import utils


output_path = 'output/instances.txt'

instances = pickle.load(open('ulm/instances.bin', 'rb'))


count = 0
needed = 0

with open(output_path, 'w') as outfile:
    for instance_id, instance in instances.items():

        sentence_tokens = []
        sentence_lemmas = []
        sentence_pos = []
        annotations = []


        for token in instance.tokens:

            sentence_tokens.append(token.text)
            sentence_lemmas.append(token.lemma)
            sentence_pos.append(token.pos)
            annotations.append(list(token.synsets))

            needed += len(token.synsets)

        for (target_lemma,
             target_pos,
             token_annotation,
             sentence_tokens,
             training_example,
             target_index) in utils.generate_training_instances_v2(sentence_tokens,
                                                                   sentence_lemmas,
                                                                   sentence_pos,
                                                                   annotations):

            outfile.write(training_example + '\n')
            count += 1


assert needed == count
