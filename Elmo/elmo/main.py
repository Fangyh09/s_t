from allennlp.commands.elmo import ElmoEmbedder
import time

# import tensorflow as tf

options_file = "data/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
#
# t1 = time.time()
# elmo = Elmo(options_file, weight_file, 2, dropout=0)
# t2 = time.time()
#
# # use batch_to_ids to convert sentences to character ids
sentences = [['First', 'sentence', '.', 'gg', 'ff'], ['Another', '.' ,  'gg', 'ff'],
             ['First',
                                                            'sentence', '.', 'gg', 'ff'
              ], ['Another', '.', 'gg', 'ff']]
# character_ids = batch_to_ids(sentences)
# # print("character_ids ", character_ids, tf.shape(character_ids))
# embeddings = elmo(character_ids)
# # print("embeddings", embeddings,
# #       type(embeddings), tf.shape(embeddings['elmo_representations']))
#
# t3 = time.time()
#
# print("load=%d, predict=%d\n" % (t2 - t1, t3 - t2))
# embeddings['elmo_representations'] is length two list of tensors.
# Each element contains one layer of ELMo representations with shape
# (2, 3, 1024).
#   2    - the batch size
#   3    - the sequence length of the batch
#   1024 - the length of each ELMo vector


elmo = ElmoEmbedder(options_file, weight_file, cuda_device=0)
embeddings = elmo.batch_to_embeddings(sentences)
print(embeddings)
