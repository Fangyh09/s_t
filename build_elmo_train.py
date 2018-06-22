from allennlp.commands.elmo import ElmoEmbedder

options_file = "Elmo/data/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "Elmo/data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

print("loadding elmo")
elmo = ElmoEmbedder(options_file, weight_file, cuda_device=0)
print("finish loading")

print("ok??????????????????????")
elmo_embedding = elmo.embed_sentence(["I", "Love", "You"])
print("ok!!!!!!!!!!!!!!!!!!!!")


def minibatches(data, minibatch_size, elmo):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    # import alog
    # alog.info("!!! I am here")
    x_batch, y_batch = [], []
    z_batch = []
    for (x, y, z) in data:
        if len(x_batch) == minibatch_size:
            activations, mask = elmo.batch_to_embeddings(z_batch)
            activations = np.transpose(activations, (0, 2, 1, 3))
            activations = activations.reshape(activations.shape[0],
                                              activations.shape[1],
                                              -1)
            z_batch = activations.cpu().numpy()
            # print("shape", z_batch.shape)
            yield x_batch, y_batch, z_batch

            x_batch, y_batch = [], []
            z_batch = []

        if type(x[0]) == tuple:
            x = zip(*x)

        x_batch += [x]
        y_batch += [y]
        z_batch += [z]


    # z_batch = np.array(z_batch)
    # alog.info("!!! I am out")

    if len(x_batch) != 0:
        activations, mask = elmo.batch_to_embeddings(z_batch)
        activations = np.transpose(activations, (0, 2, 1, 3))
        activations = activations.reshape(activations.shape[0],
                                          activations.shape[1],
                                          -1)
        z_batch = activations.cpu().numpy()
        # print("shape", z_batch.shape)
        yield x_batch, y_batch, z_batch



import numpy as np
from config import Config
from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
import ray
import ray.tune as tune
from tqdm import tqdm

config = Config()

dev = CoNLLDataset(config.filename_dev,
                   config.processing_word,
                   config.processing_tag,
                   config.max_iter)
train = CoNLLDataset(config.filename_train,
                     config.processing_word,
                     config.processing_tag,
                     config.max_iter)
test = CoNLLDataset(config.filename_test,
                     config.processing_word,
                     config.processing_tag,
                     config.max_iter, test=True)

dataset = {
    'dev': dev,
    'test': test,
    'train': train
}
batch_size = 10
mode = 'train'
out_file = "./elmo_"+ mode + ".embedding"

elmo_embedding_all = []
for i, (words, labels, elmo_embedding) in tqdm(enumerate(minibatches(dataset[mode], batch_size, elmo))):
    elmo_embedding_all.append(elmo_embedding)
elmo_embedding_all = np.array(elmo_embedding_all)

# np.savez(out_file, elmo_embedding_all)

import numpy as np
import h5py

print(type(elmo_embedding_all))
fout = h5py.File(out_file,'w')
for i in range(0, len(elmo_embedding_all)):
    fout[str(i)] = elmo_embedding_all[i]
fout.close()
# with h5py.File(out_file, 'w') as hf:
#     hf.create_dataset(mode,  data=elmo_embedding_all)