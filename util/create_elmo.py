
def save_elmo(filename):
    from allennlp.commands.elmo import ElmoEmbedder

    options_file = "Elmo/data/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "Elmo/data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    print("loadding elmo")
    elmo = ElmoEmbedder(options_file, weight_file, cuda_device=0)
    print("finish loading")

    print("ok??????????????????????")
    elmo_embedding = elmo.embed_sentence(["I", "Love", "You"])
    print("ok!!!!!!!!!!!!!!!!!!!!")

    all_embedding = []
    with open(filename) as f:
        words, tags = [], []
        orig_words = []
        # elmo_embedding = []
        for line in f:
            line = line.strip()
            if (len(line) == 0 or line.startswith("-DOCSTART-")):
                if len(words) != 0:
                    # niter += 1
                    # if self.max_iter is not None and niter > self.max_iter:
                    #     break
                    # todo remote it
                    # use_elmo = True
                    # if use_elmo:
                    embedding = elmo.embed_sentence(orig_words)
                    all_embedding.append(embedding)
                    # yield words, tags, self.elmo.embed_sentence(orig_words)
                    words, tags = [], []
                    orig_words = []
            else:
                ls = line.split(' ')
                word, tag = ls[0], ls[-1]
                orig_word = copy.deepcopy(word)
                if self.processing_word is not None:
                    word = self.processing_word(word)
                if not self.test:
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                words += [word]
                tags += [tag]
                orig_words += [orig_word]


if __main__ == "__main__":
    save_elmo()