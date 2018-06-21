
from allennlp.commands.elmo import ElmoEmbedder




def main2():
    # create instance of config

    options_file = "Elmo/data/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "Elmo/data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    print("loadding elmo")
    elmo = ElmoEmbedder(options_file, weight_file, cuda_device=0)
    print("finish loading")


    # create datasets
    print("ok??????????????????????")
    elmo_embedding = elmo.embed_sentence(["I", "Love", "You"])
    print("ok!!!!!!!!!!!!!!!!!!!!")



if __name__ == "__main__":
    main2()
