import os


from model.general_utils import get_logger
from model.data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word

DEBUG_MODE = False


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

       c """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)

    # general config
    file_name = os.path.basename(__file__)
    dir_output = "results/tmptmptest/" + "bz=10-training-bieo-nocnn" + "/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 300
    dim_char = 100

    # glove files
    filename_glove = "/home/yinghong/project/tmp/s_t/data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "/home/yinghong/project/tmp/s_t/data/glove.6B.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True

    # dataset
    if DEBUG_MODE:
        filename_dev = "/home/yinghong/project/tmp/s_t/data/dev.eval.small.bieo"
        filename_test = "/home/yinghong/project/tmp/s_t/data/dev.eval.small.bieo"
        filename_train = "/home/yinghong/project/tmp/s_t/data/train.eval.small.bieo"
    else:
        filename_dev = "/home/yinghong/project/tmp/s_t/data/dev.eval.bieo"
        filename_test = "/home/yinghong/project/tmp/s_t/data/dev.eval.bieo"
        filename_train = "/home/yinghong/project/tmp/s_t/data/train.eval.bieo"

#filename_dev = filename_test = filename_train = "data/test.txt" # test

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "/home/yinghong/project/tmp/s_t/data/words.bieo.txt"
    filename_tags = "/home/yinghong/project/tmp/s_t/data/tags.bieo.txt"
    filename_chars = "/home/yinghong/project/tmp/s_t/data/chars.bieo.txt"

    # training
    train_embeddings = True
    nepochs          = 50
    dropout          = 0.5
    batch_size       = 10
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 20

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True  # if crf, training is 1.7x slower on CPU
    use_chars = True  # if char embedding, training is 3.5x slower on CPU
    use_gru = False
    use_cnn = False
    use_reg = False
    reg_ratio = 0.001

    filter_sizes = [3, 4, 5]
    hidden_size_gru = 300

    lstm_layers = 1
    input_keep_prob = 1.0
    output_keep_prob = 1.0


    # def __setitem__(self, key, value):
    #     if key == "nepochs":
    #         Config.nepochs = value
    #     elif key == "dim_char":
    #         Config.dim_char = value
    #     elif key == "batch_size":
    #         Config.batch_size = value
    #     elif key == "lr_method":
    #         Config.lr_method = value
    #     elif key == "lr":
    #         Config.lr = value
    #     elif key == "dim_char":
    #         Config.dim_char = value
    #     elif key == "hidden_size_char":
    #         Config.hidden_size_char = value
    #     elif key == "hidden_size_lstm":
    #         Config.hidden_size_lstm = value
    #     elif key == "use_gru":
    #         Config.use_gru = value
    #     elif key == "use_cnn":
    #         Config.use_cnn = value
    #     elif key == "hidden_size_gru":
    #         Config.hidden_size_gru = value
    #     elif key == "lstm_layers":
    #         Config.lstm_layers = value
    #     elif key == "filter_sizes":
    #         Config.filter_sizes = value
    #     elif key == "dir_output":
    #         Config.dir_output = value
    #     elif key == "clip":
    #         Config.clip = value
    #     elif key == "lr_decay":
    #         Config.lr_decay = value
    #     elif key == "nepoch_no_imprv":
    #         Config.nepoch_no_imprv = value
    #     else:
    #         raise ValueError("not exist this attr")

    def dump(obj, fout=False):
        class Writer():
            def __init__(self, fout=False):
                self.fout = fout

            def write(self, key, val):
                if self.fout:
                    self.fout.write(key + "=" + val + "\n")
                    self.fout.flush()
                else:
                    print(key, val)


        writer = Writer(fout)
        # for attr in dir(obj)
        writer.write("Config.nepochs", str(Config.nepochs))
        writer.write("Config.dim_char", str(Config.dim_char))
        writer.write("Config.batch_size", str(Config.batch_size))
        writer.write("Config.lr_method", str(Config.lr_method))
        writer.write("Config.lr", str(Config.lr))
        writer.write("Config.dim_char", str(Config.dim_char))
        writer.write("Config.hidden_size_char", str(Config.hidden_size_char))
        writer.write("Config.hidden_size_lstm", str(Config.hidden_size_lstm))
        writer.write("Config.use_gru", str(Config.use_gru))
        writer.write("Config.use_cnn", str(Config.use_cnn))
        writer.write("Config.hidden_size_gru", str(Config.hidden_size_gru))
        writer.write("Config.lstm_layers", str(Config.lstm_layers))
        writer.write("Config.filter_sizes", str(Config.filter_sizes))
        writer.write("Config.clip", str(Config.clip))



        # self.dict[key] = value
