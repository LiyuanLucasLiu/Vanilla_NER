"""
.. module:: seqlabel
    :synopsis: sequence labeling model
 
.. moduleauthor:: Liyuan Liu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_seq.utils as utils
from model_seq.crf import CRF

class Vanilla_SeqLabel(nn.Module):
    """
    Sequence Labeling model augumented without language model.

    Parameters
    ----------
    c_num : ``int`` , required.
        number of characters.
    c_dim : ``int`` , required.
        dimension of character embedding.
    c_hidden : ``int`` , required.
        dimension of character hidden states.
    c_layer : ``int`` , required.
        number of character lstms.
    w_num : ``int`` , required.
        number of words.
    w_dim : ``int`` , required.
        dimension of word embedding.
    w_hidden : ``int`` , required.
        dimension of word hidden states.
    w_layer : ``int`` , required.
        number of word lstms.
    y_num : ``int`` , required.
        number of tags types.
    droprate : ``float`` , required
        dropout ratio.
    unit : "str", optional, (default = 'lstm')
        type of the recurrent unit.
    """
    def __init__(self, c_num, c_dim, c_hidden, c_layer, w_num, w_dim, w_hidden, w_layer, y_num, droprate, unit='lstm'):
        super(Vanilla_SeqLabel, self).__init__()

        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}

        self.char_embed = nn.Embedding(c_num, c_dim)
        self.word_embed = nn.Embedding(w_num, w_dim)

        self.char_seq = nn.Linear(c_hidden * 2, w_dim)

        self.c_hidden = c_hidden
        self.unit_type = unit

        tmp_rnn_dropout = droprate if c_layer > 1 else 0
        self.char_fw = rnnunit_map[unit](c_dim, c_hidden, c_layer, dropout = tmp_rnn_dropout)
        self.char_bw = rnnunit_map[unit](c_dim, c_hidden, c_layer, dropout = tmp_rnn_dropout)

        tmp_rnn_dropout = droprate if w_layer > 1 else 0
        self.word_rnn = rnnunit_map[unit](w_dim * 2, w_hidden // 2, w_layer, dropout = tmp_rnn_dropout, bidirectional = True)

        self.y_num = y_num
        self.crf = CRF(w_hidden, y_num)

        self.drop = nn.Dropout(p = droprate)

    def to_params(self):
        """
        To parameters.
        """
        return {
            "model_type": "char-lstm-crf",
            "word_embed_num": self.word_embed.num_embeddings,
            "word_embed_dim": self.word_embed.embedding_dim,
            "char_embed_num": self.char_embed.num_embeddings,
            "char_embed_dim": self.char_embed.embedding_dim,
            "char_hidden": self.c_hidden,
            "char_layers": self.char_fw.num_layers,
            "word_hidden": self.word_rnn.hidden_size,
            "word_layers": self.word_rnn.num_layers,
            "droprate": self.drop.p,
            "y_num": self.y_num,
            "label_schema": "iobes",
            "unit_type": self.unit_type
        }

    def set_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        tmp = sentence.size()
        self.word_seq_length = tmp[0]
        self.batch_size = tmp[1]

    def load_pretrained_word_embedding(self, pre_word_embeddings):
        """
        Load pre-trained word embedding.
        """
        self.word_embed.weight = nn.Parameter(pre_word_embeddings)

    def rand_init(self):
        """
        Random initialization.
        """
        utils.init_embedding(self.char_embed.weight)
        utils.init_lstm(self.char_fw)
        utils.init_lstm(self.char_bw)
        utils.init_lstm(self.word_rnn)
        utils.init_linear(self.char_seq)
        self.crf.rand_init()

    def forward(self, f_c, f_p, b_c, b_p, f_w):
        """
        Calculate the output (crf potentials).

        Parameters
        ----------
        f_c : ``torch.LongTensor``, required.
            Character-level inputs in the forward direction.
        f_p : ``torch.LongTensor``, required.
            Ouput position of character-level inputs in the forward direction.
        b_c : ``torch.LongTensor``, required.
            Character-level inputs in the backward direction.
        b_p : ``torch.LongTensor``, required.
            Ouput position of character-level inputs in the backward direction.
        f_w: ``torch.LongTensor``, required.
            Word-level inputs for the sequence labeling model.

        Returns
        -------
        output: ``torch.FloatTensor``.
            A float tensor of shape (sequence_len, batch_size, from_tag_size, to_tag_size)
        """
        
        self.set_batch_seq_size(f_w)

        f_c_e = self.drop(self.char_embed(f_c))
        b_c_e = self.drop(self.char_embed(b_c))

        f_c_e, _ = self.char_fw(f_c_e)
        b_c_e, _ = self.char_bw(b_c_e)

        f_c_e = f_c_e.view(-1, self.c_hidden).index_select(0, f_p).view(self.word_seq_length, self.batch_size, self.c_hidden)

        b_c_e = b_c_e.view(-1, self.c_hidden).index_select(0, b_p).view(self.word_seq_length, self.batch_size, self.c_hidden)

        c_o = self.drop(torch.cat([f_c_e, b_c_e], dim = 2))
        c_o = self.char_seq(c_o)

        w_e = self.word_embed(f_w)

        rnn_in = self.drop(torch.cat([c_o, w_e], dim = 2))

        rnn_out, _ = self.word_rnn(rnn_in)

        crf_out = self.crf(self.drop(rnn_out)).view(self.word_seq_length, self.batch_size, self.y_num, self.y_num)

        return crf_out