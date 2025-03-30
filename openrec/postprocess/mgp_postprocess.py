from .ctc_postprocess import BaseRecLabelDecode


class MPGLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index."""
    SPACE = '[s]'
    GO = '[GO]'
    list_token = [GO, SPACE]

    def __init__(self,
                 character_dict_path=None,
                 use_space_char=False,
                 only_char=False,
                 **kwargs):
        super(MPGLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)
        self.only_char = only_char
        self.EOS = '[s]'
        self.PAD = '[GO]'
        if not only_char:
            # transformers==4.2.1
            from transformers import BertTokenizer, GPT2Tokenizer
            self.bpe_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.wp_tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased')

    def __call__(self, preds, batch=None, *args, **kwargs):

        if isinstance(preds, list):
            char_preds = preds[0].detach().cpu().numpy()
        else:
            char_preds = preds.detach().cpu().numpy()

        preds_idx = char_preds.argmax(axis=2)
        preds_prob = char_preds.max(axis=2)
        char_text = self.char_decode(preds_idx[:, 1:], preds_prob[:, 1:])
        if batch is None:
            return char_text
        label = batch[1]
        label = self.char_decode(label[:, 1:])
        if self.only_char:
            return char_text, label
        else:
            bpe_preds = preds[1].detach().cpu().numpy()
            wp_preds = preds[2]

            bpe_preds_idx = bpe_preds.argmax(axis=2)
            bpe_preds_prob = bpe_preds.max(axis=2)
            bpe_text = self.bpe_decode(bpe_preds_idx[:, 1:],
                                       bpe_preds_prob[:, 1:])

            wp_preds = wp_preds.detach()  #.cpu().numpy()
            wp_preds_prob, wp_preds_idx = wp_preds.max(-1)
            wp_text = self.wp_decode(wp_preds_idx[:, 1:], wp_preds_prob[:, 1:])

            final_text = self.final_decode(char_text, bpe_text, wp_text)
            return char_text, bpe_text, wp_text, final_text, label

    def add_special_char(self, dict_character):
        dict_character = self.list_token + dict_character
        return dict_character

    def final_decode(self, char_text, bpe_text, wp_text):
        result_list = []
        for (char_pred,
             char_pred_conf), (bpe_pred,
                               bpe_pred_conf), (wp_pred, wp_pred_conf) in zip(
                                   char_text, bpe_text, wp_text):
            final_text = char_pred
            final_prob = char_pred_conf
            if bpe_pred_conf > final_prob:
                final_text = bpe_pred
                final_prob = bpe_pred_conf
            if wp_pred_conf > final_prob:
                final_text = wp_pred
                final_prob = wp_pred_conf
            result_list.append((final_text, final_prob))
        return result_list

    def char_decode(self, text_index, text_prob=None):
        """ convert text-index into text-label. """
        result_list = []
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = 1.0
            for idx in range(len(text_index[batch_idx])):
                try:
                    char_idx = self.character[int(text_index[batch_idx][idx])]
                except:
                    continue
                if text_prob is not None:
                    conf_list *= text_prob[batch_idx][idx]

                if char_idx == self.EOS:  # end
                    break
                if char_idx == self.PAD:
                    continue
                char_list.append(char_idx)

            text = ''.join(char_list)
            result_list.append((text, conf_list))
        return result_list

    def bpe_decode(self, text_index, text_prob):
        """ convert text-index into text-label. """
        result_list = []
        for text, probs in zip(text_index, text_prob):
            text_decoded = []
            conf_list = 1.0
            for bpeindx, prob in zip(text, probs):
                tokenstr = self.bpe_tokenizer.decode([bpeindx])
                if tokenstr == '#':
                    break
                text_decoded.append(tokenstr)
                conf_list *= prob
            text = ''.join(text_decoded)
            result_list.append((text, conf_list))
        return result_list

    def wp_decode(self, text_index, text_prob=None):
        """ convert text-index into text-label. """
        result_list = []
        for batch_idx, text in enumerate(text_index):
            wp_pred = self.wp_tokenizer.decode(text)
            wp_pred_EOS = wp_pred.find('[SEP]')
            wp_pred = wp_pred[:wp_pred_EOS]
            if text_prob is not None:
                try:
                    # print(text.cpu().tolist())
                    wp_pred_EOS_index = text.cpu().tolist().index(102) + 1
                except:
                    wp_pred_EOS_index = -1
                wp_pred_max_prob = text_prob[batch_idx][:wp_pred_EOS_index]
                try:
                    wp_confidence_score = wp_pred_max_prob.cumprod(
                        dim=0)[-1].cpu().numpy().sum()
                except:
                    wp_confidence_score = 0.0
            else:
                wp_confidence_score = 1.0
            result_list.append((wp_pred, wp_confidence_score))
        return result_list
