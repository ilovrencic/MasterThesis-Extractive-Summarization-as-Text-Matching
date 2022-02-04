"""
Custom Margin Ranking Loss function for Extractive Summarization

Loss = candidate_margin_loss + gold_summary_margin_loss
"""

import tempfile
from torch import nn
import torch
from rouge import Rouge
from os.path import join
from pyrouge import Rouge155
import subprocess as sp

_ROUGE_PATH = '/home/ilovrencic/MasterThesis/pyrouge-files/rouge/tools/ROUGE-1.5.5/'


class MarginRankingLoss(nn.Module):
    def __init__(self, margin):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, candidate_score, summary_score):
        #  initializing the loss
        ones = torch.ones(candidate_score.size()).cuda(candidate_score.device)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        total_loss = loss_func(candidate_score, candidate_score, ones)

        # candidate loss - first part
        candidate_num = candidate_score.size(1)
        for i in range(1, candidate_num):
            pos_score = candidate_score[:, :-i]
            neg_score = candidate_score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)

            ones = torch.ones(pos_score.size()).cuda(candidate_score.device)
            loss_func = torch.nn.MarginRankingLoss(self.margin * i)
            total_loss += loss_func(pos_score, neg_score, ones)

        # gold summary loss - second part
        pos_score = summary_score.unsqueeze(-1).expand_as(candidate_score)
        neg_score = candidate_score
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones(pos_score.size()).cuda(candidate_score.device)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        total_loss += loss_func(pos_score, neg_score, ones)

        return total_loss


class ValidationMetric(nn.Module):
    def __init__(self, data, save_path=None):
        super(ValidationMetric, self).__init__()

        self.data = data
        self.save_path = save_path

        self.top1_correct = 0
        self.top6_correct = 0
        self.top10_correct = 0

        self.rouge = Rouge()
        self.ROUGE = 0.0
        self.error = 0

        self.index = 0

    def metrics_name(self):
        return "ValidationMetric"

    def fast_rouge(self, dec, ref):
        if dec == '' or ref == '':
            return 0.0
        scores = self.rouge.get_scores(dec, ref)
        return (scores[0]['rouge-1']['f'] + scores[0]['rouge-2']['f'] + scores[0]['rouge-l']['f']) / 3

    def forward(self, score):
        batch_size = score.size(0)

        self.top1_correct += int(torch.sum(torch.max(score, dim=1).indices == 0))
        self.top6_correct += int(torch.sum(torch.max(score, dim=1).indices <= 5))
        self.top10_correct += int(torch.sum(torch.max(score, dim=1).indices <= 9))

        # rouge over data points in batch
        for i in range(batch_size):
            max_id = int(torch.max(score[i], dim=0).indices)  # getting the id of the candidate with highest score
            if max_id >= len(self.data[self.index]["indices"].values[
                                 0]):  # if candidate id is bigger than the size of possible ids, than we have an error
                self.error += 1
                self.index += 1
                continue

            best_candidate_ids = self.data[self.index]["indices"].values[0][
                max_id]  # ids of sentences that comprise the candidate
            best_candidate_ids.sort()

            dec = []  # decoded summary (candidate summary)
            for idx in best_candidate_ids:
                dec.append(self.data[self.index]["text"].values[0][idx])
            dec = "".join(dec)
            ref = "".join(self.data[self.index]["summary"].values[0])  # golden summary

            self.ROUGE += self.fast_rouge(dec, ref)
            self.index += 1

    def get_results(self, reset=True):
        top1_accuracy = self.top1_correct / self.index
        top6_accuracy = self.top6_correct / self.index
        top10_accuracy = self.top10_correct / self.index
        ROUGE = self.ROUGE / self.index

        eval_result = {'top1_accuracy': top1_accuracy, 'top6_accuracy': top6_accuracy,
                       'top10_accuracy': top10_accuracy, 'Error': self.error, 'ROUGE': ROUGE}

        # todo: add saving these results to path
        if reset:
            self.top1_correct = 0
            self.top6_correct = 0
            self.top10_correct = 0
            self.ROUGE = 0.0
            self.error = 0
            self.index = 0

        return eval_result


class MatchRougeMetric(nn.Module):
    def __init__(self, data, dec_path, ref_path, n_total):
        super(MatchRougeMetric, self).__init__()

        self.data = data
        self.dec_path = dec_path
        self.ref_path = ref_path
        self.n_total = n_total

        self.ext = []
        self.index = 0

    def forward(self, score):
        ext = int(torch.max(score, dim=1).indices)
        self.ext.append(ext)
        self.index += 1
        print('{}/{} ({:.2f}%) decoded\r'.format(
            self.index, self.n_total, self.index / self.n_total * 100), end='')

    def metrics_name(self):
        return "MatchRougeMetric"

    def get_results(self, reset = True):
        for i, ext in enumerate(self.ext):
            sent_ids = self.data[i]["indices"].values[0][ext]
            dec, ref = [], []

            for idx in sent_ids:
                dec.append(self.data[i]["text"].values[0][idx])
            for sent in self.data[i]["summary"].values[0]:
                ref.append(sent)

            with open(join(self.dec_path, '{}.dec'.format(i)), 'w') as f:
                for sent in dec:
                    print(sent, file=f)

            with open(join(self.ref_path, '{}.ref'.format(i)), 'w') as f:
                for sent in ref:
                    print(sent, file=f)

        R_1, R_2, R_L = MatchRougeMetric.eval_rouge(self.dec_path, self.ref_path)
        eval_result = {'ROUGE-1': R_1, 'ROUGE-2': R_2, 'ROUGE-L': R_L}

        if reset:
            self.index = 0
            self.ext = []
            self.data = []

        return eval_result

    @staticmethod
    def eval_rouge(dec_path, ref_path, Print = True):
        assert _ROUGE_PATH is not None

        dec_pattern = '(\d+).dec'
        ref_pattern = '#ID#.ref'
        cmd = '-c 95 -r 1000 -n 2 -m'
        with tempfile.TemporaryDirectory() as tmp_dir:
            Rouge155.convert_summaries_to_rouge_format(
                dec_path, join(tmp_dir, 'dec'))
            Rouge155.convert_summaries_to_rouge_format(
                ref_path, join(tmp_dir, 'ref'))
            Rouge155.write_config_static(
                join(tmp_dir, 'dec'), dec_pattern,
                join(tmp_dir, 'ref'), ref_pattern,
                join(tmp_dir, 'settings.xml'), system_id=1
            )
            cmd = (join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
                   + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
                   + cmd
                   + ' -a {}'.format(join(tmp_dir, 'settings.xml')))

            output = sp.check_output(cmd.split(' '), universal_newlines=True)
            R_1 = float(output.split('\n')[3].split(' ')[3])
            R_2 = float(output.split('\n')[7].split(' ')[3])
            R_L = float(output.split('\n')[11].split(' ')[3])
            print(output)

        if Print is True:
            rouge_path = join(dec_path, '../ROUGE.txt')
            with open(rouge_path, 'w') as f:
                print(output, file=f)

        return R_1,R_2,R_L