import torch

from utils.config import FLAGS


## for soft-target
class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification """
    ## target := [Batch, num_classes], i.e. label (softmax), prediction of supernet
    ## output_log_prob (olp) := [Batch, num_classes], i.e. prediction of subnet
    ## target_unsqz = [Batch, 1, num_classes]
    ## olp_unsqz = [Batch, num_classes, 1]
    ## CE_loss = -1 X [Batch, 1, num_classes] X [Batch, num_classes, 1] -> [Batch, 1, 1], i.e. sum(-label * log Predcition)
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return cross_entropy_loss


## for CE Loss 
class CrossEntropyLossSmooth(torch.nn.modules.loss._Loss):
    """ label smooth """
    ## target := [Batch]
    ## one_hot := [Batch, num_classes], i.e. zeros_like(*).scatter :sparse to dense operation
    ## output_log_prob (olp) := [Batch, num_classes], i.e. prediction of subnet
    ## target_unsqz := [Batch, 1, num_classes]
    ## olp_unsqz := [Batch, num_classes, 1]
    ## CE_loss = -1 X [Batch, 1, num_classes] X [Batch, num_classes, 1] -> [Batch, 1, 1], i.e. sum(-label * log Predcition)
    def forward(self, output, target):
        eps = FLAGS.label_smoothing
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        target = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return cross_entropy_loss

class KLLossSoft(torch.nn.modules.loss._Loss):
    def forward(self, output, soft_logits, target=None, temperature=1., alpha=0.9):
        output, soft_logits = output / temperature, soft_logits / temperature
        soft_target_prob = torch.nn.functional.softmax(soft_logits, dim=1)
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        kd_loss = -torch.sum(soft_target_prob * output_log_prob, dim=1)
        if target is not None:
            n_class = output.size(1)
            target = torch.zeros_like(output).scatter(1, target.view(-1, 1),1)
            target = target.unsqueeze(1)
            output_log_prob = output_log_prob.unsqueeze(2)
            ce_loss = -torch.bmm(target, output_log_prob).squeeze()
            loss = alpha * temperature * temperature * kd_loss + (1.0-alpha) * ce_loss
        else:
            loss = kd_loss

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()
        return loss
