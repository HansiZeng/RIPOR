import torch 

class RankNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, eps=1e-10, padded_value_indicator=-1, reduction="mean", sigma=1.):
        device = y_pred.device
        y_pred = y_pred.clone()
        y_true = y_true.clone()
        clamp_val = 1e8 if y_pred.dtype==torch.float32 else 1e4

        padded_mask = y_true == padded_value_indicator
        #y_pred[padded_mask] = float("-inf")
        #y_true[padded_mask] = float("-inf")
        assert torch.sum(padded_mask) == 0

        # Here we sort the true and predicted relevancy scores.
        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)

        # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)
        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

        # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
        true_sorted_by_preds.clamp_(min=0.)

        #inv_pos_idxs = 1. / torch.arange(1, y_pred.shape[1] + 1).to(device)

        # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-clamp_val, max=clamp_val)
        scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
        scores_diffs.clamp_(-20.0, 20.0)
        losses = torch.log(1. + torch.exp(-scores_diffs))  #[bz, topk, topk]

        if reduction == "sum":
            loss = torch.sum(losses[padded_pairs_mask])
        elif reduction == "mean":
            loss = torch.mean(losses[padded_pairs_mask])
        else:
            raise ValueError("Reduction method can be either sum or mean")

        return loss
    
if __name__ == "__main__":
    print('hi')
        