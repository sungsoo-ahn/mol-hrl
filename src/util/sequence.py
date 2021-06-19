import torch


def compute_sequence_accuracy(logits, batched_sequence_data, pad_id):
    sequences, _ = batched_sequence_data
    batch_size = sequences.size(0)
    logits = logits[:, :-1]
    targets = sequences[:, 1:]
    preds = torch.argmax(logits, dim=-1)

    correct = preds == targets
    correct[targets == pad_id] = True
    elem_acc = correct[targets != 0].float().mean()
    seq_acc = correct.view(batch_size, -1).all(dim=1).float().mean()

    return elem_acc, seq_acc


def compute_sequence_cross_entropy(logits, batched_sequence_data, pad_id):
    sequences, lengths = batched_sequence_data
    logits = logits[:, :-1]
    targets = sequences[:, 1:]

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="sum",
        ignore_index=pad_id,
    )
    loss /= torch.sum(lengths - 1)

    return loss
