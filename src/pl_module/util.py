import torch
import torch.nn.functional as F

def compute_edge_cross_entropy(logits_edge, batched_data):
    edge_targets_sparse = batched_data["edge_index"].T
    edge_attr_targets = batched_data["edge_attr"][:,0]
    counts = torch.bincount(batched_data["batch"])
    idx = 0
    edge_idx = 0
    edge_targets = torch.zeros(len(counts), max(counts), max(counts)).cuda()
    #edge_targets += torch.eye(max(counts)).unsqueeze(0).repeat(len(counts), 1, 1).cuda()
    edge_mask = torch.zeros(len(counts), max(counts), max(counts)).cuda()
    
    for b in range(len(counts)):
        tmp_mask1 = edge_targets_sparse >= idx 
        tmp_mask2 = edge_targets_sparse < idx + counts[b]
        tmp_mask = torch.logical_and(tmp_mask1, tmp_mask2)
        edge_mol_sparse = (edge_targets_sparse[tmp_mask] - idx).reshape(-1,2)

        for (i,bond) in enumerate(edge_mol_sparse):
            
            edge_targets[b,bond[0],bond[1]] = edge_attr_targets[edge_idx + i]+1
        edge_mask[b,:counts[b],:counts[b]] = 1.0

        idx += counts[b]
        edge_idx += len(edge_mol_sparse)
    edge_mask -= torch.eye(max(counts)).unsqueeze(0).repeat(len(counts), 1, 1).cuda()
    edge_mask[edge_mask<0] = 0
    
    
    #print(logits_edge[0,22:27,22:27,:])
    
    
    #print(logits_edge)
    #print(logits_edge.view(-1, logits_edge.shape[-1]))
    #print(edge_targets)
    #print(edge_targets.view(-1))
    #raise dd    
    edge_loss_no_mask = F.cross_entropy(
        logits_edge.view(-1,logits_edge.shape[-1]), edge_targets.view(-1).long(), reduction="none", #weight = torch.Tensor([1,3,3,3,3]).cuda()
    )
    
    edge_loss = (edge_loss_no_mask * edge_mask.view(-1)).mean()

    edge_preds = torch.argmax(logits_edge, dim=-1)

    edge_preds[edge_mask == 0.0] = 0
    edge_elem_acc = (edge_preds == edge_targets).float().mean()

    edge_mol_acc = (edge_preds == edge_targets).view(edge_preds.shape[0], -1).all(dim=1).float().mean()
    
    pred_zero = (edge_preds == 0).sum()
    gt_zero = (edge_targets == 0).sum()

    pred_one = (edge_preds == 1).sum()
    gt_one = (edge_targets == 1).sum()
    print("pred")
    for i in range(len(edge_preds[0])):
        print(edge_preds[0,i,:])
    print("target")
    for j in range(len(edge_targets[0])):
        print(edge_targets[0,j,:])
    return edge_loss, edge_elem_acc, edge_mol_acc, pred_zero, gt_zero, pred_one, gt_one



def compute_node_cross_entropy(logits_atom,logits_chirality, batched_data):
    atom_targets = batched_data["x"][:,0]
    chirality_targets = batched_data["x"][:,1]
    #print(logits_atom)
    #print(torch.argmax(logits_atom, dim=-1))
    #print(atom_targets)
    atom_loss = F.cross_entropy(
        logits_atom, atom_targets
    )
    chirality_loss = F.cross_entropy(
        logits_chirality, chirality_targets
    )
    return atom_loss, chirality_loss

def compute_node_accuracy(logits_atom, logits_chirality, batched_data):
    batch_size = max(batched_data["batch"]) + 1
    
    atom_preds = torch.argmax(logits_atom, dim=-1)
    atom_targets = batched_data["x"][:,0]
    chirality_preds = torch.argmax(logits_chirality, dim=-1)
    chirality_targets = batched_data["x"][:,1]
    
    atom_correct = atom_preds == atom_targets
    chirality_correct = chirality_preds == chirality_targets
    
    atom_elem_acc = atom_correct.float().mean()
    #atom_mol_acc = atom_correct.view(batch_size, -1).all(dim=1).float().mean()
    
    chirality_elem_acc = chirality_correct.float().mean()

    idx = 0
    chirality_mol_correct = 0
    atom_mol_correct = 0
    counts = torch.bincount(batched_data["batch"])
    
    for i in range(batch_size):
        chirality_mol_correct += int(sum(chirality_correct[idx:idx+counts[i]].float()) == counts[i])
        atom_mol_correct += int(sum(atom_correct[idx:idx+counts[i]].float()) == counts[i])
        idx += counts[i]
    
    chirality_mol_acc = chirality_mol_correct / batch_size
    atom_mol_acc = atom_mol_correct / batch_size
    
    #chirality_mol_acc = chirality_correct.view(batch_size, -1).all(dim=1).float().mean()

    return atom_elem_acc, atom_mol_acc, chirality_elem_acc, chirality_mol_acc



def compute_sequence_accuracy(logits, batched_sequence_data, pad_id=0):
    batch_size = batched_sequence_data.size(0)
    logits = logits[:, :-1]
    targets = batched_sequence_data[:, 1:]
    preds = torch.argmax(logits, dim=-1)

    correct = preds == targets
    correct[targets == pad_id] = True
    elem_acc = correct[targets != 0].float().mean()
    sequence_acc = correct.view(batch_size, -1).all(dim=1).float().mean()

    return elem_acc, sequence_acc


def compute_sequence_cross_entropy(logits, batched_sequence_data, pad_id=0):
    logits = logits[:, :-1]
    targets = batched_sequence_data[:, 1:]

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_id,
    )

    return loss