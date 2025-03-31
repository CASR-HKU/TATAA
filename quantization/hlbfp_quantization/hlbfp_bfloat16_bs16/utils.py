def get_fc_layer_idx(model_str="deit", blk_depth=12):
    idx_list = []
    if model_str == "deit":
        # the fc layers contain qkv, proj, fc1, fc2 and head
        for i in range(blk_depth):
            idx_list.append(1 + 0 + i * 7)
            idx_list.append(1 + 4 + i * 7)
            idx_list.append(1 + 5 + i * 7)
            idx_list.append(1 + 6 + i * 7)
        idx_list.append(-1)
    else:
        raise NotImplementedError("model not implemented")
    return idx_list


def get_softmax_layer_idx(model_str="deit", blk_depth=12):
    idx_list = []
    if model_str == "deit":
        # the softmax layers
        for i in range(blk_depth):
            idx_list.append(1 + 2 + i * 7)
        idx_list.append(-1)
    else:
        raise NotImplementedError("model not implemented")
    return idx_list