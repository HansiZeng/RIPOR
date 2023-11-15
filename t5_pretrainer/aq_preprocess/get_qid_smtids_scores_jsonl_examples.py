import os 
import ujson 
import numpy as np 

# max_new_token = 8
for max_new_token in [4, 8, 16, 32]:
    decay = 2
    keep_top100 = True
    decay_to_factor = {
        1: {4: 4./32, 8: 4./32 + 8./32, 16: 4./32 + 8./32 + 16./32},
        2: {4: 1./2, 8: 1./2 + (1./2)**2, 16: 1./2 + (1./2)**2 + (1./2)**3, 32: 1.},
        8: {4: 0.8, 8: 0.8 + (1-0.8)*0.8, 16: 0.8 + (1-0.8)*0.8 + (1-0.8-0.2*0.8)*0.8, 32: 1.}
    }

    factor = decay_to_factor[decay][max_new_token]
    print("factor: ", factor, "max_new_token: ", max_new_token, "keep_top100: ", keep_top100)

    root_dir = "./experiments-full-t5seq-aq/"
    original_qid_smtid_rerank_path = os.path.join(root_dir, f"t5_docid_gen_encoder_1/sub_smtid_{max_new_token}_out/qid_smtid_docids_teacher_score.train.json")
    self_qid_smtid_rerank_path = os.path.join(root_dir, f"t5seq_aq_encoder_seq2seq_1/sub_smtid_{max_new_token}_out/qid_smtid_docids_teacher_score.train.json")
    out_dir = os.path.join(root_dir, f"t5seq_aq_encoder_seq2seq_1/sub_smtid_train_decay{decay}/")

    other_qid_smtid_rerank_path = None

    assert str(max_new_token) in original_qid_smtid_rerank_path and str(max_new_token) in self_qid_smtid_rerank_path 

    with open(original_qid_smtid_rerank_path) as fin:
        original_qid_to_smtid_to_rerank = ujson.load(fin)
    with open(self_qid_smtid_rerank_path) as fin:
        self_qid_to_smtid_to_rerank = ujson.load(fin)

    other_qid_to_smtid_to_rerank = None
    if other_qid_smtid_rerank_path is not None:
        with open(other_qid_smtid_rerank_path) as fin:
            other_qid_to_smtid_to_rerank = ujson.load(fin)

    print("size of original qid_to_smtid_to_rerank: ", len(original_qid_to_smtid_to_rerank))
    print("size of self_qid_to_smtid_to_rerank: ", len(self_qid_to_smtid_to_rerank))
    if other_qid_to_smtid_to_rerank is not None:
        print("size of other_qid_to_smtid_rerank: ", len(other_qid_to_smtid_to_rerank))

    all_qid_to_smtid_to_score = {}
    for qid in original_qid_to_smtid_to_rerank:
        all_qid_to_smtid_to_score[qid] = {}
        for smtid in original_qid_to_smtid_to_rerank[qid]:
            score = max([xs[1] for xs in original_qid_to_smtid_to_rerank[qid][smtid]]) * factor
            all_qid_to_smtid_to_score[qid][smtid] = score
        
        if keep_top100:
            sorted_smtid_score = sorted(all_qid_to_smtid_to_score[qid].items(), key=lambda x: x[1], reverse=True)[:100]
            all_qid_to_smtid_to_score[qid] = dict(sorted_smtid_score)

    total_smtids = 0
    new_intro_smtids = 0
    for qid in self_qid_to_smtid_to_rerank:
        for smtid in self_qid_to_smtid_to_rerank[qid]:
            score =  max([xs[1] for xs in self_qid_to_smtid_to_rerank[qid][smtid]]) * factor
            
            if smtid in all_qid_to_smtid_to_score[qid]:
                all_qid_to_smtid_to_score[qid][smtid] = max(score, all_qid_to_smtid_to_score[qid][smtid])
            else:
                all_qid_to_smtid_to_score[qid][smtid] = score 
                new_intro_smtids += 1

            total_smtids += 1

    if other_qid_to_smtid_to_rerank is not None:
        other_total_smtids = 0
        other_new_introduced_smtids = 0
        for qid in other_qid_to_smtid_to_rerank:
            for smtid in other_qid_to_smtid_to_rerank[qid]:
                score =  max([xs[1] for xs in other_qid_to_smtid_to_rerank[qid][smtid]]) * factor

                if smtid in all_qid_to_smtid_to_score[qid]:
                    all_qid_to_smtid_to_score[qid][smtid] = max(score, all_qid_to_smtid_to_score[qid][smtid])
                else:
                    all_qid_to_smtid_to_score[qid][smtid] = score 
                    other_new_introduced_smtids += 1
                
                other_total_smtids += 1
        
        print("for other data: ")
        print("total smtids in other_data = {:,}, new_introduced smtids = {:,}, ratio = {:.3f}".format(
        other_total_smtids, other_new_introduced_smtids, other_new_introduced_smtids/other_total_smtids))
        print("average length of smtids per query in other_data = {:.3f}".format(total_smtids / len(other_qid_to_smtid_to_rerank)))

    train_examples = []
    smtids_lengths = []
    for qid, smtid_to_score in all_qid_to_smtid_to_score.items():
        pairs = sorted(smtid_to_score.items(), key=lambda x: x[1], reverse=True)
        smtids = []
        scores = []
        for smtid, score in pairs:
            smtids.append(smtid)
            scores.append(score)
        train_examples.append({
            "qid": qid,
            "smtids": smtids,
            "scores": scores
        })
        smtids_lengths.append(len(smtids))

    print("size of example = {}".format(len(train_examples)))
    print("distribution of smtids_lengths: ", np.quantile(smtids_lengths, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
    print("total smtids in self_data = {:,}, new_introduced smtids = {:,}, ratio = {:.3f}".format(
        total_smtids, new_intro_smtids, new_intro_smtids/total_smtids))
    print("average length of smtids per query in self_data = {:.3f}".format(total_smtids / len(self_qid_to_smtid_to_rerank)))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(os.path.join(out_dir, f"qid_smtids_scores_{max_new_token}.train.json"), "w") as fout:
        for example in train_examples:
            fout.write(ujson.dumps(example) + "\n")