import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


# numel(),size(),view(),expand(),expand_as, reshape()
"""x = torch.rand(5, 2, 4)
t2 = x.transpose(0, 1)
print(x)
print(x.numel())
print(x.size())
print(x.view(-1, 4))
print(x.reshape([-1, 4]))"""
"""y = torch.tensor([[[1,3,6],[4,7,9], [0,0,0]],[[1,2,3], [0,0,0], [7,0,0]]])
x = torch.tensor([[0,6,5],[1,3,6]])
x_ex = x.unsqueeze(2).expand_as(y) + 1
print(x_ex)
z = torch.nonzero(x_ex-y)
print(z.size()[0])
print(y.numel() - z.size()[0])"""

"""word_att = torch.tensor([[[0,6,5],[1,3,6],[9,3,2],[1,5,6]],
                  [[0,9,2],[1,5,7],[2,2,2],[1,1,1]]])
# 2:batch size 4:max word num in the sentence, 3:feature size
word_att_q = word_att.unsqueeze(2).expand(size=[2,4,4,3])
word_att_k= word_att.unsqueeze(1).expand(size=[2,4,4,3])
dot = alpha_net_word(F.tanh(word_att_q + word_att_k))  # 2*4*4*1
weight_word = F.softmax(dot.squeeze(3), dim=2)  # 2 * 4 * 4
att_res_word = torch.bmm(weight_word, word_rnn)  # 2 * 4 * 3"""


# y = torch.zeros(size=[5,3,2,4])
# print(x.unsqueeze(1).expand_as(y))

# torch.bmm (Batch Matrix-Matrix product)
"""weight = torch.from_numpy(np.array([[0.3,0.7],[0.9,0.1]])) # 2*2
feature = torch.from_numpy(np.array([[[1,2,3],[4,5,6]],
                    [[-1,-2,-3],[-4,-5,-6]]], dtype=np.float)) # 2*2*3
weighted_feature = torch.bmm(weight.unsqueeze(1), feature) # 2*1*2 X 2*2*3 = 2*1*3
print(weighted_feature)"""

# torch log_softmax
# data = torch.from_numpy(np.array([[[1.0],[2.0],[3.0]], [[1.0],[2.0],[3.0]]]))
# data = torch.from_numpy(np.array([[1.0], [2.0], [3.0]]))
"""softmax = F.softmax(data.squeeze(2), dim=1)
print(softmax.unsqueeze(2))
log_softmax = F.log_softmax(data)
print(log_softmax)"""

# torch.cat
"""time0 = torch.from_numpy(np.array([[[1.0,1.0,1.0]],[[2.0,2.0,2.0]]])) # 2 * 1 * 3
time1 = torch.from_numpy(np.array([[[-1.0,-1.0,-1.0]],[[-2.0,-2.0,-2.0]]]))  # 2 * 1 * 3
time2 = torch.from_numpy(np.array([[[-1.0,-1.5,-2.0]],[[-4.0,-2.0,-5.0]]]))  # 2 * 1 * 3
print(time0.shape, time1.shape, time2.shape)
seq = torch.cat([time0, time1, time2], 1)
print(seq, seq.shape)"""

# torch.gather
# batch * seq_len * vocab_size = 2 * 3 * 4
"""output = torch.Tensor([[[0.3,0.4,0.5,0.6],[0.2,0.4,0.9,0.1],[0.5,0.3,0.9,0.1]],
                       [[0.7,0.8,0.2,0.1],[0.2,0.4,0.3,0.1],[0.2,0.1,0.6,0.7]]])
print(output.shape)
# batch * seq_len * 1 = 2 * 3 * 1
target = torch.LongTensor([[[1],[2],[3]],[[0],[2],[1]]])
print(target.shape)
gather_result = output.view(-1, output.size(2)).gather(1, target.view(-1, 1))
print(gather_result)"""



# torch uniform
"""fc_feats = torch.from_numpy(np.array([[1.0,3.0,4.0],[2.0,4.0,5.0]]))  # batch * feat_size
sample_prob = fc_feats.data.new(fc_feats.size(0)).uniform_(0, 1)
sammple_mask = sample_prob < 0.5
sample_ind = sammple_mask.nonzero().view(-1)
print(sample_prob)
print(sammple_mask)
print(sample_ind)"""

# torch.multinominal
# batch * vocab_size
"""prob_prev = torch.from_numpy(np.array([[1.0, 3.0, 4.0], [5.0, 8.0, 4.0]]))
result = torch.multinomial(prob_prev, 1).view(-1)
print(result)"""

# torch.index_select, index_copy_
"""it = torch.from_numpy(np.array([1,1,1,1]))
mulitnomial_result = torch.from_numpy(np.array([5,6,2,4]))  # batch
sample_ind = torch.LongTensor([2,1])
select_result = mulitnomial_result.index_select(0, sample_ind)
print(select_result)
it.index_copy_(0, sample_ind, select_result)
print(it)"""

"""sen_emb = torch.from_numpy(np.array([[[1,2,3],[4,5,6],[7,8,9]],[[-1,-2,-3],[-4,-5,-6],[-7,-8,-9]]]))
sim_sen = torch.from_numpy(np.array([[1,0,1], [0,1,1]]))
sim_sen = sim_sen.unsqueeze(2).expand_as(sen_emb)
print(sim_sen)
sim_sen_emb = sen_emb*sim_sen # 2 * 3 * 3
print(sim_sen_emb)
sum_sen_emb = sim_sen_emb.sum(dim=1)
print(sum_sen_emb)
print(sum_sen_emb.size())"""
# torch embed
"""embed = nn.Embedding(5, 6)
sent_pointer = torch.LongTensor([[[1,1],[2,2],[0,0]],[[1,0],[1,2],[2,1]]])
sent_pointer_embed = embed(sent_pointer)
print(sent_pointer_embed)
sent_pointer_embed = sent_pointer_embed.sum(dim=2)
print(sent_pointer_embed)"""

# torch gt lt eq nozero ne
"""ix = torch.tensor([1, 9, 4, 5, 2, 7])
index = torch.lt(ix, 5).long()
print(index, index.type())
mask_ix = torch.mul(ix, index)
print(mask_ix)"""

# longtensor
# a = torch.LongTensor([x for x in np.array([2,3])])
# print(a)

#reshape
# 2*4*2 batch*sequence_length*dim
"""word_embed = torch.tensor([[[1,2],[3,4],[5,6],[7,8]],[[-1,-2],[-3,-4],[-5,-6],[-7,-8]]])
word_embed = word_embed.reshape([-1, 2, 2, 2])
print(word_embed)
word_embed = word_embed.reshape([-1, 2, 2])
print(word_embed)
w_out = word_embed * 2
print(w_out)
w_out = w_out.reshape([-1, 4, 2])
print(w_out)"""

# mask
"""mask = torch.tensor([[1.0,0.0,1.0,0.0],[1.0,0.0,1.0,0.0]])
b =  torch.tensor([[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0]])
b = b * mask
b = b.unsqueeze(2)
print(b)"""

# gather
# batch =2 seq=2 leng=3
"""a = torch.tensor([[0.8, 0.1, 0.1],[0.2, 0.6, 0.2],[0.3, 0.2, 0.5],[0.1, 0.2, 0.7]])
a = - torch.log(a)
print(a)
b = torch.tensor([[0],[2],[1],[0]])
c = a.gather(1, b)
for x in b :
    assert x[0] < 2
print(c)"""

# pad
"""a = torch.tensor([[1.0,1.0,1.0],[2.0,2.0,2.0]])
b = F.pad(a, [1,0,0,0], 'constant', 0)
print(a)
print(b)"""

# sum
"""mask = torch.tensor([[[1.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0]],[[1.0,1.0,1.0,0.0],[0.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0]]])
mask_sum = torch.sum(mask, dim=2)
mask_nonzero = torch.nonzero(mask_sum)
num = mask_nonzero.size()[0]
print(mask_sum)
print(mask_nonzero)
print(num)"""

# argmax
"""a = torch.tensor([[[1,4,7,3,1],[2,8,5,6,7]],[[2,2,2,5,4],[9,8,5,5,3]]])
a_arg = torch.argmax(a, dim=2)
print(a_arg)"""

# mask
"""x = torch.Tensor([[0,0,0,4,0,5,0,0,0],[0,0,3,0,0,0,0,0,0]])
y = torch.Tensor([[0,0,0,4,0,3,0,1,0],[0,0,3,0,5,0,0,0,0]])
mask = torch.ByteTensor(x > 0).float() - 1
print(mask)
diff = x - y + mask
print(diff)
true_num = diff.numel() - torch.nonzero(diff).size()[0]
print(true_num)"""

# mask
"""mask = torch.Tensor([[0,0,0,0],[1,2,0,0],[3,4,0,0],[5,0,0,0]])
mask_num = torch.nonzero(torch.sum(mask, dim=1)).size()[0]
print(mask_num)"""


# auto tile
"""x = torch.Tensor([[[1],[2],[3]],[[4],[5],[6]]]) # 2*3*1
y = x.view(2, 1, -1)
z = x-y
print(z)"""

# arange
"""x = torch.arange(8)
print(x)"""

#
"""y = torch.Tensor([[[[1,2,3,4],[-1,-2,-3,-4]],[[1,2,3,4],[-1,-2,-3,-4]]]]) # 1*2*2*4
print(y.shape)
z = list(y.shape[:3])
z.insert(1,1)
print(z)"""

# pytorch nll loss
"""input = torch.randn(3, 5, requires_grad=True)
target = torch.tensor([1, 0, 4])
logprobs = F.log_softmax(input)
print(logprobs)
output = F.nll_loss(logprobs, target, reduction='none')
print(output)"""

# pytorch *= and mul
"""a = torch.tensor([[[1,0],[2,0]],[[3,0],[4,0]]])
b = torch.tensor([[1,0], [0,1]])
print(b.unsqueeze(-1))
c = torch.mul(a, b.unsqueeze(-1))
print(c)
a *= b.unsqueeze(-1)
print(a*-1)"""

a = torch.zeros(5)
b = torch.ones(4)
c = torch.cat([a, b], dim=0)
print(c)

