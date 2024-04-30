# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

# 全文code解析可参考此博客：https://zhuanlan.zhihu.com/p/538209695

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input mini-batches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    # 这个类的目的是构造expert输入的mini-batch，以及整合各expert的输出
    
    There are two functions:
    # 函数1：dispatch函数：将函数输入x构造为多个给各expert的输入
    dispatch - take an input Tensor and create input Tensors for each expert.
    # 函数2：将各expert的输出按照gates的权重求和在一起
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
      
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    # gates既决定哪个元素去哪个expert，又决定最后结果的加权平均
    
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    # 上面这个是调用者负责转换的
    
    See common_layers.reshape_like().
    Example use:  一个使用的例子
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
        output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    # 上面是一个使用的例子
    
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates  # gates大小为[batch, num_experts]
        self._num_experts = num_experts
        
        # sort experts
        # torch.nonzero博客：https://blog.csdn.net/qq_50001789/article/details/120606606
        # 其作用是找出所有非零元素，并返回他们的坐标——因此nonzero后的维度为[非零元素个数，2]，一般为[batch*k, 2]
        # torch.sort博客：https://blog.csdn.net/u012495579/article/details/106117511
        # 这里dim=0表示按列排序，sort后的sorted_experts和nonzero后的维度相同
        # index_sorted_experts则表示sort后元素在原数组中对应的位置
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(dim=0)
        
        # drop indices
        # torch.split博客：https://blog.csdn.net/qq_42518956/article/details/103882579
        # 这里就是按dim=1拆
        # 而split(1, dim=1)的第一个1表示切成的块大小为1，由于sorted_experts的第二维为2，故切成了两块
        # self._expert_index的大小为[batch*k, 1]
        _, self._expert_index = sorted_experts.split(1, dim=1)  # 我们这里只保留第二块，也就是对应到expert的那一列
        
        # get according batch index for each expert
        # 由于index_sorted_experts则表示torch.nonzero(gates)做sort后元素在原数组中对应的位置
        # 由于前面sort是按列排的，所以index_sorted_experts[:, 0]表示torch.nonzero(gates)第一列sort后对应于sortr前的位置
        # index_sorted_experts[:, 1]则是第二列
        # 因此这个是获得gates中非零值样本在torch.nonzero(gates)对应的位置是多少
        # self._batch_index大小为[batch*k]
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        
        # calculate num samples that each expert gets
        # 计算每个expert得到了几个样本，结果维度为[num_experts]
        self._part_sizes = (gates > 0).sum(dim=0).tolist()
        
        # expand gates to match with self._batch_index
        # 由于gates为[batch, num_experts]，而self._batch_index为[batch*k]
        # 所以做完后的结果扩展为[bacth*k, num_experts]
        gates_exp = gates[self._batch_index.flatten()]
        
        # torch.gather就是在gates_exp的维度1上以self._expert_index为索引选取出一部分的值出来，再拼接成新的tensor
        # 由于gates_exp大小为[batch*k, num_experts]，self._expert_index大小为[batch*k, 1]
        # 所以gather选取后的结果大小也为[batch*k, 1]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
          # inp的大小为[batch, hidden_size]
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
          # 也就是将batch*k的有效非零输入值拆分成num_experts个输入
          # 例如假设输入为[32, 512], k=2, num_experts=4
          # 那么每个样本会更新两个expert，相当于有32*2=64个有效样本
          # 再按gates比例（如20, 13, 15, 16）拆成4部分
          # 最后输出为[20,512], [13,512], [15,512], [16,512]四个tensor组成的list了
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        # inp的大小为[batch, hidden_size]，而self._batch_index为[batch*k]
        # 所以选取后的inp_exp大小为[batch*k, hidden_size]
        inp_exp = inp[self._batch_index].squeeze(1)
        
        # 最后，inp_exp会按照_part_sizes的比例（如20, 13, 15, 16）拆成4部分
        # 最后输出为[20,512], [13,512], [15,512], [16,512]四个tensor组成的list了
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        # 也即会将各个expert输出结果按照gates的权重加和在一起
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  
        # 具体做法是由于每个样本会由k个expert选择，因此需要将其对应的k个expert的输出结果加权求和
        If `multiply_by_gates` is set to False, the gate values are ignored.
        # PS：如果multiply_by_gates为False的话，那么gates的门控值将不参与计算
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        # 做cat后，四个expert的输出重新变回[batch*k, hidden_size]的大小
        # exp则是做e的幂次，大小不变
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            # torch.mul是对应位置的点乘，让output乘上gates的值
            # stitched为[batch*k, hidden]，_nonzero_gates为[batch*k, 1]，最后结果仍为[batch*k, hidden]
            stitched = stitched.mul(self._nonzero_gates)
        
        # 构建一个大小为[batch, hidden_size]的全0张量
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        
        # combine samples that have been processed by the same k experts
        # index_add函数文档：https://pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html
        # 简单来说，x.index_add_(dim, index, source, *, alpha=1)
        # 表示将source和alpha的乘积按照index的位置加到x的第dim维度上
        # 由于_batch_index虽然长度为batch*k，但其包含的index值范围只有[0, batch]，而每个index会出现k次
        # 因此index_add后就将[batch*k, hidden]的stitched变成了[batch, hidden]的combined的结果了
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        
        # add eps to all zero values in order to avoid nans when going back to log space
        # np.finfo(float).eps表示取出float下的非负的最小值。可以参考此博客：https://blog.csdn.net/Dontla/article/details/103062246
        # 这里就是将为结果中为0的值加上一个eps
        combined[combined == 0] = np.finfo(float).eps
        
        # back to log space
        # 最后再重新log回来，和之前的exp抵消
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
          返回值为一个list，其中每个成员的大小为输入给当前exbert的样本数量
        """
        # split nonzero gates for each expert
        # 由于self._nonzero_gates为[batch*k, 1]，并且会按照_part_sizes的比例（如20, 13, 15, 16）拆成4部分
        # 最后输出为[20,1], [13,1], [15,1], [16,1]四个tensor组成的list
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


# 每个Expert就是一个(input_size, hidden_size) + ReLU + (hidden_size, output_size) + Softmax的两层的MLP
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        # self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # 由于我们在外面另外要做sigmoid转成分类问题的，所以在里面的MLP里无需做softmax！！！
        # out = self.soft(out)
        return out


class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the output
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    # def __init__(self, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=4):
    # 加入input_list，用于完成multi-scale的MoE
    def __init__(self, input_size, output_size, num_experts, hidden_size, input_list=None, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k  # k表示选择每次从num_experts中选出k个专家
        
        # multi-scale专用：表示各个MLP分别输入多少大小的数据
        self.input_list = input_list
        # self.output_list = output_list
        
        # instantiate experts - 实例化共计num_experts个专家
        # self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        
        # 我们这里能否改成multi-scale的MoE？
        if self.input_list is None:
            self.experts = nn.ModuleList([
                MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)
            ])
        else:
            # 各个MLP对不同范围的的数据分别建模
            self.experts = nn.ModuleList([
                MLP(self.input_list[i], self.output_size, self.hidden_size) for i in range(self.num_experts)
            ])
        
        # 门控矩阵W_g
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        # noise矩阵W_noise
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        # https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html 
        # Softplus是ReLU的平滑近似
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        # register_buffer用于定义参数，并且用它定义的参数在训练中将不被更新
        # 有时我们希望模型中某些参数不更新（从开始到结束均保持不变），但又希望参数保存下来（model.state_dict()），这时就会用到register_buffer()
        # PS：此时可以用self.mean和self.std访问这两个变量
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        # 确保选择的k不多于专家总数
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.  # 鼓励分布更加均匀
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`. 其大小为[num_experts]
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        
        # if only num_experts = 1
        if x.shape[0] == 1:
            # 由于在原方法中，做sum前的x为[batch_size, num_experts]
            # 做完sum传入此函数的x则为[num_experts]
            # 如果仅有一个专家，那么根本无需做多个专家间的balance，因此直接返回cv_squared为0
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        else:
            # 否则，返回“x的方差除以x的均值的平方”
            # 如果x分配的越平均，那么方差越小，从而loss也越小
            return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is > 0.
        # 也就是统计每个专家经过了多少样本
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        # 在batch_size维度上求和，得到的结果的大小为[num_experts]
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        # 计算当随机选择时，value出现在top-k中的概率
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        # 这使我们能反向传播“平衡对每个样本当前expert是top-k expert的次数”的loss
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        # 如果不加noise，那么noise_stddev为None，结果也会变成不可微的
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.  
        # noisy_values等于clean_values加上标准差为noise_stddev的正态分布的噪音
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        # noisy_top_values其实就等价于从noisy_values选出的top-m (m >= k+1)的部分
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)  # m一般满足m >= k+1
        # noisy_top_values大小为[batch, m]，展平后则为[batch*m]了
        top_values_flat = noisy_top_values.flatten()

        # threshold_positions_if_in大小为[batch]，其值为[k, k+m, k+2m, ..., k+(batch-1)*m]
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        # torch.gather函数的官方解释：https://pytorch.org/docs/stable/generated/torch.gather.html
        # 知乎的解释：https://zhuanlan.zhihu.com/p/352877584
        # 也就是在top_values_flat的维度0上以threshold_positions_if_in为索引选取出一部分的值出来
        # 其实最后的效果就是将top_values_flat[k], [k+m], [k+2m], ...的值取出来再拼成一个新的tensor
        # 所以这里torch.gather(top_values_flat, 0, threshold_positions_if_in)的效果和top_values_flat[threshold_positions_if_in]的效果是一样的
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        # 由于noisy_values为[batch, n]，而后者为[batch, 1]且选出的是top-m中最小的一个
        # 所以就是看每个batch对应的n个expert中有几个比select的那一个大，理论上来说应该是每个batch对应有m-1个
        is_in = torch.gt(noisy_values, threshold_if_in)
        
        # 减1后变成了threshold_positions_if_out大小也为[batch]，但值为[k-1, k-1+m, k-1+2m, ..., k-1+(batch-1)*m]
        threshold_positions_if_out = threshold_positions_if_in - 1
        # 同样也是将top_values_flat[k-1], [k-1+m], [k-1+2m], ...的值取出来再拼成一个新的tensor
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        
        # is each value currently in the top k.
        # torch.distributions.normal.Normal用法：https://zhuanlan.zhihu.com/p/462654305
        normal = Normal(self.mean, self.std)
        # cdf为其累积分布函数
        # PS：由于clean_values为[batch, n]，而threshold_if_in为[batch, 1]，所以这里要做broadcast
        # 然后cdf表示到该值截止的正态分布的累积分布值应当是多少，prob_if_in也为[batch, n]大小
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        # torch.where文档：https://pytorch.org/docs/stable/generated/torch.where.html
        # 简单来说，torch.where(condition, input, other) → Tensor，如果condition成立，该位置为input，否则为other
        # 所以，最后prob仍为[batch, n]，其中is_in为True的部分的值为prob_if_in，为False的部分的值为prob_if_out
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        # 这里的做法是：我们令input为x，x在进入gate的dense层之前，会加入参数化噪声；
        # 然后计算出结果后，进入softmax之前，直接随机将部分output置于负无穷大，这样softmax之后，这些负无穷大的部分就变成0，从而达到topk的目的。
        # （这里的topk非常简单暴力，就是通过直接的赋值，而没有像dselectk这样将topk转化为可微的操作）
        # 这里加入噪声主要是为了缓解出现部分experts被频繁选择，从而导致少量experts dominate大量下游任务的情况，也可以用dropout来替代。
        # 不过这里的噪声是通过一个可学习的linear matrix来实现的，这个倒是有意思，产生噪声的过程也和model的target强耦合了，interesting
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]  
            train: a boolean - we only add noise at training time. #仅在训练过程中加noise
            noise_epsilon: a float  #noise再加上个epsilon
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        # x为[batch_size, seq_len, hidden_size]
        # 考虑到维度问题，在实际输入时应当将x的前两维合并成一维[batch_size*seq_len, hidden_size]
        # 但是为了表述方便，本文仍然用[batch_size, hidden_size]来代替表示[batch_size*seq_len, hidden_size]
        
        # 所以这里x为[batch_size, hidden_size]，而w_gate为[hidden_size, num_experts]
        clean_logits = x @ self.w_gate  # 计算W_gate和X的矩阵乘积，其结果为[batch_size, num_experts]
        if self.noisy_gating and train:
            # 参数化噪音w_noise
            raw_noise_stddev = x @ self.w_noise  # 计算W_noise和X的矩阵乘积，其结果也为[batch_size, num_experts]
            # softplus是relu的平滑近似，softplus(x) = 1/beta * log(e^(beta*x) + 1)，
            # 由于beta默认为1，所以有softplus(x) = log(e^x + 1)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))  # 加noise_epsilon是为了保证stddev >= 0
            # 加性噪音：前者为门控值clean_logits，后者为噪声noise_stddev再乘一个random矩阵
            # 相当于在门控值上，加上一个noise_stddev倍的随机矩阵
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            # 如果门控不加噪声，那么直接输出clean_logits
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        # 结果会从输入的[batch_size, num_experts]变成[batch_size, k+1]
        # topk函数用法：https://blog.csdn.net/qq_45193872/article/details/119878804
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        # 从k+1项中再选出top-k
        # 此时会从[batch_size, k+1]再变成[batch_size, k]
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        # 再计算softmax得到门控值，其大小也是[batch_size, k]
        top_k_gates = self.softmax(top_k_logits)

        # gates和zeros大小、和logits一样，未做top-k，为[batch_size, num_experts]
        zeros = torch.zeros_like(logits, requires_grad=True)
        # scatter(output, dim, index, src) → Tensor 或 output.scatter(dim, index, src)
        # scatter函数就是把src中的数据重新分配到output当中，index则表示要把src中的数据分配到output数组中的位置，若未指定，则填充0。
        # 简单来说就是按照index顺序将src数据分散地安放在output数组中
        # 可以参考本文：https://blog.csdn.net/lifeplayer_/article/details/111561685
        # 所以这里将gates中、num_experts维度中的top-k值置为非0，其他的值均为0
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        # 由于top_k_gates是已经做了softmax之后的值，所以这里模拟的正好是：
        # gates中的top-k为softmax后的值，非top-k则设为负无穷大做softmax之后的值，也就是0

        if self.noisy_gating and self.k < self.num_experts and train:
            # 得到prob之后再对第0维也即batch维求和
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            # _gates_to_load函数在做：load=(gates>0).sum(0)
            # 其实也就是在统计每个专家经过了多少样本
            load = self._gates_to_load(gates)
        
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        # 也即extra_training_loss是应当被加入到整体的training loss中并做反向传播的
        """
        
        # x为[batch_size, seq_len, hidden_size]
        # 考虑到维度问题，在实际输入时应当将x的前两维合并成一维[batch_size*seq_len, hidden_size]
        # 但是为了表述方便，本文仍然用x为[batch_size, hidden_size]来代替表示[batch_size*seq_len, hidden_size]
        
        # 计算expert importance
        # 其结果gates大小为[batch_size, num_experts]，而load的大小为[num_experts]
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        # 这里由于gates是一个[batch_size，topk_selected_expert_numbers]的matrix，
        # 那么通过sum，计算每个被topk选择出来的experts分配的样本的数量。
        # PS：torch.sum函数：https://blog.csdn.net/qq_39463274/article/details/105145029
        # dim=0表示沿着行求和（即跨行但是同一列的数据求和），列数保持不变
        importance = gates.sum(0)  # 所以这里相当于将各batch的样本求和，求和后大小为[num_experts]
        
        # 加入关于importance的loss作为额外的loss项
        # importance和load的大小均为[num_experts]
        # PS：如果noisy_gating=False时，importance=gates.sum(0)，而load=(gates>0).sum(0)，二者是比较像的！！！
        # 相当于前者是对gates以浮点数权重的形式求和，而后者则是将有权重的非0值直接计数为1之后来求和
        # PPS：前者的expert_importance虽然能鼓励每个expert尽量equally平分一个batch中的samples，然而当sample存在weights的概念时则不行
        # 这意味着可能会存在的问题就是某个expert处理少量的大sample weights的samples，而某个expert处理大量的小sample weights的samples。
        loss = self.cv_squared(importance) + self.cv_squared(load)  # 所以这一额外loss用于鼓励每个expert较为均匀的瓜分一个batch中不同的samples的
        # loss_coef为缩放系数
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        # 这里expert_inputs是按照experts拆分好的数组（如20,13,15,16，按照num_experts拆成4部分)。例如：为[20,512], [13,512], [15,512], [16,512]四个tensor组成的list
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        
        # expert_outputs和expert_inputs一样，也是四个expert对应的输出结果
        expert_outputs = []
        for i in range(self.num_experts):
            if self.input_list is None:
                expert_inp = expert_inputs[i]
            else:
                # self.input_list用于完成multi-scale的MoE模型
                inp_size = self.input_list[i]
                # 这里每个expert输入时只取其对应的长度输入
                expert_inp = expert_inputs[i][:, -inp_size:]
                # print(inp_size, expert_inputs[0].shape, expert_inp.shape)
            output = self.experts[i](expert_inp)
            expert_outputs.append(output)
        
        # 经过combine之后，输出结果大小又变回了[batch, hidden_size]
        y = dispatcher.combine(expert_outputs)
        
        # 最后，返回输出结果y、和用于实现负载平衡的aux_loss
        # 注意这个loss最后是需要加到主实验的loss里去的。
        # 由于loss这里已经乘了loss_coef，所以外面无需再乘比例，而是直接调整loss_coef参数即可。
        return y, loss