## Operator List

## v1.0
- addmm
    ```python
    torch.addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None) → Tensor
    ```

    Performs a matrix multiplication of the matrices $mat1$ and $mat2$. The matrix $input$ is added to the final result.

    $out = input + \alpha (mat1 \times mat2)$
- bmm
    ```python
    torch.bmm(input, mat2, *, out=None) → Tensor
    ```

    Performs a batch matrix-matrix product of matrices stored in $input$ and $mat2$.

    $out = input + \alpha (mat1 \times mat2)$
- ~~matmul~~
- mm
    ```python
    torch.mm(input, mat2, *, out=None) → Tensor
    ```

    Performs a matrix multiplication of the matrices `input` and `mat2`.
- cumsum
    ```python
    torch.cumsum(input, dim, *, dtype=None, out=None) → Tensor

    >>> a = torch.randint(1, 20, (10,))
    >>> a
    tensor([13,  7,  3, 10, 13,  3, 15, 10,  9, 10])
    >>> torch.cumsum(a, dim=0)
    tensor([13, 20, 23, 33, 46, 49, 64, 74, 83, 93])
    ```

    Returns the cumulative sum of elements of $input$ in the dimension $dim$.
- ~~linear~~
    ```python
    torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
    ```

    Applies an affine linear transformation to the incoming data: $y = xA^T + b$
- dropout
    ```python
    torch.nn.Dropout(p=0.5, inplace=False)
    ```

    During training, randomly zeroes some of the elements of the input tensor with probability $p$.
- gelu
    ```python
    torch.nn.GELU(approximate='none')
    ```

    Applies the Gaussian Error Linear Units function.

    $GELU(x) = x ∗ \Phi(x)$

    where $\Phi(x)$ is the Cumulative Distribution Function for Gaussian Distribution.

    When the approximate argument is `tanh`, Gelu is estimated with:

    $GELU(x) = 0.5 ∗ x ∗ (1 + Tanh(\sqrt{2/\pi} ∗ (x + 0.044715 ∗ x^3)))$
- layer_norm
    ```python
    torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None)
    ```

    Applies Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as

    $y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta$

    The mean and standard-deviation are calculated over the last `D` dimensions, where `D` is the dimension of `normalized_shape`.
- native_dropout
- ~~native_layer_norm~~
- relu
    ```python
    torch.nn.ReLU(inplace=False)
    ```

    Applies the rectified linear unit function element-wise.

    $RELU(x) = (x)^+ = max(0, x)$
- silu
    ```python
    torch.nn.SiLU(inplace=False)
    ```

    Applies the Sigmoid Linear Unit (SiLU) function, element-wise.

    The SiLU function is also known as the swish function.

    $silu(x) = x * \sigma(x)$, where $\sigma(x)$ is the logistic sigmoid.
- softmax
    ```python
    torch.nn.Softmax(dim=None)
    ```

    Applies the Softmax function to an n-dimensional input Tensor.

    Rescales them so that the elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1.

    Softmax is defined as:

    $Softmax(x) = \frac{exp(x_i)}{\sum_j exp(x_j)}$
- triu
    ```python
    torch.triu(input, diagonal=0, *, out=None)

    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[ 0.2309,  0.5207,  2.0049],
            [ 0.2072, -1.0680,  0.6602],
            [ 0.3480, -0.5211, -0.4573]])
    >>> torch.triu(a)
    tensor([[ 0.2309,  0.5207,  2.0049],
            [ 0.0000, -1.0680,  0.6602],
            [ 0.0000,  0.0000, -0.4573]])
    >>> torch.triu(a, diagonal=1)
    tensor([[ 0.0000,  0.5207,  2.0049],
            [ 0.0000,  0.0000,  0.6602],
            [ 0.0000,  0.0000,  0.0000]])
    >>> torch.triu(a, diagonal=-1)
    tensor([[ 0.2309,  0.5207,  2.0049],
            [ 0.2072, -1.0680,  0.6602],
            [ 0.0000, -0.5211, -0.4573]])
    ```

    Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.

    The upper triangular part of the matrix is defined as the elements on and above the diagonal.

    The argument diagonal controls which diagonal to consider. If diagonal = 0, all elements on and above the main diagonal are retained. A positive value excludes just as many diagonals above the main diagonal, and similarly a negative value includes just as many diagonals below the main diagonal.
- abs
    ```python
    torch.abs(input: Tensor, *, out: Optional[Tensor]) → Tensor
    ```

    Computes the absolute value of each element in `input`.

    $out_i = |input_i|$
- add
    ```python
    torch.add(input, other, *, alpha=1, out=None) → Tensor
    ```

    Adds `other`, scaled by `alpha`, to `input`.

    $out_i = input_i + alpha \times other_i$
- div
    ```python
    torch.div(input, other, *, rounding_mode=None, out=None) → Tensor
    ```

    Divides each element of the input input by the corresponding element of other.

    $out_i = \frac{input_i}{other_i}$
- exp
    ```python
    torch.exp(input, *, out=None) → Tensor
    ```

    Returns a new tensor with the exponential of the elements of the input tensor `input`.

    $y_i = e^{x_i}$
- mul
    ```python
    torch.mul(input, other, *, out=None) → Tensor
    ```

    Multiplies input by other.

    $out_i = input_i \times other_i$
- pow
    ```python
    torch.pow(input, exponent, *, out=None) → Tensor
    ```

    Takes the power of each element in `input` with `exponent` and returns a tensor with the result.

    `exponent` can be either a single `float` number or a Tensor with the same number of elements as `input`.

    When `exponent` is a scalar value, the operation applied is:

    $out_i = x_i^{exponent}$

    When `exponent` is a tensor, the operation applied is:

    $out_i = x_i^{exponent_i}$

    When `exponent` is a tensor, the shapes of `input` and `exponent` must be broadcastable.
- reciprocal
    ```python
    torch.reciprocal(input, *, out=None) → Tensor
    ```

    Returns a new tensor with the reciprocal of the elements of `input`

    $out_i = \frac{1}{input_i}$
- rsqrt
    ```python
    torch.rsqrt(input, *, out=None) → Tensor
    ```

    Returns a new tensor with the reciprocal of the square-root of each of the elements of `input`.

    $out_i = \frac{1}{\sqrt{input_i}}$
- ~~rsub~~
- sub
    ```python
    torch.sub(input, other, *, alpha=1, out=None) → Tensor
    ```

    Subtracts `other`, scaled by `alpha`, from `input`.

    $out_i = input_i - alpha \times other_i$
- mean
    ```python
    torch.mean(input, *, dtype=None) → Tensor

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.2294, -0.5481,  1.3288]])
    >>> torch.mean(a)
    tensor(0.3367)
    ```

    Returns the mean value of all elements in the `input` tensor. Input must be floating point or complex.

## v2.0
- all
    ```python
    torch.all(input: Tensor) → Tensor

    >>> a = torch.rand(1, 2).bool()
    >>> a
    tensor([[False, True]], dtype=torch.bool)
    >>> torch.all(a)
    tensor(False, dtype=torch.bool)
    >>> a = torch.arange(0, 3)
    >>> a
    tensor([0, 1, 2])
    >>> torch.all(a)
    tensor(False)
    ```

    Tests if all elements in `input` evaluate to True.

    ```python
    torch.all(input, dim, keepdim=False, *, out=None) → Tensor

    >>> a = torch.rand(4, 2).bool()
    >>> a
    tensor([[True, True],
            [True, False],
            [True, True],
            [True, True]], dtype=torch.bool)
    >>> torch.all(a, dim=1)
    tensor([ True, False,  True,  True], dtype=torch.bool)
    >>> torch.all(a, dim=0)
    tensor([ True, False], dtype=torch.bool)
    ```

    For each row of `input` in the given dimension `dim`, returns `True` if all elements in the row evaluate to `True` and `False` otherwise.
- any
    ```python
    torch.any(input: Tensor, *, out: Optional[Tensor]) → Tensor

    >>> a = torch.rand(1, 2).bool()
    >>> a
    tensor([[False, True]], dtype=torch.bool)
    >>> torch.any(a)
    tensor(True, dtype=torch.bool)
    >>> a = torch.arange(0, 3)
    >>> a
    tensor([0, 1, 2])
    >>> torch.any(a)
    tensor(True)
    ```

    Tests if any element in `input` evaluates to `True`.

    ```python
    torch.any(input, dim, keepdim=False, *, out=None) → Tensor

    >>> a = torch.randn(4, 2) < 0
    >>> a
    tensor([[ True,  True],
            [False,  True],
            [ True,  True],
            [False, False]])
    >>> torch.any(a, 1)
    tensor([ True,  True,  True, False])
    >>> torch.any(a, 0)
    tensor([True, True])
    ```

    For each row of `input` in the given dimension `dim`, returns `True` if any element in the row evaluate to `True` and `False` otherwise.

    If keepdim is `True`, the output tensor is of the same size as `input` except in the dimension(s) `dim` where it is of size 1.
- bitwise_and
    ```python
    torch.bitwise_and(input, other, *, out=None) → Tensor

    >>> torch.bitwise_and(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    tensor([1, 0,  3], dtype=torch.int8)
    >>> torch.bitwise_and(torch.tensor([True, True, False]), torch.tensor([False, True, False]))
    tensor([ False, True, False])
    ```

    Computes the bitwise AND of `input` and `other`. The input tensor must be of integral or Boolean types. For bool tensors, it computes the logical AND.
- bitwise_not
    ```python
    torch.bitwise_not(input, *, out=None) → Tensor

    >>> torch.bitwise_not(torch.tensor([-1, -2, 3], dtype=torch.int8))
    tensor([ 0,  1, -4], dtype=torch.int8)
    ```

    Computes the bitwise NOT of the given input tensor. The input tensor must be of integral or Boolean types. For bool tensors, it computes the logical NOT.
- bitwise_or
    ```python
    torch.bitwise_or(input: Tensor, other: Tensor, *, out: Optional[Tensor]) → Tensor

    >>> torch.bitwise_or(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    tensor([-1, -2,  3], dtype=torch.int8)
    >>> torch.bitwise_or(torch.tensor([True, True, False]), torch.tensor([False, True, False]))
    tensor([ True, True, False])
    ```

    Computes the bitwise OR of `input` and `other`. The input tensor must be of integral or Boolean types. For bool tensors, it computes the logical OR.
- cos
    ```python
    torch.cos(input, *, out=None) → Tensor
    ```

    Returns a new tensor with the cosine of the elements of `input`.

    $out_i = cos(input_i)$
- clamp
    ```python
    torch.clamp(input, min=None, max=None, *, out=None) → Tensor
    ```

    Clamps all elements in `input` into the range [`min`, `max`]. Letting min_value and max_value be `min` and `max`, respectively, this returns:

    $y_i = min(max(x_i, min\_value_i), max\_value_i)$

    If `min` is None, there is no lower bound. Or, if `max` is None there is no upper bound.
- eq
    ```python
    torch.eq(input, other, *, out=None) → Tensor

    >>> torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[ True, False],
            [False, True]])
    ```

    Computes element-wise equality.

    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
- ge
    ```python
    torch.ge(input, other, *, out=None) → Tensor

    >>> torch.ge(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[True, True], [False, True]])
    ```

    Computes $input \ge other$ element-wise.

    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
- gt
    ```python
    torch.gt(input, other, *, out=None) → Tensor

    >>> torch.gt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[False, True], [False, False]])
    ```

    Computes $input > other$ element-wise.

    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
- mv
    ```python
    torch.mv(input, vec, *, out=None) → Tensor

    >>> mat = torch.randn(2, 3)
    >>> vec = torch.randn(3)
    >>> torch.mv(mat, vec)
    tensor([ 1.0404, -0.6361])
    ```

    Performs a matrix-vector product of the matrix `input` and the vector `vec`.
- isinf
    ```python
    torch.isinf(input) → Tensor

    >>> torch.isinf(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
    tensor([False,  True,  False,  True,  False])
    ```

    Tests if each element of `input` is infinite (positive or negative infinity) or not.
- isnan
    ```python
    torch.isnan(input) → Tensor

    >>> torch.isnan(torch.tensor([1, float('nan'), 2]))
    tensor([False, True, False])
    ```

    Returns a new tensor with boolean elements representing if each element of input is NaN or not. Complex values are considered NaN when either their real and/or imaginary part is NaN.
- le
    ```python
    torch.le(input, other, *, out=None) → Tensor
    ```

    Computes $input \le other$ element-wise.

    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
- lt
    ```python
    torch.lt(input, other, *, out=None) → Tensor
    ```

    Computes $input < other$ element-wise.

    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
- ne
    ```python
    torch.ne(input, other, *, out=None) → Tensor

    >>> torch.ne(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[False, True], [True, False]])
    ```

    Computes $input \neq other$ element-wise.

    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
- neg
    ```python
    torch.neg(input, *, out=None) → Tensor

    >>> a = torch.randn(5)
    >>> a
    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])
    >>> torch.neg(a)
    tensor([-0.0090,  0.2262,  0.0682,  0.2866, -0.3940])
    ```

    Returns a new tensor with the negative of the elements of `input`.

    $out = -1 \times input$
- ~~or_~~
- sin
    ```python
    torch.sin(input, *, out=None) → Tensor
    ```

    Returns a new tensor with the sine of the elements of `input`.

    $out_i = sin(input_i)$
- tanh
    ```python
    torch.tanh(input, *, out=None) → Tensor
    ```

    Returns a new tensor with the hyperbolic tangent of the elements of `input`.

    $out_i = tanh(input_i)$
- amax
    ```python
    torch.amax(input, dim, keepdim=False, *, out=None) → Tensor

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.8177,  1.4878, -0.2491,  0.9130],
            [-0.7158,  1.1775,  2.0992,  0.4817],
            [-0.0053,  0.0164, -1.3738, -0.0507],
            [ 1.9700,  1.1106, -1.0318, -1.0816]])
    >>> torch.amax(a, 1)
    tensor([1.4878, 2.0992, 0.0164, 1.9700])
    ```

    Returns the maximum value of each slice of the `input` tensor in the given dimension(s) `dim`.
- argmax
    ```python
    torch.argmax(input) → LongTensor

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-1.3948,  0.2663, -0.2686,  0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [ 0.4907,  1.3398, -1.0691, -0.3132],
            [-1.6092,  0.5419, -0.2993,  0.3195]])
    >>> torch.argmax(a)
    tensor(9)
    ```

    Returns the indices of the maximum value of all elements in the `input` tensor.

    ```python
    torch.argmax(input, dim, keepdim=False) → LongTensor

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [ 0.4907, -1.3948, -1.0691, -0.3132],
            [-1.6092,  0.5419, -0.2993,  0.3195]])
    >>> torch.argmax(a, dim=1)
    tensor([ 0,  2,  0,  1])
    ```

    Returns the indices of the maximum values of a tensor across a dimension.
- argmin
    ```python
    torch.argmin(input, dim=None, keepdim=False) → LongTensor

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.1139,  0.2254, -0.1381,  0.3687],
            [ 1.0100, -1.1975, -0.0102, -0.4732],
            [-0.9240,  0.1207, -0.7506, -1.0213],
            [ 1.7809, -1.2960,  0.9384,  0.1438]])
    >>> torch.argmin(a)
    tensor(13)
    >>> torch.argmin(a, dim=1)
    tensor([ 2,  1,  3,  1])
    >>> torch.argmin(a, dim=1, keepdim=True)
    tensor([[2],
            [1],
            [3],
            [1]])
    ```

    Returns the indices of the minimum value(s) of the flattened tensor or along a dimension
- max
    ```python
    torch.max(input) → Tensor

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.6763,  0.7445, -2.2369]])
    >>> torch.max(a)
    tensor(0.7445)
    ```

    Returns the maximum value of all elements in the `input` tensor.
- min
    ```python
    torch.min(input) → Tensor

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.6750,  1.0857,  1.7197]])
    >>> torch.min(a)
    tensor(0.6750)
    ```

    Returns the minimum value of all elements in the `input` tensor.
- outer
    ```python
    torch.outer(input, vec2, *, out=None) → Tensor

    >>> v1 = torch.arange(1., 5.)
    >>> v2 = torch.arange(1., 4.)
    >>> torch.outer(v1, v2)
    tensor([[  1.,   2.,   3.],
            [  2.,   4.,   6.],
            [  3.,   6.,   9.],
            [  4.,   8.,  12.]])
    ```

    Outer product of `input` and `vec2`. If `input` is a vector of size $n$ and `vec2` is a vector of size $m$, then `out` must be a matrix of size $(n \times m)$.
- prod
    ```python
    torch.prod(input: Tensor, *, dtype: Optional[_dtype]) → Tensor

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[-0.8020,  0.5428, -1.5854]])
    >>> torch.prod(a)
    tensor(0.6902)
    ```

    Returns the product of all elements in the `input` tensor.

    ```python
    torch.prod(input, dim, keepdim=False, *, dtype=None) → Tensor

    >>> a = torch.randn(4, 2)
    >>> a
    tensor([[ 0.5261, -0.3837],
            [ 1.1857, -0.2498],
            [-1.1646,  0.0705],
            [ 1.1131, -1.0629]])
    >>> torch.prod(a, 1)
    tensor([-0.2018, -0.2962, -0.0821, -1.1831])
    ```

    Returns the product of each row of the `input` tensor in the given dimension `dim`.
- sum
    ```python
    torch.sum(input, *, dtype=None) → Tensor

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.1133, -0.9567,  0.2958]])
    >>> torch.sum(a)
    tensor(-0.5475)
    ```

    Returns the sum of all elements in the input tensor.

    ```python
    torch.sum(input, dim, keepdim=False, *, dtype=None) → Tensor

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
            [-0.2993,  0.9138,  0.9337, -1.6864],
            [ 0.1132,  0.7892, -0.1003,  0.5688],
            [ 0.3637, -0.9906, -0.4752, -1.5197]])
    >>> torch.sum(a, 1)
    tensor([-0.4598, -0.1381,  1.3708, -2.6217])
    >>> b = torch.arange(4 * 5 * 6).view(4, 5, 6)
    >>> torch.sum(b, (2, 1))
    tensor([  435.,  1335.,  2235.,  3135.])
    ```

    Returns the sum of each row of the `input` tensor in the given dimension `dim`. If `dim` is a list of dimensions, reduce over all of them.
- var_mean
    ```python
    torch.var_mean(input, dim=None, *, correction=1, keepdim=False, out=None)

    >>> a = torch.tensor(
    ...     [[ 0.2035,  1.2959,  1.8101, -0.4644],
    ...      [ 1.5027, -0.3270,  0.5905,  0.6538],
    ...      [-1.5745,  1.3330, -0.5596, -0.6548],
    ...      [ 0.1264, -0.5080,  1.6420,  0.1992]])
    >>> torch.var_mean(a, dim=0, keepdim=True)
    (tensor([[1.5926, 1.0056, 1.2005, 0.3646]]),
    tensor([[ 0.0645,  0.4485,  0.8707, -0.0665]]))
    ```

    Calculates the variance and mean over the dimensions specified by `dim`. `dim` can be a single dimension, list of dimensions, or None to reduce over all dimensions.

    The variance ($\sigma^2$) is calculated as

    $\sigma^2 = \frac{1}{\max(0, N-\delta N)}\sum^{N-1}_{i=0}(x_i-\bar{x})^2$

    where $x$ is the sample set of elements, $\bar{x}$ is the sample mean, $N$ is the number of samples and $\delta N$ is the `correction`.
- ~~vector_norm~~
- ~~CrossEntropyLoss~~
- group_norm
    ```python
    torch.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None)

    >>> input = torch.randn(20, 6, 10, 10)
    >>> # Separate 6 channels into 3 groups
    >>> m = nn.GroupNorm(3, 6)
    >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
    >>> m = nn.GroupNorm(6, 6)
    >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
    >>> m = nn.GroupNorm(1, 6)
    >>> # Activating the module
    >>> output = m(input)
    ```

    Applies Group Normalization over a mini-batch of inputs.

    $y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta$

    The input channels are separated into `num_groups` groups, each containing `num_channels` / `num_groups` channels. `num_channels` must be divisible by `num_groups`. The mean and standard-deviation are calculated separately over the each group. $\gamma$ and $\beta$ are learnable per-channel affine transform parameter vectors of size `num_channels` if `affine` is `True`. The variance is calculated via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.
- log_softmax
    ```python
    torch.nn.LogSoftmax(dim=None)
    ```

    Applies the $log(Softmax(x))$ function to an n-dimensional input Tensor.

    The LogSoftmax formulation can be simplified as:

    $LogSoftmax(x_i) = log(\frac{exp(x_i)}{\sum_j exp(x_j)})$
- ~~native_group_norm~~
- sigmoid
    ```python
    torch.sigmoid(input, *, out=None) → Tensor
    Alias for:
    torch.special.expit(input, *, out=None) → Tensor

    >>> t = torch.randn(4)
    >>> t
    tensor([ 0.9213,  1.0887, -0.8858, -1.7683])
    >>> torch.special.expit(t)
    tensor([ 0.7153,  0.7481,  0.2920,  0.1458])
    ```

    Computes the expit (also known as the logistic sigmoid function) of the elements of `input`.

    $out_i = \frac{1}{1 + e^{-input_i}}$

## v3.0
- ~~_conv_depthwise2d~~
- ~~_convolution~~
- conv1d
    ```python
    torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    ```

    Applies a 1D convolution over an input signal composed of several input planes.
- conv2d
    ```python
    torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    ```

    Applies a 2D convolution over an input signal composed of several input planes.
- ~~convolution~~
- ~~cudnn_convolution (N/A)~~
- multinomial
    ```python
    torch.multinomial(input, num_samples, replacement=False, *, generator=None, out=None) → LongTensor

    >>> weights = torch.tensor([0, 10, 3, 0], dtype=torch.float) # create a tensor of weights
    >>> torch.multinomial(weights, 2)
    tensor([1, 2])
    >>> torch.multinomial(weights, 5) # ERROR!
    RuntimeError: cannot sample n_sample > prob_dist.size(-1) samples without replacement
    >>> torch.multinomial(weights, 4, replacement=True)
    tensor([ 2,  1,  1,  1])
    ```

    Returns a tensor where each row contains `num_samples` indices sampled from the multinomial (a stricter definition would be multivariate, refer to torch.distributions.multinomial.Multinomial for more details) probability distribution located in the corresponding row of tensor `input`.
- nonzero
    ```python
    torch.nonzero(input, *, out=None, as_tuple=False) → LongTensor or tuple of LongTensors

    >>> torch.nonzero(torch.tensor([1, 1, 1, 0, 1]))
    tensor([[ 0],
            [ 1],
            [ 2],
            [ 4]])
    >>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
    ...                             [0.0, 0.4, 0.0, 0.0],
    ...                             [0.0, 0.0, 1.2, 0.0],
    ...                             [0.0, 0.0, 0.0,-0.4]]))
    tensor([[ 0,  0],
            [ 1,  1],
            [ 2,  2],
            [ 3,  3]])
    >>> torch.nonzero(torch.tensor([1, 1, 1, 0, 1]), as_tuple=True)
    (tensor([0, 1, 2, 4]),)
    >>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
    ...                             [0.0, 0.4, 0.0, 0.0],
    ...                             [0.0, 0.0, 1.2, 0.0],
    ...                             [0.0, 0.0, 0.0,-0.4]]), as_tuple=True)
    (tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3]))
    >>> torch.nonzero(torch.tensor(5), as_tuple=True)
    (tensor([0]),)
    ```

    Returns a 2-D tensor where each row is the index for a nonzero value.
- normal
    ```python
    torch.normal(mean, std, *, generator=None, out=None) → Tensor

    >>> torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
    tensor([  1.0425,   3.5672,   2.7969,   4.2925,   4.7229,   6.2134,
              8.0505,   8.1408,   9.0563,  10.0566])
    ```

    Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given.

    The `mean` is a tensor with the mean of each output element's normal distribution

    The `std` is a tensor with the standard deviation of each output element's normal distribution
- rand
    ```python
    torch.rand(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) → Tensor

    >>> torch.rand(4)
    tensor([ 0.5204,  0.2503,  0.3525,  0.5673])
    >>> torch.rand(2, 3)
    tensor([[ 0.8237,  0.5781,  0.6879],
            [ 0.3816,  0.7249,  0.0998]])
    ```

    Returns a tensor filled with random numbers from a uniform distribution on the interval $[0,1)$, the shape of the tensor is defined by the variable argument `size`.
- rand_like
    ```python
    torch.rand_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) → Tensor
    ```

    Returns a tensor with the same size as `input` that is filled with random numbers from a uniform distribution on the interval $[0,1)$. `torch.rand_like(input)` is equivalent to `torch.rand(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)`.
- randn
    ```python
    torch.randn(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) → Tensor
    ```

    Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).
- topk
    ```python
    torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None)

    >>> x = torch.arange(1., 6.)
    >>> x
    tensor([ 1.,  2.,  3.,  4.,  5.])
    >>> torch.topk(x, 3)
    torch.return_types.topk(values=tensor([5., 4., 3.]), indices=tensor([4, 3, 2]))
    ```

    Returns the `k` largest elements of the given `input` tensor along a given dimension.

    If `dim` is not given, the last dimension of the `input` is chosen.

    A namedtuple of `(values, indices)` is returned with the `values` and `indices` of the largest `k` elements of each row of the `input` tensor in the given dimension `dim`.
- uniform_
    ```python
    torch.Tensor.uniform_(from=0, to=1, *, generator=None) → Tensor
    ```

    Fills `self` tensor with numbers sampled from the continuous uniform distribution:

    $f(x) = \frac{1}{to − from}$
- ~~_efficient_attention_forward~~
- ~~_scaled_dot_product_efficient_attention~~
- embedding
    ```python
    torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None)

    >>> # an Embedding module containing 10 tensors of size 3
    >>> embedding = nn.Embedding(10, 3)
    >>> # a batch of 2 samples of 4 indices each
    >>> input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    >>> embedding(input)
    tensor([[[-0.0251, -1.6902,  0.7172],
            [-0.6431,  0.0748,  0.6969],
            [ 1.4970,  1.3448, -0.9685],
            [-0.3677, -2.7265, -0.1685]],

            [[ 1.4970,  1.3448, -0.9685],
            [ 0.4362, -0.4004,  0.9400],
            [-0.6431,  0.0748,  0.6969],
            [ 0.9124, -2.3616,  1.1151]]])

    >>> # example with padding_idx
    >>> embedding = nn.Embedding(10, 3, padding_idx=0)
    >>> input = torch.LongTensor([[0, 2, 0, 5]])
    >>> embedding(input)
    tensor([[[ 0.0000,  0.0000,  0.0000],
            [ 0.1535, -2.0309,  0.9315],
            [ 0.0000,  0.0000,  0.0000],
            [-0.1655,  0.9897,  0.0635]]])

    >>> # example of changing `pad` vector
    >>> padding_idx = 0
    >>> embedding = nn.Embedding(3, 3, padding_idx=padding_idx)
    >>> embedding.weight
    Parameter containing:
    tensor([[ 0.0000,  0.0000,  0.0000],
            [-0.7895, -0.7089, -0.0364],
            [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
    >>> with torch.no_grad():
    ...     embedding.weight[padding_idx] = torch.ones(3)
    >>> embedding.weight
    Parameter containing:
    tensor([[ 1.0000,  1.0000,  1.0000],
            [-0.7895, -0.7089, -0.0364],
            [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
    ```

    A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices. The input to the module is a list of indices, and the output is the corresponding word embeddings.
- nll_loss
    ```python
    torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    ```

    The negative log likelihood loss. It is useful to train a classification problem with C classes.
- nll_loss_forward
- nll_loss_nd
- scaled_dot_product_attention
    ```python
    torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False) -> Tensor

    >>> # Optionally use the context manager to ensure one of the fused kernels is run
    >>> query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
    >>> key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
    >>> value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
    >>> with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
    >>>     F.scaled_dot_product_attention(query,key,value)
    ```

    Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed, and applying dropout if a probability greater than 0.0 is specified. The optional scale argument can only be specified as a keyword argument.
- upsample_nearest2d
    ```python
    torch.nn.UpsamplingNearest2d(size=None, scale_factor=None)

    >>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
    >>> input
    tensor([[[[1., 2.],
              [3., 4.]]]])

    >>> m = nn.UpsamplingNearest2d(scale_factor=2)
    >>> m(input)
    tensor([[[[1., 1., 2., 2.],
              [1., 1., 2., 2.],
              [3., 3., 4., 4.],
              [3., 3., 4., 4.]]]])
    ```

    Applies a 2D nearest neighbor upsampling to an input signal composed of several input channels.

    To specify the scale, it takes either the `size` or the `scale_factor` as it's constructor argument.

    When `size` is given, it is the output size of the image (h, w).
- ~~_fft_c2r~~
- ~~_fft_r2c~~
- ~~conj_physical~~
- erf
    ```python
    torch.erf(input, *, out=None) → Tensor
    Alias for:
    torch.special.erf(input, *, out=None) → Tensor

    >>> torch.special.erf(torch.tensor([0, -1., 10.]))
    tensor([ 0.0000, -0.8427,  1.0000])
    ```

    Computes the error function of `input`. The error function is defined as follows:

    $erf(x) = \frac{2}{\sqrt{\pi}}\int_0^x e^{-t^2}dt$
- ~~fft_irfft~~
- ~~fft_rfft~~
- resolve_conj
    ```python
    torch.resolve_conj(input) → Tensor

    >>> x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
    >>> y = x.conj()
    >>> y.is_conj()
    True
    >>> z = y.resolve_conj()
    >>> z
    tensor([-1 - 1j, -2 - 2j, 3 + 3j])
    >>> z.is_conj()
    False
    ```

    Returns a new tensor with materialized conjugation if `input`'s conjugate bit is set to `True`, else returns `input`. The output tensor will always have its conjugate bit set to `False`.
- resolve_neg
    ```python
    torch.resolve_neg(input) → Tensor

    >>> x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
    >>> y = x.conj()
    >>> z = y.imag
    >>> z.is_neg()
    True
    >>> out = z.resolve_neg()
    >>> out
    tensor([-1., -2., 3.])
    >>> out.is_neg()
    False
    ```

    Returns a new tensor with materialized negation if `input`'s negative bit is set to `True`, else returns `input`. The output tensor will always have its negative bit set to `False`.
- arange
    ```python
    torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

    >>> torch.arange(5)
    tensor([ 0,  1,  2,  3,  4])
    >>> torch.arange(1, 4)
    tensor([ 1,  2,  3])
    >>> torch.arange(1, 2.5, 0.5)
    tensor([ 1.0000,  1.5000,  2.0000])
    ```

    Returns a 1-D tensor of size $\lceil\frac{end-start}{step}\rceil$ with values from the interval `[start, end)` taken with common difference `step` beginning from `start`.
- cat
    ```python
    torch.cat(tensors, dim=0, *, out=None) → Tensor


    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 0)
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 1)
    tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
            -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
            -0.5790,  0.1497]])
    ```

    Concatenates the given sequence of tensors in `tensors` in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be a 1-D empty tensor with size `(0,)`.
- ~~chunk (N/A)~~
- ~~concat~~
- constant_pad_nd
    ```python
    torch.nn.functional.pad(input, pad, mode='constant', value=None) → Tensor

    >>> t4d = torch.empty(3, 3, 4, 2)
    >>> p1d = (1, 1) # pad last dim by 1 on each side
    >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
    >>> print(out.size())
    torch.Size([3, 3, 4, 4])
    >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
    >>> out = F.pad(t4d, p2d, "constant", 0)
    >>> print(out.size())
    torch.Size([3, 3, 8, 4])
    >>> t4d = torch.empty(3, 3, 4, 2)
    >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
    >>> out = F.pad(t4d, p3d, "constant", 0)
    >>> print(out.size())
    torch.Size([3, 9, 7, 3])
    ```

    Pads tensor.
- ~~contiguous (N/A)~~
- ~~copy_ (N/A)~~
- fill
    ```python
    torch.Tensor.fill_(value) → Tensor
    ```

    Fills `self` tensor with the specified value.
- flip
    ```python
    torch.flip(input, dims) → Tensor

    >>> x = torch.arange(8).view(2, 2, 2)
    >>> x
    tensor([[[ 0,  1],
             [ 2,  3]],

            [[ 4,  5],
             [ 6,  7]]])
    >>> torch.flip(x, [0, 1])
    tensor([[[ 6,  7],
             [ 4,  5]],

            [[ 2,  3],
             [ 0,  1]]])
    ```

    Reverse the order of an n-D tensor along given axis in dims.
- full
    ```python
    torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

    >>> torch.full((2, 3), 3.141592)
    tensor([[ 3.1416,  3.1416,  3.1416],
            [ 3.1416,  3.1416,  3.1416]])
    ```

    Creates a tensor of size `size` filled with `fill_value`. The tensor's dtype is inferred from `fill_value`.
- full_like
    ```python
    torch.full_like(input, fill_value, \*, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format) → Tensor
    ```

    Returns a tensor with the same size as `input` filled with `fill_value`. `torch.full_like(input, fill_value)` is equivalent to `torch.full(input.size(), fill_value, dtype=input.dtype, layout=input.layout, device=input.device)`.
- gather
    ```python
    torch.gather(input, dim, index, *, sparse_grad=False, out=None) → Tensor

    >>> t = torch.tensor([[1, 2], [3, 4]])
    >>> torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
    tensor([[ 1,  1],
            [ 4,  3]])
    ```

    Gathers values along an axis specified by `dim`.

    For a 3-D tensor the output is specified by:

    ```python
    out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
    ```

    `input` and `index` must have the same number of dimensions. It is also required that `index.size(d) <= input.size(d)` for all dimensions `d != dim`. `out` will have the same shape as `index`. Note that `input` and `index` do not broadcast against each other.
- hstack
    ```python
    torch.hstack(tensors, *, out=None) → Tensor

    >>> a = torch.tensor([1, 2, 3])
    >>> b = torch.tensor([4, 5, 6])
    >>> torch.hstack((a,b))
    tensor([1, 2, 3, 4, 5, 6])
    >>> a = torch.tensor([[1],[2],[3]])
    >>> b = torch.tensor([[4],[5],[6]])
    >>> torch.hstack((a,b))
    tensor([[1, 4],
            [2, 5],
            [3, 6]])
    ```

    Stack tensors in sequence horizontally (column wise).

    This is equivalent to concatenation along the first axis for 1-D tensors, and along the second axis for all other tensors.
- ~~index_put_~~
- index_select
    ```python
    torch.index_select(input, dim, index, *, out=None) → Tensor

    >>> x = torch.randn(3, 4)
    >>> x
    tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
            [-0.4664,  0.2647, -0.1228, -1.1068],
            [-1.1734, -0.6571,  0.7230, -0.6004]])
    >>> indices = torch.tensor([0, 2])
    >>> torch.index_select(x, 0, indices)
    tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
            [-1.1734, -0.6571,  0.7230, -0.6004]])
    >>> torch.index_select(x, 1, indices)
    tensor([[ 0.1427, -0.5414],
            [-0.4664, -0.1228],
            [-1.1734,  0.7230]])
    ```

    Returns a new tensor which indexes the `input` tensor along dimension `dim` using the entries in `index` which is a LongTensor.

    The returned tensor has the same number of dimensions as the original tensor (`input`). The `dim`th dimension has the same size as the length of `index`; other dimensions have the same size as in the original tensor.
- masked_fill
    ```python
    torch.Tensor.masked_fill_(mask, value) → Tensor
    ```

    Fills elements of `self` tensor with `value` where `mask` is True. The shape of `mask` must be broadcastable with the shape of the underlying tensor.
- ~~narrow~~
- ones
    ```python
    torch.ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

    >>> torch.ones(2, 3)
    tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])

    >>> torch.ones(5)
    tensor([ 1.,  1.,  1.,  1.,  1.])
    ```

    Returns a tensor filled with the scalar value `1`, with the shape defined by the variable argument `size`.
- pad
    ```python
    torch.nn.functional.pad(input, pad, mode='constant', value=None) → Tensor

    >>> t4d = torch.empty(3, 3, 4, 2)
    >>> p1d = (1, 1) # pad last dim by 1 on each side
    >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
    >>> print(out.size())
    torch.Size([3, 3, 4, 4])
    >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
    >>> out = F.pad(t4d, p2d, "constant", 0)
    >>> print(out.size())
    torch.Size([3, 3, 8, 4])
    >>> t4d = torch.empty(3, 3, 4, 2)
    >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
    >>> out = F.pad(t4d, p3d, "constant", 0)
    >>> print(out.size())
    torch.Size([3, 9, 7, 3])
    ```

    Pads tensor.
- ~~permute (N/A)~~
- repeat
    ```python
    torch.Tensor.repeat(*repeats) → Tensor

    >>> x = torch.tensor([1, 2, 3])
    >>> x.repeat(4, 2)
    tensor([[ 1,  2,  3,  1,  2,  3],
            [ 1,  2,  3,  1,  2,  3],
            [ 1,  2,  3,  1,  2,  3],
            [ 1,  2,  3,  1,  2,  3]])
    >>> x.repeat(4, 2, 1).size()
    torch.Size([4, 2, 3])
    ```

    Repeats this tensor along the specified dimensions.
- repeat_interleave
    ```python
    torch.repeat_interleave(input, repeats, dim=None, *, output_size=None) → Tensor

    >>> x = torch.tensor([1, 2, 3])
    >>> x.repeat_interleave(2)
    tensor([1, 1, 2, 2, 3, 3])
    >>> y = torch.tensor([[1, 2], [3, 4]])
    >>> torch.repeat_interleave(y, 2)
    tensor([1, 1, 2, 2, 3, 3, 4, 4])
    >>> torch.repeat_interleave(y, 3, dim=1)
    tensor([[1, 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4]])
    >>> torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0)
    tensor([[1, 2],
            [3, 4],
            [3, 4]])
    >>> torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0, output_size=3)
    tensor([[1, 2],
            [3, 4],
            [3, 4]])
    ```

    Repeat elements of a tensor.

    ```python
    torch.repeat_interleave(repeats, *) → Tensor

    >>> torch.repeat_interleave(torch.tensor([1, 2, 3]))
    tensor([0, 1, 1, 2, 2, 2])
    ```

    Repeats 0 `repeats[0]` times, 1 `repeats[1]` times, 2 `repeats[2]` times, etc.
- ~~resize_ (N/A)~~
- scatter
    ```python
    torch.Tensor.scatter_(dim, index, src, *, reduce=None) → Tensor

    >>> src = torch.arange(1, 11).reshape((2, 5))
    >>> src
    tensor([[ 1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10]])
    >>> index = torch.tensor([[0, 1, 2, 0]])
    >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
    tensor([[1, 0, 0, 4, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0]])
    >>> index = torch.tensor([[0, 1, 2], [0, 1, 4]])
    >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
    tensor([[1, 2, 3, 0, 0],
            [6, 7, 0, 0, 8],
            [0, 0, 0, 0, 0]])

    >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
    ...            1.23, reduce='multiply')
    tensor([[2.0000, 2.0000, 2.4600, 2.0000],
            [2.0000, 2.0000, 2.0000, 2.4600]])
    >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
    ...            1.23, reduce='add')
    tensor([[2.0000, 2.0000, 3.2300, 2.0000],
            [2.0000, 2.0000, 2.0000, 3.2300]])
    ```

    Writes all values from the tensor `src` into `self` at the indices specified in the `index` tensor. For each value in `src`, its output index is specified by its index in `src` for `dimension != dim` and by the corresponding value in `index` for `dimension = dim`.

    For a 3-D tensor, `self` is updated as:

    ```python
    self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
    ```

    This is the reverse operation of the manner described in `gather()`.
- index_add
    ```python
    torch.Tensor.index_add_(dim, index, source, *, alpha=1) → Tensor

    >>> x = torch.ones(5, 3)
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    >>> index = torch.tensor([0, 4, 2])
    >>> x.index_add_(0, index, t)
    tensor([[  2.,   3.,   4.],
            [  1.,   1.,   1.],
            [  8.,   9.,  10.],
            [  1.,   1.,   1.],
            [  5.,   6.,   7.]])
    >>> x.index_add_(0, index, t, alpha=-1)
    tensor([[  1.,   1.,   1.],
            [  1.,   1.,   1.],
            [  1.,   1.,   1.],
            [  1.,   1.,   1.],
            [  1.,   1.,   1.]])
    ```

    Accumulate the elements of `alpha` times `source` into the `self` tensor by adding to the indices in the order given in `index`. For example, if `dim == 0`, `index[i] == j`, and `alpha=-1`, then the `i`th row of `source` is subtracted from the `j`th row of `self`.

    The `dim`th dimension of `source` must have the same size as the length of `index` (which must be a vector), and all other dimensions must match `self`, or an error will be raised.

    For a 3-D tensor the `output` is given as:

    ```python
    self[index[i], :, :] += alpha * src[i, :, :]  # if dim == 0
    self[:, index[i], :] += alpha * src[:, i, :]  # if dim == 1
    self[:, :, index[i]] += alpha * src[:, :, i]  # if dim == 2
    ```
- ~~select (N/A)~~
- select_scatter
    ```python
    torch.select_scatter(input, src, dim, index) → Tensor

    >>> a = torch.zeros(2, 2)
    >>> b = torch.ones(2)
    >>> a.select_scatter(b, 0, 0)
    tensor([[1., 1.],
            [0., 0.]])
    ```

    Embeds the values of the `src` tensor into `input` at the given index. This function returns a tensor with fresh storage; it does not create a view.
- ~~slice~~
- slice_scatter
    ```python
    torch.slice_scatter(input, src, dim=0, start=None, end=None, step=1) → Tensor

    >>> a = torch.zeros(8, 8)
    >>> b = torch.ones(2, 8)
    >>> a.slice_scatter(b, start=6)
    tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1.]])

    >>> b = torch.ones(8, 2)
    >>> a.slice_scatter(b, dim=1, start=2, end=6, step=2)
    tensor([[0., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 1., 0., 0., 0.]])
    ```

    Embeds the values of the `src` tensor into `input` at the given dimension. This function returns a tensor with fresh storage; it does not create a view.
- sort
    ```python
    torch.sort(input, dim=-1, descending=False, stable=False, *, out=None)

    >>> x = torch.randn(3, 4)
    >>> sorted, indices = torch.sort(x)
    >>> sorted
    tensor([[-0.2162,  0.0608,  0.6719,  2.3332],
            [-0.5793,  0.0061,  0.6058,  0.9497],
            [-0.5071,  0.3343,  0.9553,  1.0960]])
    >>> indices
    tensor([[ 1,  0,  2,  3],
            [ 3,  1,  0,  2],
            [ 0,  3,  1,  2]])

    >>> sorted, indices = torch.sort(x, 0)
    >>> sorted
    tensor([[-0.5071, -0.2162,  0.6719, -0.5793],
            [ 0.0608,  0.0061,  0.9497,  0.3343],
            [ 0.6058,  0.9553,  1.0960,  2.3332]])
    >>> indices
    tensor([[ 2,  0,  0,  1],
            [ 0,  1,  1,  2],
            [ 1,  2,  2,  0]])
    >>> x = torch.tensor([0, 1] * 9)
    >>> x.sort()
    torch.return_types.sort(
        values=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        indices=tensor([ 2, 16,  4,  6, 14,  8,  0, 10, 12,  9, 17, 15, 13, 11,  7,  5,  3,  1]))
    >>> x.sort(stable=True)
    torch.return_types.sort(
        values=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        indices=tensor([ 0,  2,  4,  6,  8, 10, 12, 14, 16,  1,  3,  5,  7,  9, 11, 13, 15, 17]))
    ```

    Sorts the elements of the `input` tensor along a given dimension in ascending order by value.

    If `dim` is not given, the last dimension of the input is chosen.

    If `descending` is `True` then the elements are sorted in descending order by value.

    If `stable` is `True` then the sorting routine becomes stable, preserving the order of equivalent elements.

    A namedtuple of `(values, indices)` is returned, where the `values` are the sorted values and `indices` are the indices of the elements in the original `input` tensor.
- ~~split (N/A)~~
- ~~split_with_sizes (N/A)~~
- stack
    ```python
    torch.stack(tensors, dim=0, *, out=None) → Tensor

    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 0.3367,  0.1288,  0.2345],
            [ 0.2303, -1.1229, -0.1863]])
    >>> torch.stack((x, x)) # same as torch.stack((x, x), dim=0)
    tensor([[[ 0.3367,  0.1288,  0.2345],
             [ 0.2303, -1.1229, -0.1863]],

            [[ 0.3367,  0.1288,  0.2345],
             [ 0.2303, -1.1229, -0.1863]]])
    >>> torch.stack((x, x)).size()
    torch.Size([2, 2, 3])
    >>> torch.stack((x, x), dim=1)
    tensor([[[ 0.3367,  0.1288,  0.2345],
             [ 0.3367,  0.1288,  0.2345]],

            [[ 0.2303, -1.1229, -0.1863],
             [ 0.2303, -1.1229, -0.1863]]])
    >>> torch.stack((x, x), dim=2)
    tensor([[[ 0.3367,  0.3367],
             [ 0.1288,  0.1288],
             [ 0.2345,  0.2345]],

            [[ 0.2303,  0.2303],
             [-1.1229, -1.1229],
             [-0.1863, -0.1863]]])
    >>> torch.stack((x, x), dim=-1)
    tensor([[[ 0.3367,  0.3367],
             [ 0.1288,  0.1288],
             [ 0.2345,  0.2345]],

            [[ 0.2303,  0.2303],
             [-1.1229, -1.1229],
             [-0.1863, -0.1863]]])
    ```

    Concatenates a sequence of tensors along a new dimension.

    All tensors need to be of the same size.
- tile
    ```python
    torch.tile(input, dims) → Tensor

    >>> x = torch.tensor([1, 2, 3])
    >>> x.tile((2,))
    tensor([1, 2, 3, 1, 2, 3])
    >>> y = torch.tensor([[1, 2], [3, 4]])
    >>> torch.tile(y, (2, 2))
    tensor([[1, 2, 1, 2],
            [3, 4, 3, 4],
            [1, 2, 1, 2],
            [3, 4, 3, 4]])
    ```

    Constructs a tensor by repeating the elements of `input`. The `dims` argument specifies the number of repetitions in each dimension.

    If `dims` specifies fewer dimensions than `input` has, then ones are prepended to `dims` until all dimensions are specified. For example, if `input` has shape `(8, 6, 4, 2)` and `dims` is `(2, 2)`, then `dims` is treated as `(1, 1, 2, 2)`.

    Analogously, if `input` has fewer dimensions than `dims` specifies, then `input` is treated as if it were unsqueezed at dimension zero until it has as many dimensions as `dims` specifies. For example, if `input` has shape `(4, 2)` and dims is `(3, 3, 2, 2)`, then `input` is treated as if it had the shape `(1, 1, 4, 2)`.
- ~~transpose (N/A)~~
- ~~unfold (N/A)~~
- where
    ```python
    torch.where(condition, input, other, *, out=None) → Tensor

    >>> x = torch.randn(3, 2)
    >>> y = torch.ones(3, 2)
    >>> x
    tensor([[-0.4620,  0.3139],
            [ 0.3898, -0.7197],
            [ 0.0478, -0.1657]])
    >>> torch.where(x > 0, 1.0, 0.0)
    tensor([[0., 1.],
            [1., 0.],
            [1., 0.]])
    >>> torch.where(x > 0, x, y)
    tensor([[ 1.0000,  0.3139],
            [ 0.3898,  1.0000],
            [ 0.0478,  1.0000]])
    >>> x = torch.randn(2, 2, dtype=torch.double)
    >>> x
    tensor([[ 1.0779,  0.0383],
            [-0.8785, -1.1089]], dtype=torch.float64)
    >>> torch.where(x > 0, x, 0.)
    tensor([[1.0779, 0.0383],
            [0.0000, 0.0000]], dtype=torch.float64)
    ```

    Return a tensor of elements selected from either `input` or `other`, depending on `condition`.

    The operation is defined as:

    $$
    out_i = \left\{
        \begin{array}{rcl}
            input_i & {if\ condition_i} \\
            other_i & {otherwise}
        \end{array}
    \right.
    $$
- zeros
    ```python
    torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

    >>> torch.zeros(2, 3)
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])

    >>> torch.zeros(5)
    tensor([ 0.,  0.,  0.,  0.,  0.])
    ```

    Returns a tensor filled with the scalar value `0`, with the shape defined by the variable argument `size`.

## v4.0
- rms_norm
    ```python
    torch.nn.modules.normalization.RMSNorm(normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None)
    ```

    Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

    This layer implements the operation

    $y_i = \frac{x_i}{RMS(x)} * \gamma_i$, where $RMS(x) = \sqrt{\epsilon + \frac{1}{n}\sum_{i=1}^nx_i^2}$

    The RMS is taken over the last `D` dimensions, where `D` is the dimension of `normalized_shape`. For example, if `normalized_shape` is `(3, 5)` (a 2-dimensional shape), the RMS is computed over the last 2 dimensions of the input.
- ~~skip_rms_norm~~
- ~~skip_layer_norm~~
- ~~gelu_and_mul~~
- ~~silu_and_mul~~
- ~~apply_rotary_pos_emb~~

## v5.0
- ~~unique~~
- isin
    ```python
    torch.isin(elements, test_elements, *, assume_unique=False, invert=False) → Tensor

    >>> torch.isin(torch.tensor([[1, 2], [3, 4]]), torch.tensor([2, 3]))
    tensor([[False,  True],
            [ True, False]])
    ```

    Tests if each element of `elements` is in `test_elements`. Returns a boolean tensor of the same shape as `elements` that is True for elements in `test_elements` and False otherwise.
- allclose
    ```python
    torch.allclose(input: Tensor, other: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) → bool

    >>> torch.allclose(torch.tensor([10000., 1e-07]), torch.tensor([10000.1, 1e-08]))
    False
    >>> torch.allclose(torch.tensor([10000., 1e-08]), torch.tensor([10000.1, 1e-09]))
    True
    >>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]))
    False
    >>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]), equal_nan=True)
    True
    ```

    This function checks if `input` and `other` satisfy the condition:

    $|input_i - other_i| \le atol + rtol \times |other_i|$
- isclose
    ```python
    torch.isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) → Tensor

    >>> torch.isclose(torch.tensor((1., 2, 3)), torch.tensor((1 + 1e-10, 3, 4)))
    tensor([ True, False, False])
    >>> torch.isclose(torch.tensor((float('inf'), 4)), torch.tensor((float('inf'), 6)), rtol=.5)
    tensor([True, True])
    ```

    Returns a new tensor with boolean elements representing if each element of `input` is "close" to the corresponding element of `other`. Closeness is defined as:

    $|input_i - other_i| \le rtol \times |other_i| + atol$

    where `input` and `other` are finite. Where `input` and/or `other` are nonfinite they are close if and only if they are equal, with NaNs being considered equal to each other when `equal_nan` is True.
- isfinite
    ```python
    torch.isfinite(input) → Tensor

    >>> torch.isfinite(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
    tensor([True,  False,  True,  False,  False])
    ```

    Returns a new tensor with boolean elements representing if each element is `finite` or not.

    Real values are finite when they are not NaN, negative infinity, or infinity. Complex values are finite when both their real and imaginary parts are finite.
- exponential_
    ```python
    torch.Tensor.exponential_(lambd=1, *, generator=None) → Tensor
    ```

    Fills `self` tensor with elements drawn from the PDF (probability density function):

    $f(x) = \lambda e^{-\lambda x}, x > 0$
- vstack
    ```python
    torch.vstack(tensors, *, out=None) → Tensor

    >>> a = torch.tensor([1, 2, 3])
    >>> b = torch.tensor([4, 5, 6])
    >>> torch.vstack((a,b))
    tensor([[1, 2, 3],
            [4, 5, 6]])
    >>> a = torch.tensor([[1],[2],[3]])
    >>> b = torch.tensor([[4],[5],[6]])
    >>> torch.vstack((a,b))
    tensor([[1],
            [2],
            [3],
            [4],
            [5],
            [6]])
    ```

    Stack tensors in sequence vertically (row wise).

    This is equivalent to concatenation along the first axis after all 1-D tensors have been reshaped by `torch.atleast_2d()`.
- weight_norm_interface
    ```python
    torch.nn.utils.weight_norm(module, name='weight', dim=0)

    >>> m = weight_norm(nn.Linear(20, 40), name='weight')
    >>> m
    Linear(in_features=20, out_features=40, bias=True)
    >>> m.weight_g.size()
    torch.Size([40, 1])
    >>> m.weight_v.size()
    torch.Size([40, 20])
    ```

    Apply weight normalization to a parameter in the given module.

    $w=g\frac{v}{||v||}$

    Weight normalization is a reparameterization that decouples the magnitude of a weight tensor from its direction. This replaces the parameter specified by name (e.g. 'weight') with two parameters: one specifying the magnitude (e.g. 'weight_g') and one specifying the direction (e.g. 'weight_v'). Weight normalization is implemented via a hook that recomputes the weight tensor from the magnitude and direction before every `forward()` call.

    By default, with `dim=0`, the norm is computed independently per output channel/plane. To compute a norm over the entire weight tensor, use `dim=None`.
- minimum
    ```python
    torch.minimum(input, other, *, out=None) → Tensor

    >>> a = torch.tensor((1, 2, -1))
    >>> b = torch.tensor((3, 0, 4))
    >>> torch.minimum(a, b)
    tensor([1, 0, -1])
    ```

    Computes the element-wise minimum of `input` and `other`.
- maximum
    ```python
    torch.maximum(input, other, *, out=None) → Tensor

    >>> a = torch.tensor((1, 2, -1))
    >>> b = torch.tensor((3, 0, 4))
    >>> torch.maximum(a, b)
    tensor([3, 2, 4])
    ```

    Computes the element-wise maximum of `input` and `other`.
- ones_like
    ```python
    torch.ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) → Tensor

    >>> input = torch.empty(2, 3)
    >>> torch.ones_like(input)
    tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])
    ```

    Returns a tensor filled with the scalar value `1`, with the same size as `input`. `torch.ones_like(input)` is equivalent to `torch.ones(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)`.
- zeros_like
    ```python
    torch.zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) → Tensor

    >>> input = torch.empty(2, 3)
    >>> torch.zeros_like(input)
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])
    ```

    Returns a tensor filled with the scalar value `0`, with the same size as `input`. `torch.zeros_like(input)` is equivalent to `torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)`.
- randn_like
    ```python
    torch.randn_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) → Tensor
    ```

    Returns a tensor with the same size as `input` that is filled with random numbers from a normal distribution with mean `0` and variance `1`. Please refer to `torch.randn()` for the sampling process of complex dtypes. `torch.randn_like(input)` is equivalent to `torch.randn(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)`.
- floor_divide
    ```python
    torch.floor_divide(input, other, *, out=None) → Tensor
    ```

    Computes `input` divided by `other`, elementwise, and floors the result.

    $out_i = floor(\frac{input_i}{other_i})$
- masked_select
    ```python
    torch.masked_select(input, mask, *, out=None) → Tensor

    >>> x = torch.randn(3, 4)
    >>> x
    tensor([[ 0.3552, -2.3825, -0.8297,  0.3477],
            [-1.2035,  1.2252,  0.5002,  0.6248],
            [ 0.1307, -2.0608,  0.1244,  2.0139]])
    >>> mask = x.ge(0.5)
    >>> mask
    tensor([[False, False, False, False],
            [False, True, True, True],
            [False, False, False, True]])
    >>> torch.masked_select(x, mask)
    tensor([ 1.2252,  0.5002,  0.6248,  2.0139])
    ```

    Returns a new 1-D tensor which indexes the `input` tensor according to the boolean mask `mask` which is a BoolTensor.

    The shapes of the `mask` tensor and the `input` tensor don’t need to match, but they must be broadcastable.
- ~~trunc_divide~~
- remainder
    ```python
    torch.remainder(input, other, *, out=None) → Tensor

    >>> torch.remainder(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
    tensor([ 1.,  0.,  1.,  1.,  0.,  1.])
    >>> torch.remainder(torch.tensor([1, 2, 3, 4, 5]), -1.5)
    tensor([ -0.5000, -1.0000,  0.0000, -0.5000, -1.0000 ])
    ```

    Computes Python's modulus operation entrywise. The result has the same sign as the divisor `other` and its absolute value is less than that of `other`.

    It may also be defined in terms of `torch.div()` as

    ```python
    torch.remainder(a, b) == a - a.div(b, rounding_mode="floor") * b
    ```

    Supports broadcasting to a common shape, type promotion, and integer and float inputs.
