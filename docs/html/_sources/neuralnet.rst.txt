Neural Networks
===============

Fastinference offers a limited support for Deep Learning and Neural Network architectures. The current focus is on feed-forward MLPs and ConvNets in the context of small, embedded systems and FPGAs, but we are always open to enhance our support for new Deep Learning architectures. 

..
    If you are interested in deploying the latest Deep Learning models for production then this project is probably not mature enough for you. For deep learning there are a ton of frameworks available such as [glow](https://github.com/pytorch/glow), [tensorflow-lite](https://www.tensorflow.org/), [ONNX Runtime](https://github.com/microsoft/onnxruntime), [NGraph](https://github.com/NervanaSystems/ngraph), [MACE](https://github.com/XiaoMi/mace), [NCNN](https://github.com/Tencent/ncnn), [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt), [OpenVINO Toolkit](https://github.com/openvinotoolkit/openvino) and probably more. 

**Important:** `ONNX <https://onnx.ai/>`_ is the open standard for machine learning interoperability and supported by all major Deep Learning frameworks. However, the ONNX format is still under development and a given deep architecture can often be represented with various computational graphs. Hence, this standard is sometimes ambiguous. This implementation has been tested with `PyTorch <https://pytorch.org/>`_ and visualized with `Netron <https://netron.app/>`_. For exporting a Neural Net we usually use

    .. code-block:: python

        dummy_x = torch.randn(1, x_train.shape[1], requires_grad=False)
        torch.onnx.export(model, dummy_x, os.path.join(out_path,name), training=torch.onnx.TrainingMode.PRESERVE, export_params=True,opset_version=11, do_constant_folding=True, input_names = ['input'],  output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})

Some notes on Binarized Neural Networks
---------------------------------------

Binarized Neural Networks (BNNs) are Neural Networks with weights constraint to {-1,+1} so that the forward pass of the entire network can be executed via boolean operations (usually XNOR + popcount). A typical structure of these networks are as follows:

    Input -> Linear / Conv -> BatchNorm -> Step -> ... -> Linear / Conv -> BatchNorm -> Step -> Output

where the Linear / Conv layers only have "binary" weights and biases {-1,+1} and the step function is Heaviside function. BNNs are usually not supported by the major frameworks out of the box, but require some additional libraries as well as some tweaks in the ONNX format. For example, `larq <https://larq.dev/>`_ offers binarization for keras / tensorflow and `Brevitas <https://github.com/Xilinx/brevitas>`_ enables binarization for PyTorch. Alternatively, we can directly implement binarization as shown in the example below. Unfortunately, ONNX does not support the custom operators from these libraries so that we have to sanitize these before exporting. In fastinference we simply replace each binary layer, e.g. :code:`BinaryLinear`, with its regular counterpart :code:`torch.nn.Linear`. Moreover, PyTorch cannot export the Heaviside function yet into an ONNX file. Hence we mimic this function with a series of "Constant -> Greater -> Constant -> Constant -> Where" layers which is then parsed and merged back into a Step layer by fastinference. For a complete example check out `train_mlp.py <https://github.com/sbuschjaeger/fastinference/blob/main/tests/train_mlp.py>`_ or `train_cnn.py <https://github.com/sbuschjaeger/fastinference/blob/main/tests/train_cnn.py>`_.

    .. code-block:: python

        class BinarizeF(Function):
            @staticmethod
            def forward(ctx, input):
                output = input.new(input.size())
                output[input > 0] = 1
                output[input <= 0] = -1
                return output

            @staticmethod
            def backward(ctx, grad_output):
                #return grad_output, None
                grad_input = grad_output.clone()
                return grad_input#, None

        binarize = BinarizeF.apply

        class BinaryLinear(nn.Linear):
            def __init__(self, *args, **kwargs):
                super(BinaryLinear, self).__init__(*args, **kwargs)

            def forward(self, input):
                if self.bias is None:
                    binary_weight = binarize(self.weight)

                    return F.linear(input, binary_weight)
                else:
                    binary_weight = binarize(self.weight)
                    binary_bias = binarize(self.bias)
                    return F.linear(input, binary_weight, binary_bias)

        class BinaryTanh(nn.Module):
            def __init__(self, *args, **kwargs):
                super(BinaryTanh, self).__init__()
                self.hardtanh = nn.Hardtanh(*args, **kwargs)

            def forward(self, input):
                output = self.hardtanh(input)
                output = binarize(output)
                return output


        class SimpleMLP(nn.Module):

            def __init__(self, input_dim, n_classes):
                super().__init__()
                self.layer_1 = BinaryLinear(input_dim, 128)
                self.bn_1 = nn.BatchNorm1d(128)
                self.activation_1 = BinaryTanh()
                self.layer_2 = BinaryLinear(128, 256)
                self.bn_2 = nn.BatchNorm1d(256)
                self.activation_2 = BinaryTanh()
                self.layer_3 = BinaryLinear(256, n_classes)

            def forward(self, x):
                x = self.layer_1(x)
                x = self.bn_1(x)
                x = self.activation_1(x)
                x = self.layer_2(x)
                x = self.bn_2(x)
                x = self.activation_2(x)
                x = self.layer_3(x)
                x = torch.log_softmax(x, dim=1)

                return x

        def sanatize_onnx(model):

            # Usually I would use https://pytorch.org/docs/stable/generated/torch.heaviside.html for exporting here, but this is not yet supported in ONNX files. 
            class Sign(nn.Module):
                def forward(self, input):
                    return torch.where(input > 0, torch.tensor([1.0]), torch.tensor([-1.0]))

            for name, m in reversed(model._modules.items()):
                print("Checking {}".format(name))

                if isinstance(m, BinaryLinear):
                    print("Replacing {}".format(name))
                    # layer_old = m
                    layer_new = nn.Linear(m.in_features, m.out_features, hasattr(m, 'bias'))
                    if (hasattr(m, 'bias')):
                        layer_new.bias.data = binarize(m.bias.data)
                    layer_new.weight.data = binarize(m.weight.data)
                    model._modules[name] = layer_new

                if isinstance(m, BinaryTanh):
                    model._modules[name] = Sign()

            return model

        model = SimpleMLP(input_dim, n_classes)
        # Train the model 
        model = sanatize_onnx(model)
        torch.onnx.export(model,dummy_x,os.path.join(out_path,name), export_params=True,opset_version=11, do_constant_folding=True, input_names = ['input'],  output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})

Available optimizations
-----------------------

.. autofunction:: fastinference.optimizers.neuralnet.merge_nodes.optimize
.. autofunction:: fastinference.optimizers.neuralnet.remove_nodes.optimize


The NeuralNet object
--------------------

.. automodule:: fastinference.models.nn.NeuralNet
   :special-members: __init__
   :members:
   :undoc-members:
   