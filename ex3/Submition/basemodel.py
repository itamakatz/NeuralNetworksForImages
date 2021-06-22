import enum
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable

class BaseModel(nn.Module):

  def __init__(self, child_name, model_name):
    super(BaseModel, self).__init__()
    self.model_name = MType.parse(model_name)
    self.child_name = child_name
    self.set_mappings()

  def set_model(self):
    '''sets the model depending on the running model type'''
    print(f"Running model: {self.child_name} - {self.model_name}")
    # if the mapping exist, run it. else throw an exception
    if(self.model_name in self._set_model_mapping):
      return self._set_model_mapping[self.model_name]()
    else:
      raise Exception(f"No such model named: {self.model_name}")

  def forward(self, x):
    '''forwards depending on the running model type'''
    # if the mapping exist, run it. else throw an exception
    if(self.model_name in self._forward_mapping):
      return self._forward_mapping[self.model_name](x)
    else:
      raise Exception(f"No such model named: {self.model_name}")

  def set_mappings(self):
    '''defines switching of the models to the correct functions'''
    self._set_model_mapping = {
      MType.Model_1: self.set_model_1,
      MType.Model_2: self.set_model_2,
      MType.Model_3: self.set_model_3,
      MType.Model_4: self.set_model_4,
      MType.Model_5: self.set_model_5,
      MType.Model_6: self.set_model_6,
      MType.Model_7: self.set_model_7,
      MType.Model_8: self.set_model_8,
      MType.Model_9: self.set_model_9,
      MType.Model_10: self.set_model_10,
      MType.Model_11: self.set_model_11,
      MType.Model_12: self.set_model_12,
      MType.Model_13: self.set_model_13,
      MType.Model_14: self.set_model_14,
      MType.Model_15: self.set_model_15,
      MType.Model_16: self.set_model_16,
      MType.Model_17: self.set_model_17,
      MType.Model_18: self.set_model_18,
      MType.Model_19: self.set_model_19,
      MType.Model_20: self.set_model_20,
      MType.Model_21: self.set_model_21,
    }
    self._forward_mapping = {
      MType.Model_1: self.forward_1,
      MType.Model_2: self.forward_2,
      MType.Model_3: self.forward_3,
      MType.Model_4: self.forward_4,
      MType.Model_5: self.forward_5,
      MType.Model_6: self.forward_6,
      MType.Model_7: self.forward_7,
      MType.Model_8: self.forward_8,
      MType.Model_9: self.forward_9,
      MType.Model_10: self.forward_10,
      MType.Model_11: self.forward_11,
      MType.Model_12: self.forward_12,
      MType.Model_13: self.forward_13,
      MType.Model_14: self.forward_14,
      MType.Model_15: self.forward_15,
      MType.Model_16: self.forward_16,
      MType.Model_17: self.forward_17,
      MType.Model_18: self.forward_18,
      MType.Model_19: self.forward_19,
      MType.Model_20: self.forward_20,
      MType.Model_21: self.forward_21,
    }

  def set_model_1(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_2(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_3(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_4(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_5(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_6(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_7(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_8(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_9(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_10(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_11(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_12(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_13(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_14(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_15(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_16(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_17(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_18(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_19(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_20(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def set_model_21(self): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")

  def forward_1(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_2(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_3(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_4(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_5(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_6(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_7(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_8(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_9(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_10(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_11(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_12(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_13(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_14(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_15(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_16(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_17(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_18(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_19(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_20(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")
  def forward_21(self, x): raise Exception(f"Function not implemented in inheriting instance named: {self.model_name}")

  def load(self, backup):
    for m_from, m_to in zip(backup.modules(), self.modules()):
       if isinstance(m_to, nn.Linear):
        m_to.weight.data = m_from.weight.data.clone()
        if m_to.bias is not None:
          m_to.bias.data = m_from.bias.data.clone()

  def summary(self, input_size, to_cuda):
    def register_hook(module):
      def hook(module, input, output):
        if module._modules:  # only want base layers
          return
        class_name = str(module.__class__).split('.')[-1].split("'")[0]
        module_idx = len(summary)
        m_key = '%s-%i' % (class_name, module_idx + 1)
        summary[m_key] = OrderedDict()
        summary[m_key]['input_shape'] = list(input[0].size())
        summary[m_key]['input_shape'][0] = None
        if output.__class__.__name__ == 'tuple':
          summary[m_key]['output_shape'] = list(output[0].size())
        else:
          summary[m_key]['output_shape'] = list(output.size())
        summary[m_key]['output_shape'][0] = None

        params = 0
        # iterate through parameters and count num params
        for name, p in module._parameters.items():
          params += torch.numel(p.data)
          summary[m_key]['trainable'] = p.requires_grad

        summary[m_key]['nb_params'] = params

      if not isinstance(module, torch.nn.Sequential) and \
         not isinstance(module, torch.nn.ModuleList) and \
         not (module == self):
        hooks.append(module.register_forward_hook(hook))

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
      # x = [Variable(torch.rand(1, *in_size)) for in_size in input_size]
      x = [Variable(torch.rand(*in_size)) for in_size in input_size]
    else:
      # x = Variable(torch.randn(1, *input_size))
      x = Variable(torch.randn(input_size))

    if(torch.cuda.is_available() and to_cuda): x = x.cuda()
    
    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    self.apply(register_hook)
    # make a forward pass
    self(x)
    # remove these hooks
    for h in hooks:
      h.remove()

    # print out neatly
    def get_names(module, name, acc):
      if not module._modules:
        acc.append(name)
      else:
        for key in module._modules.keys():
          p_name = key if name == "" else name + "." + key
          get_names(module._modules[key], p_name, acc)
    names = []
    get_names(self, "", names)

    col_width = 25  # should be >= 12
    summary_width = 61

    def crop(s):
      return s[:col_width] if len(s) > col_width else s

    print('_' * summary_width)
    print('{0: <{3}} {1: <{3}} {2: <{3}}'.format(
      'Layer (type)', 'Output Shape', 'Param #', col_width))
    print('=' * summary_width)
    total_params = 0
    trainable_params = 0
    for (i, l_type), l_name in zip(enumerate(summary), names):
      d = summary[l_type]
      total_params += d['nb_params']
      if 'trainable' in d and d['trainable']:
        trainable_params += d['nb_params']
      print('{0: <{3}} {1: <{3}} {2: <{3}}'.format(
        crop(l_name + ' (' + l_type[:-2] + ')'), crop(str(d['output_shape'])),
        crop(str(d['nb_params'])), col_width))
      if i < len(summary) - 1:
        print('_' * summary_width)
    print('=' * summary_width)
    print('Total params: ' + str(total_params))
    print('Trainable params: ' + str(trainable_params))
    print('Non-trainable params: ' + str((total_params - trainable_params)))
    print('_' * summary_width)

  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MType(enum.Enum):
  '''Enum class to choose the running model'''

  Model_1 = "M1"
  Model_2 = "M2"
  Model_3 = "M3"
  Model_4 = "M4"
  Model_5 = "M5"
  Model_6 = "M6"
  Model_7 = "M7"
  Model_8 = "M8"
  Model_9 = "M9"
  Model_10 = "M10"
  Model_11 = "M11"
  Model_12 = "M12"
  Model_13 = "M13"
  Model_14 = "M14"
  Model_15 = "M15"
  Model_16 = "M16"
  Model_17 = "M17"
  Model_18 = "M18"
  Model_19 = "M19"
  Model_20 = "M20"
  Model_21 = "M21"

  @staticmethod
  # parse a string name and get the corresponding enum
  def parse(model_name):
    models_dict = {
      "M1": MType.Model_1,
      "M2": MType.Model_2,
      "M3": MType.Model_3,
      "M4": MType.Model_4,
      "M5": MType.Model_5,
      "M6": MType.Model_6,
      "M7": MType.Model_7,
      "M8": MType.Model_8,
      "M9": MType.Model_9,
      "M10": MType.Model_10,
      "M11": MType.Model_11,
      "M12": MType.Model_12,
      "M13": MType.Model_13,
      "M14": MType.Model_14,
      "M15": MType.Model_15,
      "M16": MType.Model_16,
      "M17": MType.Model_17,
      "M18": MType.Model_18,
      "M19": MType.Model_19,
      "M20": MType.Model_20,
      "M21": MType.Model_21,
    }

    if(model_name.upper() in models_dict):
      return models_dict[model_name.upper()]
    else:
      raise Exception(f"No such model named: {model_name}")