#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：ZhangJinYun lbert time:20-2-4

"""
    adversarial_training

"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.python import keras
import array
import functools
import gzip
import operator
import os
import struct
import tempfile
import sys
import warnings
from abc import ABCMeta

import numpy as np
from c_hans.compat import reduce_max, reduce_sum, softmax_cross_entropy_with_logits
from c_hans import utils_tf

from tensorflow.python.platform import app, flags
from cleverhans_tutorials import check_installation
from dataset_analysis import Setup_mnist_fashion
from dataset_analysis import Setup_mnist
from model_evaluation import evaluation
from fgsm import fgsm


import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FLAGS = flags.FLAGS

NB_EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = .001

Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
Dense = tf.keras.layers.Dense
Activation = tf.keras.layers.Activation
Flatten = tf.keras.layers.Flatten
KerasModel = tf.keras.models.Model
load_model=tf.keras.models.load_model
utils=tf.keras.utils




def adversarial_training(model,dataset,file_name,nb_epochs=5, batch_size=128,
                         learning_rate=.001, testing=False,
                         label_smoothing=0.1):
    """
    MNIST CleverHans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param testing: if true, training error is calculated
    :param label_smoothing: float, amount of label smoothing for cross entropy
    :return: an AccuracyReport object
    """

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()  # ---

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)
    # # Force TensorFlow to use single thread to improve reproducibility
    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    #
    # if keras.backend.image_data_format() != 'channels_last':
    #     raise NotImplementedError("this tutorial requires keras to be configured to channels_last format")
    #
    # Create TF session and set as Keras backend session
    sess = tf.Session(config=config)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    keras.backend.set_session(sess)

    # Get MNIST test data
    # train_start=0
    # train_end=60000
    # test_start=0
    # test_end=10000
    # mnist = MNIST(train_start=train_start, train_end=train_end,
    #               test_start=test_start, test_end=test_end)  # --- MNIST() change
    # x_train, y_train = mnist.get_set('train')
    # x_test, y_test = mnist.get_set('test')

    x_train = dataset.train_data
    y_train = dataset.train_labels
    x_test = dataset.test_data
    y_test = dataset.test_labels

    # Obtain Image Parameters
    img_rows, img_cols, nchannels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]

    # Label smoothing
    y_train -= label_smoothing * (y_train - 1. / nb_classes)

    print("The dataset is ready")
    # Define Keras model
    # model = cnn_model(img_rows=img_rows, img_cols=img_cols,
    #                   channels=nchannels, nb_filters=64,
    #                   nb_classes=nb_classes)  # ---model change
    # print("Defined Keras model.")


    # To be able to call the model in the custom loss, we need to call it once
    # before, see https://github.com/tensorflow/tensorflow/issues/23769
    model(model.input)  # cnn_model.input?

    # Initialize the Fast Gradient Sign Method (FGSM) attack object
    wrap = KerasModelWrapper(model)  # ---包装模型
    if not isinstance(wrap, Model):
      raise TypeError("The wrap argument should be an instance of"
                      " the cleverhans.model.Model class.")

    fgsm = FastGradientMethod(wrap, sess=sess)  # ---
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}

    adv_acc_metric = get_adversarial_acc_metric(model, fgsm,
                                                fgsm_params)  # ???adv_acc_metric return keras.metric.categorical_crossentropy

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', adv_acc_metric]  # ???以adv_acc_metric作为正确率计算有什么影响
        # metrics=['accuracy']
    )
    print('model.metrics_names', model.metrics_names)

    # Train an MNIST model
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=nb_epochs,
              validation_data=(x_test, y_test),
              verbose=1)

    # Evaluate the accuracy on legitimate and adversarial test examples
    _,acc,adv_acc=model.evaluate(x_test, y_test,
                                     batch_size=batch_size,
                                     verbose=1)
    report.clean_train_clean_eval = acc
    report.clean_train_adv_eval = adv_acc
    print('Test accuracy on legitimate examples: %0.4f' % acc)
    print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)

    # Calculate training error
    if testing:
        _, train_acc, train_adv_acc = model.evaluate(x_train, y_train,
                                                     batch_size=batch_size,
                                                     verbose=0)
        report.train_clean_train_clean_eval = train_acc
        report.train_clean_train_adv_eval = train_adv_acc

    # using adversarial training
    print("Repeating the process, using adversarial training")
    #训练好了的第一次模型拿来进行第二次对抗训练
    model_2=model
    model_2(model_2.input)
    wrap_2 = KerasModelWrapper(model_2)
    fgsm_2 = FastGradientMethod(wrap_2, sess=sess)

    # Use a loss function based on legitimate and adversarial examples
    adv_loss_2 = get_adversarial_loss(model_2, fgsm_2, fgsm_params)  # 改变损失函数,进行对抗训练
    adv_acc_metric_2 = get_adversarial_acc_metric(model_2, fgsm_2, fgsm_params)

    model_2.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=adv_loss_2,
        metrics=['accuracy', adv_acc_metric_2]
    )

    # Train an MNIST model
    model_2.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=nb_epochs,
                validation_data=(x_test, y_test),
                verbose=1)

    # Evaluate the accuracy on legitimate and adversarial test examples
    _, acc, adv_acc = model_2.evaluate(x_test, y_test,
                                       batch_size=batch_size,
                                       verbose=1)
    report.adv_train_clean_eval = acc  # 对抗训练正常样本评估
    report.adv_train_adv_eval = adv_acc  # 对抗训练对抗样本评估
    print('Test accuracy on legitimate examples: %0.4f' % acc)
    print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)

    # Calculate training error
    if testing:
        _, train_acc, train_adv_acc = model_2.evaluate(x_train, y_train,
                                                       batch_size=batch_size,
                                                       verbose=0)
        report.train_adv_train_clean_eval = train_acc
        report.train_adv_train_adv_eval = train_adv_acc

    #重新覆盖compile以免load_model无法读取自定义loss
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate),
                  metrics=['accuracy'])

    # 保存对抗训练模型
    if file_name != None:
        model_save_path = file_name.replace('.h5', "_adv_training_" + str(nb_epochs) + '.h5')
        model_2.save(model_save_path)
        print('the saved strengthen model is:', model_save_path)

    return model_2,report

def get_adversarial_acc_metric(model, fgsm, fgsm_params):
    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Accuracy on the adversarial examples
        preds_adv = model(x_adv)  #
        return keras.metrics.categorical_accuracy(y, preds_adv)

    return adv_acc


def get_adversarial_loss(model, fgsm, fgsm_params):
    def adv_loss(y, preds):
        # Cross-entropy on the legitimate examples
        cross_ent = keras.losses.categorical_crossentropy(y, preds)

        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = keras.losses.categorical_crossentropy(y, preds_adv)

        return 0.5 * cross_ent + 0.5 * cross_ent_adv

    return adv_loss

class AccuracyReport(object):

  """
  An object summarizing the accuracy results for experiments involving
  training on clean examples or adversarial examples, then evaluating
  on clean or adversarial examples.
  """

  def __init__(self):
    self.clean_train_clean_eval = 0.#干净样本原始模型训练
    self.clean_train_adv_eval = 0.#对抗样本原始模型训练
    self.adv_train_clean_eval = 0.#干净样本增强模型训练
    self.adv_train_adv_eval = 0.#对抗样本增强模型训练

    # Training data accuracy results to be used by tutorials
    self.train_clean_train_clean_eval = 0.
    self.train_clean_train_adv_eval = 0.
    self.train_adv_train_clean_eval = 0.
    self.train_adv_train_adv_eval = 0.

#
# class Dataset(object):
#   """Abstract base class representing a dataset.
#   """
#
#   # The number of classes in the dataset. Should be specified by subclasses.
#   NB_CLASSES = None
#
#   def __init__(self, kwargs=None):
#     if kwargs is None:
#       kwargs = {}
#     if "self" in kwargs:
#       del kwargs["self"]
#     self.kwargs = kwargs
#
#   def get_factory(self):
#     """Returns a picklable callable that recreates the dataset.
#     """
#
#     return Factory(type(self), self.kwargs)
#
#   def get_set(self, which_set):
#     """Returns the training set or test set as an (x_data, y_data) tuple.
#     :param which_set: 'train' or 'test'
#     """
#     return (getattr(self, 'x_' + which_set),
#             getattr(self, 'y_' + which_set))
#
#   def to_tensorflow(self):
#     raise NotImplementedError()
#
#   @classmethod
#   def in_memory_dataset(cls, x, y, shuffle=None, repeat=True):
#     assert x.shape[0] == y.shape[0]
#     d = tf.data.Dataset.range(x.shape[0])
#     if repeat:
#       d = d.repeat()
#     if shuffle:
#       d = d.shuffle(shuffle)
#
#     def lookup(p):
#       return x[p], y[p]
#     d = d.map(lambda i: tf.py_func(lookup, [i], [tf.float32] * 2))
#     return d
#
#
# class MNIST(Dataset):
#   """The MNIST dataset"""
#
#   NB_CLASSES = 10
#
#   def __init__(self, train_start=0, train_end=60000, test_start=0,
#                test_end=10000, center=False, max_val=1.):
#     kwargs = locals()
#     if '__class__' in kwargs:
#       del kwargs['__class__']
#     super(MNIST, self).__init__(kwargs)
#     x_train, y_train, x_test, y_test = data_mnist(datadir='data',
#                                                   train_start=train_start,
#                                                   train_end=train_end,
#                                                   test_start=test_start,
#                                                   test_end=test_end)
#
#     if center:
#       x_train = x_train * 2. - 1.
#       x_test = x_test * 2. - 1.
#     x_train *= max_val
#     x_test *= max_val
#
#     self.x_train = x_train.astype('float32')
#     self.y_train = y_train.astype('float32')
#     self.x_test = x_test.astype('float32')
#     self.y_test = y_test.astype('float32')
#
#   def to_tensorflow(self, shuffle=4096):
#     return (self.in_memory_dataset(self.x_train, self.y_train, shuffle),
#             self.in_memory_dataset(self.x_test, self.y_test, repeat=False))
#
#
# class Factory(object):
#   """
#   A callable that creates an object of the specified type and configuration.
#   """
#
#   def __init__(self, cls, kwargs):
#     self.cls = cls
#     self.kwargs = kwargs
#
#   def __call__(self):
#     """Returns the created object.
#     """
#     return self.cls(**self.kwargs)

class NoSuchLayerError(ValueError):
  """Raised when a layer that does not exist is requested."""

class Model(object):
  """
  An abstract interface for model wrappers that exposes model symbols
  needed for making an attack. This abstraction removes the dependency on
  any specific neural network package (e.g. Keras) from the core
  code of CleverHans. It can also simplify exposing the hidden features of a
  model when a specific package does not directly expose them.
  """
  __metaclass__ = ABCMeta
  O_LOGITS, O_PROBS, O_FEATURES = 'logits probs features'.split()

  def __init__(self, scope=None, nb_classes=None, hparams=None,
               needs_dummy_fprop=False):
    """
    Constructor.
    :param scope: str, the name of model.
    :param nb_classes: integer, the number of classes.
    :param hparams: dict, hyper-parameters for the model.
    :needs_dummy_fprop: bool, if True the model's parameters are not
        created until fprop is called.
    """
    self.scope = scope or self.__class__.__name__
    self.nb_classes = nb_classes
    self.hparams = hparams or {}
    self.needs_dummy_fprop = needs_dummy_fprop

  def __call__(self, *args, **kwargs):
    """
    For compatibility with functions used as model definitions (taking
    an input tensor and returning the tensor giving the output
    of the model on that input).
    """

    warnings.warn("Model.__call__ is deprecated. "
                  "The call is ambiguous as to whether the output should "
                  "be logits or probabilities, and getting the wrong one "
                  "can cause serious problems. "
                  "The output actually is probabilities, which are a very "
                  "dangerous thing to use as part of any interface for "
                  "cleverhans, because softmax probabilities are prone "
                  "to gradient masking."
                  "On or after 2019-04-24, this method will change to raise "
                  "an exception explaining why Model.__call__ should not be "
                  "used.")

    return self.get_probs(*args, **kwargs)

  def get_logits(self, x, **kwargs):
    """
    :param x: A symbolic representation (Tensor) of the network input
    :return: A symbolic representation (Tensor) of the output logits
    (i.e., the values fed as inputs to the softmax layer).
    """
    outputs = self.fprop(x, **kwargs)
    if self.O_LOGITS in outputs:
      return outputs[self.O_LOGITS]
    raise NotImplementedError(str(type(self)) + "must implement `get_logits`"
                              " or must define a " + self.O_LOGITS +
                              " output in `fprop`")

  def get_predicted_class(self, x, **kwargs):
    """
    :param x: A symbolic representation (Tensor) of the network input
    :return: A symbolic representation (Tensor) of the predicted label
    """
    return tf.argmax(self.get_logits(x, **kwargs), axis=1)

  def get_probs(self, x, **kwargs):
    """
    :param x: A symbolic representation (Tensor) of the network input
    :return: A symbolic representation (Tensor) of the output
    probabilities (i.e., the output values produced by the softmax layer).
    """
    d = self.fprop(x, **kwargs)
    if self.O_PROBS in d:
      output = d[self.O_PROBS]
      min_prob = tf.reduce_min(output)
      max_prob = tf.reduce_max(output)
      asserts = [utils_tf.assert_greater_equal(min_prob,
                                               tf.cast(0., min_prob.dtype)),
                 utils_tf.assert_less_equal(max_prob,
                                            tf.cast(1., min_prob.dtype))]
      with tf.control_dependencies(asserts):
        output = tf.identity(output)
      return output
    elif self.O_LOGITS in d:
      return tf.nn.softmax(logits=d[self.O_LOGITS])
    else:
      raise ValueError('Cannot find probs or logits.')

  def fprop(self, x, **kwargs):
    """
    Forward propagation to compute the model outputs.
    :param x: A symbolic representation of the network input
    :return: A dictionary mapping layer names to the symbolic
             representation of their output.
    """
    raise NotImplementedError('`fprop` not implemented.')

  def get_params(self):
    """
    Provides access to the model's parameters.
    :return: A list of all Variables defining the model parameters.
    """

    if hasattr(self, 'params'):
      return list(self.params)

    # Catch eager execution and assert function overload.
    try:
      if tf.executing_eagerly():
        raise NotImplementedError("For Eager execution - get_params "
                                  "must be overridden.")
    except AttributeError:
      pass

    # For graph-based execution
    scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   self.scope + "/")

    if len(scope_vars) == 0:
      self.make_params()
      scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     self.scope + "/")
      assert len(scope_vars) > 0

    # Make sure no parameters have been added or removed
    if hasattr(self, "num_params"):
      if self.num_params != len(scope_vars):
        print("Scope: ", self.scope)
        print("Expected " + str(self.num_params) + " variables")
        print("Got " + str(len(scope_vars)))
        for var in scope_vars:
          print("\t" + str(var))
        assert False
    else:
      self.num_params = len(scope_vars)

    return scope_vars

  def make_params(self):
    """
    Create all Variables to be returned later by get_params.
    By default this is a no-op.
    Models that need their fprop to be called for their params to be
    created can set `needs_dummy_fprop=True` in the constructor.
    """

    if self.needs_dummy_fprop:
      if hasattr(self, "_dummy_input"):
        return
      self._dummy_input = self.make_input_placeholder()
      self.fprop(self._dummy_input)

  def get_layer_names(self):
    """Return the list of exposed layers for this model."""
    raise NotImplementedError

  def get_layer(self, x, layer, **kwargs):
    """Return a layer output.
    :param x: tensor, the input to the network.
    :param layer: str, the name of the layer to compute.
    :param **kwargs: dict, extra optional params to pass to self.fprop.
    :return: the content of layer `layer`
    """
    return self.fprop(x, **kwargs)[layer]

  def make_input_placeholder(self):
    """Create and return a placeholder representing an input to the model.

    This method should respect context managers (e.g. "with tf.device")
    and should not just return a reference to a single pre-created
    placeholder.
    """

    raise NotImplementedError(str(type(self)) + " does not implement "
                              "make_input_placeholder")

  def make_label_placeholder(self):
    """Create and return a placeholder representing class labels.

    This method should respect context managers (e.g. "with tf.device")
    and should not just return a reference to a single pre-created
    placeholder.
    """

    raise NotImplementedError(str(type(self)) + " does not implement "
                              "make_label_placeholder")

  def __hash__(self):
    return hash(id(self))

  def __eq__(self, other):
    return self is other

class KerasModelWrapper(Model):
  """
  An implementation of `Model` that wraps a Keras model. It
  specifically exposes the hidden features of a model by creating new models.
  The symbolic graph is reused and so there is little overhead. Splitting
  in-place operations can incur an overhead.
  """

  def __init__(self, model):
    """
    Create a wrapper for a Keras model
    :param model: A Keras model
    """
    super(KerasModelWrapper, self).__init__(None, None, {})

    if model is None:
      raise ValueError('model argument must be supplied.')

    self.model = model
    self.keras_model = None

  def _get_softmax_name(self):
    """
    Looks for the name of the softmax layer.
    :return: Softmax layer name
    """
    for layer in self.model.layers:
      cfg = layer.get_config()
      if 'activation' in cfg and cfg['activation'] == 'softmax':
        return layer.name

    raise Exception("No softmax layers found")

  def _get_abstract_layer_name(self):
    """
    Looks for the name of abstracted layer.
    Usually these layers appears when model is stacked.
    :return: List of abstracted layers
    """
    abstract_layers = []
    for layer in self.model.layers:
      if 'layers' in layer.get_config():
        abstract_layers.append(layer.name)

    return abstract_layers

  def _get_logits_name(self):
    """
    Looks for the name of the layer producing the logits.
    :return: name of layer producing the logits
    """
    softmax_name = self._get_softmax_name()
    softmax_layer = self.model.get_layer(softmax_name)

    if not isinstance(softmax_layer, Activation):
      # In this case, the activation is part of another layer
      return softmax_name

    if not hasattr(softmax_layer, '_inbound_nodes'):
      raise RuntimeError("Please update keras to version >= 2.1.3")

    node = softmax_layer._inbound_nodes[0]

    logits_name = node.inbound_layers[0].name

    return logits_name

  def get_logits(self, x):
    """
    :param x: A symbolic representation of the network input.
    :return: A symbolic representation of the logits
    """
    logits_name = self._get_logits_name()
    logits_layer = self.get_layer(x, logits_name)

    # Need to deal with the case where softmax is part of the
    # logits layer
    if logits_name == self._get_softmax_name():
      softmax_logit_layer = self.get_layer(x, logits_name)

      # The final op is the softmax. Return its input
      logits_layer = softmax_logit_layer._op.inputs[0]

    return logits_layer

  def get_probs(self, x):
    """
    :param x: A symbolic representation of the network input.
    :return: A symbolic representation of the probs
    """
    name = self._get_softmax_name()

    return self.get_layer(x, name)

  def get_layer_names(self):
    """
    :return: Names of all the layers kept by Keras
    """
    layer_names = [x.name for x in self.model.layers]
    return layer_names

  def fprop(self, x):
    """
    Exposes all the layers of the model returned by get_layer_names.
    :param x: A symbolic representation of the network input
    :return: A dictionary mapping layer names to the symbolic
             representation of their output.
    """

    if self.keras_model is None:
      # Get the input layer
      new_input = self.model.get_input_at(0)

      # Make a new model that returns each of the layers as output
      abstract_layers = self._get_abstract_layer_name()
      if abstract_layers:
        warnings.warn(
            "Abstract layer detected, picking last ouput node as default."
            "This could happen due to using of stacked model.")

      layer_outputs = []
      # For those abstract model layers, return their last output node as
      # default.
      for x_layer in self.model.layers:
        if x_layer.name not in abstract_layers:
          layer_outputs.append(x_layer.output)
        else:
          layer_outputs.append(x_layer.get_output_at(-1))

      self.keras_model = KerasModel(new_input, layer_outputs)

    # and get the outputs for that model on the input x
    outputs = self.keras_model(x)

    # Keras only returns a list for outputs of length >= 1, if the model
    # is only one layer, wrap a list
    if len(self.model.layers) == 1:
      outputs = [outputs]

    # compute the dict to return
    fprop_dict = dict(zip(self.get_layer_names(), outputs))

    return fprop_dict

  def get_layer(self, x, layer):
    """
    Expose the hidden features of a model given a layer name.
    :param x: A symbolic representation of the network input
    :param layer: The name of the hidden layer to return features at.
    :return: A symbolic representation of the hidden features
    :raise: NoSuchLayerError if `layer` is not in the model.
    """
    # Return the symbolic representation for this layer.
    output = self.fprop(x)
    try:
      requested = output[layer]
    except KeyError:
      raise NoSuchLayerError()
    return requested


class Attack(object):
  """
  Abstract base class for all attack classes.
  """
  __metaclass__ = ABCMeta

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    """
    :param model: An instance of the cleverhans.model.Model class.
    :param sess: The (possibly optional) tf.Session to run graphs in.
    :param dtypestr: Floating point precision to use (change to float64
                     to avoid numerical instabilities).
    :param back: (deprecated and will be removed on or after 2019-03-26).
                 The backend to use. Currently 'tf' is the only option.
    """
    if 'back' in kwargs:
      if kwargs['back'] == 'tf':
        warnings.warn("Argument back to attack constructors is not needed"
                      " anymore and will be removed on or after 2019-03-26."
                      " All attacks are implemented using TensorFlow.")
      else:
        raise ValueError("Backend argument must be 'tf' and is now deprecated"
                         "It will be removed on or after 2019-03-26.")

    self.tf_dtype = tf.as_dtype(dtypestr)
    self.np_dtype = np.dtype(dtypestr)

    if sess is not None and not isinstance(sess, tf.Session):
      raise TypeError("sess is not an instance of tf.Session")

    from c_hans import attacks_tf
    attacks_tf.np_dtype = self.np_dtype
    attacks_tf.tf_dtype = self.tf_dtype

    if not isinstance(model, Model):
      raise TypeError("The model argument should be an instance of"
                      " the cleverhans.model.Model class.")

    # Prepare attributes
    self.model = model
    self.sess = sess
    self.dtypestr = dtypestr

    # We are going to keep track of old graphs and cache them.
    self.graphs = {}

    # When calling generate_np, arguments in the following set should be
    # fed into the graph, as they are not structural items that require
    # generating a new graph.
    # This dict should map names of arguments to the types they should
    # have.
    # (Usually, the target class will be a feedable keyword argument.)
    self.feedable_kwargs = tuple()

    # When calling generate_np, arguments in the following set should NOT
    # be fed into the graph, as they ARE structural items that require
    # generating a new graph.
    # This list should contain the names of the structural arguments.
    self.structural_kwargs = []

  def generate(self, x, **kwargs):

    error = "Sub-classes must implement generate."
    raise NotImplementedError(error)
    # Include an unused return so pylint understands the method signature
    return x

  def get_or_guess_labels(self, x, kwargs):
    """
    Get the label to use in generating an adversarial example for x.
    The kwargs are fed directly from the kwargs of the attack.
    If 'y' is in kwargs, then assume it's an untargeted attack and
    use that as the label.
    If 'y_target' is in kwargs and is not none, then assume it's a
    targeted attack and use that as the label.
    Otherwise, use the model's prediction as the label and perform an
    untargeted attack.
    """
    if 'y' in kwargs and 'y_target' in kwargs:
      raise ValueError("Can not set both 'y' and 'y_target'.")
    elif 'y' in kwargs:
      labels = kwargs['y']
    elif 'y_target' in kwargs and kwargs['y_target'] is not None:
      labels = kwargs['y_target']
    else:
      preds = self.model.get_probs(x)
      preds_max = reduce_max(preds, 1, keepdims=True)
      original_predictions = tf.to_float(tf.equal(preds, preds_max))
      labels = tf.stop_gradient(original_predictions)
      del preds
    if isinstance(labels, np.ndarray):
      nb_classes = labels.shape[1]
    else:
      nb_classes = labels.get_shape().as_list()[1]
    return labels, nb_classes

  def parse_params(self, params=None):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.

    :param params: a dictionary of attack-specific parameters
    :return: True when parsing was successful
    """

    if params is not None:
      warnings.warn("`params` is unused and will be removed "
                    " on or after 2019-04-26.")
    return True


class FastGradientMethod(Attack):
  """
  This attack was originally implemented by Goodfellow et al. (2014) with the
  infinity norm (and is known as the "Fast Gradient Sign Method"). This
  implementation extends the attack to other norms, and is therefore called
  the Fast Gradient Method.
  Paper link: https://arxiv.org/abs/1412.6572

  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    """
    Create a FastGradientMethod instance.
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """

    super(FastGradientMethod, self).__init__(model, sess, dtypestr, **kwargs)
    self.feedable_kwargs = ('eps', 'y', 'y_target', 'clip_min', 'clip_max')
    self.structural_kwargs = ['ord', 'sanity_checks', 'clip_grad']

  def generate(self, x, **kwargs):
    """
    Returns the graph for Fast Gradient Method adversarial examples.

    :param x: The model's symbolic inputs.
    :param kwargs: See `parse_params`
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    labels, _nb_classes = self.get_or_guess_labels(x, kwargs)

    return fgm(
        x,
        self.model.get_logits(x),
        y=labels,
        eps=self.eps,
        ord=self.ord,
        clip_min=self.clip_min,
        clip_max=self.clip_max,
        clip_grad=self.clip_grad,
        targeted=(self.y_target is not None),
        sanity_checks=self.sanity_checks)

  def parse_params(self,
                   eps=0.3,
                   ord=np.inf,
                   y=None,
                   y_target=None,
                   clip_min=None,
                   clip_max=None,
                   clip_grad=False,
                   sanity_checks=True,
                   **kwargs):
      self.eps = eps
      self.ord = ord
      self.y = y
      self.y_target = y_target
      self.clip_min = clip_min
      self.clip_max = clip_max
      self.clip_grad = clip_grad
      self.sanity_checks = sanity_checks

      if self.y is not None and self.y_target is not None:
          raise ValueError("Must not set both y and y_target")
      # Check if order of the norm is acceptable given current implementation
      if self.ord not in [np.inf, int(1), int(2)]:
          raise ValueError("Norm order must be either np.inf, 1, or 2.")

      if self.clip_grad and (self.clip_min is None or self.clip_max is None):
          raise ValueError("Must set clip_min and clip_max if clip_grad is set")

      if len(kwargs.keys()) > 0:
          warnings.warn("kwargs is unused and will be removed on or after "
                        "2019-04-26.")

      return True

def fgm(x,
        logits,
        y=None,
        eps=0.3,
        ord=np.inf,
        clip_min=None,
        clip_max=None,
        clip_grad=False,
        targeted=False,
        sanity_checks=True):
    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        asserts.append(utils_tf.assert_greater_equal(
            x, tf.cast(clip_min, x.dtype)))

    if clip_max is not None:
        asserts.append(utils_tf.assert_less_equal(x, tf.cast(clip_max, x.dtype)))

    # Make sure the caller has not passed probs by accident
    assert logits.op.type != 'Softmax'

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = reduce_max(logits, 1, keepdims=True)
        y = tf.to_float(tf.equal(logits, preds_max))
        y = tf.stop_gradient(y)
    y = y / reduce_sum(y, 1, keepdims=True)

    # Compute loss
    loss = softmax_cross_entropy_with_logits(labels=y, logits=logits)
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    if clip_grad:
        grad = utils_tf.zero_out_clipped_grads(grad, x, clip_min, clip_max)

    optimal_perturbation = optimize_linear(grad, eps, ord)

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        # We don't currently support one-sided clipping
        assert clip_min is not None and clip_max is not None
        adv_x = utils_tf.clip_by_value(adv_x, clip_min, clip_max)

    if sanity_checks:
        with tf.control_dependencies(asserts):
            adv_x = tf.identity(adv_x)

    return adv_x


def optimize_linear(grad, eps, ord=np.inf):
  """
  Solves for the optimal input to a linear function under a norm constraint.

  Optimal_perturbation = argmax_{eta, ||eta||_{ord} < eps} dot(eta, grad)

  :param grad: tf tensor containing a batch of gradients
  :param eps: float scalar specifying size of constraint region
  :param ord: int specifying order of norm
  :returns:
    tf tensor containing optimal perturbation
  """

  # In Python 2, the `list` call in the following line is redundant / harmless.
  # In Python 3, the `list` call is needed to convert the iterator returned by `range` into a list.
  red_ind = list(range(1, len(grad.get_shape())))
  avoid_zero_div = 1e-12
  if ord == np.inf:
    # Take sign of gradient
    optimal_perturbation = tf.sign(grad)
    # The following line should not change the numerical results.
    # It applies only because `optimal_perturbation` is the output of
    # a `sign` op, which has zero derivative anyway.
    # It should not be applied for the other norms, where the
    # perturbation has a non-zero derivative.
    optimal_perturbation = tf.stop_gradient(optimal_perturbation)
  elif ord == 1:
    abs_grad = tf.abs(grad)
    sign = tf.sign(grad)
    max_abs_grad = tf.reduce_max(abs_grad, red_ind, keepdims=True)
    tied_for_max = tf.to_float(tf.equal(abs_grad, max_abs_grad))
    num_ties = tf.reduce_sum(tied_for_max, red_ind, keepdims=True)
    optimal_perturbation = sign * tied_for_max / num_ties
  elif ord == 2:
    square = tf.maximum(avoid_zero_div,
                        reduce_sum(tf.square(grad),
                                   reduction_indices=red_ind,
                                   keepdims=True))
    optimal_perturbation = grad / tf.sqrt(square)
  else:
    raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                              "currently implemented.")

  # Scale perturbation to be the solution for the norm=eps rather than
  # norm=1 problem
  scaled_perturbation = utils_tf.mul(eps, optimal_perturbation)
  return scaled_perturbation

# def main(args=None):
#     dataset =Setup_mnist(train_start=0, train_end=60000, test_start=0, test_end=10000)
#     model_path='models_test/tf_keras_mnist_model.h5'
#     model=load_model(model_path)
#     model_defend,_=adversarial_training(model, dataset, model_path, nb_epochs=FLAGS.nb_epochs,
#                          batch_size=FLAGS.batch_size,
#                          learning_rate=FLAGS.learning_rate)
#
#     eval = evaluation(model, dataset, model_defend, fgsm)
#
#
# if __name__=="__main__":
#
#     flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
#                          'Number of epochs to train model')
#     flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
#     flags.DEFINE_float('learning_rate', LEARNING_RATE,
#                        'Learning rate for training')
#
#     tf.app.run()
