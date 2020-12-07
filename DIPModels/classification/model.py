## TODO: remove **kwargs, use config instead
""" Classification models by fine tunning different backbones. """

import inspect
import os

from keras import applications
from keras import callbacks
from keras import losses
from keras import metrics
from keras import optimizers
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.models import Model

from . import utils
from ..utils_g import utils_keras

## TODO: Add support for ResNet101, ResNet152
## TODO: Add support to ResNeXt
class FineTunning(object):
    """ Fine Tunning models for classification tasks.
        Currently supported models are: ResNet50, ResNet101,
        densenet121, densenet169, densenet201, CheXNet, inception_v3,
    """
    def __init__(self, backbone, **kwargs):
        self.backbone_name = backbone
        input_shape = kwargs['model_input_shape']
        input_tensor = (kwargs['input_tensor']
                        if 'input_tensor' in kwargs else None)
        pooling = (kwargs['backbone_toplayer_pooling']
                   if 'backbone_toplayer_pooling' in kwargs else 'avg')
        
        ## Load backbone model
        self.backbone = self._load_backbone(input_tensor=input_tensor,
                                            input_shape=input_shape,
                                            pooling=pooling)
    
    def __call__(self, **kwargs):
        self.build_model(**kwargs)
        return self
    
    def build_model(self, **kwargs):
        """ Create a backbone + fc layers structure for classification.
            Arguments:
                nb_classes: number of predict classes (required).
                model_input_shape: the image input shape tuple (required).
                input_tensor: optional Keras tensor to use as image input.
                weights: a tuple contains weights ('backbone', 'all', None) and
                         the weights path.
                backbone_toplayer_pooling: the pooling method for backbone top
                                           layer (pooling in keras.applications)
                nb_fclayers: number of fc layers on top of the backbone.
                fclayer_dropout_rate: the dropout rate used in top fc layers.
        """
        self.model_name = kwargs['model_name']
        self.nb_classes = kwargs['nb_classes']
        weights, weights_path = kwargs['weights']
        nb_fclayers = (kwargs['nb_fclayers']
                       if 'nb_fclayers' in kwargs else 0)
        dropout_rate = (kwargs['fclayer_dropout_rate']
                        if 'fclayer_dropout_rate' in kwargs else None)
        
        ## Load backbone model
        ## Add top layers and load weights
        if weights == 'backbone':
            print("Load weights from: " + weights_path)
            self.backbone.load_weights(weights_path)
        
        x = self._add_fc_layers(nb_fclayers, dropout_rate)
        # Add a logistic layer
        preds = Dense(self.nb_classes, activation='softmax')(x)
        model = Model(inputs=self.backbone.input, outputs=preds)
        
        ## TODO: maybe support load_weights_by_name
        if weights == 'all':
            print("Load weights from: " + weights_path)
            model.load_weights(weights_path)
        
        self.keras_model = model

    
    def _load_backbone(self, input_shape, input_tensor=None, pooling='avg'):
        model_args = dict(include_top=False, weights=None, pooling=pooling,
                          input_tensor=input_tensor, input_shape=input_shape)
        """ Load backbone models. """
        ## Add an extra GlobalAveragePooling2D layer to flatten the model
        ## by setting pooling = 'avg': [None, h, w, 2048] -> [None, 2048]
        if self.backbone_name == 'resnet50':
            model = applications.resnet50.ResNet50(**model_args)
        elif self.backbone_name == 'inception_v3':
            model = applications.inception_v3.InceptionV3(**model_args)
        elif self.backbone_name == 'CheXNet':
            model = applications.densenet.DenseNet121(**model_args)
        elif self.backbone_name == 'densenet121':
            model = applications.densenet.DenseNet121(**model_args)
        elif self.backbone_name == 'densenet169':
            model = applications.densenet.DenseNet169(**model_args)
        elif self.backbone_name == 'densenet201':
            model = applications.densenet.DenseNet201(**model_args)
        else:
            raise ValueError("Backbone " + self.backbone + " is not supported")
        return model
    
    def _add_fc_layers(self, nb_fclayers, dropout_rate):
        """ Add a classification header on top of the backbone. """
        x = self.backbone.output
        for _ in range(nb_fclayers):
            # Add a fully-connected layer
            x = Dense(1024, activation='relu')(x)
            # Add dropout layer
            x = Dropout(dropout_rate)(x)
        return x
    
    ## TODO: support freeze_layers by name
    def set_trainable(self, freeze_layers=None):
        """ Freeze layers don't neet to be trained. """
        if not freeze_layers:
            freeze_layers = len(self.backbone.layers)
        for layer in self.keras_model.layers[:freeze_layers]:
            layer.trainable = False
        for layer in self.keras_model.layers[freeze_layers:]:
            layer.trainable = True
        return
    
    def get_optimizer(self, **kwargs):
        """ Construct optimizers based on config. """
        optimizer_name = kwargs['optimizer']
        optimizer_args = kwargs['optimizer_args']
        optimizer_class = getattr(optimizers, optimizer_name)
        args_list = set(inspect.getargspec(optimizer_class.__init__).args)
        # args_list = inspect.signature(optimizer_class).parameters.keys()
        for k in set(optimizer_args.keys()) - args_list:
            del optimizer_args[k]
        return optimizer_class(**optimizer_args)

    def get_loss(self, name):
        """ Define model loss function. """
        return getattr(losses, name)

    def get_callbacks(self, log_dir, checkpoint_dir):
        """ Define validation callbacks. """
        callbacks_list = [
            callbacks.ModelCheckpoint(checkpoint_dir, monitor='val_acc', 
                                      verbose=1, save_best_only=True),
            callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0),
            callbacks.TensorBoard(log_dir, histogram_freq=0,
                                  write_graph=False, write_images=False)
        ]
        return callbacks_list

    def train(self, train_generator, validation_generator, 
              freeze_layers=None, **kwargs):
        """ Train the model. """
        self.optimizer = self.get_optimizer(**kwargs)
        self.loss = self.get_loss(kwargs['loss'])
        # self.metrics = [getattr(metrics, args['metrics']]
        self.set_trainable(freeze_layers)
        self.keras_model.compile(optimizer=self.optimizer,
                                 loss=self.loss, metrics=['acc'])
        
        log_dir, checkpoint_dir, initial_epoch = \
            utils_keras.get_log_dir(self.model_name, kwargs['model_dir'], 
                                    kwargs['weights_path'])
        callbacks = self.get_callbacks(log_dir, checkpoint_dir)
        
        # batch_size = kwargs['batch_size']
        epochs = kwargs['epochs']
        steps_per_epoch = kwargs['steps_per_epoch']
        validation_steps = kwargs['validation_steps']
        
        lr = kwargs['optimizer_args']['lr']
        print("\nStarting at epoch {}. lr={}\n".format(initial_epoch, lr))
        print("Checkpoint Path: {}".format(checkpoint_dir))
        
        self.keras_model.fit_generator(generator=train_generator,
                                       epochs=epochs,
                                       steps_per_epoch=steps_per_epoch,
                                       validation_data=validation_generator,
                                       validation_steps=validation_steps,
                                       initial_epoch=initial_epoch,
                                       callbacks=callbacks)


