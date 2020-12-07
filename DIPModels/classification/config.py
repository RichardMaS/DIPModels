import argparse
import os

## TODO: Rewrite as config and move argparse into utils_misc
class Config(object):
    """ Command line argument parser.
        
        Sample command:
        cmd = "first_try -backbone CheXNet -model_input_shape=256*256*3 -nb_classes=2 -epochs=20 -batch_size=5 -optimizer=sgd --backbone_toplayer_pooling=max --nb_fclayers=1 --fclayer_dropout_rate=0 --optimizer_args lr=0.001 momentum=0.9 decay=0.0 nesterov=True rho=0.9 epsilon=None beta_1=0.9 beta_2=0.999 amsgrad=False schedule_decay=0.004"
        
    """
    def __init__(self, cmd=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('model_name', help="Give a name to the model")
        parser.add_argument('-backbone',
                            choices=['CheXNet', 'resnet50',
                                     'inception_v3', 'densenet121',
                                     'densenet169', 'densenet201'],
                            help="The backbone feature layers used for tunning")
        
        parser.add_argument('-weights', default='backbone',
                            choices=['backbone', 'all', None],
                            help="The weights upload into the model")
        parser.add_argument('-model_input_shape', default='256*256*3',
                            help="The input shape of images")
        parser.add_argument('-nb_classes', type=int, default=2,
                            help="The number of classes")
        parser.add_argument('-epochs', type=int, default=10,
                            help="The number of epochs for training")
        parser.add_argument('-batch_size', type=int, default=5,
                            help="The batch size for training")
        parser.add_argument('-optimizer', default='adam',
                            help="The optimizer used for training")

        parser.add_argument('--model_dir', default=None,
                            help="The output folder to save weights")
        parser.add_argument('--weights_path', default=None,
                            help="The pretrained weights path")
        parser.add_argument('--prev_weights_path', default=None,
                            help="The pretrained weights path")        
        parser.add_argument('--backbone_toplayer_pooling',
                            choices=['avg', 'max'], default='avg',
                            help="Pooling method used on top of feature layers")
        parser.add_argument('--nb_fclayers', type=int, default=1,
                            help="Number of fc layers on top of the backbone")
        parser.add_argument('--fclayer_dropout_rate', type=float, default=None,
                            help="Drop out rate for the top dense layer")
        parser.add_argument('--optimizer_args', nargs='*', default=[],
                            help="Optimizer args: learning rate, decay, etc")
        parser.add_argument('--training_args', nargs='*', default=[],
                            help="Other training args: learning rate, decay, etc")
        
        args = vars(parser.parse_args(cmd))
        self.args_model, self.args_train = self.organize_args(args)

    def organize_args(self, args):
        args['model_input_shape'] = tuple(int(x) for x in args['model_input_shape'].split('*'))
        args['model_input_size'] = args['model_input_shape'][:2]
        if args['weights'] is None:
            args['weights_path'] = None
        args['weights'] = (args['weights'], args['weights_path'])
        args['optimizer_args'] = self._convert_named_str_to_pars(args['optimizer_args'])
        
        ## TODO: Add support to regression
        args['class_mode'] = 'categorical'
        args['loss'] = args['class_mode'] + '_crossentropy'
        
        args['steps_per_epoch'] = 100
        args['validation_steps'] = 50
        
        ## separate args into model, train, io
        args_name = [['model_name', 'backbone', 'model_input_shape', 'weights',
                      'nb_classes', 'fclayer_dropout_rate', 'nb_fclayers',
                      'backbone_toplayer_pooling'],
                     ['epochs', 'batch_size', 'class_mode', 'loss', 'model_input_size', 
                      'optimizer', 'optimizer_args', 'model_dir', 'weights_path', 
                      'steps_per_epoch', 'validation_steps']]
        args_list = []
        for names in args_name:
            args_sub = dict()
            for k in names:
                args_sub[k] = args[k]
            args_list.append(args_sub)
        return args_list

    def _convert_named_str_to_pars(self, par_list):
        def __converter(x):
            k, v = x.split('=')
            if v == 'False':
                v = False
            elif v == 'True':
                v = True
            elif v == 'None':
                v = None 
            else:
                v = float(v)
            return (k, v)
        return dict(__converter(x) for x in par_list)
