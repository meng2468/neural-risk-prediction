import wandb
import sys

if __name__ == '__main__':
    sweep_configuration = {
        'method': 'random',
        'name': 'mimic-rnn-relu',
        'entity': 'risk-prediction',
        'metric': {'goal': 'minimize', 'name': 'Final Val Loss'},
        'parameters': 
        {
            'learning_rate': {'max': 1e-3, 'min': 1e-6},
            'hidden_size': {'values': [512, 1024, 2048, 4096, 8192]},
            'dropout': {'values': [0, .1, .2, .3, .4, .5]},
            'model_name': {'values': ['rnn', 'lstm','gru']},
            'batch_size': {'value': 50},
            'input_size': {'value': 51},
            'dataset': {'value':'mimic'}
        }
    }
    project_name = 'mimic-rnn-relu'

    sweep_id = wandb.sweep(sweep_configuration, project=project_name)
    print('MIMIC Sweep'+':', sweep_id, project_name)

    sweep_configuration = {
        'method': 'random',
        'name': 'eicu-rnn-relu',
        'entity': 'risk-prediction',
        'metric': {'goal': 'minimize', 'name': 'Final Val Loss'},
        'parameters': 
        {
            'learning_rate': {'max': 1e-3, 'min': 1e-6},
            'hidden_size': {'values': [512, 1024, 2048, 4096, 8192]},
            'dropout': {'values': [0, .1, .2, .3, .4, .5]},
            'model_name': {'values': ['rnn']},
            'batch_size': {'value': 50},
            'input_size': {'value': 45},
            'dataset': {'value':'eicu'}
        }
    }
    project_name = 'eicu-rnn-relu'


    sweep_id = wandb.sweep(sweep_configuration, project=project_name)
    print('EICU Sweep'+':', sweep_id, project_name)