import wandb
import sys

if __name__ == '__main__':
    sweep_configuration = {
        'method': 'random',
        'name': 'mimic-sweep',
        'entity': 'risk-prediction',
        'metric': {'goal': 'minimize', 'name': 'Final Val Loss'},
        'parameters': 
        {
            'learning_rate': {'max': 1e-1, 'min': 1e-6},
            'hidden_size': {'values': [512, 1024, 2048, 4096, 8192]},
            'dropout': {'max': .5, 'min': 0.},
            'model_name': {'values': ['gru','lstm','rnn']},
            'batch_size': {'value': 50},
            'input_size': {'value': 51},
            'dataset': {'value':'mimic'}
        }
    }
    project_name = 'mimic-random-sweep-test-2'

    sweep_id = wandb.sweep(sweep_configuration, project=project_name)
    print('MIMIC Sweep'+':', sweep_id, project_name)

    sweep_configuration = {
        'method': 'random',
        'name': 'eicu-sweep',
        'entity': 'risk-prediction',
        'metric': {'goal': 'minimize', 'name': 'Final Val Loss'},
        'parameters': 
        {
            'learning_rate': {'max': 1e-1, 'min': 1e-6},
            'hidden_size': {'values': [512, 1024, 2048, 4096, 8192]},
            'dropout': {'max': .5, 'min': 0.},
            'model_name': {'values': ['gru','lstm','rnn']},
            'batch_size': {'value': 50},
            'input_size': {'value': 45},
            'dataset': {'value':'eicu'}
        }
    }
    project_name = 'eicu-random-sweep-test-2'


    sweep_id = wandb.sweep(sweep_configuration, project=project_name)
    print('EICU Sweep'+':', sweep_id, project_name)