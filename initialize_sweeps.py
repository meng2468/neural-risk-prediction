import wandb
import sys

if __name__ == '__main__':
    inputs = sys.argv
    if len(inputs) == 1:
        print('No dataset specified, cancelling')
    else:
        sweep_type = ' '.join(sys.argv[1:])
        print('Initialising sweep for', sweep_type)
    
    if 'mim' in sweep_type:
        sweep_configuration = {
            'method': 'random',
            'name': 'mimic-sweep',
            'project': 'mimic-random-sweep-test',
            'entity': 'risk-prediction',
            'metric': {'goal': 'minimize', 'name': 'Val Loss'},
            'parameters': 
            {
                'learning_rate': {'max': 0.01, 'min': 0.00001},
                'hidden_size': {'values': [512, 1024, 2048, 4096, 8192]},
                'dropout': {'max': .5, 'min': 0.},
                'model_name': {'values': ['gru','lstm','rnn']},
                'batch_size': {'value': 50},
                'input_size': {'value': 51},
                'dataset': {'value':'mimic'}
            }
        }
        project_name = 'mimic-random-sweep-test'
    else:
        sweep_configuration = {
            'method': 'random',
            'name': 'eicu-sweep',
            'entity': 'risk-prediction',
            'metric': {'goal': 'minimize', 'name': 'Val Loss'},
            'parameters': 
            {
                'learning_rate': {'max': 0.01, 'min': 0.00001},
                'hidden_size': {'values': [512, 1024, 2048, 4096, 8192]},
                'dropout': {'max': .5, 'min': 0.},
                'model_name': {'values': ['gru','lstm','rnn']},
                'batch_size': {'value': 50},
                'input_size': {'value': 45},
                'dataset': {'value':'eicu'}
            }
        }
        project_name = 'eicu-random-sweep-test'


    sweep_id = wandb.sweep(sweep_configuration, project=project_name)
    print(sweep_type+':', sweep_id, project_name)