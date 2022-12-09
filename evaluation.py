import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

def save_plot_loss(train_loss, test_loss, model_name):    
    fig = go.Figure()

    fig.add_trace(
    go.Scatter(
        x=list(range(len(train_loss))),
        y=train_loss,
        mode='lines',
        name='Train',
        line=dict(width=2, color='#3A0CA3'),
    ))

    fig.add_trace(
    go.Scatter(
        x=list(range(len(test_loss))),
        y=test_loss,
        mode='lines',
        name='Validation',
        line=dict(width=2, color='#808F85'),
    ))

    fig.update_layout(
        template='simple_white',
        width=900,
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    fig.update_xaxes(
        minor=dict(ticklen=0, tickcolor="black"),
        title='Epoch',
        # range=[0,10],
        dtick=1
    )
    fig.update_yaxes(
        title='Loss',
        # dtick=20,
        range=[0,max(train_loss)*1.1],
        dtick=.1
    )

    fig.write_image("evaluation/"+model_name+"_loss.jpeg")
