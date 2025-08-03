import numpy as np
import pandas as pd
import gradio as gr
import builtins


def FCFS(df):
    df = df.copy()
    j = 0
    for i in range(len(df)):
        j += df['PT'].iloc[i]
        df.at[i, 'FT'] = j
    df['Lateness'] = (df['FT'] - df['DD']).clip(lower=0)
    return df['FT'].mean(), df['PT'].sum() / df['FT'].sum(), df['Lateness'].mean()

def SPT(df):
    df = df.copy().sort_values(by=['PT']).reset_index(drop=True)
    j = 0
    for i in range(len(df)):
        j += df['PT'].iloc[i]
        df.at[i, 'FT'] = j
    df['Lateness'] = (df['FT'] - df['DD']).clip(lower=0)
    return df['FT'].mean(), df['PT'].sum() / df['FT'].sum(), df['Lateness'].mean()

def LPT(df):
    df = df.copy().sort_values(by=['PT'], ascending=False).reset_index(drop=True)
    j = 0
    for i in range(len(df)):
        j += df['PT'].iloc[i]
        df.at[i, 'FT'] = j
    df['Lateness'] = (df['FT'] - df['DD']).clip(lower=0)
    return df['FT'].mean(), df['PT'].sum() / df['FT'].sum(), df['Lateness'].mean()

def SCR(df):
    df = df.copy().sort_values(by=['CR']).reset_index(drop=True)
    j = 0
    for i in range(len(df)):
        j += df['PT'].iloc[i]
        df.at[i, 'FT'] = j
    df['Lateness'] = (df['FT'] - df['DD']).clip(lower=0)
    return df['FT'].mean(), df['PT'].sum() / df['FT'].sum(), df['Lateness'].mean()

def SS(df):
    df = df.copy().sort_values(by=['Slack']).reset_index(drop=True)
    j = 0
    for i in range(len(df)):
        j += df['PT'].iloc[i]
        df.at[i, 'FT'] = j
    df['Lateness'] = (df['FT'] - df['DD']).clip(lower=0)
    return df['FT'].mean(), df['PT'].sum() / df['FT'].sum(), df['Lateness'].mean()



def run_simulations(n, simulations):
    methods = ['FCFS', 'SPT', 'LPT', 'SCR', 'SS']
    raw_results = {method: {'Flow Time': 0, 'Throughput Rate': 0, 'Lateness': 0} for method in methods}

    for _ in range(int(simulations)):
        pt = np.random.randint(2, 10, size=int(n))
        total_processing_time = pt.sum()
        dd = np.random.randint(int(total_processing_time * 0.3), int(total_processing_time * 0.9), size=int(n))
        slack = dd - pt
        cr = dd / pt

        df = pd.DataFrame({
            'PT': pt,
            'DD': dd,
            'Slack': slack,
            'CR': cr
        })

        for method_name, method in zip(methods, [FCFS, SPT, LPT, SCR, SS]):
            avg_flow_time, throughput_rate, avg_lateness = method(df)
            raw_results[method_name]['Flow Time'] += avg_flow_time
            raw_results[method_name]['Throughput Rate'] += throughput_rate
            raw_results[method_name]['Lateness'] += avg_lateness

    
    for method in raw_results:
        for metric in raw_results[method]:
            raw_results[method][metric] /= int(simulations)


    results_table = []
    for metric in ['Flow Time', 'Throughput Rate', 'Lateness']:
        if metric == 'Throughput Rate':
            best_method = max(raw_results, key=lambda x: raw_results[x][metric])
        else:
            best_method = min(raw_results, key=lambda x: raw_results[x][metric])
        results_table.append([metric, best_method, raw_results[best_method][metric]])


    avg_scores = {method: builtins.sum(raw_results[method].values()) / len(raw_results[method]) for method in methods}
    best_avg_method = min(avg_scores, key=avg_scores.get)
    results_table.append(["Average Score", best_avg_method, avg_scores[best_avg_method]])

    return pd.DataFrame(results_table, columns=["Metric", "Best Method", "Score"])



inputs = [
    gr.Number(label="Number of Jobs (n)", value=10, info="Enter the number of jobs to simulate."),
    gr.Number(label="Simulations", value=1000, info="Specify the number of simulation runs."),
]

outputs = gr.Dataframe(label="Best Scheduling Method Results")

app = gr.Interface(
    fn=run_simulations,
    inputs=inputs,
    outputs=outputs,
    title="Production Scheduling Optimizer",
    description=(
        "Optimize production scheduling with five different strategies. "
        "Select the number of jobs (n) and simulations to identify the best scheduling method "
        "for each metric and the best overall method based on average performance."
    ),
    article=(
        "<p><b>How to use:</b></p>"
        "<ol>"
        "<li>Input the number of jobs to simulate production scheduling for.</li>"
        "<li>Enter the number of simulation runs for accuracy.</li>"
        "<li>Results will show the best scheduling method based on Flow Time, Throughput Rate, and Lateness, "
        "along with the method that has the best Average Score across these metrics.</li>"
        "</ol>"
        "<p><b>Metrics:</b> Flow Time measures average time jobs remain in the system, Throughput Rate reflects processing efficiency, "
        "and Lateness calculates delay relative to deadlines.</p>"
    )
)


app.launch(debug=True)
