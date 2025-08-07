import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import builtins


def schedule(df, sort_by=None, ascending=True):
    df = df.copy()
    if sort_by:
        df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
    time = 0
    for i in range(len(df)):
        time += df['PT'].iloc[i]
        df.at[i, 'FT'] = time
    df['Lateness'] = (df['FT'] - df['DD']).clip(lower=0)
    return df

def FCFS(df): return schedule(df)
def SPT(df): return schedule(df, 'PT')
def LPT(df): return schedule(df, 'PT', ascending=False)
def SCR(df): return schedule(df, 'CR')
def SS(df): return schedule(df, 'Slack')

method_functions = {'FCFS': FCFS, 'SPT': SPT, 'LPT': LPT, 'SCR': SCR, 'SS': SS}


def plot_gantt(df, title="Gantt Chart"):
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.5 + 1))
    start = 0
    for i in range(len(df)):
        ax.broken_barh([(start, df['PT'].iloc[i])], (i - 0.4, 0.8), facecolors='tab:blue')
        ax.text(start + df['PT'].iloc[i] / 2, i, f"J{i+1}", ha='center', va='center', color='white')
        start += df['PT'].iloc[i]
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"Job {i+1}" for i in range(len(df))])
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.grid(True)
    return fig


def run_simulation(n, simulations):
    methods = list(method_functions.keys())
    raw_results = {m: {'Flow Time': 0, 'Throughput Rate': 0, 'Lateness': 0} for m in methods}
    
    for _ in range(simulations):
        pt = np.random.randint(2, 10, size=n)
        total_pt = pt.sum()
        dd = np.random.randint(int(total_pt * 0.3), int(total_pt * 0.9), size=n)
        slack = dd - pt
        cr = dd / pt

        df = pd.DataFrame({'PT': pt, 'DD': dd, 'Slack': slack, 'CR': cr})

        for m in methods:
            res = method_functions[m](df)
            avg_ft = res['FT'].mean()
            tput = res['PT'].sum() / res['FT'].sum()
            lateness = res['Lateness'].mean()
            raw_results[m]['Flow Time'] += avg_ft
            raw_results[m]['Throughput Rate'] += tput
            raw_results[m]['Lateness'] += lateness

    for m in raw_results:
        for k in raw_results[m]:
            raw_results[m][k] /= simulations

    results_table = []
    for metric in ['Flow Time', 'Throughput Rate', 'Lateness']:
        best = max(raw_results, key=lambda x: raw_results[x][metric]) if metric == 'Throughput Rate' else min(raw_results, key=lambda x: raw_results[x][metric])
        results_table.append([metric, best, raw_results[best][metric]])
    
    avg_scores = {m: builtins.sum(raw_results[m].values()) / 3 for m in methods}
    best_avg = min(avg_scores, key=avg_scores.get)
    results_table.append(["Average Score", best_avg, avg_scores[best_avg]])

    return pd.DataFrame(results_table, columns=["Metric", "Best Method", "Score"])


def main(mode, method, n, simulations, custom_data, uploaded_file):
    if mode == "Random Simulation":
        return run_simulation(int(n), int(simulations)), None

    try:
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file.name)
        else:
            df = pd.DataFrame(custom_data, columns=["PT", "DD"]).dropna()
        
        df["PT"] = pd.to_numeric(df["PT"], errors="coerce")
        df["DD"] = pd.to_numeric(df["DD"], errors="coerce")
        df = df.dropna()

        if df.empty:
            return "Please input at least one valid job with numeric PT and DD.", None

        df['Slack'] = df['DD'] - df['PT']
        df['CR'] = df['DD'] / df['PT']
        scheduled = method_functions[method](df)

        avg_ft = scheduled['FT'].mean()
        tput = scheduled['PT'].sum() / scheduled['FT'].sum()
        lateness = scheduled['Lateness'].mean()

        table = pd.DataFrame({
            "Metric": ["Flow Time", "Throughput Rate", "Lateness"],
            "Value": [avg_ft, tput, lateness]
        })

        fig = plot_gantt(scheduled, title=f"Gantt Chart for {method}")
        return table, fig

    except Exception as e:
        return f"Error: {str(e)}", None


with gr.Blocks(title="Production Scheduling Optimizer") as demo:
    gr.Markdown("##  Production Scheduling Optimizer with Gantt Chart")
    gr.Markdown("Compare job scheduling strategies with random simulation or custom job data from table or CSV.")

    with gr.Row():
        mode = gr.Dropdown(["Random Simulation", "Custom Data"], value="Random Simulation", label="Mode")
        method = gr.Dropdown(list(method_functions.keys()), value="FCFS", label="Scheduling Method")

    with gr.Row():
        n = gr.Number(label="Number of Jobs", value=10, precision=0)
        simulations = gr.Number(label="Number of Simulations", value=1000, precision=0)

    custom_data = gr.Dataframe(headers=["PT", "DD"], label="Custom Job Data (PT, DD)", visible=False, row_count=5)
    uploaded_file = gr.File(label="Or Upload CSV (with PT and DD columns)", file_types=[".csv"], visible=False)

    run_btn = gr.Button("Run Simulation")

    output_table = gr.Dataframe(label="Results")
    gantt_chart = gr.Plot(label="Gantt Chart (Custom Data Only)")

    def toggle_custom_inputs(mode_val):
        show = mode_val == "Custom Data"
        return [gr.update(visible=show), gr.update(visible=show)]

    mode.change(toggle_custom_inputs, inputs=mode, outputs=[custom_data, uploaded_file])

    run_btn.click(
        fn=main,
        inputs=[mode, method, n, simulations, custom_data, uploaded_file],
        outputs=[output_table, gantt_chart]
    )

demo.launch(debug=True)
