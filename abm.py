import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# File used for saving persistent UI settings.
SETTINGS_FILE = "settings.json"

# Default values for parameters (excluding file paths).
DEFAULTS = {
    # Simulation Tab
    "Number of Agents": "200",
    "Simulation Duration (days)": "30",
    "Number of Runs": "1",
    "Random Seed": "42",
    
    # Initialization
    "Proportion Female": "0.5",
    "Sleep Method": "Normal",
    "Sleep Normal": "8.0, 1.0, 5.0, 10.0",  # hours: mean, std, min, max
    "Sleep Empirical Values": "5,6,7,8,9,10",  # in hours
    "Sleep Empirical Probabilities": "0.07,0.25,0.37,0.24,0.05,0.02",
    
    "SM Method": "Normal",
    "SM Normal": "60, 15, 15, 120",  # minutes: mean, std, min, max
    "SM Empirical Values": "15,30,45,60,75,90,105,120",  # in minutes
    "SM Empirical Probabilities": "0.02,0.05,0.2,0.35,0.2,0.1,0.05,0.03",
    
    "Household Rule Method": "Empirical",
    "Household Empirical Values": "0.0,0.25,0.5,0.75,1.0,1.25,-1",  # in hours; -1 means no restriction
    "Household Empirical Probabilities": "0.3,0.05,0.10,0.05,0.15,0.05,0.3",
    
    "Network Method": "Random",
    "Average Network Size": "5",
    
    # Model
    "Baseline SM Use (Male, hrs)": "2.0",
    "Baseline SM Use (Female, hrs)": "2.5",
    "Weight Social Norm": "0.5",
    "Day-of-Week Effect (hrs)": "1.0",
    "Global Baseline Sleep (hrs)": "8.0",
    "Sleep Effect Coefficient": "0.7",
    "Noise Range for SM (hrs)": "0.5",
    "Noise Range for Sleep (hrs)": "0.5"
}

# ==============================================================================
# Agent Class Definition
# ==============================================================================
class AdolescentAgent:
    def __init__(self, agent_id, sex, sleep_duration, night_sm, household_rule, baseline_sm):
        """
        Represents an adolescent agent with static and dynamic attributes.
        
        Static Attributes:
          - agent_id: Unique identifier (int)
          - sex: 'M' or 'F'
          - household_rule: Maximum allowed night-time SM use (hrs); if None, then no restriction.
          - baseline_sm: Baseline SM use (hrs) (set based on sex)
          
        Dynamic Attributes:
          - sleep_duration: Current sleep duration (hrs)
          - night_sm: Current night-time SM use (hrs)
          - sleep_history: List of sleep durations (hrs) over simulation days
          - sm_history: List of night-time SM use (hrs) over days
          - social_network: List of agent IDs representing friends
          
        Note: The initial state is stored in the attributes (sleep_duration and night_sm)
        but NOT in the history lists so that simulation history length equals the number
        of simulation days.
        """
        self.agent_id = agent_id
        self.sex = sex
        self.household_rule = household_rule
        self.baseline_sm = baseline_sm
        self.sleep_duration = sleep_duration
        self.night_sm = night_sm
        self.sleep_history = []  # History will be built during simulation
        self.sm_history = []
        self.social_network = []  # To be assigned later

# ==============================================================================
# Utility Functions for Sampling and File Loading
# ==============================================================================
def sample_from_normal(mean, std, min_val, max_val):
    """Sample a value from a normal distribution and clip it between min and max."""
    value = random.gauss(mean, std)
    return max(min_val, min(max_val, value))

def sample_empirical(value_list, prob_list):
    """Sample a value from an empirical distribution using provided probabilities."""
    return random.choices(value_list, weights=prob_list, k=1)[0]

def load_initialization_file(file_path):
    """
    Load agent initialization data from a CSV file.
    Expected columns (case-insensitive): id/agentid, sex, sleep_duration, social_media_use, household_restriction.
    Returns a tuple: (header_set, agents_data).
    """
    agents_data = []
    header_set = set()
    try:
        with open(file_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            if reader.fieldnames is not None:
                header_set = {h.lower() for h in reader.fieldnames}
            for row in reader:
                row = {k.lower(): v for k, v in row.items()}
                agents_data.append(row)
    except Exception as e:
        messagebox.showerror("File Error", f"Error reading initialization file:\n{e}")
    return header_set, agents_data

def load_social_network_file(file_path):
    """
    Load social network data from a CSV file.
    Supports formats:
      1. Columns: AgentID, Friends (semicolon-separated)
      2. Columns: ChildID, AlterID, Weight (weights ignored)
      3. Columns: agentid, friendid, weight (weights ignored)
    Returns a dict mapping agent_id to a list of friend IDs.
    """
    network = {}
    try:
        with open(file_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            fields = [f.lower() for f in reader.fieldnames]
            if "friends" in fields:
                for row in reader:
                    row = {k.lower(): v for k, v in row.items()}
                    try:
                        aid = int(row.get("agentid", row.get("id", 0)))
                    except:
                        continue
                    friends_str = row.get("friends", "")
                    friends = [int(f.strip()) for f in friends_str.split(";") if f.strip().isdigit()]
                    network[aid] = friends
            elif "childid" in fields and "alterid" in fields:
                for row in reader:
                    row = {k.lower(): v for k, v in row.items()}
                    try:
                        child_id = int(row["childid"])
                        alter_id = int(row["alterid"])
                    except:
                        continue
                    network.setdefault(child_id, []).append(alter_id)
            elif "agentid" in fields and "friendid" in fields:
                for row in reader:
                    row = {k.lower(): v for k, v in row.items()}
                    try:
                        aid = int(row["agentid"])
                        friend_id = int(row["friendid"])
                    except:
                        continue
                    network.setdefault(aid, []).append(friend_id)
            else:
                messagebox.showerror("File Error", "Social network file format not recognized.")
    except Exception as e:
        messagebox.showerror("File Error", f"Error reading social network file:\n{e}")
    return network

# ==============================================================================
# Agent Update Function (per day)
# ==============================================================================
def update_agent(agent, agents_dict, params, day_of_week):
    """
    Update an agent for one day.
    New SM use = baseline + weight*(social norm - baseline) + day effect + noise, capped by household rule.
    New sleep = global_baseline_sleep - (sleep_effect_coef * new SM) + noise.
    """
    if agent.social_network:
        friend_sm = [agents_dict.get(fid).night_sm for fid in agent.social_network if agents_dict.get(fid) is not None]
        norm = np.mean(friend_sm) if friend_sm else agent.baseline_sm
    else:
        norm = agent.baseline_sm

    new_sm = agent.baseline_sm + params['weight_social_norm'] * (norm - agent.baseline_sm)
    if day_of_week % 7 in [6, 0]:  # Saturday and Sunday
        new_sm += params['day_of_week_effect']
    new_sm += random.uniform(-params['noise_sm'], params['noise_sm'])
    if agent.household_rule is not None:
        new_sm = min(new_sm, agent.household_rule)
    new_sm = max(0.0, new_sm)
    
    new_sleep = params['global_baseline_sleep'] - params['sleep_effect_coef'] * new_sm
    new_sleep += random.uniform(-params['noise_sleep'], params['noise_sleep'])
    new_sleep = max(0.0, new_sleep)
    
    # Update state and append to history for this day.
    agent.night_sm = new_sm
    agent.sleep_duration = new_sleep
    agent.sm_history.append(new_sm)
    agent.sleep_history.append(new_sleep)

# ==============================================================================
# Social Network Assignment
# ==============================================================================
def assign_social_networks(agents, method, avg_network_size, network_file=""):
    """
    Assign social networks to agents.
    - "Random": assign a random set of friends.
    - "From File": load network from the specified CSV file.
    """
    num_agents = len(agents)
    if method == "From File":
        if not network_file or not os.path.exists(network_file):
            messagebox.showerror("File Error", "Please provide a valid social network CSV file.")
            return
        network_data = load_social_network_file(network_file)
        for agent in agents:
            agent.social_network = network_data.get(agent.agent_id, [])
    else:
        for agent in agents:
            network_size = random.randint(max(1, avg_network_size - 1), avg_network_size + 1)
            possible_friends = [other.agent_id for other in agents if other.agent_id != agent.agent_id]
            agent.social_network = random.sample(possible_friends, min(network_size, len(possible_friends))) if possible_friends else []

# ==============================================================================
# Single Simulation Run
# ==============================================================================
def run_simulation_run(params, init_options):
    """
    Run a single simulation over a number of days.
    Returns a dict with:
      - 'days': list of day numbers (starting from 1 to sim_duration)
      - 'pop_avg_sleep': list of daily population average sleep durations
      - 'pop_avg_sm': list of daily population average night-time SM use
      - 'agents': list of agents with individual histories (each with exactly sim_duration entries)
    Returns None if a required file is missing/invalid.
    """
    sim_days = params['sim_duration']
    agents = []
    
    if init_options['init_source'] == "From File":
        init_file = init_options['init_file']
        if not init_file or not os.path.exists(init_file):
            messagebox.showerror("File Error", "Please provide a valid initialization CSV file.")
            return None
        header_set, init_data = load_initialization_file(init_file)
        if not init_data:
            messagebox.showerror("File Error", "Initialization file is empty or in incorrect format.")
            return None

        if init_options['sleep_method'] == "From File" and "sleep_duration" not in header_set:
            messagebox.showerror("File Error", "Initialization file is missing 'sleep_duration' column.")
            return None
        if init_options['sm_method'] == "From File" and "social_media_use" not in header_set:
            messagebox.showerror("File Error", "Initialization file is missing 'social_media_use' column.")
            return None
        if init_options['household_rule_method'] == "From File" and "household_restriction" not in header_set:
            messagebox.showerror("File Error", "Initialization file is missing 'household_restriction' column.")
            return None

        params['num_agents'] = len(init_data)
        for row in init_data:
            try:
                aid = int(row.get("id", row.get("agentid", 0)))
                sex = row.get("sex", "").strip()
                if init_options['sleep_method'] == "From File":
                    sleep_val = float(row.get("sleep_duration", 8.0))
                else:
                    sleep_val = 8.0
                if init_options['sm_method'] == "From File":
                    sm_val = float(row.get("social_media_use", 0.0))
                else:
                    sm_val = 0.0
                if init_options['household_rule_method'] == "From File":
                    hr = row.get("household_restriction", "").strip()
                    hr_val = None if hr == "" or hr.lower() in ["no restriction", "none"] else float(hr)
                else:
                    hr_val = None
                base_sm = params['baseline_sm_female'] if sex.upper() == "F" else params['baseline_sm_male']
                agent = AdolescentAgent(aid, sex, sleep_val, sm_val, hr_val, base_sm)
                agents.append(agent)
            except Exception as e:
                print(f"Error parsing row {row}: {e}")
                return None
    else:
        num_agents = params['num_agents']
        prop_female = init_options['prop_female']
        sleep_method = init_options['sleep_method']
        sm_method = init_options['sm_method']
        for i in range(num_agents):
            aid = i
            sex = "F" if random.random() < prop_female else "M"
            base_sm = params['baseline_sm_female'] if sex == "F" else params['baseline_sm_male']
            if sleep_method == "Normal":
                sleep_val = sample_from_normal(
                    init_options['sleep_normal_mean'],
                    init_options['sleep_normal_std'],
                    init_options['sleep_normal_min'],
                    init_options['sleep_normal_max']
                )
            elif sleep_method == "Empirical":
                if not init_options.get('sleep_empirical_values') or not init_options.get('sleep_empirical_probs'):
                    messagebox.showerror("Parameter Error", "Please provide empirical values and probabilities for sleep duration.")
                    return None
                try:
                    sleep_values = [float(x.strip()) for x in init_options['sleep_empirical_values'].split(",")]
                    sleep_probs = [float(x.strip()) for x in init_options['sleep_empirical_probs'].split(",")]
                except Exception as e:
                    messagebox.showerror("Parameter Error", f"Error parsing empirical sleep parameters:\n{e}")
                    return None
                sleep_val = sample_empirical(sleep_values, sleep_probs)
            else:
                sleep_val = 8.0

            if sm_method == "Normal":
                sm_minutes = sample_from_normal(
                    init_options['sm_normal_mean'],
                    init_options['sm_normal_std'],
                    init_options['sm_normal_min'],
                    init_options['sm_normal_max']
                )
                sm_val = sm_minutes / 60.0
            elif sm_method == "Empirical":
                if not init_options.get('sm_empirical_values') or not init_options.get('sm_empirical_probs'):
                    messagebox.showerror("Parameter Error", "Please provide empirical values and probabilities for night-time SM use.")
                    return None
                try:
                    sm_values = [float(x.strip()) for x in init_options['sm_empirical_values'].split(",")]
                    sm_probs = [float(x.strip()) for x in init_options['sm_empirical_probs'].split(",")]
                except Exception as e:
                    messagebox.showerror("Parameter Error", f"Error parsing empirical SM parameters:\n{e}")
                    return None
                sm_minutes = sample_empirical(sm_values, sm_probs)
                sm_val = sm_minutes / 60.0
            else:
                sm_val = 0.0

            if init_options['household_rule_method'] == "Empirical":
                if not init_options.get('household_empirical_values') or not init_options.get('household_empirical_probs'):
                    messagebox.showerror("Parameter Error", "Please provide empirical values and probabilities for household rules.")
                    return None
                try:
                    hr_values_raw = [x.strip() for x in init_options['household_empirical_values'].split(",")]
                    hr_values = [None if x.lower() in ["-1", "none"] else float(x) for x in hr_values_raw]
                    hr_probs = [float(x.strip()) for x in init_options['household_empirical_probs'].split(",")]
                except Exception as e:
                    messagebox.showerror("Parameter Error", f"Error parsing empirical household rule parameters:\n{e}")
                    return None
                chosen_hr = sample_empirical(hr_values, hr_probs)
                hr_val = chosen_hr
            else:
                hr_val = None
            
            agent = AdolescentAgent(aid, sex, sleep_val, sm_val, hr_val, base_sm)
            agents.append(agent)
    
    assign_social_networks(agents, init_options['network_method'],
                           init_options['avg_network_size'], init_options.get('network_file', ""))
    
    # Use exactly sim_days iterations (days 1 to sim_days).
    days = list(range(1, sim_days + 1))
    pop_avg_sleep = []
    pop_avg_sm = []
    for day in days:
        day_of_week = day % 7
        agents_dict = {agent.agent_id: agent for agent in agents}
        for agent in agents:
            update_agent(agent, agents_dict, params, day_of_week)
        pop_avg_sleep.append(np.mean([a.sleep_duration for a in agents]))
        pop_avg_sm.append(np.mean([a.night_sm for a in agents]))
    
    return {
        'days': days,
        'pop_avg_sleep': pop_avg_sleep,
        'pop_avg_sm': pop_avg_sm,
        'agents': agents
    }

# ==============================================================================
# Multiple Simulation Runs and Averaging
# ==============================================================================
def run_multiple_simulations(num_runs, params, init_options):
    results = []
    base_seed = params['random_seed']
    for r in range(num_runs):
        random.seed(base_seed + r)
        np.random.seed(base_seed + r)
        res = run_simulation_run(params, init_options)
        if res is None:
            return None
        results.append(res)
    return results

def average_simulation_results(sim_runs):
    days = sim_runs[0]['days']
    avg_sleep = np.mean([run['pop_avg_sleep'] for run in sim_runs], axis=0)
    avg_sm = np.mean([run['pop_avg_sm'] for run in sim_runs], axis=0)
    return days, avg_sleep, avg_sm

# ==============================================================================
# CSV Output Functions for Per-Agent and Per-Run Data
# ==============================================================================
def save_per_agent_csv(sim_runs, file_path):
    """
    Save per-agent data across runs.
    Writes columns: Run, AgentID, Day, SleepDuration, NightSMUse.
    """
    try:
        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Run", "AgentID", "Day", "SleepDuration", "NightSMUse"])
            for run_index, result in enumerate(sim_runs, start=1):
                for agent in result['agents']:
                    for day, (sleep_val, sm_val) in enumerate(zip(agent.sleep_history, agent.sm_history), start=1):
                        writer.writerow([run_index, agent.agent_id, day, sleep_val, sm_val])
    except Exception as e:
        messagebox.showerror("File Error", f"Error saving per-agent output CSV:\n{e}")

def save_per_run_csv(sim_runs, file_path):
    """
    Save per-run (population average) data.
    Writes columns: Run, Day, AvgSleepDuration, AvgNightSMUse.
    """
    try:
        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Run", "Day", "AvgSleepDuration", "AvgNightSMUse"])
            for run_index, run in enumerate(sim_runs, start=1):
                for day, sleep_val, sm_val in zip(run['days'], run['pop_avg_sleep'], run['pop_avg_sm']):
                    writer.writerow([run_index, day, sleep_val, sm_val])
    except Exception as e:
        messagebox.showerror("File Error", f"Error saving per-run output CSV:\n{e}")

# ==============================================================================
# Plotting Functions
# ==============================================================================
def plot_simulation_results(result, title_suffix=""):
    days = result['days']
    pop_avg_sleep = result['pop_avg_sleep']
    pop_avg_sm = result['pop_avg_sm']
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(days, pop_avg_sleep, marker='o')
    plt.xlabel("Day")
    plt.ylabel("Average Sleep Duration (hrs)")
    plt.title("Sleep Duration " + title_suffix)
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(days, pop_avg_sm, marker='o')
    plt.xlabel("Day")
    plt.ylabel("Average Night-time SM Use (hrs)")
    plt.title("Social Media Use " + title_suffix)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_agent_results(agent, title_suffix=""):
    days = list(range(1, len(agent.sleep_history) + 1))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(days, agent.sleep_history, marker='o')
    plt.xlabel("Day")
    plt.ylabel("Sleep Duration (hrs)")
    plt.title(f"Agent {agent.agent_id} Sleep Duration " + title_suffix)
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(days, agent.sm_history, marker='o')
    plt.xlabel("Day")
    plt.ylabel("Night-time SM Use (hrs)")
    plt.title(f"Agent {agent.agent_id} Night-time SM Use " + title_suffix)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ==============================================================================
# Persistent Settings Functions
# ==============================================================================
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
            # Simulation Tab
            if "Number of Agents" in settings:
                basic_entries["Number of Agents"].delete(0, tk.END)
                basic_entries["Number of Agents"].insert(0, settings["Number of Agents"])
            if "Simulation Duration (days)" in settings:
                basic_entries["Simulation Duration (days)"].delete(0, tk.END)
                basic_entries["Simulation Duration (days)"].insert(0, settings["Simulation Duration (days)"])
            if "Number of Runs" in settings:
                basic_entries["Number of Runs"].delete(0, tk.END)
                basic_entries["Number of Runs"].insert(0, settings["Number of Runs"])
            if "Random Seed" in settings:
                basic_entries["Random Seed"].delete(0, tk.END)
                basic_entries["Random Seed"].insert(0, settings["Random Seed"])
            # Initialization Tab
            if "Initialization File Path" in settings:
                init_file_entry.delete(0, tk.END)
                init_file_entry.insert(0, settings["Initialization File Path"])
            if "Proportion Female" in settings:
                prop_female_entry.delete(0, tk.END)
                prop_female_entry.insert(0, settings["Proportion Female"])
            if "Sleep (Normal)" in settings:
                sleep_normal_entry.delete(0, tk.END)
                sleep_normal_entry.insert(0, settings["Sleep (Normal)"])
            if "Sleep (Empirical) - Values" in settings:
                sleep_empirical_entry.delete(0, tk.END)
                sleep_empirical_entry.insert(0, settings["Sleep (Empirical) - Values"])
            if "Sleep (Empirical) - Probabilities" in settings:
                sleep_empirical_probs_entry.delete(0, tk.END)
                sleep_empirical_probs_entry.insert(0, settings["Sleep (Empirical) - Probabilities"])
            if "Night SM (Normal)" in settings:
                sm_normal_entry.delete(0, tk.END)
                sm_normal_entry.insert(0, settings["Night SM (Normal)"])
            if "Night SM (Empirical) - Values" in settings:
                sm_empirical_entry.delete(0, tk.END)
                sm_empirical_entry.insert(0, settings["Night SM (Empirical) - Values"])
            if "Night SM (Empirical) - Probabilities" in settings:
                sm_empirical_probs_entry.delete(0, tk.END)
                sm_empirical_probs_entry.insert(0, settings["Night SM (Empirical) - Probabilities"])
            if "Household Rule (Empirical) - Values" in settings:
                household_empirical_entry.delete(0, tk.END)
                household_empirical_entry.insert(0, settings["Household Rule (Empirical) - Values"])
            if "Household Rule (Empirical) - Probabilities" in settings:
                household_empirical_probs_entry.delete(0, tk.END)
                household_empirical_probs_entry.insert(0, settings["Household Rule (Empirical) - Probabilities"])
            if "Social Network File Path" in settings:
                network_file_entry.delete(0, tk.END)
                network_file_entry.insert(0, settings["Social Network File Path"])
            if "Average Network Size" in settings:
                avg_network_entry.delete(0, tk.END)
                avg_network_entry.insert(0, settings["Average Network Size"])
            # Output Options Tab
            if "Specific Agent ID" in settings:
                specific_agent_entry.delete(0, tk.END)
                specific_agent_entry.insert(0, settings["Specific Agent ID"])
            if "Specific Run Number" in settings:
                specific_run_entry.delete(0, tk.END)
                specific_run_entry.insert(0, settings["Specific Run Number"])
            if "Per Agent Output CSV File Path" in settings:
                per_agent_output_entry.delete(0, tk.END)
                per_agent_output_entry.insert(0, settings["Per Agent Output CSV File Path"])
            if "Per Run Output CSV File Path" in settings:
                per_run_output_entry.delete(0, tk.END)
                per_run_output_entry.insert(0, settings["Per Run Output CSV File Path"])
        except Exception as e:
            print("Error loading settings:", e)

def save_settings():
    settings = {}
    # Simulation Tab
    settings["Number of Agents"] = basic_entries["Number of Agents"].get()
    settings["Simulation Duration (days)"] = basic_entries["Simulation Duration (days)"].get()
    settings["Number of Runs"] = basic_entries["Number of Runs"].get()
    settings["Random Seed"] = basic_entries["Random Seed"].get()
    # Initialization Tab
    settings["Initialization File Path"] = init_file_entry.get()
    settings["Proportion Female"] = prop_female_entry.get()
    settings["Sleep (Normal)"] = sleep_normal_entry.get()
    settings["Sleep (Empirical) - Values"] = sleep_empirical_entry.get()
    settings["Sleep (Empirical) - Probabilities"] = sleep_empirical_probs_entry.get()
    settings["Night SM (Normal)"] = sm_normal_entry.get()
    settings["Night SM (Empirical) - Values"] = sm_empirical_entry.get()
    settings["Night SM (Empirical) - Probabilities"] = sm_empirical_probs_entry.get()
    settings["Household Rule (Empirical) - Values"] = household_empirical_entry.get()
    settings["Household Rule (Empirical) - Probabilities"] = household_empirical_probs_entry.get()
    settings["Social Network File Path"] = network_file_entry.get()
    settings["Average Network Size"] = avg_network_entry.get()
    # Output Options Tab
    settings["Specific Agent ID"] = specific_agent_entry.get()
    settings["Specific Run Number"] = specific_run_entry.get()
    settings["Per Agent Output CSV File Path"] = per_agent_output_entry.get()
    settings["Per Run Output CSV File Path"] = per_run_output_entry.get()
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f)
    except Exception as e:
        print("Error saving settings:", e)

# ==============================================================================
# Tkinter UI Setup
# ==============================================================================
root = tk.Tk()
root.title("Adolescent Social Media & Sleep ABM")

notebook = ttk.Notebook(root)
notebook.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# --------------------------
# Tab 1: Simulation Parameters
# --------------------------
basic_frame = ttk.Frame(notebook, padding="10")
notebook.add(basic_frame, text="Simulation")

basic_params = [
    ("Number of Agents", DEFAULTS["Number of Agents"]),
    ("Simulation Duration (days)", DEFAULTS["Simulation Duration (days)"]),
    ("Number of Runs", DEFAULTS["Number of Runs"]),
    ("Random Seed", DEFAULTS["Random Seed"])
]
basic_entries = {}
row = 0
for label_text, default in basic_params:
    ttk.Label(basic_frame, text=label_text).grid(row=row, column=0, sticky="w", pady=2)
    ent = ttk.Entry(basic_frame, width=25)
    ent.insert(0, default)
    ent.grid(row=row, column=1, pady=2)
    basic_entries[label_text] = ent
    row += 1

# --------------------------
# Tab 2: Initialization Parameters
# --------------------------
init_frame = ttk.Frame(notebook, padding="10")
notebook.add(init_frame, text="Initialization")

r = 0
ttk.Label(init_frame, text="Initialization Source:", font=('Helvetica', 10, 'bold')).grid(row=r, column=0, sticky="w", pady=(0,5))
init_source_var = tk.StringVar(value="Random")
init_source_menu = ttk.OptionMenu(init_frame, init_source_var, "Random", "Random", "From File")
init_source_menu.grid(row=r, column=1, sticky="w", pady=(0,5))
r += 1
ttk.Label(init_frame, text="Initialization File Path:").grid(row=r, column=0, sticky="w", pady=2)
init_file_entry = ttk.Entry(init_frame, width=25)
init_file_entry.insert(0, "")
init_file_entry.grid(row=r, column=1, pady=2)
r += 1
ttk.Label(init_frame, text="Proportion Female (0-1) [fraction]:").grid(row=r, column=0, sticky="w", pady=2)
prop_female_entry = ttk.Entry(init_frame, width=25)
prop_female_entry.insert(0, DEFAULTS["Proportion Female"])
prop_female_entry.grid(row=r, column=1, pady=2)
r += 1

# Group for Sleep Options
sleep_label = ttk.LabelFrame(init_frame, text="Sleep Options")
sleep_label.grid(row=r, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
r_sleep = 0
ttk.Label(sleep_label, text="Sleep Duration Init Method:").grid(row=r_sleep, column=0, sticky="w", pady=2)
sleep_method_var = tk.StringVar(value=DEFAULTS["Sleep Method"])
sleep_method_menu = ttk.OptionMenu(sleep_label, sleep_method_var, "Normal", "Normal", "Empirical", "From File")
sleep_method_menu.grid(row=r_sleep, column=1, sticky="w", pady=2)
r_sleep += 1
ttk.Label(sleep_label, text="(Normal) Mean, Std, Min, Max [hours]:").grid(row=r_sleep, column=0, sticky="w", pady=2)
sleep_normal_entry = ttk.Entry(sleep_label, width=25)
sleep_normal_entry.insert(0, DEFAULTS["Sleep Normal"])
sleep_normal_entry.grid(row=r_sleep, column=1, pady=2)
r_sleep += 1
ttk.Label(sleep_label, text="(Empirical) Values [hours]:").grid(row=r_sleep, column=0, sticky="w", pady=2)
sleep_empirical_entry = ttk.Entry(sleep_label, width=25)
sleep_empirical_entry.insert(0, DEFAULTS["Sleep Empirical Values"])
sleep_empirical_entry.grid(row=r_sleep, column=1, pady=2)
r_sleep += 1
ttk.Label(sleep_label, text="(Empirical) Probabilities:").grid(row=r_sleep, column=0, sticky="w", pady=2)
sleep_empirical_probs_entry = ttk.Entry(sleep_label, width=25)
sleep_empirical_probs_entry.insert(0, DEFAULTS["Sleep Empirical Probabilities"])
sleep_empirical_probs_entry.grid(row=r_sleep, column=1, pady=2)
r_sleep += 1
r += 1

# Group for Night SM Options
sm_label = ttk.LabelFrame(init_frame, text="Night SM Options")
sm_label.grid(row=r, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
r_sm = 0
ttk.Label(sm_label, text="Night SM Use Init Method:").grid(row=r_sm, column=0, sticky="w", pady=2)
sm_method_var = tk.StringVar(value=DEFAULTS["SM Method"])
sm_method_menu = ttk.OptionMenu(sm_label, sm_method_var, "Normal", "Normal", "Empirical", "From File")
sm_method_menu.grid(row=r_sm, column=1, sticky="w", pady=2)
r_sm += 1
ttk.Label(sm_label, text="(Normal) Mean, Std, Min, Max [minutes]:").grid(row=r_sm, column=0, sticky="w", pady=2)
sm_normal_entry = ttk.Entry(sm_label, width=25)
sm_normal_entry.insert(0, DEFAULTS["SM Normal"])
sm_normal_entry.grid(row=r_sm, column=1, pady=2)
r_sm += 1
ttk.Label(sm_label, text="(Empirical) Values [minutes]:").grid(row=r_sm, column=0, sticky="w", pady=2)
sm_empirical_entry = ttk.Entry(sm_label, width=25)
sm_empirical_entry.insert(0, DEFAULTS["SM Empirical Values"])
sm_empirical_entry.grid(row=r_sm, column=1, pady=2)
r_sm += 1
ttk.Label(sm_label, text="(Empirical) Probabilities:").grid(row=r_sm, column=0, sticky="w", pady=2)
sm_empirical_probs_entry = ttk.Entry(sm_label, width=25)
sm_empirical_probs_entry.insert(0, DEFAULTS["SM Empirical Probabilities"])
sm_empirical_probs_entry.grid(row=r_sm, column=1, pady=2)
r_sm += 1
r += 1

# Group for Household Rule Options
household_label = ttk.LabelFrame(init_frame, text="Household Rule Options")
household_label.grid(row=r, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
r_hr = 0
ttk.Label(household_label, text="Household Restriction Init Method:").grid(row=r_hr, column=0, sticky="w", pady=2)
hr_method_var = tk.StringVar(value=DEFAULTS["Household Rule Method"])
hr_method_menu = ttk.OptionMenu(household_label, hr_method_var, "Empirical", "Empirical", "From File")
hr_method_menu.grid(row=r_hr, column=1, sticky="w", pady=2)
r_hr += 1
ttk.Label(household_label, text="(Empirical) Values [hours, -1=No Restriction]:").grid(row=r_hr, column=0, sticky="w", pady=2)
household_empirical_entry = ttk.Entry(household_label, width=25)
household_empirical_entry.insert(0, DEFAULTS["Household Empirical Values"])
household_empirical_entry.grid(row=r_hr, column=1, pady=2)
r_hr += 1
ttk.Label(household_label, text="(Empirical) Probabilities:").grid(row=r_hr, column=0, sticky="w", pady=2)
household_empirical_probs_entry = ttk.Entry(household_label, width=25)
household_empirical_probs_entry.insert(0, DEFAULTS["Household Empirical Probabilities"])
household_empirical_probs_entry.grid(row=r_hr, column=1, pady=2)
r_hr += 1
r += 1

# Group for Social Network Options
network_label = ttk.LabelFrame(init_frame, text="Social Network Options")
network_label.grid(row=r, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
r_nw = 0
ttk.Label(network_label, text="Social Network Init Method:").grid(row=r_nw, column=0, sticky="w", pady=2)
network_method_var = tk.StringVar(value=DEFAULTS["Network Method"])
network_method_menu = ttk.OptionMenu(network_label, network_method_var, "Random", "Random", "From File")
network_method_menu.grid(row=r_nw, column=1, sticky="w", pady=2)
r_nw += 1
ttk.Label(network_label, text="Social Network File Path:").grid(row=r_nw, column=0, sticky="w", pady=2)
network_file_entry = ttk.Entry(network_label, width=25)
network_file_entry.insert(0, "")
network_file_entry.grid(row=r_nw, column=1, pady=2)
r_nw += 1
ttk.Label(network_label, text="Average Network Size (if Random) [#]:").grid(row=r_nw, column=0, sticky="w", pady=2)
avg_network_entry = ttk.Entry(network_label, width=25)
avg_network_entry.insert(0, DEFAULTS["Average Network Size"])
avg_network_entry.grid(row=r_nw, column=1, pady=2)
r_nw += 1
r += 1

# --------------------------
# Tab 3: Model Parameters
# --------------------------
model_frame = ttk.Frame(notebook, padding="10")
notebook.add(model_frame, text="Model Parameters")

model_params = [
    ("Baseline SM Use (Male, hrs)", DEFAULTS["Baseline SM Use (Male, hrs)"]),
    ("Baseline SM Use (Female, hrs)", DEFAULTS["Baseline SM Use (Female, hrs)"]),
    ("Weight Social Norm", DEFAULTS["Weight Social Norm"]),
    ("Day-of-Week Effect (hrs)", DEFAULTS["Day-of-Week Effect (hrs)"]),
    ("Global Baseline Sleep (hrs)", DEFAULTS["Global Baseline Sleep (hrs)"]),
    ("Sleep Effect Coefficient", DEFAULTS["Sleep Effect Coefficient"]),
    ("Noise Range for SM (hrs)", DEFAULTS["Noise Range for SM (hrs)"]),
    ("Noise Range for Sleep (hrs)", DEFAULTS["Noise Range for Sleep (hrs)"])
]
model_entries = {}
row = 0
for label_text, default in model_params:
    ttk.Label(model_frame, text=label_text).grid(row=row, column=0, sticky="w", pady=2)
    ent = ttk.Entry(model_frame, width=25)
    ent.insert(0, default)
    ent.grid(row=row, column=1, pady=2)
    model_entries[label_text] = ent
    row += 1

# --------------------------
# Tab 4: Output Options & Graphics
# --------------------------
output_frame = ttk.Frame(notebook, padding="10")
notebook.add(output_frame, text="Output Options")

ttk.Label(output_frame, text="Output Mode:", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, sticky="w", pady=2)
output_mode_var = tk.StringVar(value="Single")
output_mode_menu = ttk.OptionMenu(output_frame, output_mode_var, "Single", "Single", "Multiple", "Average")
output_mode_menu.grid(row=0, column=1, sticky="w", pady=2)

ttk.Label(output_frame, text="Specific Agent ID (optional) [integer]:").grid(row=1, column=0, sticky="w", pady=2)
specific_agent_entry = ttk.Entry(output_frame, width=25)
specific_agent_entry.insert(0, "")
specific_agent_entry.grid(row=1, column=1, pady=2)

ttk.Label(output_frame, text="Specific Run Number (optional) [integer]:").grid(row=2, column=0, sticky="w", pady=2)
specific_run_entry = ttk.Entry(output_frame, width=25)
specific_run_entry.insert(0, "")
specific_run_entry.grid(row=2, column=1, pady=2)

ttk.Label(output_frame, text="Per Agent Output CSV File Path:").grid(row=3, column=0, sticky="w", pady=2)
per_agent_output_entry = ttk.Entry(output_frame, width=25)
per_agent_output_entry.insert(0, "")
per_agent_output_entry.grid(row=3, column=1, pady=2)

ttk.Label(output_frame, text="Per Run Output CSV File Path:").grid(row=4, column=0, sticky="w", pady=2)
per_run_output_entry = ttk.Entry(output_frame, width=25)
per_run_output_entry.insert(0, "")
per_run_output_entry.grid(row=4, column=1, pady=2)

# ==============================================================================
# Button Handlers
# ==============================================================================
def run_simulation_handler():
    """
    Reads UI inputs, validates file paths and empirical parameters,
    enforces that 'From File' attributes require 'From File' init,
    runs the simulation(s) for exactly the entered number of days,
    displays plots, and generates both CSV outputs.
    """
    try:
        num_agents = int(basic_entries["Number of Agents"].get())
        sim_duration = int(basic_entries["Simulation Duration (days)"].get())
        num_runs = int(basic_entries["Number of Runs"].get())
        random_seed_val = int(basic_entries["Random Seed"].get())
    except Exception as e:
        messagebox.showerror("Parameter Error", f"Error reading simulation parameters:\n{e}")
        return

    try:
        params = {
            'sim_duration': sim_duration,
            'num_agents': num_agents,
            'random_seed': random_seed_val,
            'baseline_sm_male': float(model_entries["Baseline SM Use (Male, hrs)"].get()),
            'baseline_sm_female': float(model_entries["Baseline SM Use (Female, hrs)"].get()),
            'weight_social_norm': float(model_entries["Weight Social Norm"].get()),
            'day_of_week_effect': float(model_entries["Day-of-Week Effect (hrs)"].get()),
            'global_baseline_sleep': float(model_entries["Global Baseline Sleep (hrs)"].get()),
            'sleep_effect_coef': float(model_entries["Sleep Effect Coefficient"].get()),
            'noise_sm': float(model_entries["Noise Range for SM (hrs)"].get()),
            'noise_sleep': float(model_entries["Noise Range for Sleep (hrs)"].get())
        }
    except Exception as e:
        messagebox.showerror("Parameter Error", f"Error reading model parameters:\n{e}")
        return

    init_options = {}
    init_options['init_source'] = init_source_var.get()
    init_options['init_file'] = init_file_entry.get().strip()
    try:
        init_options['prop_female'] = float(prop_female_entry.get())
    except Exception as e:
        messagebox.showerror("Parameter Error", f"Error reading proportion female:\n{e}")
        return

    init_options['sleep_method'] = sleep_method_var.get()
    init_options['sm_method'] = sm_method_var.get()
    init_options['household_rule_method'] = hr_method_var.get()

    from_file_attrs = []
    if init_options['sleep_method'] == "From File":
        from_file_attrs.append("sleep_duration")
    if init_options['sm_method'] == "From File":
        from_file_attrs.append("social_media_use")
    if init_options['household_rule_method'] == "From File":
        from_file_attrs.append("household_restriction")
    if from_file_attrs:
        if init_options['init_source'] != "From File":
            messagebox.showerror("Parameter Error",
                "Selected 'From File' for an attribute, but Initialization Source is not 'From File'.\n"
                "Please set Initialization Source to 'From File' and provide a valid CSV."
            )
            return
        if not init_options['init_file'] or not os.path.exists(init_options['init_file']):
            messagebox.showerror("File Error", "Provide a valid initialization CSV file if attributes are 'From File'.")
            return

    if init_options['sleep_method'] == "Normal":
        try:
            sleep_params = [float(x.strip()) for x in sleep_normal_entry.get().split(",")]
            init_options['sleep_normal_mean'] = sleep_params[0]
            init_options['sleep_normal_std'] = sleep_params[1]
            init_options['sleep_normal_min'] = sleep_params[2]
            init_options['sleep_normal_max'] = sleep_params[3]
        except Exception as e:
            messagebox.showerror("Parameter Error", f"Error reading sleep normal parameters:\n{e}")
            return
    elif init_options['sleep_method'] == "Empirical":
        init_options['sleep_empirical_values'] = sleep_empirical_entry.get()
        init_options['sleep_empirical_probs'] = sleep_empirical_probs_entry.get()

    if init_options['sm_method'] == "Normal":
        try:
            sm_params = [float(x.strip()) for x in sm_normal_entry.get().split(",")]
            init_options['sm_normal_mean'] = sm_params[0]
            init_options['sm_normal_std'] = sm_params[1]
            init_options['sm_normal_min'] = sm_params[2]
            init_options['sm_normal_max'] = sm_params[3]
        except Exception as e:
            messagebox.showerror("Parameter Error", f"Error reading SM normal parameters:\n{e}")
            return
    elif init_options['sm_method'] == "Empirical":
        init_options['sm_empirical_values'] = sm_empirical_entry.get()
        init_options['sm_empirical_probs'] = sm_empirical_probs_entry.get()

    if init_options['household_rule_method'] == "Empirical":
        init_options['household_empirical_values'] = household_empirical_entry.get()
        init_options['household_empirical_probs'] = household_empirical_probs_entry.get()

    init_options['network_method'] = network_method_var.get()
    init_options['network_file'] = network_file_entry.get().strip()
    if init_options['network_method'] == "From File":
        if not init_options['network_file'] or not os.path.exists(init_options['network_file']):
            messagebox.showerror("File Error", "Provide a valid social network CSV file.")
            return

    try:
        init_options['avg_network_size'] = int(avg_network_entry.get())
    except Exception as e:
        messagebox.showerror("Parameter Error", f"Error reading average network size:\n{e}")
        return

    specific_agent_str = specific_agent_entry.get().strip()
    specific_agent = int(specific_agent_str) if specific_agent_str != "" else None
    specific_run_str = specific_run_entry.get().strip()
    specific_run = int(specific_run_str) if specific_run_str != "" else None

    per_agent_output = per_agent_output_entry.get().strip()
    per_run_output = per_run_output_entry.get().strip()

    random.seed(random_seed_val)
    np.random.seed(random_seed_val)

    output_mode = output_mode_var.get()

    if num_runs > 1:
        sim_runs = run_multiple_simulations(num_runs, params, init_options)
    else:
        result = run_simulation_run(params, init_options)
        if result is None:
            return
        sim_runs = [result]

    if sim_runs is None:
        return

    if output_mode == "Single":
        res = sim_runs[0]
        if specific_agent is not None:
            agent_found = next((a for a in res['agents'] if a.agent_id == specific_agent), None)
            if agent_found is not None:
                plot_agent_results(agent_found, title_suffix="(Single Run)")
            else:
                messagebox.showerror("Agent Not Found", f"Agent {specific_agent} not found.")
                return
        else:
            plot_simulation_results(res, title_suffix="(Population Average - Single Run)")
    elif output_mode == "Multiple":
        if specific_run is not None and 1 <= specific_run <= len(sim_runs):
            selected_run = sim_runs[specific_run - 1]
            if specific_agent is not None:
                agent_found = next((a for a in selected_run['agents'] if a.agent_id == specific_agent), None)
                if agent_found is not None:
                    plot_agent_results(agent_found, title_suffix=f"(Run {specific_run})")
                else:
                    messagebox.showerror("Agent Not Found", f"Agent {specific_agent} not found in run {specific_run}.")
                    return
            else:
                plot_simulation_results(selected_run, title_suffix=f"(Run {specific_run} - Population Average)")
        else:
            plt.figure(figsize=(12, 5))
            plt.subplot(1,2,1)
            for r in sim_runs:
                plt.plot(r['days'], r['pop_avg_sleep'], alpha=0.5)
            plt.xlabel("Day")
            plt.ylabel("Avg Sleep Duration (hrs)")
            plt.title("Sleep Duration (Multiple Runs)")
            plt.grid(True)
            plt.subplot(1,2,2)
            for r in sim_runs:
                plt.plot(r['days'], r['pop_avg_sm'], alpha=0.5)
            plt.xlabel("Day")
            plt.ylabel("Avg Night SM Use (hrs)")
            plt.title("Night SM Use (Multiple Runs)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    elif output_mode == "Average":
        days, avg_sleep, avg_sm = average_simulation_results(sim_runs)
        plt.figure(figsize=(12, 5))
        plt.subplot(1,2,1)
        plt.plot(days, avg_sleep, marker='o', color='blue')
        plt.xlabel("Day")
        plt.ylabel("Avg Sleep Duration (hrs)")
        plt.title("Sleep Duration (Average over Runs)")
        plt.grid(True)
        plt.subplot(1,2,2)
        plt.plot(days, avg_sm, marker='o', color='red')
        plt.xlabel("Day")
        plt.ylabel("Avg Night SM Use (hrs)")
        plt.title("Night SM Use (Average over Runs)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        messagebox.showerror("Output Mode Error", "Invalid output mode selected.")
    
    if per_agent_output != "":
        save_per_agent_csv(sim_runs, per_agent_output)
    if per_run_output != "":
        save_per_run_csv(sim_runs, per_run_output)
    
    save_settings()

def reset_to_defaults():
    """
    Revert all parameter values to default ones, except file paths.
    """
    basic_entries["Number of Agents"].delete(0, tk.END)
    basic_entries["Number of Agents"].insert(0, DEFAULTS["Number of Agents"])
    basic_entries["Simulation Duration (days)"].delete(0, tk.END)
    basic_entries["Simulation Duration (days)"].insert(0, DEFAULTS["Simulation Duration (days)"])
    basic_entries["Number of Runs"].delete(0, tk.END)
    basic_entries["Number of Runs"].insert(0, DEFAULTS["Number of Runs"])
    basic_entries["Random Seed"].delete(0, tk.END)
    basic_entries["Random Seed"].insert(0, DEFAULTS["Random Seed"])
    
    prop_female_entry.delete(0, tk.END)
    prop_female_entry.insert(0, DEFAULTS["Proportion Female"])
    
    sleep_method_var.set(DEFAULTS["Sleep Method"])
    sleep_normal_entry.delete(0, tk.END)
    sleep_normal_entry.insert(0, DEFAULTS["Sleep Normal"])
    sleep_empirical_entry.delete(0, tk.END)
    sleep_empirical_entry.insert(0, DEFAULTS["Sleep Empirical Values"])
    sleep_empirical_probs_entry.delete(0, tk.END)
    sleep_empirical_probs_entry.insert(0, DEFAULTS["Sleep Empirical Probabilities"])
    
    sm_method_var.set(DEFAULTS["SM Method"])
    sm_normal_entry.delete(0, tk.END)
    sm_normal_entry.insert(0, DEFAULTS["SM Normal"])
    sm_empirical_entry.delete(0, tk.END)
    sm_empirical_entry.insert(0, DEFAULTS["SM Empirical Values"])
    sm_empirical_probs_entry.delete(0, tk.END)
    sm_empirical_probs_entry.insert(0, DEFAULTS["SM Empirical Probabilities"])
    
    hr_method_var.set(DEFAULTS["Household Rule Method"])
    household_empirical_entry.delete(0, tk.END)
    household_empirical_entry.insert(0, DEFAULTS["Household Empirical Values"])
    household_empirical_probs_entry.delete(0, tk.END)
    household_empirical_probs_entry.insert(0, DEFAULTS["Household Empirical Probabilities"])
    
    network_method_var.set(DEFAULTS["Network Method"])
    avg_network_entry.delete(0, tk.END)
    avg_network_entry.insert(0, DEFAULTS["Average Network Size"])
    
    model_entries["Baseline SM Use (Male, hrs)"].delete(0, tk.END)
    model_entries["Baseline SM Use (Male, hrs)"].insert(0, DEFAULTS["Baseline SM Use (Male, hrs)"])
    model_entries["Baseline SM Use (Female, hrs)"].delete(0, tk.END)
    model_entries["Baseline SM Use (Female, hrs)"].insert(0, DEFAULTS["Baseline SM Use (Female, hrs)"])
    model_entries["Weight Social Norm"].delete(0, tk.END)
    model_entries["Weight Social Norm"].insert(0, DEFAULTS["Weight Social Norm"])
    model_entries["Day-of-Week Effect (hrs)"].delete(0, tk.END)
    model_entries["Day-of-Week Effect (hrs)"].insert(0, DEFAULTS["Day-of-Week Effect (hrs)"])
    model_entries["Global Baseline Sleep (hrs)"].delete(0, tk.END)
    model_entries["Global Baseline Sleep (hrs)"].insert(0, DEFAULTS["Global Baseline Sleep (hrs)"])
    model_entries["Sleep Effect Coefficient"].delete(0, tk.END)
    model_entries["Sleep Effect Coefficient"].insert(0, DEFAULTS["Sleep Effect Coefficient"])
    model_entries["Noise Range for SM (hrs)"].delete(0, tk.END)
    model_entries["Noise Range for SM (hrs)"].insert(0, DEFAULTS["Noise Range for SM (hrs)"])
    model_entries["Noise Range for Sleep (hrs)"].delete(0, tk.END)
    model_entries["Noise Range for Sleep (hrs)"].insert(0, DEFAULTS["Noise Range for Sleep (hrs)"])

# ==============================================================================
# Buttons
# ==============================================================================
run_button = ttk.Button(root, text="Run Simulation", command=run_simulation_handler)
run_button.grid(row=1, column=0, pady=5, sticky="e")

reset_button = ttk.Button(root, text="Reset to Defaults", command=reset_to_defaults)
reset_button.grid(row=1, column=0, pady=5, sticky="w", padx=(10,0))

load_settings()
root.mainloop()
