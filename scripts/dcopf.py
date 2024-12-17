# 3. Function
# ===========================

def dcopf(gen_time_series, branch, bus, demand_time_series, delta_t=1):
    import pulp
    from pandas.tseries.offsets import DateOffset
    import numpy as np
    import pandas as pd

    DCOPF = pulp.LpProblem("DCOPF", pulp.LpMinimize)

    # Identify storage and non-storage units
    # Storage units are those with emax > 0
    storage_data = gen_time_series[gen_time_series['emax'] > 0]
    S = storage_data['id'].unique()
    non_storage_data = gen_time_series[gen_time_series['emax'] == 0]
    G = non_storage_data['id'].unique()

    N = bus['bus_i'].values
    T = sorted(demand_time_series['time'].unique())
    next_time = T[-1] + DateOffset(hours=delta_t)
    extended_T = list(T) + [next_time]

    # Create GEN variables for non-storage generators
    GEN = {}
    for g in G:
        gen_rows = gen_time_series[(gen_time_series['id'] == g)]
        # We assume the same pmin/pmax/gencost across all T or a row for each T
        for t in T:
            row = gen_rows.loc[gen_rows['time'] == t]
            if row.empty:
                print(f"Error: Generator {g} has no data for time {t}.")
                return None
            pmin = row['pmin'].values[0]
            pmax = row['pmax'].values[0]
            GEN[g, t] = pulp.LpVariable(f"GEN_{g}_{t}_var", lowBound=pmin, upBound=pmax)

    # Voltage angle variables
    THETA = {(i, t): pulp.LpVariable(f"THETA_{i}_{t}_var", lowBound=None) for i in N for t in T}

    # FLOW variables
    FLOW = {}
    for idx, row in branch.iterrows():
        i = int(row['fbus'])  
        j = int(row['tbus'])
        for t in T:
            FLOW[i, j, t] = pulp.LpVariable(f"FLOW_{i}_{j}_{t}_var", lowBound=None)

    # DC Power Flow Constraints
    for idx_branch, row in branch.iterrows():
        i = int(row['fbus'])
        j = int(row['tbus'])
        susceptance = row['sus']
        for t in T:
            DCOPF += FLOW[i, j, t] == susceptance * (THETA[i, t] - THETA[j, t]), \
                    f"Flow_Constraint_{i}_{j}_Time_{t}"

    # Storage Variables and Constraints
    # For each storage unit, extract parameters from gen_time_series
    P_charge = {}
    P_discharge = {}
    E = {}

    for s in S:
        # Assume storage parameters are time-invariant and take from first row
        s_row = gen_time_series[(gen_time_series['id'] == s)].iloc[0]
        E_max = s_row['emax']
        E_initial = s_row['einitial']
        eta = s_row['eta']
        P_max = s_row['pmax']  # positive direction
        # pmin is negative pmax, so max charge/discharge capacity is abs(pmin)

        # Create storage variables
        for t in T:
            P_charge[s, t] = pulp.LpVariable(f"P_charge_{s}_{t}_var", lowBound=0, upBound=P_max)
            P_discharge[s, t] = pulp.LpVariable(f"P_discharge_{s}_{t}_var", lowBound=0, upBound=P_max)

        for t in extended_T:
            E[s, t] = pulp.LpVariable(f"E_{s}_{t}_var", lowBound=0, upBound=E_max)

        # Initial SoC constraint
        DCOPF += E[s, T[0]] == E_initial, f"Initial_Storage_SoC_{s}"

        # Storage dynamics
        for idx_t, t in enumerate(T):
            next_t = extended_T[idx_t + 1]
            DCOPF += E[s, next_t] == E[s, t] + eta * P_charge[s, t] * delta_t - (1 / eta) * P_discharge[s, t] * delta_t, \
                    f"Storage_Dynamics_{s}_Time_{t}"

        # Optional: Final SoC constraint (if desired)
        DCOPF += E[s, extended_T[-1]] == E_initial, f"Final_Storage_SoC_{s}"

    # Slack bus angle = 0
    slack_bus = 1
    for t in T:
        DCOPF += THETA[slack_bus, t] == 0, f"Slack_Bus_Angle_Time_{t}"

    # Objective Function: sum generation costs (non-storage + storage if any gencost)
    DCOPF += pulp.lpSum(
        gen_time_series.loc[(gen_time_series['id'] == g) & (gen_time_series['time'] == t), 'gencost'].values[0] * GEN[g, t]
        for g in G for t in T
    ), "Total_Generation_Cost"
    # Note: Storage cost is typically zero as defined.

    # Power Balance Constraints
    for t in T:
        for i in N:
            # Sum of generation at bus i (non-storage)
            gen_sum = pulp.lpSum(
                GEN[g, t]
                for g in G
                if gen_time_series.loc[(gen_time_series['id'] == g) & (gen_time_series['time'] == t), 'bus'].values[0] == i
            )

            # Demand at bus i
            demand = demand_time_series.loc[(demand_time_series['bus'] == i) & (demand_time_series['time'] == t), 'pd']
            demand_value = demand.values[0] if not demand.empty else 0

            # Add storage contributions for this bus
            storages_at_bus_i = gen_time_series.loc[(gen_time_series['bus'] == i) & (gen_time_series['emax'] > 0), 'id'].unique()
            if len(storages_at_bus_i) > 0:
                gen_sum += pulp.lpSum(P_discharge[s, t] - P_charge[s, t] for s in storages_at_bus_i)

            flow_out = pulp.lpSum(FLOW[i, j, t] for j in branch.loc[branch['fbus'] == i, 'tbus'])
            flow_in = pulp.lpSum(FLOW[j, i, t] for j in branch.loc[branch['tbus'] == i, 'fbus'])

            # Power balance constraint
            DCOPF += gen_sum - demand_value + flow_in - flow_out == 0, f"Power_Balance_Bus_{i}_Time_{t}"

    # Flow limits
    for idx_branch, row in branch.iterrows():
        i = row['fbus']
        j = row['tbus']
        rate_a = row['ratea']
        for t in T:
            DCOPF += FLOW[i, j, t] <= rate_a, f"Flow_Limit_{i}_{j}_Upper_Time_{t}"
            DCOPF += FLOW[i, j, t] >= -rate_a, f"Flow_Limit_{i}_{j}_Lower_Time_{t}"

    # Solve the problem
    DCOPF.solve(pulp.PULP_CBC_CMD(msg=True))

    # Check status
    if pulp.LpStatus[DCOPF.status] != 'Optimal':
        print(f"Optimization did not find an optimal solution. Status: {pulp.LpStatus[DCOPF.status]}")
        return None

    # Extracting Results
    # Non-storage generation
    generation = pd.DataFrame([
        {
            'time': t,
            'id': g,
            'node': gen_time_series.loc[(gen_time_series['id'] == g) & (gen_time_series['time'] == t), 'bus'].values[0],
            'gen': pulp.value(GEN[g, t])
        }
        for g in G for t in T
    ])

    # Also add storage net output as "generation" for reporting, if desired
    # Storage doesn't have GEN variables but we can represent net output (discharge - charge)
    storage_gen_list = []
    for s in S:
        s_bus = gen_time_series.loc[gen_time_series['id'] == s, 'bus'].iloc[0]
        for t in T:
            storage_gen_list.append({
                'time': t,
                'id': s,
                'node': s_bus,
                'gen': pulp.value(P_discharge[s, t]) - pulp.value(P_charge[s, t])
            })
    storage_generation = pd.DataFrame(storage_gen_list)

    # Combine non-storage and storage generation
    generation = pd.concat([generation, storage_generation], ignore_index=True)

    angles = pd.DataFrame([
        {'time': t, 'bus': i, 'theta': pulp.value(THETA[i, t])}
        for i in N for t in T
    ])

    flows_df = pd.DataFrame([
        {'time': t, 'from_bus': i, 'to_bus': j, 'flow': pulp.value(FLOW[i, j, t])}
        for (i, j, t) in FLOW
    ])

    # Extract Storage Results
    storage_list = []
    for s in S:
        for idx_t, tt in enumerate(extended_T):
            E_val = pulp.value(E[s, tt])
            Pch = pulp.value(P_charge[s, tt]) if tt in T else None
            Pdis = pulp.value(P_discharge[s, tt]) if tt in T else None
            storage_list.append({
                'storage_id': s,
                'time': tt,
                'E': E_val,
                'P_charge': Pch,
                'P_discharge': Pdis
            })

    storage_df = pd.DataFrame(storage_list)

    # Shift E values if you want E at the start of the interval
    storage_corrected = []
    for s_id, group in storage_df.groupby('storage_id'):
        group = group.sort_values('time').reset_index(drop=True)
        group['E'] = group['E'].shift(-1)
        group = group.iloc[:-1]  # remove last row with NaN
        storage_corrected.append(group)

    storage_df = pd.concat(storage_corrected, ignore_index=True)

    total_cost = pulp.value(DCOPF.objective)
    status = pulp.LpStatus[DCOPF.status]

    return {
        'generation': generation,
        'angles': angles,
        'flows': flows_df,
        'storage': storage_df,
        'cost': total_cost,
        'status': status
    }