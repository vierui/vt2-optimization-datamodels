def dcopf(gen_time_series, branch, bus, demand_time_series, delta_t=1):
    
    import pulp
    from pandas.tseries.offsets import DateOffset
    import numpy as np
    import pandas as pd

    # Create LP problem
    DCOPF = pulp.LpProblem("DCOPF", pulp.LpMinimize)

    # Identify storage vs non-storage units
    storage_data = gen_time_series[gen_time_series['emax'] > 0]
    S = storage_data['id'].unique()  # set of storage IDs
    non_storage_data = gen_time_series[gen_time_series['emax'] == 0]
    G = non_storage_data['id'].unique()  # set of non-storage gen IDs

    # Time and bus sets
    N = bus['bus_i'].values
    T = sorted(demand_time_series['time'].unique())

    # Extended time steps to handle SoC at final step
    next_time = T[-1] + DateOffset(hours=delta_t)
    extended_T = list(T) + [next_time]

    #
    # 1. Create GEN variables for non-storage generators
    #
    GEN = {}
    for g in G:
        gen_rows = gen_time_series[gen_time_series['id'] == g]
        # We assume rows are either time-invariant or there is one row per time step
        for t in T:
            row_t = gen_rows.loc[gen_rows['time'] == t]
            if row_t.empty:
                raise ValueError(f"[DCOPF] Missing generator data for gen={g}, time={t}")
            pmin = row_t['pmin'].iloc[0]
            pmax = row_t['pmax'].iloc[0]
            GEN[g, t] = pulp.LpVariable(f"GEN_{g}_{t}_var", lowBound=pmin, upBound=pmax)

    #
    # 2. Voltage angle variables (one per bus-time)
    #
    THETA = {
        (i, t): pulp.LpVariable(f"THETA_{i}_{t}_var", lowBound=None)
        for i in N for t in T
    }

    #
    # 3. FLOW variables (one per branch-time)
    #
    FLOW = {}
    for idx, row_b in branch.iterrows():
        i = int(row_b['fbus'])
        j = int(row_b['tbus'])
        for t in T:
            FLOW[i, j, t] = pulp.LpVariable(f"FLOW_{i}_{j}_{t}_var", lowBound=None)

    #
    # 4. DC Power Flow Constraints
    #
    for idx_b, row_b in branch.iterrows():
        i = int(row_b['fbus'])
        j = int(row_b['tbus'])
        susceptance = row_b['sus']
        for t in T:
            DCOPF += FLOW[i, j, t] == susceptance * (THETA[i, t] - THETA[j, t]), \
                     f"Flow_Constraint_{i}_{j}_Time_{t}"

    #
    # 5. Storage Variables/Constraints
    #
    P_charge = {}
    P_discharge = {}
    E = {}

    for s in S:
        # One row from gen_time_series for static storage parameters
        s_row = gen_time_series.loc[gen_time_series['id'] == s].iloc[0]
        E_max = s_row['emax']
        E_initial = s_row['einitial']
        eta = s_row['eta']
        P_max = s_row['pmax']  # for discharging
        # pmin is typically -pmax if you want symmetrical charge/discharge

        # Create variables
        for t in T:
            P_charge[s, t] = pulp.LpVariable(f"P_charge_{s}_{t}_var", 
                                             lowBound=0, upBound=abs(P_max))
            P_discharge[s, t] = pulp.LpVariable(f"P_discharge_{s}_{t}_var", 
                                                lowBound=0, upBound=abs(P_max))
        # SoC variables at each extended time
        for t in extended_T:
            E[s, t] = pulp.LpVariable(f"E_{s}_{t}_var", lowBound=0, upBound=E_max)

        # Initial SoC
        DCOPF += E[s, T[0]] == E_initial, f"Initial_Storage_SoC_{s}"

        # SoC dynamics
        for idx_t, t in enumerate(T):
            next_t = extended_T[idx_t + 1]
            # E[next_t] = E[t] + eta*charge - (1/eta)*discharge
            DCOPF += E[s, next_t] == E[s, t] + eta * P_charge[s, t] * delta_t - \
                                          (1 / eta) * P_discharge[s, t] * delta_t, \
                     f"Storage_Dynamics_{s}_Time_{t}"

        # Final SoC constraint (optional; can comment out if you don't want to fix final SoC)
        DCOPF += E[s, extended_T[-1]] == E_initial, f"Final_Storage_SoC_{s}"

    #
    # 6. Slack bus angle = 0
    #
    slack_bus = 1
    for t in T:
        DCOPF += THETA[slack_bus, t] == 0, f"Slack_Bus_Angle_Time_{t}"

    #
    # 7. Objective: sum generation costs
    #
    # If you want a cost for storage usage, you could add it similarly. 
    # For now, only non-storage gencost is included.
    DCOPF += pulp.lpSum(
        gen_time_series.loc[
            (gen_time_series['id'] == g) & 
            (gen_time_series['time'] == t),
            'gencost'
        ].values[0] * GEN[g, t]
        for g in G for t in T
    ), "Total_Generation_Cost"

    #
    # 8. Power Balance Constraints
    #
    for t in T:
        for i in N:
            # sum of non-storage generation at bus i
            gen_sum = pulp.lpSum(
                GEN[g, t]
                for g in G
                if gen_time_series.loc[
                    (gen_time_series['id'] == g) & (gen_time_series['time'] == t),
                    'bus'
                ].values[0] == i
            )

            # Demand at bus i
            demand_val = demand_time_series.loc[
                (demand_time_series['bus'] == i) & (demand_time_series['time'] == t),
                'pd'
            ]
            pd_val = demand_val.values[0] if not demand_val.empty else 0

            # Storage net power if any storage at bus i
            storages_at_bus_i = gen_time_series.loc[
                (gen_time_series['bus'] == i) & (gen_time_series['emax'] > 0),
                'id'
            ].unique()

            if len(storages_at_bus_i) > 0:
                # net output = discharge - charge
                gen_sum += pulp.lpSum(
                    (P_discharge[s, t] - P_charge[s, t]) for s in storages_at_bus_i
                )

            # flows in/out
            flow_out = pulp.lpSum(FLOW[i, j, t] for j in branch.loc[branch['fbus'] == i, 'tbus'])
            flow_in  = pulp.lpSum(FLOW[j, i, t] for j in branch.loc[branch['tbus'] == i, 'fbus'])

            # power balance
            DCOPF += gen_sum - pd_val + flow_in - flow_out == 0, \
                     f"Power_Balance_Bus_{i}_Time_{t}"

    #
    # 9. Flow limits
    #
    for idx_b, row_b in branch.iterrows():
        i = row_b['fbus']
        j = row_b['tbus']
        rate_a = row_b['ratea']
        for t in T:
            DCOPF += FLOW[i, j, t] <= rate_a,  f"Flow_Limit_{i}_{j}_Upper_Time_{t}"
            DCOPF += FLOW[i, j, t] >= -rate_a, f"Flow_Limit_{i}_{j}_Lower_Time_{t}"

    #
    # 10. Solve
    #
    DCOPF.solve(pulp.PULP_CBC_CMD(msg=True))

    # Check solution status
    status_str = pulp.LpStatus[DCOPF.status]
    if status_str != 'Optimal':
        print(f"DCOPF not optimal. Status = {status_str}")
        return None

    #
    # 11. Extract results
    #
    # (a) Non-storage generation
    import math

    generation = []
    for g in G:
        # For each time t
        g_bus = gen_time_series.loc[gen_time_series['id'] == g, 'bus'].iloc[0]
        for t in T:
            val = pulp.value(GEN[g, t])
            generation.append({
                'time': t,
                'id': g,
                'node': g_bus,
                'gen': 0 if math.isnan(val) else val
            })
    generation = pd.DataFrame(generation)

    # (b) Storage net output as "generation" for reporting
    storage_generation = []
    for s in S:
        s_bus = gen_time_series.loc[gen_time_series['id'] == s, 'bus'].iloc[0]
        for t in T:
            ch = pulp.value(P_charge[s, t])
            dis = pulp.value(P_discharge[s, t])
            if math.isnan(ch): ch = 0
            if math.isnan(dis): dis = 0
            net_out = dis - ch
            storage_generation.append({
                'time': t,
                'id': s,
                'node': s_bus,
                'gen': net_out
            })
    storage_generation = pd.DataFrame(storage_generation)

    # Combine them
    generation = pd.concat([generation, storage_generation], ignore_index=True)

    # (c) Angles
    angles = []
    for i in N:
        for t in T:
            val_theta = pulp.value(THETA[i, t])
            angles.append({
                'time': t,
                'bus': i,
                'theta': 0 if math.isnan(val_theta) else val_theta
            })
    angles = pd.DataFrame(angles)

    # (d) Flows
    flows_list = []
    for (i, j, t) in FLOW:
        val_flow = pulp.value(FLOW[i, j, t])
        flows_list.append({
            'time': t,
            'from_bus': i,
            'to_bus': j,
            'flow': 0 if math.isnan(val_flow) else val_flow
        })
    flows_df = pd.DataFrame(flows_list)

    # (e) Storage states
    storage_list = []
    for s in S:
        for idx_t, tt in enumerate(extended_T):
            E_val = pulp.value(E[s, tt])
            Pch = pulp.value(P_charge[s, tt]) if tt in T else None
            Pdis = pulp.value(P_discharge[s, tt]) if tt in T else None
            storage_list.append({
                'storage_id': s,
                'time': tt,
                'E': 0 if math.isnan(E_val) else E_val,
                'P_charge': None if (Pch is None or math.isnan(Pch)) else Pch,
                'P_discharge': None if (Pdis is None or math.isnan(Pdis)) else Pdis
            })
    storage_df = pd.DataFrame(storage_list)

    # If you want E[t] to represent SoC at the beginning of hour t, shift them:
    storage_corrected = []
    for s_id, group in storage_df.groupby('storage_id'):
        group = group.sort_values('time').reset_index(drop=True)
        group['E'] = group['E'].shift(-1)
        # drop last row (NaN after shift)
        group = group.iloc[:-1]
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