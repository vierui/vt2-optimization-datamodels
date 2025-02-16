def dcopf(gen_time_series, branch, bus, demand_time_series, delta_t=1):
    import pulp
    from pandas.tseries.offsets import DateOffset
    import numpy as np
    import pandas as pd
    import math

    print("[DCOPF] Entering dcopf function...")
    print(f"[DCOPF] gen_time_series length = {len(gen_time_series)}, demand_time_series length = {len(demand_time_series)}")

    # Create LP problem
    DCOPF = pulp.LpProblem("DCOPF", pulp.LpMinimize)

    # Identify storage vs. non-storage units
    storage_data = gen_time_series[gen_time_series['emax'] > 0]
    S = storage_data['id'].unique()     # set of storage IDs
    non_storage_data = gen_time_series[gen_time_series['emax'] == 0]
    G = non_storage_data['id'].unique() # set of non-storage gen IDs

    print(f"[DCOPF] Found storage units: {S}, non-storage units: {G}")
    print("[DCOPF] Storage data sample:")
    print(storage_data[['id', 'bus', 'emax', 'pmax', 'eta']].head())

    # Time and bus sets
    N = bus['bus_i'].values
    T = sorted(demand_time_series['time'].unique())
    if not T:
        print("[DCOPF] No time steps found in demand_time_series. Returning None.")
        return None

    next_time = T[-1] + DateOffset(hours=delta_t)
    extended_T = list(T) + [next_time]

    # 1. Create GEN variables for non-storage generators
    GEN = {}
    for g in G:
        gen_rows = gen_time_series[gen_time_series['id'] == g]

        # We assume one row per time step or time-invariant parameters
        for t in T:
            row_t = gen_rows.loc[gen_rows['time'] == t]
            if row_t.empty:
                print(f"[DCOPF] Missing data for generator={g}, time={t}. Returning None.")
                return None

            pmin = row_t['pmin'].iloc[0]
            pmax = row_t['pmax'].iloc[0]
            GEN[g, t] = pulp.LpVariable(f"GEN_{g}_{t}_var", lowBound=pmin, upBound=pmax)

    # 2. Voltage angle variables
    THETA = {
        (i, t): pulp.LpVariable(f"THETA_{i}_{t}_var", lowBound=None)
        for i in N for t in T
    }

    # 3. FLOW variables
    FLOW = {}
    for idx, row_b in branch.iterrows():
        i = int(row_b['fbus'])
        j = int(row_b['tbus'])
        for t in T:
            FLOW[i, j, t] = pulp.LpVariable(f"FLOW_{i}_{j}_{t}_var", lowBound=None)

    # 4. DC Power Flow Constraints
    for idx_b, row_b in branch.iterrows():
        i = int(row_b['fbus'])
        j = int(row_b['tbus'])
        susceptance = row_b['sus']

        for t in T:
            DCOPF += FLOW[i, j, t] == susceptance * (THETA[i, t] - THETA[j, t]), \
                     f"Flow_Constraint_{i}_{j}_Time_{t}"

    # 5. Storage Variables/Constraints
    P_charge = {}
    P_discharge = {}
    E = {}

    for s in S:
        s_row = gen_time_series.loc[gen_time_series['id'] == s].iloc[0]
        E_max = s_row['emax']
        E_initial = s_row['einitial']
        eta = s_row['eta']
        P_max = s_row['pmax']  # Discharging max

        # Create charge/discharge variables
        for t in T:
            P_charge[s, t] = pulp.LpVariable(f"P_charge_{s}_{t}_var", lowBound=0, upBound=abs(P_max))
            P_discharge[s, t] = pulp.LpVariable(f"P_discharge_{s}_{t}_var", lowBound=0, upBound=abs(P_max))

        # SoC variables
        for t in extended_T:
            E[s, t] = pulp.LpVariable(f"E_{s}_{t}_var", lowBound=0, upBound=E_max)

        # Initial SoC
        DCOPF += E[s, T[0]] == E_initial, f"Initial_Storage_SoC_{s}"

        # SoC dynamics: E[next] = E[t] + eta*charge - (1/eta)*discharge
        for idx_t, t in enumerate(T):
            next_t = extended_T[idx_t + 1]
            DCOPF += E[s, next_t] == E[s, t] + eta * P_charge[s, t] * delta_t - (1/eta) * P_discharge[s, t] * delta_t, \
                     f"Storage_Dynamics_{s}_Time_{t}"

        # Final SoC (optional)
        DCOPF += E[s, extended_T[-1]] == E_initial, f"Final_Storage_SoC_{s}"

    # 6. Slack bus angle = 0
    slack_bus = 1  # Adding back the slack bus constraint
    for t in T:
        DCOPF += THETA[slack_bus, t] == 0, f"Slack_Bus_Angle_Time_{t}"

    # 7. Objective: Include both generation costs and storage costs
    generation_cost = pulp.lpSum(
        gen_time_series.loc[
            (gen_time_series['id'] == g) & (gen_time_series['time'] == t), 
            'gencost'
        ].values[0] * GEN[g, t]
        for g in G for t in T
    )

    # Simple storage cost to prevent unnecessary cycling
    storage_cost = pulp.lpSum(
        0.001 * (P_discharge[s, t] + P_charge[s, t])
        for s in S for t in T
    )

    DCOPF += generation_cost + storage_cost, "Total_Cost"

    # Get load buses from bus dataframe
    load_buses = bus[bus['type'] == 1]['bus_i'].values
    
    # 8. Power Balance Constraints
    for t in T:
        for i in N:  # Include all buses, even those without generators
            # sum non-storage gen at bus i
            gen_sum = pulp.lpSum(
                GEN[g, t]
                for g in G
                if gen_time_series.loc[
                    (gen_time_series['id'] == g) & (gen_time_series['time'] == t),
                    'bus'
                ].values[0] == i
            )

            # Get demand at bus i - only if it's a load bus
            pd_val = 0
            if i in load_buses:
                demands_at_bus = demand_time_series.loc[
                    (demand_time_series['bus'] == i) & (demand_time_series['time'] == t),
                    'pd'
                ]
                pd_val = demands_at_bus.sum() if not demands_at_bus.empty else 0

            # Storage at bus i => discharge - charge
            storages_at_bus_i = gen_time_series.loc[
                (gen_time_series['bus'] == i) & (gen_time_series['emax'] > 0),
                'id'
            ].unique()
            
            if len(storages_at_bus_i) > 0:
                gen_sum += pulp.lpSum(
                    (P_discharge[s, t] - P_charge[s, t]) for s in storages_at_bus_i
                )

            # Power flow balance at each bus
            flow_out = pulp.lpSum(FLOW[i, j, t] for j in branch.loc[branch['fbus'] == i, 'tbus'])
            flow_in  = pulp.lpSum(FLOW[j, i, t] for j in branch.loc[branch['tbus'] == i, 'fbus'])

            DCOPF += (gen_sum - pd_val + flow_in - flow_out == 0), f"Power_Balance_Bus_{i}_Time_{t}"

    # 9. Flow limits
    for _, row_b in branch.iterrows():
        i = row_b['fbus']
        j = row_b['tbus']
        rate_a = row_b['ratea']
        for t in T:
            DCOPF += FLOW[i, j, t] <= rate_a,  f"Flow_Limit_{i}_{j}_Upper_Time_{t}"
            DCOPF += FLOW[i, j, t] >= -rate_a, f"Flow_Limit_{i}_{j}_Lower_Time_{t}"

    # 10. Solve
    print("[DCOPF] About to solve the LP problem with CBC solver...")
    solver_result = DCOPF.solve(pulp.PULP_CBC_CMD(msg=True))

    status_code = DCOPF.status
    status_str = pulp.LpStatus[status_code]
    print(f"[DCOPF] Solver returned status code = {status_code}, interpreted as '{status_str}'")

    # If code != 1 => Not recognized as Optimal
    if status_code != 1:
        print(f"[DCOPF] Not optimal => returning None.")
        return None

    # 11. Extract results
    print("[DCOPF] Extraction of results - building final dictionary...")

    # a) Non-storage generation
    generation = []
    for g in G:
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

    # b) Storage net output
    storage_generation = []
    for s in S:
        s_bus = gen_time_series.loc[gen_time_series['id'] == s, 'bus'].iloc[0]
        for t in T:
            ch = pulp.value(P_charge[s, t])
            dis = pulp.value(P_discharge[s, t])
            if math.isnan(ch):
                ch = 0
            if math.isnan(dis):
                dis = 0
            net_out = dis - ch
            storage_generation.append({
                'time': t,
                'id': s,
                'node': s_bus,
                'gen': net_out
            })
    storage_generation = pd.DataFrame(storage_generation)
    generation = pd.concat([generation, storage_generation], ignore_index=True)

    # c) Angles
    angles = []
    for i_bus in N:
        for t in T:
            val_theta = pulp.value(THETA[i_bus, t])
            angles.append({
                'time': t,
                'bus': i_bus,
                'theta': 0 if math.isnan(val_theta) else val_theta
            })
    angles = pd.DataFrame(angles)

    # d) Flows
    flows_list = []
    for (i_bus, j_bus, t) in FLOW:
        val_flow = pulp.value(FLOW[i_bus, j_bus, t])
        flows_list.append({
            'time': t,
            'from_bus': i_bus,
            'to_bus': j_bus,
            'flow': 0 if math.isnan(val_flow) else val_flow
        })
    flows_df = pd.DataFrame(flows_list)

    # e) Storage states
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

    # Always define columns so groupby('storage_id') won't fail
    storage_df = pd.DataFrame(
        storage_list,
        columns=["storage_id", "time", "E", "P_charge", "P_discharge"]
    )

    if len(S) > 0 and not storage_df.empty:
        # Shift E if you want SoC at start of each interval
        storage_corrected = []
        for s_id, group in storage_df.groupby('storage_id'):
            group = group.sort_values('time').reset_index(drop=True)
            group['E'] = group['E'].shift(-1)
            # remove last row
            group = group.iloc[:-1]
            storage_corrected.append(group)
        storage_df = pd.concat(storage_corrected, ignore_index=True)

    total_cost = pulp.value(DCOPF.objective)
    if total_cost is None:
        print("[DCOPF] Warning: Could not extract objective value. Setting cost to infinity.")
        total_cost = float('inf')
    
    status = pulp.LpStatus[DCOPF.status]

    print(f"[DCOPF] Final cost = {total_cost}, status = {status}")
    print("[DCOPF] Done, returning result dictionary.")

    return {
        'generation': generation,
        'angles': angles,
        'flows': flows_df,
        'storage': storage_df,
        'cost': total_cost,
        'status': status
    }