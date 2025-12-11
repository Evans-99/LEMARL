def compute_reward(state, action):
    """
    LEMARL
    Compute reward based on multi-objective satisfaction for IES low-carbon scheduling.
    Args:
        state (dict): System state containing:
            - 'P_gen_total': total generation (MW)
            - 'P_load_total': total load (MW)
            - 'H_gen_total': total heat generation (MW)
            - 'H_load_total': total heat demand (MW)
            - 'carbon_emission': total carbon emission (ton)
            - 'total_cost': total operational cost ($)
            - 'renewable_curtailment': curtailed renewable energy (MWh)
            - 'voltage': array of node voltages (p.u.)
            - 'current_squared': array of line currents squared (I^2)
            - 'temperature_supply': supply network temperatures (℃)
            - 'temperature_return': return network temperatures (℃)
            - 'temperature_indoor': indoor temperatures for TCL (℃)
            - 'P_imbalance': active power imbalance at each node (MW)
            - 'Q_imbalance': reactive power imbalance at each node (MVAr)
            - 'H_imbalance': thermal power imbalance at each node (MW)
            - 'constraints_violated': list of violated constraint names
            - 'unit_outputs': dict of actual unit outputs
        action (dict): Actions taken by all agents
    Returns:
        reward (float): Scalar reward in [0, 1]
                       1.0 if all sub-objectives satisfied
                       0.0 otherwise
    """
    import numpy as np

    #Thresholds based on Case 1
    CARBON_THRESHOLD = 101.75 * 1.05
    COST_THRESHOLD = 13327.71 * 1.05
    RENEWABLE_CURTAILMENT_THRESHOLD = 1.50 * 1.2

    LOAD_MIN = 2.5
    LOAD_MAX = 8.2

    # Power balance tolerance
    POWER_BALANCE_TOL = 0.01
    THERMAL_BALANCE_TOL = 0.02

    # Voltage limits
    V_MIN_SQ = 0.95 ** 2
    V_MAX_SQ = 1.05 ** 2

    # Temperature limits
    T_SUPPLY_MIN = 60
    T_SUPPLY_MAX = 90
    T_RETURN_MIN = 40
    T_RETURN_MAX = 60
    T_INDOOR_MIN = 20
    T_INDOOR_MAX = 24

    # Current limit
    I_MAX_SQ = 400

    try:
        # Cost Minimization
        total_cost = state.get('total_cost', float('inf'))
        cost_satisfied = (total_cost <= COST_THRESHOLD)

        if not cost_satisfied:
            cost_violation = total_cost - COST_THRESHOLD
            print(
                f"Cost violation: {cost_violation:.2f} $ (total: {total_cost:.2f} $, threshold: {COST_THRESHOLD:.2f} $)")

        # Carbon Emission Reduction
        carbon_emission = state.get('carbon_emission', float('inf'))
        carbon_satisfied = (carbon_emission <= CARBON_THRESHOLD)

        if not carbon_satisfied:
            carbon_violation = carbon_emission - CARBON_THRESHOLD
            print(
                f"Carbon violation: {carbon_violation:.2f} ton (total: {carbon_emission:.2f} ton, threshold: {CARBON_THRESHOLD:.2f} ton)")

        # Renewable Energy Utilization
        renewable_curtailment = state.get('renewable_curtailment', 0)
        renewable_satisfied = (renewable_curtailment <= RENEWABLE_CURTAILMENT_THRESHOLD)

        if not renewable_satisfied:
            curtailment_violation = renewable_curtailment - RENEWABLE_CURTAILMENT_THRESHOLD
            print(f"Renewable curtailment violation: {curtailment_violation:.2f} MWh")

        # Power Balance
        P_gen_total = state.get('P_gen_total', 0)
        P_load_total = state.get('P_load_total', 1)
        Q_gen_total = state.get('Q_gen_total', 0)
        Q_load_total = state.get('Q_load_total', 1)

        # Active power balance
        P_imbalance_rel = abs(P_gen_total - P_load_total) / max(P_load_total, 1e-6)
        power_balance_satisfied = (P_imbalance_rel < POWER_BALANCE_TOL)

        # Reactive power balance
        Q_imbalance_rel = abs(Q_gen_total - Q_load_total) / max(Q_load_total, 1e-6)
        reactive_balance_satisfied = (Q_imbalance_rel < POWER_BALANCE_TOL)

        if not power_balance_satisfied:
            print(f"Power imbalance: {P_imbalance_rel * 100:.4f}% (tolerance: {POWER_BALANCE_TOL * 100}%)")

        if not reactive_balance_satisfied:
            print(f"Reactive power imbalance: {Q_imbalance_rel * 100:.4f}% (tolerance: {POWER_BALANCE_TOL * 100}%)")

        # Thermal Balance
        H_gen_total = state.get('H_gen_total', 0)
        H_load_total = state.get('H_load_total', 1)

        H_imbalance_rel = abs(H_gen_total - H_load_total) / max(H_load_total, 1e-6)
        thermal_balance_satisfied = (H_imbalance_rel < THERMAL_BALANCE_TOL)

        if not thermal_balance_satisfied:
            print(f"Thermal imbalance: {H_imbalance_rel * 100:.4f}% (tolerance: {THERMAL_BALANCE_TOL * 100}%)")

        # Safety Constraints

        voltage_sq = state.get('voltage_squared', np.array([1.0]))  # v_i,t from eq. 32
        voltage_violations = np.sum((voltage_sq < V_MIN_SQ) | (voltage_sq > V_MAX_SQ))
        voltage_satisfied = (voltage_violations == 0)

        if not voltage_satisfied:
            v_min_actual = np.sqrt(np.min(voltage_sq))
            v_max_actual = np.sqrt(np.max(voltage_sq))
            print(
                f"Voltage violations: {voltage_violations} nodes (range: [{v_min_actual:.4f}, {v_max_actual:.4f}] p.u.)")

        current_sq = state.get('current_squared', np.array([0]))  # l_ij,t from eq. 32
        current_violations = np.sum(current_sq > I_MAX_SQ)
        current_satisfied = (current_violations == 0)

        if not current_satisfied:
            i_max_actual = np.sqrt(np.max(current_sq))
            print(f"Current violations: {current_violations} lines (max: {i_max_actual:.2f} A)")

        temp_supply = state.get('temperature_supply', np.array([75]))  # ℃
        temp_supply_violations = np.sum((temp_supply < T_SUPPLY_MIN) | (temp_supply > T_SUPPLY_MAX))
        temp_supply_satisfied = (temp_supply_violations == 0)

        if not temp_supply_satisfied:
            print(
                f"Supply temp violations: {temp_supply_violations} nodes (range: [{np.min(temp_supply):.1f}, {np.max(temp_supply):.1f}]℃)")

        temp_return = state.get('temperature_return', np.array([50]))  # ℃
        temp_return_violations = np.sum((temp_return < T_RETURN_MIN) | (temp_return > T_RETURN_MAX))
        temp_return_satisfied = (temp_return_violations == 0)

        if not temp_return_satisfied:
            print(
                f"Return temp violations: {temp_return_violations} nodes (range: [{np.min(temp_return):.1f}, {np.max(temp_return):.1f}]℃)")

        temp_indoor = state.get('temperature_indoor', np.array([22] * 7))  # ℃
        temp_indoor_violations = np.sum((temp_indoor < T_INDOOR_MIN) | (temp_indoor > T_INDOOR_MAX))
        temp_indoor_satisfied = (temp_indoor_violations == 0)

        if not temp_indoor_satisfied:
            print(
                f"Indoor temp violations: {temp_indoor_violations} TCLs (range: [{np.min(temp_indoor):.1f}, {np.max(temp_indoor):.1f}]℃)")

        unit_violations = 0

        # Check CHP constraints
        if 'CHP' in state.get('unit_outputs', {}):
            for unit_id, output in state['unit_outputs']['CHP'].items():
                P_out = output.get('P', 0)
                H_out = output.get('H', 0)

                if not (1.0 <= P_out <= 5.5):
                    unit_violations += 1
                    print(f"{unit_id} power violation: P={P_out:.2f} MW (limits: [1.0, 5.5])")

                if not (0.0 <= H_out <= 4.5):
                    unit_violations += 1
                    print(f"{unit_id} heat violation: H={H_out:.2f} MW (limits: [0, 4.5])")

                # Feasible region constraint
                P_min_feasible = 0.8 * H_out + 0.5
                P_max_feasible = 5.5 - 0.6 * H_out
                if not (P_min_feasible <= P_out <= P_max_feasible):
                    unit_violations += 1
                    print(f"{unit_id} feasible region violation: P={P_out:.2f}, H={H_out:.2f}")

        # Check DG constraints
        if 'DG' in state.get('unit_outputs', {}):
            for unit_id, P_out in state['unit_outputs']['DG'].items():
                if unit_id == 'DG_1':
                    if not (0.0 <= P_out <= 4.5):
                        unit_violations += 1
                        print(f"{unit_id} violation: P={P_out:.2f} MW (limits: [0, 4.5])")
                elif unit_id == 'DG_2':
                    if not (0.0 <= P_out <= 4.0):
                        unit_violations += 1
                        print(f"{unit_id} violation: P={P_out:.2f} MW (limits: [0, 4.0])")

        # Check EB constraints
        if 'EB' in state.get('unit_outputs', {}):
            for unit_id, P_out in state['unit_outputs']['EB'].items():
                if not (0.0 <= P_out <= 3.0):
                    unit_violations += 1
                    print(f"{unit_id} violation: P={P_out:.2f} MW (limits: [0, 3.0])")

                    unit_constraints_satisfied = (unit_violations == 0)
                    # 6.7 Ramping constraints (eq. 2, 19)
                    ramping_violations = state.get('ramping_violations', 0)
                    ramping_satisfied = (ramping_violations == 0)

                    if not ramping_satisfied:
                        print(f"Ramping violations: {ramping_violations} units")

        # DR constraints
        dr_violations = 0
        if 'DR' in state.get('unit_outputs', {}):
            for node_idx, output in state['unit_outputs']['DR'].items():
                P_up = output.get('P_up', 0)
                P_down = output.get('P_down', 0)

                # Cannot increase and decrease simultaneously (eq. 23)
                if P_up > 0 and P_down > 0:
                    dr_violations += 1
                    print(f"DR node {node_idx}: simultaneous up and down")

        dr_satisfied = (dr_violations == 0)

        # Aggregate all safety constraints
        safety_satisfied = (
            voltage_satisfied and
            current_satisfied and
            temp_supply_satisfied and
            temp_return_satisfied and
            temp_indoor_satisfied and
            unit_constraints_satisfied and
            ramping_satisfied and
            dr_satisfied
        )


        # Final Reward Calculation
        all_objectives_satisfied = (
            cost_satisfied and
            carbon_satisfied and
            renewable_satisfied and
            power_balance_satisfied and
            reactive_balance_satisfied and
            thermal_balance_satisfied and
            safety_satisfied
        )

        if all_objectives_satisfied:
            reward = 1.0
            print("All objectives satisfied - Reward = 1.0")
        else:
            reward = 0.0
            print("Some objectives not satisfied - Reward = 0.0")
            print(f"  Cost: {cost_satisfied}, Carbon: {carbon_satisfied}, Renewable: {renewable_satisfied}")
            print(f"  Power balance: {power_balance_satisfied}, Thermal balance: {thermal_balance_satisfied}")
            print(f"  Safety: {safety_satisfied}")


        performance_metrics = {
            'cost': total_cost,
            'carbon': carbon_emission,
            'curtailment': renewable_curtailment,
            'power_imbalance': P_imbalance_rel,
            'thermal_imbalance': H_imbalance_rel,
            'voltage_violations': voltage_violations,
            'temperature_violations': (temp_supply_violations + temp_return_violations + temp_indoor_violations),
            'unit_violations': unit_violations,
            'all_satisfied': all_objectives_satisfied
        }

        return reward, performance_metrics

    except Exception as e:
        print(f"Error in reward computation: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, {'error': str(e)}