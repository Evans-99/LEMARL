def generate_prior_policy(observation):
    """
    Prior policy for IES scheduling based on merit order and basic constraints.
    Based on IEEE 33-bus power system with 12-node thermal system.

    Args:
        observation Contains:
            'P_load': array of electrical load at each node (MW)
            'H_load': array of thermal load at each node (MW)
            'P_ren_forecast': forecasted renewable generation (MW)
            'T_outdoor': outdoor temperature (℃)
            'T_indoor': indoor temperature for TCL nodes (℃)
            'T_supply': supply network temperature (℃)
            'T_return': return network temperature (℃)
            'NCI_e': node carbon intensity for electrical network (ton/MWh)
            'NCI_h': node carbon intensity for thermal network (ton/MWh)
            'electricity_price': current electricity price ($/MWh)
            'carbon_price': current carbon price ($/ton)
            'unit_info': dict with unit parameters
            'previous_action': previous time step action for ramping check

    Returns:
        action: Action for each agent type
    """
    import numpy as np

    action = {}

    try:
        # Extract information
        P_load_total = np.sum(observation['P_load'])
        H_load_total = np.sum(observation['H_load'])
        P_ren_available = np.sum(observation['P_ren_forecast'])
        unit_info = observation['unit_info']
        prev_action = observation.get('previous_action', None)

        CHP_PARAMS = {
            'CHP_1': {
                'P_min': 1.0, 'P_max': 5.5,
                'H_min': 0.0, 'H_max': 4.5,
                'R_ramp': 0.5,
                'c_m': 0.8,
                'c_k': 0.5,
                'c_v': 0.6,
                'GCI': 0.95
            }
        }

        DG_PARAMS = {
            'DG_1': {
                'P_min': 0.0, 'P_max': 4.5,
                'Q_min': 0.0, 'Q_max': 4.5 / 0.85,
                'R_ramp': 0.4,
                'GCI': 0.70
            },
            'DG_2': {
                'P_min': 0.0, 'P_max': 4.0,
                'Q_min': 0.0, 'Q_max': 4.0 / 0.85,
                'R_ramp': 0.35,
                'GCI': 0.53
            }
        }

        EB_PARAMS = {
            'EB_1': {
                'P_min': 0.0, 'P_max': 3.0,
                'H_max': 2.5,
                'eta': 0.95,
                'c_EB': 20
            }
        }

        REN_PARAMS = {
            'WT_1': {'P_max': 2.5, 'GCI': 0.0},
            'WT_2': {'P_max': 3.0, 'GCI': 0.0},
            'PV_1': {'P_max': 2.5, 'GCI': 0.0},
            'PV_2': {'P_max': 2.0, 'GCI': 0.0}
        }

        TCL_PARAMS = {
            'TCL_1': {'alpha': 0.9884, 'beta': 0.9259, 'gamma': 0.0116,
                      'H_min': 0, 'H_max': 25, 'T_min': 20, 'T_max': 24, 'T_init': 21},
            'TCL_2': {'alpha': 0.9884, 'beta': 0.9259, 'gamma': 0.0116,
                      'H_min': 0, 'H_max': 25, 'T_min': 20, 'T_max': 24, 'T_init': 21},
            'TCL_3': {'alpha': 0.99, 'beta': 0.8333, 'gamma': 0.01,
                      'H_min': 0, 'H_max': 30, 'T_min': 20, 'T_max': 24, 'T_init': 21},
            'TCL_4': {'alpha': 0.99, 'beta': 0.8333, 'gamma': 0.01,
                      'H_min': 0, 'H_max': 30, 'T_min': 20, 'T_max': 24, 'T_init': 21},
            'TCL_5': {'alpha': 0.99, 'beta': 0.8333, 'gamma': 0.01,
                      'H_min': 0, 'H_max': 30, 'T_min': 20, 'T_max': 24, 'T_init': 21},
            'TCL_6': {'alpha': 0.9913, 'beta': 0.7576, 'gamma': 0.0087,
                      'H_min': 0, 'H_max': 35, 'T_min': 20, 'T_max': 24, 'T_init': 21},
            'TCL_7': {'alpha': 0.9913, 'beta': 0.7576, 'gamma': 0.0087,
                      'H_min': 0, 'H_max': 35, 'T_min': 20, 'T_max': 24, 'T_init': 21}
        }

        # Grid parameters
        GRID_PARAMS = {
            'P_max': 0.15 * (5.5 + 4.5 + 4.0 + 2.5 + 3.0 + 2.5 + 2.0),
            'Q_max': 0.15 * (5.5 + 4.5 + 4.0 + 2.5 + 3.0 + 2.5 + 2.0) / 0.85,
            'GCI': 0.6
        }

        # DR parameters
        DR_ADJUSTABLE_RATIO = 0.20

        # Voltage and temperature limits
        V_MIN, V_MAX = 0.95, 1.05
        T_SUPPLY_MIN, T_SUPPLY_MAX = 60, 90
        T_RETURN_MIN, T_RETURN_MAX = 40, 60

        # Maximize renewable utilization
        action['Renewable'] = {}
        P_ren_total = 0

        for unit_id in ['WT_1', 'WT_2', 'PV_1', 'PV_2']:
            if unit_id in REN_PARAMS:
                # Use forecasted value, capped by maximum capacity
                forecast_idx = {'WT_1': 0, 'WT_2': 1, 'PV_1': 2, 'PV_2': 3}
                P_forecast = observation['P_ren_forecast'][forecast_idx[unit_id]]
                P_output = np.clip(P_forecast, 0, REN_PARAMS[unit_id]['P_max'])
                action['Renewable'][unit_id] = float(P_output)
                P_ren_total += P_output

        # Calculate residual electrical demand
        P_residual = max(0, P_load_total - P_ren_total)

        # Merit-order dispatch for DG units
        # Sort by carbon intensity (lower carbon first)
        dg_units_sorted = sorted(
            DG_PARAMS.items(),
            key=lambda x: x[1]['GCI']
        )

        action['DG'] = {}
        for unit_id, params in dg_units_sorted:
            if P_residual > 0:
                # Determine dispatch amount
                P_dispatch = min(params['P_max'], P_residual)

                # Apply ramping constraint if previous action exists
                if prev_action and unit_id in prev_action.get('DG', {}):
                    P_prev = prev_action['DG'][unit_id]
                    P_dispatch = np.clip(
                        P_dispatch,
                        P_prev - params['R_ramp'],
                        P_prev + params['R_ramp']
                    )

                # Ensure within limits
                P_dispatch = np.clip(P_dispatch, params['P_min'], params['P_max'])
                action['DG'][unit_id] = float(P_dispatch)
                P_residual -= P_dispatch
            else:
                action['DG'][unit_id] = float(params['P_min'])

        # CHP dispatch for combined heat and power
        action['CHP'] = {}
        H_residual = H_load_total

        for unit_id, params in CHP_PARAMS.items():
            # Prioritize satisfying thermal demand
            H_dispatch = min(params['H_max'], H_residual)
            H_dispatch = max(params['H_min'], H_dispatch)

            # Calculate corresponding electrical output based on feasible region
            P_min_coupled = params['c_m'] * H_dispatch + params['c_k']
            P_max_coupled = params['P_max'] - params['c_v'] * H_dispatch

            # Choose middle point for stable operation
            P_dispatch = (P_min_coupled + P_max_coupled) / 2
            P_dispatch = np.clip(P_dispatch, params['P_min'], params['P_max'])

            # Apply ramping constraint
            if prev_action and unit_id in prev_action.get('CHP', {}):
                P_prev = prev_action['CHP'][unit_id]['P']
                P_dispatch = np.clip(
                    P_dispatch,
                    P_prev - params['R_ramp'],
                    P_prev + params['R_ramp']
                )

            action['CHP'][unit_id] = {
                'P': float(P_dispatch),
                'H': float(H_dispatch)
            }
            H_residual -= H_dispatch
            P_residual -= P_dispatch

        # Electric Boiler for remaining thermal demand
        action['EB'] = {}

        for unit_id, params in EB_PARAMS.items():
            if H_residual > 0:
                # Determine thermal output needed
                H_dispatch = min(params['H_max'], H_residual)

                # Calculate electrical input
                P_dispatch = H_dispatch / params['eta']
                P_dispatch = np.clip(P_dispatch, params['P_min'], params['P_max'])

                action['EB'][unit_id] = float(P_dispatch)
                H_residual -= H_dispatch
                P_residual += P_dispatch
            else:
                action['EB'][unit_id] = 0.0

        # TCL adjustment
        action['TCL'] = {}
        T_outdoor = observation.get('T_outdoor', 0)

        for unit_id, params in TCL_PARAMS.items():
            # Get current indoor temperature
            tcl_idx = int(unit_id.split('_')[1]) - 1
            T_current = observation.get('T_indoor', [21] * 7)[tcl_idx]

            # Calculate required thermal power to maintain comfort
            T_target = (params['T_min'] + params['T_max']) / 2

            H_required = (T_target - params['alpha'] * T_current -
                          params['gamma'] * T_outdoor) / params['beta']

            H_dispatch = np.clip(H_required, params['H_min'], params['H_max'])
            action['TCL'][unit_id] = float(H_dispatch)

        # Demand Response
        action['DR'] = {}

        for node_idx, P_node_load in enumerate(observation['P_load']):
            unit_id = f'DR_node_{node_idx}'
            P_adjustable_max = DR_ADJUSTABLE_RATIO * P_node_load

            # Simple strategy: shift load from high-price to low-price periods
            # For prior policy, use conservative approach (minimal adjustment)
            if observation['electricity_price'] > 100:
                # Reduce load slightly
                P_down = min(0.05 * P_node_load, P_adjustable_max)
                P_up = 0.0
            elif observation['electricity_price'] < 50:
                P_up = min(0.05 * P_node_load, P_adjustable_max)
                P_down = 0.0
            else:
                P_up = 0.0
                P_down = 0.0

            action['DR'][unit_id] = {
                'P_up': float(P_up),
                'P_down': float(P_down)
            }

        # Grid purchase for remaining deficit
        total_local_gen = (
                P_ren_total +
                sum(action['DG'].values()) +
                sum(chp['P'] for chp in action['CHP'].values()) -
                sum(action['EB'].values())
        )

        P_grid_needed = P_load_total - total_local_gen
        P_grid = np.clip(P_grid_needed, 0, GRID_PARAMS['P_max'])

        action['Grid'] = {
            'P': float(P_grid),
            'Q': float(P_grid * np.tan(np.arccos(0.85)))
        }

        # Check all constraints
        # Power balance check
        total_gen = (
                sum(action['Renewable'].values()) +
                sum(action['DG'].values()) +
                sum(chp['P'] for chp in action['CHP'].values()) +
                action['Grid']['P'] -
                sum(action['EB'].values())
        )

        power_balance_error = abs(total_gen - P_load_total) / max(P_load_total, 1e-6)

        if power_balance_error > 0.01:
            print(f"Warning: Power balance error = {power_balance_error:.4f}")

        # Thermal balance check
        total_heat = (
                sum(chp['H'] for chp in action['CHP'].values()) +
                sum(action['EB'].values()) * EB_PARAMS['EB_1']['eta']
        )

        thermal_balance_error = abs(total_heat - H_load_total) / max(H_load_total, 1e-6)

        if thermal_balance_error > 0.02:
            print(f"Warning: Thermal balance error = {thermal_balance_error:.4f}")

    except Exception as e:
        print(f"Error in prior policy: {e}")
        import traceback
        traceback.print_exc()
        # Return safe minimal action
        action = {
            'Renewable': {'WT_1': 0, 'WT_2': 0, 'PV_1': 0, 'PV_2': 0},
            'DG': {'DG_1': 0, 'DG_2': 0},
            'CHP': {'CHP_1': {'P': 1.0, 'H': 0.0}},
            'EB': {'EB_1': 0.0},
            'TCL': {f'TCL_{i}': 0 for i in range(1, 8)},
            'DR': {f'DR_node_{i}': {'P_up': 0, 'P_down': 0} for i in range(33)},
            'Grid': {'P': 5.0, 'Q': 5.0 * np.tan(np.arccos(0.85))}
        }

    return action