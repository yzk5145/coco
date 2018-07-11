from gurobipy import *
import logging
import numpy
def isotonic_l1(arr, var_type="INTEGER", cumsum=True, nonnegative=True,
        starts_at=None, ends_at=None, weighted_total=None):
    """ Performs L1 isotonic regression on arr

    Inputs:
            arr: 1-d array or 2-d array where row 0 is an histogram of a region
                the rest of the rows are histograms of its sub regions.
            nonnegative: specifies whether returned array x must have nonnegative
                values
            starts_at: either None or a real value. If not None, then x[0] is
                set to starts_at
            ends_at: either None or a real value. If not None then X[arr.size-1]
                is set to ends_at
            weighted_total: either None or a real value. If not None, then

                1) sum_i  i * (x[i] - x[i-1])  = weighted_total (simplify the
                    cancellations); if cumsum == True
                2) sum_i (x[i]) = weighted_total;  if cumsum == False
            cumsum: the input is the cumulative sum instead of the H array
    Outputs:
            x: a nondecreasing array that minimizes L1 distance to arr
            x is supposed to be a cumulative sum where x[i] = number of groups
                of size <= i.
    """
    joint_flag = True
    grb_var_type = GRB.INTEGER if var_type == "INTEGER" else GRB.CONTINUOUS


    if (isinstance(arr, list)) or (type(arr).__module__ == numpy.__name__ and arr.ndim == 1):
        #transform to 2-d
        arr = numpy.array([arr])
        weighted_total = weighted_total if weighted_total is None \
            else [weighted_total]
        ends_at = [ends_at] if ends_at is not None else None

        joint_flag = False

    row, length = arr.shape[0], arr.shape[1]

    m = Model("isotonic_l1")
    q = m.addVars(row, length, lb=(-GRB.INFINITY), vtype=grb_var_type) #row, col indices
    x = m.addVars(row, length, vtype=grb_var_type)

    m.setObjective(quicksum(q), GRB.MINIMIZE)

    for r_idx in range(row):
        if nonnegative:
            m.addConstrs(x[r_idx, i] >= 0 for i in range(length))

        # set up the non-decreasing constraints for x
        m.addConstrs(x[r_idx, i] <= x[r_idx, i+1]  for i in range(length-1))

        # add n constraints for q_i >= arr_i - x_i and n constraints for q_i >= -arr_i + x_i
        m.addConstrs(q[r_idx, i] >= arr[r_idx, i] - x[r_idx, i] for i in range(length))
        m.addConstrs(q[r_idx, i] >= -arr[r_idx, i] + x[r_idx, i] for i in range(length))

        # total weight constraint
        if weighted_total is not None:
            if cumsum:
                expr = (length - 1) * x[r_idx, length - 1] - quicksum([x[r_idx, j] \
                    for j in range(length - 1)])
            else:
                expr = quicksum([x[r_idx, j] for j in range(length)])
            m.addConstr(expr == weighted_total[r_idx])

        if starts_at is not None:
            m.addConstr(x[r_idx, 0] == starts_at) #every region shares the same

        if ends_at is not None:
            m.addConstr(x[r_idx, length-1] == ends_at[r_idx])

    # enforce consistency for sum of sub units == unit
    if joint_flag:
        for i in range(length):
            m.addConstr(x[0, i] == quicksum([ x[sub, i] for sub in range(1, row) ]))

    m.Params.OutputFlag = 0 #disable solver output
    m.optimize()

    if m.status != 2:
        print("Model not OPTIMAL - status: ", m.status)

    # collect x result
    opt_x = numpy.zeros((row, length))
    for r_idx in range(row):
        for i in range(length):
            try:
                opt_x[r_idx, i] = x[r_idx, i].x
            except AttributeError:
                logging.warning("=======x AttributeError=========")
                opt_x[r_idx, i] = numpy.nan


    return opt_x if joint_flag else opt_x.flatten()


def isotonic_l2(arr, var_type="INTEGER", cumsum=True, nonnegative=True, starts_at=None, ends_at=None, weighted_total=None):
    """
        Inputs and Outputs are the same as isotonic_l1()
    """
    joint_flag = True
    grb_var_type = GRB.INTEGER if var_type == "INTEGER" else GRB.CONTINUOUS

    if (isinstance(arr, list)) or (type(arr).__module__ == numpy.__name__ and arr.ndim == 1):
        #transform to 2-d
        arr = numpy.array([arr])
        weighted_total = weighted_total if weighted_total is None else [weighted_total]
        ends_at = [ends_at] if ends_at is not None else None
        joint_flag = False

    row, length = arr.shape[0], arr.shape[1]

    m = Model("isotonic_l2")
    q = m.addVars(row, length, lb=(-GRB.INFINITY), vtype=grb_var_type)
    x = m.addVars(row, length, vtype=grb_var_type)

    m.setObjective(quicksum([q[r, i]*q[r, i] for r in range(row) for i in range(length)]), GRB.MINIMIZE)

    for r_idx in range(row):
        if nonnegative:
            m.addConstrs(x[r_idx, i] >= 0  for i in range(length))

        # set up the non-decreasing constraints for x
        m.addConstrs(x[r_idx, i] <= x[r_idx, i+1]  for i in range(length-1))

        # add n constraints for q_i = arr_i - x_i
        m.addConstrs(q[r_idx, i] == arr[r_idx, i] - x[r_idx, i] for i in range(length))

        # total weight constraint
        if weighted_total is not None:

            if cumsum:
                expr = (length - 1) * x[r_idx, length - 1] - quicksum([x[r_idx, j]\
                    for j in range(length - 1)])
            else:
                expr = quicksum([x[r_idx, j] for j in range(length)])
            m.addConstr(expr == weighted_total[r_idx])

        if starts_at is not None:
            m.addConstr(x[r_idx, 0] == starts_at) #every region shares the same

        if ends_at is not None:
            m.addConstr(x[r_idx, length-1] == ends_at[r_idx])



    # enforce consistency for sum of sub units == unit
    if joint_flag:
        #arr[0, ] is unit, sub units are: arr[1, ] arr[2, ]...
        for i in range(length):
            m.addConstr(x[0, i] == quicksum([ x[sub, i] for sub in range(1, row) ]))



    m.Params.OutputFlag = 0
    m.optimize()

    if m.status != 2:
        print("Model not OPTIMAL - status: ", m.status)

    # collect x result
    opt_x = numpy.zeros((row, length))
    for r_idx in range(row):
        for i in range(length):
            try:
                opt_x[r_idx, i] = x[r_idx, i].x
            except AttributeError:
                logging.warning("=======x AttributeError=========")
                opt_x[r_idx, i] = numpy.nan


    return opt_x if joint_flag else opt_x.flatten()
