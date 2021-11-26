from dag import Device
from dag import Buffer
from dag import Node


# -------------------------------------------------------------------------------------------
def calculate_dag_SVD(matrix_size: float, profile_empty_host: (bool, [(float, Device)]),
                      profile_mat_mul: (bool, [(float, Device)]), profile_mat_transp: (bool, [(float, Device)])) -> (
        [Node], str):
    # memory buffers
    u = Buffer('U', matrix_size)
    e = Buffer('E', matrix_size)
    v = Buffer('V', matrix_size)
    vt = Buffer('Vt', matrix_size)
    ue = Buffer('UE', matrix_size)
    uevt = Buffer('UEVt', matrix_size)

    # nodes
    n0 = Node([], [u, e, v], profile_empty_host, [], '$V_{s}$', 0)
    n1 = Node([u, e], [ue], profile_mat_mul, [n0], '$UE$', 1)
    n2 = Node([v], [vt], profile_mat_transp, [n0], '$V^T$', 2)
    n3 = Node([ue, vt], [uevt], profile_mat_mul, [n1, n2], '$UEV^T$', 3)
    n4 = Node([uevt], [], profile_empty_host, [n3], '$V_{e}$', 4)

    title_plot = r'SVD = $\mathbf{UEV}^T$'

    dag = [n0, n1, n2, n3, n4]

    overall_in_order_time = 0.0
    for node in dag:
        node.summary()
        overall_in_order_time += node.cost()
    print('In-Order estimated time: ', overall_in_order_time)

    return dag, title_plot


# -------------------------------------------------------------------------------------------
def calculate_dag_CLYAP(matrix_size: float, profile_empty_host: [], profile_mat_mul: (bool, [(float, Device)]),
                        profile_mat_transp: (bool, [(float, Device)]),
                        profile_mat_add: (bool, [(float, Device)])) -> ([Node], str):
    # memory buffers
    a = Buffer('A', matrix_size)
    x = Buffer('X', matrix_size)
    ax = Buffer('AX', matrix_size)
    at = Buffer('At', matrix_size)
    xat = Buffer('XAt', matrix_size)
    q = Buffer('Q', matrix_size)
    axxatq = Buffer('EQ', matrix_size)

    # nodes
    n0 = Node([], [a, x, q], profile_empty_host, [], '$V_{s}$', 0)
    n1 = Node([a, x], [ax], profile_mat_mul, [n0], '$AX$', 1)
    n2 = Node([a], [at], profile_mat_transp, [n0], '$A^T$', 2)
    n3 = Node([x, at], [xat], profile_mat_mul, [n0, n2], '$XA^T$', 3)
    n4 = Node([xat, ax, q], [axxatq], profile_mat_add, [n0, n1, n3], '$EQ$', 4)
    n5 = Node([axxatq], [], profile_empty_host, [n4], '$V_{e}$', 5)

    dag = [n0, n1, n2, n3, n4, n5]

    overall_in_order_time = 0.0
    for node in dag:
        node.summary()
        overall_in_order_time += node.cost()
    print('In-Order estimated time: ', overall_in_order_time)

    title_plot = r'APP: ' + 'CLYAP = $\mathbf{AX}+\mathbf{XA}^T+\mathbf{Q}$'
    return dag, title_plot


# -------------------------------------------------------------------------------------------
def calculate_dag_MEQ(matrix_size: float, profile_empty_host: (bool, [(float, Device)]),
                      profile_mat_mul: (bool, [(float, Device)]), profile_mat_transp: (bool, [(float, Device)])) \
        -> ([Node], str):
    # memory buffers
    a = Buffer('A', matrix_size)
    b = Buffer('B', matrix_size)
    c = Buffer('C', matrix_size)
    a2 = Buffer('A2', matrix_size)
    bt = Buffer('Bt', matrix_size)
    cb = Buffer('CB', matrix_size)
    bbt = Buffer('BBt', matrix_size)
    a2bbt = Buffer('A2BBt', matrix_size)
    a2bbtcb = Buffer('EQ', matrix_size)

    # nodes
    n0 = Node([], [a, b, c], profile_empty_host, [], '$V_{s}$', 0)
    n1 = Node([a], [a2], profile_mat_mul, [n0], '$A^2$', 1)
    n2 = Node([b], [bt], profile_mat_transp, [n0], '$B^T$', 2)
    n3 = Node([c, b], [cb], profile_mat_mul, [n0], '$CB$', 3)
    n4 = Node([bt, b], [bbt], profile_mat_mul, [n0, n2], '$BB^T$', 4)
    n5 = Node([a2, bbt], [a2bbt], profile_mat_mul, [n1, n4], '$A^2BB^T$', 5)
    n6 = Node([a2bbt, cb], [a2bbtcb], profile_mat_mul, [n3, n5], 'EQ', 6)
    n7 = Node([a2bbtcb], [], profile_empty_host, [n6], '$V_{e}$', 7)

    dag = [n0, n1, n2, n3, n4, n5, n6, n7]

    overall_in_order_time = 0.0
    for node in dag:
        node.summary()
        overall_in_order_time += node.cost()
    print('In-Order estimated time: ', overall_in_order_time)

    title_plot = r'APP: ' + 'MEQ = $\mathbf{A}^2*\mathbf{BB}^T*\mathbf{CB}$'
    return dag, title_plot


# -------------------------------------------------------------------------------------------
def calculate_dag_ABE(matrix_size: float, profile_empty_host: (bool, [(float, Device)]),
                      profile_mat_mul: (bool, [(float, Device)]), profile_mat_transp: (bool, [(float, Device)]),
                      profile_mat_add: (bool, [(float, Device)])) -> ([Node], str):
    # memory buffers
    a = Buffer('A', matrix_size)
    x = Buffer('X', matrix_size)
    ax = Buffer('AX', matrix_size)
    b = Buffer('B', matrix_size)
    bt = Buffer('Bt', matrix_size)
    at = Buffer('At', matrix_size)
    xat = Buffer('XAt', matrix_size)
    xb = Buffer('XB', matrix_size)
    xbt = Buffer('XBt', matrix_size)
    xbxbt = Buffer('XBXBt', matrix_size)
    atxxaxbxbt = Buffer('EQ', matrix_size)

    # nodes
    n0 = Node([], [a, x, b], profile_empty_host, [], '$V_{s}$', 0)
    n1 = Node([a], [at], profile_mat_transp, [n0], '$A^T$', 1)
    n2 = Node([a, x], [ax], profile_mat_mul, [n0], '$AX$', 2)
    n3 = Node([b], [bt], profile_mat_transp, [n0], '$B^T$', 3)
    n4 = Node([x, at], [xat], profile_mat_mul, [n0, n1], '$XA^T$', 4)
    n5 = Node([bt, x], [xbt], profile_mat_mul, [n0, n3], '$XB^T$', 5)
    n6 = Node([b, x], [xb], profile_mat_mul, [n0], '$XB$', 6)
    n7 = Node([xb, xbt], [xbxbt], profile_mat_mul, [n5, n6], '$XBXB^T$', 7)
    n8 = Node([ax, xat, xbxbt], [atxxaxbxbt], profile_mat_add, [n2, n4, n7], 'EQ', 8)
    n9 = Node([atxxaxbxbt], [], profile_empty_host, [n8], '$V_{e}$', 9)

    dag = [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9]

    overall_in_order_time = 0.0
    for node in dag:
        node.summary()
        overall_in_order_time += node.cost()
    print('In-Order estimated time: ', overall_in_order_time)

    title_plot = r'APP: ' + 'ABE = $\mathbf{A}^T\mathbf{X}+\mathbf{XA}-\mathbf{XB}*\mathbf{B}^T\mathbf{X}$'
    return dag, title_plot


# -------------------------------------------------------------------------------------------
def calculate_dag_GABE(matrix_size: float, profile_empty_host: (bool, [(float, Device)]),
                       profile_mat_mul: (bool, [(float, Device)]), profile_mat_transp: (bool, [(float, Device)]),
                       profile_mat_add: (bool, [(float, Device)])) -> ([Node], str):
    # memory buffers
    a = Buffer('A', matrix_size)
    x = Buffer('X', matrix_size)
    g = Buffer('G', matrix_size)
    e = Buffer('E', matrix_size)

    at = Buffer('At', matrix_size)
    xe = Buffer('XE', matrix_size)
    atxe = Buffer('AtXE', matrix_size)

    et = Buffer('Et', matrix_size)
    xa = Buffer('XA', matrix_size)
    etxa = Buffer('EtXA', matrix_size)

    xg = Buffer('XG', matrix_size)
    etxg = Buffer('EtXG', matrix_size)
    etxgxe = Buffer('EtXGXE', matrix_size)

    eq_add = Buffer('EQ', matrix_size)

    # nodes
    n0 = Node([], [a, x, g, e], profile_empty_host, [], '$V_{s}$', 0)
    n1 = Node([x, e], [xe], profile_mat_mul, [n0], 'XE', 1)
    n2 = Node([a], [at], profile_mat_transp, [n0], '$A^T$', 2)
    n3 = Node([x, a], [xa], profile_mat_mul, [n0], 'XA', 3)
    n4 = Node([e], [et], profile_mat_transp, [n0], '$E^T$', 4)
    n5 = Node([x, g], [xg], profile_mat_mul, [n0], '$XG$', 5)
    n6 = Node([at, xe], [atxe], profile_mat_mul, [n2, n1], '$A^TXE$', 6)
    n7 = Node([et, xa], [etxa], profile_mat_mul, [n4, n3], '$E^TXA$', 7)
    n8 = Node([et, xg], [etxg], profile_mat_mul, [n4, n5], '$E^TXG$', 8)
    n9 = Node([etxg, xe], [etxgxe], profile_mat_mul, [n8, n1], '$E^TXGXE$', 9)
    n10 = Node([etxgxe, etxa, atxe], [eq_add], profile_mat_add, [n9, n7, n6], 'EQ', 10)
    n11 = Node([eq_add], [], profile_empty_host, [n10], '$V_{e}$', 11)

    dag = [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11]

    overall_in_order_time = 0.0
    for node in dag:
        node.summary()
        overall_in_order_time += node.cost()
    print('In-Order estimated time: ', overall_in_order_time)

    title_plot = r'APP: ' + 'GABE = $\mathbf{A}^T\mathbf{XE}+\mathbf{E}^T\mathbf{XA}-\mathbf{E}^T\mathbf{XG}*\mathbf{XE}$'
    return dag, title_plot


# -------------------------------------------------------------------------------------------
def calculate_dag_BICG(matrix_size: float, vector_size: float, profile_empty_host: (bool, [(float, Device)]),
                       profile_mat_vect: (bool, [(float, Device)]), profile_vector_vect: (bool, [(float, Device)]),
                       profile_mat_add: (bool, [(float, Device)])) -> ([Node], str):
    # memory buffers
    a = Buffer('A', matrix_size)
    b = Buffer('b', vector_size)
    xk = Buffer('xk', vector_size)
    pk = Buffer('pk', vector_size)

    axk = Buffer('Axk', vector_size)
    xka = Buffer('xkA', vector_size)
    apk = Buffer('Apk', vector_size)

    vk = Buffer('vk', vector_size)  # vk = b-xkA
    rk = Buffer('rk', vector_size)  # rk = b-Axk
    pkapk = Buffer('pkApk', vector_size)
    vkrk = Buffer('vkrk', vector_size)
    lk = Buffer('lk', vector_size)  # lk = rk*vk/rkArk
    lkpk = Buffer('lkpk', vector_size)
    lkapk = Buffer('lkApk', vector_size)
    rkn = Buffer('rkn', vector_size)  # rk+1

    # nodes
    n0 = Node([], [a, xk, b, pk], profile_empty_host, [], '$V_{s}$', 0)
    n1 = Node([a, xk], [axk], profile_mat_vect, [n0], '$\mathbf{A}x_{k}$', 1)
    n2 = Node([xk, a], [xka], profile_mat_vect, [n0], '$x_{k}\mathbf{A}$', 2)
    n3 = Node([a, pk], [apk], profile_mat_vect, [n0], '$\mathbf{A}p_{k}$', 3)

    n4 = Node([b, xka], [vk], profile_mat_add, [n0, n2], '$b-x_{k}\mathbf{A}$', 4)
    n5 = Node([b, axk], [rk], profile_mat_add, [n0, n1], '$b-\mathbf{A}x_{k}$', 5)
    n6 = Node([pk, apk], [pkapk], profile_vector_vect, [n0, n3], '$p_{k}\mathbf{A}p_{k}$', 6)
    n7 = Node([vk, rk], [vkrk], profile_vector_vect, [n4, n5], '$v_{k}r_{k}$', 7)
    n8 = Node([vkrk, pkapk], [lk], profile_vector_vect, [n7, n6], '$v_{k}r_{k} / p_{k}\mathbf{A}p_{k}$', 8)
    n9 = Node([pk, lk], [lkpk], profile_vector_vect, [n0, n8], '$l_{k}p_{k}$', 9)
    n10 = Node([lk, apk], [lkapk], profile_vector_vect, [n8, n3], '$l_{k}\mathbf{A}p_{k}$', 10)

    n11 = Node([rk, lkapk], [rkn], profile_mat_add, [n5, n10], '$r_{k}-l_{k}\mathbf{A}p_{k}$', 11)
    n12 = Node([rkn, lkpk], [], profile_empty_host, [n11, n9], 'Ve', 12)

    dag = [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12]

    overall_in_order_time = 0.0
    for node in dag:
        node.summary()
        overall_in_order_time += node.cost()
    print('In-Order estimated time: ', overall_in_order_time)

    title_plot = r'APP: ' + 'BICG'
    return dag, title_plot


# -------------------------------------------------------------------------------------------
def calculate_dag_TRC(matrix_size: float, profile_empty_host: (bool, [(float, Device)]),
                      profile_mat_mul: (bool, [(float, Device)]),
                      profile_mat_add: (bool, [(float, Device)])) -> ([Node], str):
    # memory buffers
    a = Buffer('A', matrix_size)
    b = Buffer('B', matrix_size)
    c = Buffer('C', matrix_size)

    ab = Buffer('AB', matrix_size)
    ca = Buffer('CA', matrix_size)
    ac = Buffer('AC', matrix_size)
    cb = Buffer('CB', matrix_size)

    abc = Buffer('ABC', matrix_size)
    cab = Buffer('CAB', matrix_size)
    bca = Buffer('BCA', matrix_size)
    bac = Buffer('BAC', matrix_size)
    cba = Buffer('CBA', matrix_size)
    acb = Buffer('ACB', matrix_size)

    eq_add1 = Buffer('ABC_CAB', matrix_size)
    eq_add2 = Buffer('EQ_BCA', matrix_size)
    eq_add3 = Buffer('EQ_BAC', matrix_size)
    eq_add4 = Buffer('EQ_CBA', matrix_size)
    eq_add5 = Buffer('EQ_ACB', matrix_size)

    # nodes
    n0 = Node([], [a, b, c], profile_empty_host, [], '$V_{s}$', 0)
    n1 = Node([a, b], [ab], profile_mat_mul, [n0], '$AB$', 1)
    n2 = Node([c, a], [ca], profile_mat_mul, [n0], '$CA$', 2)
    n3 = Node([a, c], [ac], profile_mat_mul, [n0], '$AC$', 3)
    n4 = Node([c, b], [cb], profile_mat_mul, [n0], '$CB$', 4)
    n5 = Node([ab, c], [abc], profile_mat_mul, [n0, n1], '$ABC$', 5)
    n6 = Node([c, ab], [cab], profile_mat_mul, [n0, n1], '$CAB$', 6)
    n7 = Node([b, ca], [bca], profile_mat_mul, [n0, n2], '$BCA$', 7)
    n8 = Node([b, ac], [bac], profile_mat_mul, [n0, n3], '$BAC$', 8)
    n9 = Node([cb, a], [cba], profile_mat_mul, [n0, n4], '$CBA$', 9)
    n10 = Node([a, cb], [acb], profile_mat_mul, [n0, n4], '$ACB$', 10)
    n11 = Node([abc, cab], [eq_add1], profile_mat_add, [n5, n6], '$ABC+CAB$', 11)
    n12 = Node([eq_add1, bca], [eq_add2], profile_mat_add, [n7, n11], '$+BCA$', 12)
    n13 = Node([eq_add2, bac], [eq_add3], profile_mat_add, [n8, n12], '$+BAC$', 13)
    n14 = Node([eq_add3, cba], [eq_add4], profile_mat_add, [n9, n13], '$+CBA$', 14)
    n15 = Node([eq_add4, acb], [eq_add5], profile_mat_add, [n10, n14], '$+ACB$', 15)
    n16 = Node([eq_add5], [], profile_empty_host, [n15], '$V_{e}$', 16)

    dag = [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16]

    overall_in_order_time = 0.0
    for node in dag:
        node.summary()
        overall_in_order_time += node.cost()
    print('In-Order estimated time: ', overall_in_order_time)

    title_plot = r'APP: ' + 'TRC = $\mathbf{ABC}+\mathbf{BCA}+\mathbf{CAB}-\mathbf{BAC}-\mathbf{ACB}-\mathbf{CBA}$'
    return dag, title_plot
