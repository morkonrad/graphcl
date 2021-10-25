import dag
import heft
import gantt

import networkx as nx
import matplotlib.axes as mataxs
import matplotlib.pyplot as plt

import os
import numpy as np

from time import time as timer
from enum import Enum
from dag import Device
from dag import Buffer
from dag import Node
from dag import calculate_offload_profile

_DBG_ = False

application_names = {0: 'SVD',
                     1: 'CLYAP',
                     2: 'MEQ',
                     3: 'ABE',
                     4: 'GABE',
                     5: 'BiCG',
                     6: 'TRC'}


class Platforms(Enum):
    PLATFORM_A = 129  # obj_129
    PLATFORM_B = 61  # obj_061
    PLATFORM_C = 119  # obj_119


class MKMDSetup:
    platform: Platforms = Platforms.PLATFORM_A
    matrix_width: int = 0
    matrix_height: matrix_width
    application_name: str
    use_subkernel_mkmd: bool = True
    threshold_offload: float = 1.0
    show_gant_dag: bool = False

    def __init__(self, mkmd_app_name: str,
                 matrix_width: int,
                 offload_threshold: float = 1.0,
                 selected_platform: Platforms = Platforms.PLATFORM_A,
                 use_subkernel_schedule: bool = False,
                 show_gant_dag: bool = False):
        self.platform = selected_platform
        self.matrix_width = matrix_width
        self.matrix_height = self.matrix_width
        self.application_name = mkmd_app_name
        self.use_subkernel_mkmd = use_subkernel_schedule
        self.threshold_offload = offload_threshold
        self.show_gant_dag = show_gant_dag
        return


def update_node_dependencies(inputs: [Buffer], predecessors_old: [Node],
                             key_items: [Node], insertion_map: {Node: Node}) -> ([Buffer], [Node]):
    inputs_new = []
    predecessors_new = []
    """
    update_node_dependencies method uses:
    - replaced_nodes_map  
    - predecessors_old
    to create a new list with new predecessors_after_replace and input buffers after nodes replace
                
    check all predecessors, if any dependent node is replaced by the sub-kernel nodes update his
    inputs and predecessors
    """
    for predecessor in predecessors_old:
        # pn = predecessor.name()
        # print(node_name, pn)
        is_replaced_by_subkernel_nodes = False

        # check if node is replaced by the sub-kernel nodes ?
        for key in key_items:
            if key == predecessor:
                is_replaced_by_subkernel_nodes = True

                # insert a new node dependence and update data-flow
                predecessors_new.append(insertion_map[key])

                for buffer in insertion_map[key].output_buffers():
                    inputs_new.append(buffer)

        if not is_replaced_by_subkernel_nodes:
            # copy old node dependence and data-flow
            predecessors_new.append(predecessor)

            for buffer in predecessor.output_buffers():
                for input in inputs:
                    if buffer.name() == input.name():
                        inputs_new.append(buffer)

    return inputs_new, predecessors_new


def try_insert_subkernel_nodes(dag: [Node], node: Node, start_end_times: (float, float), replaced_nodes: [Node],
                               replaced_nodes_map: {Node: Node}, threshold_offload: float,
                               profile_merge: (bool, [(float, Device)]), idle_slots: {int: [(float, float)]}) -> \
        ({Node: Node}, [Node]):
    node_name = node.name()
    outputs = node.output_buffers()
    predecessors_old = node.predecessor_nodes()

    # update node dependencies and data-flow if predecessor was replaced
    inputs_after_replace, predecessors_after_replace = update_node_dependencies(node.input_buffers(),
                                                                                predecessors_old,
                                                                                replaced_nodes,
                                                                                replaced_nodes_map)

    # create new sub-kernel nodes
    # with updated predecessors and inputs (data-flow)
    if _DBG_:
        print('Check partition of: ', node_name)
    profiling_data = node.node_weights_V()

    sub_kernel_profile, offload_ndr = calculate_offload_profile(profiling_data, threshold_offload,
                                                                idle_slots, start_end_times, node.executor_id())

    is_contiguous_output = node.has_contiguous_output()
    i = 0
    sub_kernel_nodes = []
    for offload in offload_ndr:
        new_outputs = []
        j = 0
        for buffer in outputs:

            if is_contiguous_output:
                sub_ker_size = buffer.size() * offload
                new_outputs.append(Buffer('sk_out' + str(i) + str(j), sub_ker_size))
            else:
                new_outputs.append(Buffer('sk_out' + str(i) + str(j), buffer.size()))
            j += 1

        sub_kernel_node = Node(inputs_after_replace, new_outputs, (is_contiguous_output, sub_kernel_profile),
                               predecessors_after_replace,
                               node_name + '$_{' + str(i) + '}$', 0,
                               sub_kernel_profile[i])
        i += 1
        sub_kernel_nodes.append(sub_kernel_node)

    # create merge_node
    # gather inputs_after_replace
    merge_inputs = []
    for node_sk in sub_kernel_nodes:
        for buffer in node_sk.output_buffers():
            merge_inputs.append(buffer)

    # FIXME: here is a potential BUG -> assumption MISO(multiple input single output)
    #  Only a single output exist, possible MIMO ,each output have different size

    merge_out = Buffer('merge_out', new_outputs[0].size())
    node_name_merge = 'M_' + node_name
    merge_node = Node(merge_inputs, [merge_out], profile_merge,
                      sub_kernel_nodes, node_name_merge)

    node_cost = node.cost()
    costs_sub_kernels = []
    for sk in sub_kernel_nodes:
        costs_sub_kernels.append(sk.cost())

    # max because all nodes execute in parallel
    sub_kernel_exec_cost = max(costs_sub_kernels)
    sub_kernel_exec_cost += merge_node.cost()

    is_node_replaced_by_subnodes = False

    # Check cost of sub_kernel execution if lower than original add sub-kernel and merge nodes
    if sub_kernel_exec_cost < node_cost:
        if _DBG_:
            print('Partitioned node: ', node_name)
        for sk in sub_kernel_nodes:
            dag.append(sk)
        dag.append(merge_node)

        # add on place of old node a merge_node
        replaced_nodes_map[node] = merge_node
        is_node_replaced_by_subnodes = True
    else:
        # print('Not-partitioned node: ', node_name)
        for replaced_node in replaced_nodes:
            for predecessor in node.predecessor_nodes():
                if replaced_node == predecessor:
                    if _DBG_:
                        print("Predecessor after sub_kernel insertion changed, need update!")
                    node.replace_dependency(predecessor, replaced_nodes_map[predecessor])

        dag.append(node)

    return is_node_replaced_by_subnodes


# -------------------------------------------------------------------------------------------
def calculate_dag_subkernel(dag_coarse: [Node],
                            profile_merge: (bool, [(float, Device)]),
                            profile_host: (bool, [(float, Device)]),
                            threshold_offload: float, span_tasks: [Node, float, float],
                            idle_slots: {int: [(float, float)]}) -> [Node]:
    dag_subkernel = []

    if len(dag_coarse) == 0:
        return dag_subkernel

    # this dictionary holds references to new nodes inserted in place of old nodes
    replaced_nodes_map = {}

    # 1) scan input dag_nodes
    for node in dag_coarse:

        node_name = node.name()
        replaced_nodes = list(replaced_nodes_map)

        # for each input node check if sub-kernel execution is profitable
        if node.is_entry():
            # entry node just copy
            dag_subkernel.append(node)
        elif node.is_exit():

            outputs = node.output_buffers()
            predecessors_old = node.predecessor_nodes()

            # exit node, update his dependencies and data-flow if predecessor was replaced
            inputs_after_replace, predecessors_after_replace = update_node_dependencies(node.input_buffers(),
                                                                                        predecessors_old,
                                                                                        replaced_nodes,
                                                                                        replaced_nodes_map)
            # create a new node with updated predecessors and inputs (data-flow)
            exit_node = Node(inputs_after_replace, outputs, profile_host, predecessors_after_replace,
                             node_name, 0, profile_host[1][0])

            dag_subkernel.append(exit_node)
        else:
            # is_on_span = False
            try_insert_subkernel_nodes(dag_subkernel, node, (0, 0),
                                       replaced_nodes, replaced_nodes_map, threshold_offload,
                                       profile_merge, idle_slots)
            is_on_span = True
            '''
            # check if node is in span_tasks list
            for node_span, start_time, end_time in span_tasks:
                if node_span == node:
                    try_insert_subkernel_nodes(dag_subkernel, node, (start_time, end_time),
                                               replaced_nodes, replaced_nodes_map, threshold_offload,
                                               profile_merge, idle_slots)
                    is_on_span = True
                    continue'''
            if not is_on_span:
                if _DBG_:
                    print('Not in span_list', node.name())
                for replaced_node in replaced_nodes:
                    for predecessor in node.predecessor_nodes():
                        if replaced_node == predecessor:
                            if _DBG_:
                                print("Predecessors after sub_kernel insertion changed, need update!")
                            node.replace_dependency(predecessor, replaced_nodes_map[predecessor])
                dag_subkernel.append(node)

    # update node ids
    new_node_ids = 0
    for node in dag_subkernel:
        node.set_id(new_node_ids)
        new_node_ids += 1

    return dag_subkernel


# -------------------------------------------------------------------------------------------
def calculate_dag_SVD_subkernel2(matrix_size: float, profile_empty_host: (bool, [(float, Device)]),
                                 profile_mat_mul: (bool, [(float, Device)]),
                                 profile_mat_transp: (bool, [(float, Device)]),
                                 profile_merge: (bool, [(float, Device)])):
    # memory buffers
    u = Buffer('U', matrix_size)
    e = Buffer('E', matrix_size)
    v = Buffer('V', matrix_size)
    vt = Buffer('Vt', matrix_size)

    ue1 = Buffer('UE1', matrix_size)
    ue2 = Buffer('UE2', matrix_size)
    ueM = Buffer('UEM', matrix_size)

    uevt1 = Buffer('UEVT1', matrix_size)
    uevt2 = Buffer('UEVT2', matrix_size)
    uevt = Buffer('UEVT', matrix_size)

    # nodes
    n0 = Node([], [u, e, v], profile_empty_host, [], '$V_{s}$', 0)
    n1 = Node([u, e], [ue1], profile_mat_mul, [n0], '$UE_{1}$', 1, profile_mat_mul[1][1])
    n2 = Node([u, e], [ue2], profile_mat_mul, [n0], '$UE_{2}$', 2, profile_mat_mul[1][2])
    n3 = Node([ue1, ue2], [ueM], profile_merge, [n1, n2], '$UE_{M}$', 3)

    n4 = Node([v], [vt], profile_mat_transp, [n0], '$V^T$', 4)

    n5 = Node([ueM, vt], [uevt1], profile_mat_mul, [n3, n4], '$UEV^T_{1}$', 5, profile_mat_mul[1][1])
    n6 = Node([ueM, vt], [uevt2], profile_mat_mul, [n3, n4], '$UEV^T_{2}$', 6, profile_mat_mul[1][2])
    n7 = Node([uevt1, uevt2], [uevt], profile_merge, [n5, n6], '$UEV^T_{M}$', 7)

    n8 = Node([uevt], [], profile_empty_host, [n7], '$V_{e}$', 8)

    dag = [n0, n1, n2, n3, n4, n5, n6, n7, n8]

    for node in dag:
        node.summary()

    title_plot = r'APP: ' + 'SVD_SUBK = $\mathbf{UEV}^T$'
    return dag, title_plot


def calculate_dag_SVD_subkernel3(matrix_size: float, profile_empty_host: (bool, [(float, Device)]),
                                 profile_mat_mul: (bool, [(float, Device)]),
                                 profile_mat_transp: (bool, [(float, Device)]),
                                 profile_merge: (bool, [(float, Device)])):
    # memory buffers
    u = Buffer('U', matrix_size)
    e = Buffer('E', matrix_size)
    v = Buffer('V', matrix_size)
    vt = Buffer('Vt', matrix_size)

    ue1 = Buffer('UE1', matrix_size)
    ue2 = Buffer('UE2', matrix_size)
    ue3 = Buffer('UE3', matrix_size)
    ueM = Buffer('UEM', matrix_size)

    uevt1 = Buffer('UEVT1', matrix_size)
    uevt2 = Buffer('UEVT2', matrix_size)
    uevt3 = Buffer('UEVT3', matrix_size)
    uevt = Buffer('UEVT', matrix_size)

    # nodes
    n0 = Node([], [u, e, v], profile_empty_host, [], '$V_{s}$', 0)

    sub_kernel_profile, offload_ndr = calculate_offload_profile(profile_mat_mul[1], 0.01, [], (0, 0), 0)
    subk_prof = (profile_mat_mul[0], sub_kernel_profile)

    n1 = Node([u, e], [ue1], subk_prof, [n0], '$UE_{1}$', 1, sub_kernel_profile[0])
    n2 = Node([u, e], [ue2], subk_prof, [n0], '$UE_{2}$', 2, sub_kernel_profile[1])
    n3 = Node([u, e], [ue3], subk_prof, [n0], '$UE_{3}$', 3, sub_kernel_profile[2])
    n4 = Node([ue1, ue2, ue3], [ueM], profile_merge, [n1, n2, n3], '$UE_{M}$', 4)

    n5 = Node([v], [vt], profile_mat_transp, [n0], '$V^T$', 5)

    n6 = Node([ueM, vt], [uevt1], subk_prof, [n4, n5], '$UEV^T_{1}$', 6, sub_kernel_profile[0])
    n7 = Node([ueM, vt], [uevt2], subk_prof, [n4, n5], '$UEV^T_{2}$', 7, sub_kernel_profile[1])
    n8 = Node([ueM, vt], [uevt3], subk_prof, [n4, n5], '$UEV^T_{3}$', 8, sub_kernel_profile[2])
    n9 = Node([uevt1, uevt2, uevt3], [uevt], profile_merge, [n6, n7, n8], '$UEV^T_{M}$', 9)

    n10 = Node([uevt], [], profile_empty_host, [n9], '$V_{e}$', 10)

    dag = [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10]

    for node in dag:
        node.summary()

    title_plot = r'APP: ' + 'SVD_SUBK = $\mathbf{UEV}^T$'
    return dag, title_plot


def calculate_dag_SVD(matrix_size: float, profile_empty_host: (bool, [(float, Device)]),
                      profile_mat_mul: (bool, [(float, Device)]), profile_mat_transp: (bool, [(float, Device)])) -> (
        [Node], str):
    # memory buffers
    u = Buffer('U', matrix_size)
    e = Buffer('E', matrix_size)
    v = Buffer('V', matrix_size)
    vt = Buffer('Vt', matrix_size)
    ue = Buffer('UE', matrix_size)
    uevt = Buffer('UE', matrix_size)

    # nodes
    n0 = Node([], [u, e, v], profile_empty_host, [], '$V_{s}$', 0)
    n1 = Node([u, e], [ue], profile_mat_mul, [n0], '$UE$', 1)
    n2 = Node([v], [vt], profile_mat_transp, [n0], '$V^T$', 2)
    n3 = Node([ue, vt], [uevt], profile_mat_mul, [n1, n2], '$UEV^T$', 3)
    n4 = Node([uevt], [], profile_empty_host, [n3], '$V_{e}$', 4)

    title_plot = r'SVD = $\mathbf{UEV}^T$'

    dag = [n0, n1, n2, n3, n4]
    # overall_in_order_time = 0.0
    for node in dag:
        node.summary()
        # overall_in_order_time += node.cost()
    # print('In-Order estimated time: ', overall_in_order_time)
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

    for node in dag:
        node.summary()

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

    for node in dag:
        node.summary()

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

    for node in dag:
        node.summary()

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
    atxe = Buffer('At', matrix_size)

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

    for node in dag:
        node.summary()

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
    rkn = Buffer('rk+1', vector_size)  # rk+1

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

    for node in dag:
        node.summary()

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

    for node in dag:
        node.summary()

    title_plot = r'APP: ' + 'TRC = $\mathbf{ABC}+\mathbf{BCA}+\mathbf{CAB}-\mathbf{BAC}-\mathbf{ACB}-\mathbf{CBA}$'
    return dag, title_plot


def calculate_HEFT_schedule(dag_list: [], map_processors: {}, avoid: bool = False, print_duration: bool = True) -> (
        {int: heft.ScheduleEvent}, nx.DiGraph, float, float):
    file_name_node_matrix = '/tmp/nv.csv'
    file_name_edge_matrix = '/tmp/ne.csv'
    file_name_bw_resources = '/tmp/bw.csv'
    # print('OS name:\t', os.name)
    if os.name == "nt":
        path_tmp = os.environ['TMP']
        # print(path_tmp)
        file_name_node_matrix = path_tmp + '\\nv.csv'
        file_name_edge_matrix = path_tmp + '\\ne.csv'
        file_name_bw_resources = path_tmp + '\\bw.csv'

    dag.build_csv_files(dag_list, map_processors, file_name_node_matrix, file_name_edge_matrix, file_name_bw_resources)

    # call HEFT.py
    comp_matrix_V = heft.readCsvToNumpyMatrix(file_name_node_matrix)
    comm_matrix_BW = heft.readCsvToNumpyMatrix(file_name_bw_resources)
    dag_di_graph = heft.readDagMatrix(file_name_edge_matrix)

    start = timer()
    schedule = {}
    if not avoid:
        schedule, _, _ = heft.schedule_dag(dag_di_graph, communication_matrix=comm_matrix_BW,
                                           computation_matrix=comp_matrix_V)  # , rank_metric=RankMetric.BEST)
    end = timer()
    schedule_duaration = round((end - start) * 1e3, 2)

    cnt_nodes = len(dag_list)
    if print_duration:
        print('-----')
        print('DAG nodes:', cnt_nodes)
        print(f"Calculate schedule took {schedule_duaration} msec!")

    return schedule, dag_di_graph, cnt_nodes, schedule_duaration


def visualize_Graph(dag_list: [], dag_MAT: nx.DiGraph, title_plot: str, ax: mataxs.Axes = None):
    labeldict = dag.get_labels_dict(dag_list)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        plt.subplot(ax)
        ax.set_title(title_plot)
        nx.draw(dag_MAT, pos=nx.nx_pydot.graphviz_layout(dag_MAT, prog='dot'),
                with_labels=True, labels=labeldict, arrows=False)

        # fig.savefig('dag.png', dpi=500)
        plt.show()
        return

    ax.set_title(title_plot)
    nx.draw(dag_MAT, pos=nx.nx_pydot.graphviz_layout(dag_MAT, prog='dot'),
            with_labels=True, labels=labeldict, arrows=False)
    return


def visualize_Gantt(dag_list: [], schedule: {}, processor_names: [str], title_plot: str, ax: mataxs.Axes = None):
    # -------------------------------------------------
    # Visualize
    labeldict = dag.get_labels_dict(dag_list)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 9))
        plt.subplot(ax)
        ax.set_title(title_plot)

        plt.title(title_plot, fontsize='small')
        gantt.showGanttChart(schedule, labeldict, processor_names, ax)

        # fig.savefig('gantt.png', dpi=500)
        plt.show()
        return

    plt.title(title_plot, fontsize='small')
    gantt.showGanttChart(schedule, labeldict, processor_names, ax)
    return


def visualize(dag_list: [], schedule: {}, dag_MAT: nx.DiGraph, processor_names: [str],
              title_plot: str, subplot: bool = False):
    if subplot:
        fig, ax = plt.subplots(2, 1, figsize=(15, 10))
        visualize_Gantt(dag_list, schedule, processor_names, title_plot, ax[0])
        visualize_Graph(dag_list, dag_MAT, title_plot, ax[1])
        # fig.savefig('graphcl.png', dpi=500)
        return

    visualize_Gantt(dag_list, schedule, processor_names, title_plot)
    visualize_Graph(dag_list, dag_MAT, title_plot)
    return

def merge_two_dicts(x, y):
    z = x.copy()  # start with keys and values of x
    z.update(y)  # modifies z with keys and values of y
    return z


def setup_workload_profiles(matrix_width: float, platform: Platforms):
    # select platform

    if platform == Platforms.PLATFORM_A:
        cpu = dag.Device('CPU 6134G', 100e9, 0)
        gpu1 = dag.Device('GPU WX-7100', 12e9, 1)  # PCIE-x16
        gpu2 = dag.Device('GPU R9-290', 12e9, 2)  # PCIE-x16

        map_processors = {0: cpu, 1: gpu1, 2: gpu2}
        processor_names = [cpu.name(), gpu1.name(), gpu2.name()]
        profile_host = (True, [(0, cpu), (0, gpu1), (0, gpu2)])

        if matrix_width == 2048:  # MM 15,18
            profile_MM = (True, [(122, cpu), (15, gpu1), (18, gpu2)])
            profile_ME = (True, [(1.53, cpu), (0.5, gpu1), (0.6, gpu2)])  # matrix merge
            profile_MT = (False, [(2.6, cpu), (1.5, gpu1), (2.5, gpu2)])
            profile_MADD = (True, [(1.4, cpu), (1.5, gpu1), (1.6, gpu2)])
            profile_vec_vec = (True, [(0.13, cpu), (0.21, gpu1), (0.23, gpu2)])
            profile_mat_vec = (True, [(0.53, cpu), (0.65, gpu1), (0.67, gpu2)])

        elif matrix_width == 4096:
            profile_MM = (True, [(1996, cpu), (126, gpu1), (156, gpu2)])
            profile_ME = (True, [(7.5, cpu), (1.47, gpu1), (1.16, gpu2)])  # matrix merge
            profile_MT = (False, [(10.6, cpu), (25.3, gpu1), (15.1, gpu2)])
            profile_MADD = (True, [(15.4, cpu), (4.0, gpu1), (3.6, gpu2)])
            profile_vec_vec = (True, [(0.13, cpu), (0.21, gpu1), (0.23, gpu2)])
            profile_mat_vec = (True, [(4.7, cpu), (2.45, gpu1), (2.27, gpu2)])

    if platform == Platforms.PLATFORM_B:
        cpu = dag.Device('CPU i7-4930k', 100e9, 0)
        gpu2 = dag.Device('GPU 1080', 12e9, 1)  # PCIE-x16
        gpu3 = dag.Device('GPU TitanX', 12e9, 2)  # PCIE-x16
        gpu1 = dag.Device('GPU GTX780TI', 6e9, 3)  # PCIE-x8

        map_processors = {0: cpu, 1: gpu1, 2: gpu2, 3: gpu3}
        processor_names = [cpu.name(), gpu1.name(), gpu2.name(), gpu3.name()]

        profile_host = (True, [(0, cpu), (0, gpu1), (0, gpu2), (0, gpu3)])

        if matrix_width == 2048:

            profile_MM = (True, [(590, cpu), (51.5, gpu1), (19, gpu2), (23.7, gpu3)])
            profile_MT = (True, [(3.2, cpu), (0.18, gpu1), (0.14, gpu2), (0.14, gpu3)])

            profile_MADD = (True, [(0.27, cpu), (0.28, gpu1), (0.95, gpu2), (0.9, gpu3)])
            profile_ME = (True, [(1.7, cpu), (0.2, gpu1), (0.21, gpu2), (0.19, gpu3)])  # matrix merge

            profile_mat_vec = (True, [(4.9, cpu), (0.13, gpu1), (0.09, gpu2), (0.12, gpu3)])
            profile_vec_vec = (True, [(0.3, cpu), (0.1, gpu1), (0.05, gpu2), (0.08, gpu3)])

        elif matrix_width == 4096:

            profile_MM = (True, [(4986, cpu), (380, gpu1), (139, gpu2), (187, gpu3)])
            profile_MT = (True, [(11.8, cpu), (0.71, gpu1), (0.59, gpu2), (0.58, gpu3)])

            profile_MADD = (True, [(16.3, cpu), (1.8, gpu1), (3.6, gpu2), (3.5, gpu3)])
            profile_ME = (True, [(6.5, cpu), (0.79, gpu1), (0.83, gpu2), (0.78, gpu3)])  # matrix merge

            profile_mat_vec = (True, [(19.5, cpu), (0.35, gpu1), (0.28, gpu2), (0.34, gpu3)])
            profile_vec_vec = (True, [(1.1, cpu), (0.23, gpu1), (0.097, gpu2), (0.15, gpu3)])

    if platform == Platforms.PLATFORM_C:
        cpu = dag.Device('CPU i9-7980XE', 100e9, 0)
        gpu1 = dag.Device('GPU 1080TI', 12e9, 1)  # PCIE-x16
        gpu2 = dag.Device('GPU 1080', 6e9, 2)  # PCIE-x8
        gpu3 = dag.Device('GPU TitanX', 6e9, 3)  # PCIE-x8
        map_processors = {0: cpu, 1: gpu1, 2: gpu2, 3: gpu3}
        processor_names = [cpu.name(), gpu1.name(), gpu2.name(), gpu3.name()]

        profile_host = (True, [(0, cpu), (0, gpu1), (0, gpu2), (0, gpu3)])

        if matrix_width == 2048:
            """
            --------------------------------------------------------------------
            MM 2048x2048
                        :	time[ms]:	POWER[Watt]
            CPU		    :	110		: 	130
            GTX-1080TI	:	13.5	:	200
            GTX-1080	:	19		:	151
            GTX-TITANX	:	23.7	:	210

            --------------------------------------------------------------------
            MM 2048x2048
                        :	time[ms]:	POWER[Watt]
            CPU		    :	0.7		: 	120
            GTX-1080TI	:	0.096	:	176
            GTX-1080	:	0.14	:	118
            GTX-TITANX	:	0.14	:	178
            """

            profile_MM = (True, [(110, cpu), (13.5, gpu1), (19, gpu2), (23.7, gpu3)])
            profile_MT = (True, [(0.7, cpu), (0.09, gpu1), (0.14, gpu2), (0.14, gpu3)])

            profile_MADD = (True, [(1.6, cpu), (1.1, gpu1), (0.95, gpu2), (0.9, gpu3)])
            profile_ME = (True, [(0.5, cpu), (0.14, gpu1), (0.21, gpu2), (0.19, gpu3)])  # matrix merge

            profile_mat_vec = (True, [(0.74, cpu), (0.08, gpu1), (0.09, gpu2), (0.12, gpu3)])
            profile_vec_vec = (True, [(0.18, cpu), (0.05, gpu1), (0.05, gpu2), (0.08, gpu3)])

        elif matrix_width == 4096:
            """
            --------------------------------------------------------------------
            MM 4096x4096
                        :	time[ms]:	POWER[Watt]
            CPU		    :	1495	: 	130
            GTX-1080TI	:	100		:	230
            GTX-1080	:	139		:	160
            GTX-TITANX	:	187		:	199

            --------------------------------------------------------------------
            MT 4096x4096
                        :	time[ms]:	POWER[Watt]
            CPU		    :	4.8 	: 	141
            GTX-1080TI	:	0.39	:	78
            GTX-1080	:	0.59	:	123
            GTX-TITANX	:	0.58	:	184
            """

            profile_MM = (True, [(1495, cpu), (100, gpu1), (139, gpu2), (187, gpu3)])
            profile_MT = (True, [(4.8, cpu), (0.39, gpu1), (0.59, gpu2), (0.58, gpu3)])

            profile_MADD = (True, [(8.3, cpu), (3.8, gpu1), (3.6, gpu2), (3.5, gpu3)])
            profile_ME = (True, [(2.7, cpu), (0.56, gpu1), (0.83, gpu2), (0.78, gpu3)])  # matrix merge

            profile_mat_vec = (True, [(4.74, cpu), (0.21, gpu1), (0.28, gpu2), (0.34, gpu3)])
            profile_vec_vec = (True, [(0.32, cpu), (0.1, gpu1), (0.097, gpu2), (0.15, gpu3)])

    return profile_ME, profile_host, profile_MM, profile_MT, \
           profile_mat_vec, profile_vec_vec, profile_MADD, map_processors, processor_names


def call_mkmd(setup: MKMDSetup) -> (np.ndarray, np.ndarray, float, float):
    # Platform specific params:
    # input matrix size (WxH) or vector (H,1)
    matrix_width = setup.matrix_width
    matrix_height = setup.matrix_height
    matrix_size = matrix_width * matrix_height * 4  # size in bytes
    vector_size = matrix_height * 4  # size in bytes

    profile_ME, profile_host, profile_MM, profile_MT, \
    profile_mat_vec, profile_vec_vec, profile_MADD, \
    map_processors, processor_names = setup_workload_profiles(matrix_width, setup.platform)

    if setup.application_name == "SVD":
        dag_list, title_plot = calculate_dag_SVD(matrix_size, profile_host, profile_MM, profile_MT)
        # dag_list, title_plot = mkmd.calculate_dag_SVD_subkernel3(matrix_size, profile_host, profile_MM, profile_MT,
        # profile_ME)
    elif setup.application_name == "CLYAP":
        dag_list, title_plot = calculate_dag_CLYAP(matrix_size, profile_host, profile_MM, profile_MT, profile_MADD)
    elif setup.application_name == "MEQ":
        dag_list, title_plot = calculate_dag_MEQ(matrix_size, profile_host, profile_MM, profile_MT)
    elif setup.application_name == "ABE":
        dag_list, title_plot = calculate_dag_ABE(matrix_size, profile_host, profile_MM, profile_MT, profile_MADD)
    elif setup.application_name == "GABE":
        dag_list, title_plot = calculate_dag_GABE(matrix_size, profile_host, profile_MM, profile_MT, profile_MADD)
    elif setup.application_name == "BiCG":
        dag_list, title_plot = calculate_dag_BICG(matrix_size, vector_size, profile_host, profile_mat_vec,
                                                  profile_vec_vec, profile_MADD)
    elif setup.application_name == "TRC":
        dag_list, title_plot = calculate_dag_TRC(matrix_size, profile_host, profile_MM, profile_MADD)

    map_in_order_times = dag.calculate_dag_execution_times(dag_list, map_processors)

    '''overall_in_order_time = 0.0
    for node in dag_list:
        # node.summary()
        overall_in_order_time += node.cost()
    '''
    # Calculate coarse_schedule
    schedule, dag_di_graph, cnt_nodes, schedule_duration = \
        calculate_HEFT_schedule(dag_list, map_processors, False, False)

    end_time_coarse = dag.print_find_end_times(schedule, processor_names)

    if setup.show_gant_dag:
        visualize(dag_list, schedule, dag_di_graph, processor_names, title_plot, False)

    span_tasks = []
    idle_slots = []

    end_time_fine = 0
    if setup.use_subkernel_mkmd is True:

        dag_list_sk = calculate_dag_subkernel(dag_list, profile_ME, profile_host,
                                              setup.threshold_offload, span_tasks, idle_slots)
        if len(dag_list_sk) == len(dag_list):
            print("----------------------", setup.application_name, "----------------------")
            print("!!! Sub kernel execution not profitable !!!")
            print("--------------------------------------------")
        else:
            schedule, dag_di_graph, cnt_nodes, schedule_duration = \
                calculate_HEFT_schedule(dag_list_sk, map_processors, False, False)

            end_time_fine = dag.print_find_end_times(schedule, processor_names)
            if setup.show_gant_dag:
                visualize(dag_list_sk, schedule, dag_di_graph, processor_names, title_plot, False)

    map_speedup = {}
    map_in_order_exec_processors = {}
    speedups = np.fromiter(map_speedup.values(), dtype=float)
    in_order_exe = np.fromiter(map_in_order_exec_processors.values(), dtype=float)

    return speedups, in_order_exe, cnt_nodes, schedule_duration
